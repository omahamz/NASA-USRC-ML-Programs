"""
Check the constraints of a configuration against real SolidWorks rebuilds.

The constraints in constraints.py are derived by hand; the rebuild results of SwGen are ground truth: a design whose
row has status ok / warning is buildable, one whose rebuild failed is not.  This tool

  points    writes a random sample of the design box, with NO constraint filtering, for SwGen to try
  compare   reads SwGen's swgen_results.jsonl and reports how well "the constraints accept it" predicts
            "SolidWorks builds it" - for the whole set and for each constraint:

              violated by built rows    the constraint rejects designs that do build (too strict, or wrong)
              satisfied by failed rows  designs that do not build still pass it (fine alone if another
                                        constraint catches them)
              only violation of failed  failed rows that just this constraint rejects (it earns its place)

Workflow (also the way to check the constraints of a new configuration, e.g. T = 1, once they are derived):

    python src/validate_constraints.py points --n 120 --out validate/points.csv
    SwGen.exe generate --part N0AShell.SLDPRT --csv validate/points.csv --out validate/out
    python src/validate_constraints.py compare validate/out/swgen_results.jsonl --config 0:0 --OD 40 --L 50 --G 3
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from constraints import PARAM_BOUNDS, PARAM_COLS, constraints_for, evaluate, parse_configs

BUILT_STATUSES = ("ok", "warning")


def random_box_points(n: int, seed: int | None = None) -> pd.DataFrame:
    """n uniform random points of the design box (R, A rounded to 0.01; CC, VC integers)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "R":  np.round(rng.uniform(*PARAM_BOUNDS["R"], n), 2),
        "A":  np.round(rng.uniform(*PARAM_BOUNDS["A"], n), 2),
        "CC": rng.integers(PARAM_BOUNDS["CC"][0], PARAM_BOUNDS["CC"][1] + 1, n),
        "VC": rng.integers(PARAM_BOUNDS["VC"][0], PARAM_BOUNDS["VC"][1] + 1, n),
    })


def _in_box(R: float, A: float, CC: int, VC: int) -> bool:
    b = PARAM_BOUNDS
    return b["R"][0] <= R <= b["R"][1] and b["A"][0] <= A <= b["A"][1] and b["CC"][0] <= CC <= b["CC"][1] and b["VC"][0] <= VC <= b["VC"][1]


def edge_points_61(OD: float, L: float, G: float = 0.0, rel: float = 0.01) -> pd.DataFrame:
    """
    Edge cases for the twisted-hexagon set (6, 1) = C1, C4', C5: columns R, A, CC, VC, source ("edge"), label.

      box corner      the 16 corners of the design box
      C1 -/+1%        R just inside / outside the C1 bound, in contexts where C4' and C5 hold on both sides
      C5 -/+1%        R just inside / outside the vertical-packing bound C5 (C1 and C4' hold on both sides)
      C4' tilt -/+1%  A just above / below the smallest tilt C4' accepts (C1 and C5 hold on both sides)
      Vsep=2R -/+1%   R just below / above (L-2G)/(2VC), where C4' switches between "any tilt" and "tilt needed"

    The contexts are found numerically, so a bracket is only emitted when the other constraints do not
    already decide it.  R and A are rounded to 0.01 (SolidWorks stores 2 decimals); points are unique.
    """
    from constraints import C1, C4P, C5, SQRT3, PI

    def others_ok(R, A, CC, VC, *, skip):
        return all(c.satisfied(R, A, CC, VC, OD, L, G)[0] for c in (C1, C4P, C5) if c.name != skip)

    rows: list[tuple] = []

    def add(R, A, CC, VC, label):
        R, A, CC, VC = round(float(R), 2), round(float(A), 2), int(CC), int(VC)
        if _in_box(R, A, CC, VC):
            rows.append((R, A, CC, VC, label))

    b = PARAM_BOUNDS
    for R in b["R"]:
        for A in b["A"]:
            for CC in b["CC"]:
                for VC in b["VC"]:
                    add(R, A, CC, VC, "box corner")

    for CC in (9, 12, 16, 22):                                   # C1 bound inside the box needs CC >= 9
        bound = PI * OD / (SQRT3 * CC)
        for A in (90.0, 80.0, 70.0):
            for VC in (5, 4, 6, 7, 8, 9, 10):
                lo, hi = bound * (1 - rel), bound * (1 + rel)
                if others_ok(lo, A, CC, VC, skip="C1") and C4P.satisfied(hi, A, CC, VC, OD, L, G)[0] and C5.satisfied(hi, A, CC, VC, OD, L, G)[0]:
                    add(lo, A, CC, VC, f"C1 -1% (CC={CC}, A={A:g})")
                    add(hi, A, CC, VC, f"C1 +1% (CC={CC}, A={A:g})")
                    break

    c5_contexts = []
    for A in (90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0):
        for VC in (4, 5, 6, 7, 8, 9, 10):
            cot = 1.0 / np.tan(np.radians(A))
            bound = (L - 2 * G) / (VC - 1) / (-(cot / (SQRT3 + cot) - 1) + 2 / (VC - 1))
            lo, hi = bound * (1 - rel), bound * (1 + rel)
            for CC in (4, 5, 6, 8, 10, 12, 16, 22):
                if all(c.satisfied(R_, A, CC, VC, OD, L, G)[0] for R_ in (lo, hi) for c in (C1, C4P)):
                    c5_contexts.append((lo, hi, A, CC, VC))
                    break
    for i in np.linspace(0, len(c5_contexts) - 1, min(10, len(c5_contexts))).round().astype(int) if c5_contexts else []:
        lo, hi, A, CC, VC = c5_contexts[i]
        add(lo, A, CC, VC, f"C5 -1% (A={A:g}, VC={VC})")
        add(hi, A, CC, VC, f"C5 +1% (A={A:g}, VC={VC})")

    contexts = []
    for R in np.arange(2.5, 6.01, 0.5):
        for CC in (6, 8, 10, 12, 16, 20):
            for VC in (7, 8, 9, 10):
                vs = (L - 2 * (R + G)) / (VC - 1)
                den = vs - 2 * R
                if den >= 0:
                    continue
                bound = (-SQRT3 * vs + 2 * SQRT3 * R - PI * OD / (2 * CC)) / den
                a_min = float(np.degrees(np.arctan2(1.0, bound)))
                if 33 < a_min < 87 and all(others_ok(R, a, CC, VC, skip="C4'") for a in (a_min * (1 - rel), a_min * (1 + rel))):
                    contexts.append((a_min, float(R), CC, VC))
    contexts.sort()
    for i in np.linspace(0, len(contexts) - 1, min(10, len(contexts))).round().astype(int) if contexts else []:
        a_min, R, CC, VC = contexts[i]
        add(R, a_min * (1 - rel), CC, VC, f"C4' tilt -1% (R={R:g}, CC={CC}, VC={VC}, Amin={a_min:.1f})")
        add(R, a_min * (1 + rel), CC, VC, f"C4' tilt +1% (R={R:g}, CC={CC}, VC={VC}, Amin={a_min:.1f})")

    for VC in (4, 6, 8, 10):
        r_switch = (L - 2 * G) / (2 * VC)
        for A in (30.0, 60.0):
            add(r_switch * (1 - rel), A, 12, VC, f"Vsep=2R -1% (VC={VC}, A={A:g})")
            add(r_switch * (1 + rel), A, 12, VC, f"Vsep=2R +1% (VC={VC}, A={A:g})")

    out = pd.DataFrame(rows, columns=["R", "A", "CC", "VC", "label"]).drop_duplicates(["R", "A", "CC", "VC"], keep="first")
    out.insert(4, "source", "edge")
    return out.reset_index(drop=True)


def suite_points(config: tuple[int, int], OD: float, L: float, G: float, n_sweep: int = 120, seed: int = 2026) -> pd.DataFrame:
    """Edge cases followed by `n_sweep` uniform random points (no constraint filtering), de-duplicated.
    Columns R, A, CC, VC, source ("edge" / "sweep"), label."""
    if tuple(config) != (6, 1):
        raise NotImplementedError("edge cases are defined for (6, 1) only")
    edge = edge_points_61(OD, L, G)
    seen = set(map(tuple, edge[["R", "A", "CC", "VC"]].to_numpy().tolist()))
    keep = []
    draw = 0
    while len(keep) < n_sweep:
        for r in random_box_points(n_sweep, seed + draw).itertuples(index=False):
            key = (float(r.R), float(r.A), int(r.CC), int(r.VC))
            if key not in seen and len(keep) < n_sweep:
                seen.add(key)
                keep.append(key)
        draw += 1
    sweep = pd.DataFrame(keep, columns=["R", "A", "CC", "VC"]).assign(source="sweep", label="random sweep")
    return pd.concat([edge, sweep], ignore_index=True)


def load_results(path: str) -> pd.DataFrame:
    """One row per SwGen result: row, R, A, CC, VC, status, built, failed_features."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            params = r.get("params", {})
            if not all(c in params for c in PARAM_COLS):
                continue                                   # malformed CSV row: no parameters to judge
            rows.append({
                "row": r["row"], **{c: params[c] for c in PARAM_COLS}, "status": r["status"],
                "built": r["status"] in BUILT_STATUSES,
                "failed_features": ",".join(sorted({e["feature"] for e in r.get("rebuild_errors", []) if e.get("feature")})),
            })
    return pd.DataFrame(rows)


def compare(results_path: str, config: tuple[int, int], OD: float, L: float, G: float = 0.0) -> dict:
    """Compare the constraints of `config` with the rebuild outcomes in `results_path`."""
    cset = constraints_for(config)
    df = load_results(results_path)
    if df.empty:
        raise ValueError(f"no usable rows in {results_path}")
    df = df.assign(N=config[0], T=config[1])
    report = evaluate(df, OD, L, G)
    accepted = report["feasible"].to_numpy(dtype=bool)
    built = df["built"].to_numpy(dtype=bool)

    per_constraint = {}
    for con in cset:
        sat = report[con.name].astype(bool).to_numpy()
        others_ok = np.ones(len(df), dtype=bool)
        for other in cset:
            if other.name != con.name:
                others_ok &= report[other.name].astype(bool).to_numpy()
        per_constraint[con.name] = {
            "expr": con.expr,
            "violated_by_built": int((~sat & built).sum()),
            "satisfied_by_failed": int((sat & ~built).sum()),
            "only_violation_of_failed": int((~sat & others_ok & ~built).sum()),
        }

    return {
        "config": config,
        "n": len(df),
        "built": int(built.sum()),
        "failed": int((~built).sum()),
        "accepted": int(accepted.sum()),
        "accepted_and_built": int((accepted & built).sum()),
        "accepted_but_failed": int((accepted & ~built).sum()),
        "rejected_but_built": int((~accepted & built).sum()),
        "agreement": int((accepted == built).sum()),
        "per_constraint": per_constraint,
        "false_accepts": df[accepted & ~built][["row", *PARAM_COLS, "failed_features"]].to_dict("records"),
        "false_rejects": df[~accepted & built][["row", *PARAM_COLS]].to_dict("records"),
    }


def print_report(rep: dict) -> None:
    n = rep["n"]
    print(f"Configuration (N={rep['config'][0]}, T={rep['config'][1]}): {n} rows, "
          f"{rep['built']} rebuilt in SolidWorks, {rep['failed']} failed")
    print(f"  constraints accept {rep['accepted']}: {rep['accepted_and_built']} built, "
          f"{rep['accepted_but_failed']} FAILED  (constraints too loose)")
    print(f"  constraints reject {n - rep['accepted']}: {rep['rejected_but_built']} BUILT  "
          f"(constraints too strict), {n - rep['accepted'] - rep['rejected_but_built']} failed")
    print(f"  agreement: {rep['agreement']}/{n}")
    print("\n  per constraint          violated by built   satisfied by failed   only violation of failed")
    for name, m in rep["per_constraint"].items():
        print(f"  {name:<22} {m['violated_by_built']:>17} {m['satisfied_by_failed']:>21} {m['only_violation_of_failed']:>26}   {m['expr']}")
    for title, rows in (("accepted but failed to build", rep["false_accepts"]), ("rejected but built", rep["false_rejects"])):
        if rows:
            print(f"\n  {title} (first 10):")
            print(pd.DataFrame(rows).head(10).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate constraints against real SolidWorks rebuilds (SwGen results).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_points = sub.add_parser("points", help="write a random, unfiltered sample of the design box for SwGen")
    p_points.add_argument("--n", type=int, default=120)
    p_points.add_argument("--seed", type=int, default=2026)
    p_points.add_argument("--out", required=True, help="output CSV (R, A, CC, VC)")

    p_suite = sub.add_parser("suite", help="write edge cases + a random sweep for a configuration (currently (6, 1))")
    p_suite.add_argument("--config", default="6:1")
    p_suite.add_argument("--OD", type=float, required=True)
    p_suite.add_argument("--L", type=float, required=True)
    p_suite.add_argument("--G", "--gap", type=float, default=0.0)
    p_suite.add_argument("--n-sweep", type=int, default=120)
    p_suite.add_argument("--seed", type=int, default=2026)
    p_suite.add_argument("--out", required=True, help="output CSV for SwGen (R, A, CC, VC); <out>.meta.csv gets source / label")

    p_cmp = sub.add_parser("compare", help="compare constraints with SwGen's swgen_results.jsonl")
    p_cmp.add_argument("results", help="swgen_results.jsonl from `SwGen generate`")
    p_cmp.add_argument("--config", required=True, help="configuration of the part that was built, e.g. 0:0")
    p_cmp.add_argument("--OD", type=float, required=True)
    p_cmp.add_argument("--L", type=float, required=True)
    p_cmp.add_argument("--G", "--gap", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.command == "points":
        random_box_points(args.n, args.seed).to_csv(args.out, index=False)
        print(f"{args.n} unfiltered design points written to {args.out}")
        return 0

    if args.command == "suite":
        try:
            (config,) = parse_configs(args.config)
            pts = suite_points(config, args.OD, args.L, args.G, args.n_sweep, args.seed)
        except (ValueError, NotImplementedError) as exc:
            print(f"ERROR: {exc}")
            return 2
        pts[PARAM_COLS].to_csv(args.out, index=False)
        meta = pts.copy()
        meta.insert(0, "row", range(1, len(meta) + 1))
        meta.to_csv(args.out[:-4] + ".meta.csv" if args.out.lower().endswith(".csv") else args.out + ".meta.csv", index=False)
        print(f"{len(pts)} points ({(pts['source'] == 'edge').sum()} edge, {(pts['source'] == 'sweep').sum()} sweep) written to {args.out}")
        return 0

    try:
        (config,) = parse_configs(args.config)
    except (ValueError, NotImplementedError) as exc:       # unknown config, or a placeholder without constraints
        print(f"ERROR: --config: {exc}")
        return 2
    print_report(compare(args.results, config, args.OD, args.L, args.G))
    return 0


if __name__ == "__main__":
    sys.exit(main())
