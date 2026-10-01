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

    try:
        (config,) = parse_configs(args.config)
    except (ValueError, NotImplementedError) as exc:       # unknown config, or a placeholder without constraints
        print(f"ERROR: --config: {exc}")
        return 2
    print_report(compare(args.results, config, args.OD, args.L, args.G))
    return 0


if __name__ == "__main__":
    sys.exit(main())
