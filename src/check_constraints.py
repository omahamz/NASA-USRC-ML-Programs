# Standard Library
import argparse
import os
import sys

# 3rd Party
import pandas as pd

# Local
from constraints import (
    CONSTRAINT_SETS,
    config_columns,
    config_label,
    constraint_names,
    evaluate,
)

# Example usage: python check_constraints.py src_data/Sample_OD40L50G3_New.csv --G 3 --L 50 --OD 40 --out src_data/violations.csv
#
# The constraints themselves live in constraints.py (single source of truth shared with
# sample.py).  Each row is checked against the constraint set of its own (N, T)
# configuration; a CSV without T / N columns is treated as the legacy (N=6, T=0) family.
# Rows of a placeholder configuration (constraints not derived yet) are reported as
# UNCHECKED and count as failing.


def check_constraints(
    csv_path: str,
    G: float,
    L: float,
    OD: float,
    output_path: str | None = None,
) -> pd.DataFrame | None:
    """Check every row of ``csv_path`` and print a per-configuration violation summary.

    Writes the failing rows (violated *or* unchecked) to ``output_path`` (default
    ``<input>_violations.csv``) and returns them, or returns None when every row passes.
    """

    if not os.path.isfile(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required = {"R", "CC", "VC"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        sys.exit(1)

    try:
        cfg = config_columns(df)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    absent = [c for c in ("N", "T") if c not in df.columns]
    if absent:
        print(f"NOTE: column(s) {absent} absent - treating those rows as the legacy (N=6, T=0) configuration.")

    report = evaluate(df, OD=OD, L=L, G=G)

    # 1-based row index matching the original sample order
    df.insert(0, "SampleRow", range(1, len(df) + 1))
    n_total = len(df)

    # --- Summary per configuration ---
    print("\n" + "=" * 62)
    print("  CONSTRAINT VIOLATION SUMMARY")
    print("=" * 62)
    print(f"  Constants used:  G={G},  L={L},  OD={OD}")
    print(f"  Total rows:      {n_total}")

    used_names: list[str] = []
    for (n, t), cset in CONSTRAINT_SETS.items():
        rows = ((cfg["N"] == n) & (cfg["T"] == t)).to_numpy()
        n_rows = int(rows.sum())
        if n_rows == 0:
            continue
        print(f"\n  Configuration (N={n}, T={t}) [{config_label((n, t))}]: {n_rows} row(s)")
        if cset is None:
            print("    UNCHECKED - constraints for this configuration are not derived yet")
            continue
        for con in cset:
            used_names.append(con.name)
            n_bad = int((report.loc[rows, con.name] == False).sum())     # noqa: E712
            print(f"    Violate {con.name}: {n_bad:>6}  ({100 * n_bad / n_rows:5.1f}%)   {con.expr}")
        n_any = int((~report.loc[rows, "feasible"]).sum())
        print(f"    Violate any: {n_any:>5}  ({100 * n_any / n_rows:5.1f}%)")

    n_unchecked = int((~report["defined"]).sum())
    n_failing   = int((~report["feasible"]).sum())
    print(f"\n  Failing rows (violated or unchecked): {n_failing}  ({100 * n_failing / max(n_total, 1):.1f}%)")
    if n_unchecked:
        print(f"  Unchecked rows (placeholder configuration): {n_unchecked}")
    print("=" * 62 + "\n")

    failing = report["feasible"] == False        # noqa: E712
    if not failing.any():
        print("All rows satisfy their constraints. No output CSV written.")
        return None

    # Failing rows keep the original columns plus one flag per applicable constraint
    # (blank where the constraint does not apply to that row's configuration).
    out = df[failing.to_numpy()].copy()
    for name in constraint_names():
        if name in used_names:
            flag = report.loc[failing, name].map(lambda v: pd.NA if pd.isna(v) else (not bool(v)))
            out[f"Violates_{name}"] = flag.astype("boolean").to_numpy()
    out["Unchecked"] = (~report.loc[failing, "defined"]).to_numpy()

    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_violations{ext}"

    out.to_csv(output_path, index=False)
    print(f"Violations written to: {output_path}")
    print(f"({len(out)} rows)\n")

    print(out.to_string(index=False))
    print()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check sample rows against the geometric constraints of their (N, T) configuration.\n\n"
            "(N=6, T=0)  C1: R < pi*OD/(sqrt(3)*CC)\n"
            "            C2: R < (1/VC)(L/2-G) + (sqrt(3)*pi*OD/(12*CC))(1-1/VC)\n"
            "            C3: R < (L-2G)/(VC+1)\n"
            "(N=0, T=0)  E1: R < (L-2G)/(VC+1)\n"
            "            E2: R < pi*OD/(2*CC)\n"
            "            E3: 4R^2 > (pi*OD/(2*CC))^2 + ((L-2(R+G))/(VC-1))^2\n"
            "(T=1)       placeholder - constraints not derived yet (rows are UNCHECKED)\n\n"
            "Columns N and T are optional; without them every row is treated as (N=6, T=0)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("csv",          help="Path to the input CSV file")
    parser.add_argument("--G",  "--gap", type=float, required=True, metavar="G",
                        help="Gap constant (e.g. --G 2)")
    parser.add_argument("--L",          type=float, required=True, metavar="L",
                        help="Length constant (e.g. --L 50)")
    parser.add_argument("--OD",         type=float, required=True, metavar="OD",
                        help="Outer diameter constant (e.g. --OD 42)")
    parser.add_argument("--out", "-o",  default=None, metavar="OUTPUT_CSV",
                        help="Output CSV path (default: <input>_violations.csv)")

    args = parser.parse_args()

    check_constraints(
        csv_path=args.csv,
        G=args.G,
        L=args.L,
        OD=args.OD,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
