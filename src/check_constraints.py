# Standard Library
import argparse
import os
import sys

# 3rd Party
import pandas as pd

# Example usage: python check_constraints.py src_data/Sample_OD42L50.csv --G 2 --L 50 --OD 40 --out src_data/Sample_OD42L50_violations.csv

# ---------------------------------------------------------------------------
# Constraint functions  (return True when the row VIOLATES the constraint)
# ---------------------------------------------------------------------------

def violates_c1(row: pd.Series, G: float, L: float, OD: float) -> bool:
    """
    Constraint 1:  VC < (L - 2*(G + R)) / (2R - (5π·OD)/(6·CC))  + 1

    Violated when VC >= RHS.
    Note: denominator can be zero/negative — treated as violation if so.
    """
    R  = row["R"]
    CC = row["CC"]
    VC = row["VC"]

    numerator   = L - 2 * (G + R)
    denominator = 2 * R - (5 * 3.141592653589793 * OD) / (6 * CC)

    if denominator <= 0:
        return True                     # undefined / infinite RHS → violation

    rhs = numerator / denominator + 1
    return VC >= rhs


def violates_c2(row: pd.Series, G: float, L: float) -> bool:
    """
    Constraint 2:  VC < (L - 2*(R + G)) / R  + 1

    Violated when VC >= RHS.
    """
    R  = row["R"]
    VC = row["VC"]

    rhs = (L - 2 * (R + G)) / R + 1
    return VC >= rhs


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

def check_constraints(
    csv_path: str,
    G: float,
    L: float,
    OD: float,
    output_path: str | None = None,
) -> None:

    if not os.path.isfile(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required = {"R", "CC", "VC"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        sys.exit(1)

    # 1-based row index matching the original Sobol sample order
    df.insert(0, "SampleRow", range(1, len(df) + 1))

    # Evaluate each constraint per row
    df["Violates_C1"] = df.apply(violates_c1, axis=1, G=G, L=L, OD=OD)
    df["Violates_C2"] = df.apply(violates_c2, axis=1, G=G, L=L)
    df["Violates_Any"] = df["Violates_C1"] | df["Violates_C2"]

    # --- Summary counts ---
    n_total  = len(df)
    n_c1     = int(df["Violates_C1"].sum())
    n_c2     = int(df["Violates_C2"].sum())
    n_either = int(df["Violates_Any"].sum())
    n_both   = int((df["Violates_C1"] & df["Violates_C2"]).sum())

    print("\n" + "=" * 50)
    print("  CONSTRAINT VIOLATION SUMMARY")
    print("=" * 50)
    print(f"  Constants used:  G={G},  L={L},  OD={OD}")
    print(f"  Total rows:      {n_total}")
    print(f"  Violate C1:      {n_c1}  ({100*n_c1/n_total:.1f}%)")
    print(f"  Violate C2:      {n_c2}  ({100*n_c2/n_total:.1f}%)")
    print(f"  Violate both:    {n_both}  ({100*n_both/n_total:.1f}%)")
    print(f"  Violate either:  {n_either}  ({100*n_either/n_total:.1f}%)")
    print("=" * 50 + "\n")

    # --- Rows that violated at least one constraint ---
    violated = df[df["Violates_Any"]].copy()

    if violated.empty:
        print("All rows satisfy both constraints. No output CSV written.")
        return

    # Drop the helper boolean columns from the saved file but keep Which ones
    # were violated so the output is informative.
    violated = violated.drop(columns=["Violates_Any"])

    # Default output path next to the input file
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_violations{ext}"

    violated.to_csv(output_path, index=False)
    print(f"Violations written to: {output_path}")
    print(f"({len(violated)} rows)\n")

    # Print a quick preview
    print(violated.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check Sobol sample rows against two geometric constraints.\n\n"
            "Constraint 1: VC < (L - 2(G+R)) / (2R - 5π·OD/(6·CC))  + 1\n"
            "Constraint 2: VC < (L - 2(R+G)) / R  + 1"
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