# Standard Library
import argparse
import os
import sys

# 3rd Party
import pandas as pd

# Example usage: python top_n.py ../data_folder/output/temp/FDData5_Obj_vs_A__colour_=_A.csv Obj 5

def top_n(csv_path: str, field: str, n: int, ascending: bool = False) -> pd.DataFrame:
    """
    Return the top N rows from a CSV ranked by `field`.

    Parameters
    ----------
    csv_path  : path to the CSV file
    field     : column name to rank by
    n         : number of rows to return
    ascending : False = highest first (default), True = lowest first
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if field not in df.columns:
        raise ValueError(
            f"Field '{field}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    result = (
        df.sort_values(field, ascending=ascending)
          .head(n)
          .reset_index(drop=True)
    )
    result.index += 1           # rank starts at 1
    result.index.name = "Rank"
    return result


def interactive() -> None:
    csv_path = input("CSV file path: ").strip()

    try:
        df_preview = pd.read_csv(csv_path, nrows=0)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    print(f"Columns: {list(df_preview.columns)}")

    field = input("Field to rank by: ").strip()

    raw_n = input("How many top rows? [5]: ").strip()
    n = int(raw_n) if raw_n else 5

    order_raw = input("Order — highest first? [Y/n]: ").strip().lower()
    ascending = order_raw == "n"

    try:
        result = top_n(csv_path, field, n, ascending=ascending)
    except ValueError as e:
        print(e)
        return

    direction = "lowest" if ascending else "highest"
    print(f"\nTop {n} rows by {direction} '{field}':\n")
    print(result.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the top N rows of a CSV ranked by a chosen column."
    )
    parser.add_argument("csv",            help="Path to the CSV file")
    parser.add_argument("field",          help="Column to rank by (e.g. Obj)")
    parser.add_argument("n", type=int,    help="Number of rows to return")
    parser.add_argument(
        "--asc", action="store_true",
        help="Return lowest values instead of highest"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Ignore other args and run interactively"
    )

    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        interactive()
        return

    try:
        result = top_n(args.csv, args.field, args.n, ascending=args.asc)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)

    direction = "lowest" if args.asc else "highest"
    print(f"\nTop {args.n} rows by {direction} '{args.field}':\n")
    print(result.to_string())


if __name__ == "__main__":
    main()