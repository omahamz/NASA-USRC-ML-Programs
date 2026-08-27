"""
Latin Hypercube Sampler for a defined parameter space.

Usage:
    # Basic usage (no constraints):
    python sample.py --k 50 --output samples.csv

    # With constraints (provide OD and L constants):
    python sample.py --k 50 --output samples.csv --OD 12.0 --L 40.0

    # With a fixed random seed for reproducibility:
    python sample.py --k 50 --output samples.csv --OD 12.0 --L 40.0 --seed 42

    # Generate a full LHS of K points, then draw n=10 test points via stratified sampling:
    python sample.py --k 50 --subsample 10 --output samples.csv --subsample-output test_points.csv

    # Stratified subsampling strategies: 'random' (default), 'maxmin', or 'both'
    python sample.py --k 50 --subsample 10 --subsample-strategy maxmin --output samples.csv

    # Gold
    python sample.py --k 300 --subsample 10 --OD 40.0 --L 50.0 --seed 42 --subsample-strategy maxmin --output lhs_full.csv --subsample-output test_points.csv

    # Generate 64 Sobol points (no constraints):
    python sample.py --sobol 64 --sobol-output sobol.csv

    # Generate 64 Sobol points with constraints:
    python sample.py --sobol 64 --OD 40.0 --L 50.0 --sobol-output sobol.csv

    # Generate 64 Sobol points with constraints AND domain propagation:
    python sample.py --sobol 64 --OD 40.0 --L 50.0 --propagate --sobol-output sobol.csv

    # Create a named instance (saves myrun.csv + myrun.sobol_state.json):
    python sample.py --sobol 1024 --OD 42.0 --L 50.0 --sobol-instance Sample_OD42L50

    # Create a named instance with domain propagation:
    python sample.py --sobol 1024 --OD 40.0 --L 50.0 --G 3.0 --seed 42 --propagate --sobol-instance Sample_OD40L50G3_New

    # Append 64 more sequential points to the same instance:
    python sample.py --sobol 24 --OD 40.0 --L 50.0 --G 3.0 --propagate --sobol-instance Sample_OD40L50G3_New
    """

import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import qmc, randint as sp_randint, uniform as sp_uniform


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------
# Each entry: (name, dtype, low, high)
PARAMS = [
    ("R",  "float", 2,  8.8),
    ("A",  "float", 30, 90),
    ("CC", "int",   4,  22),
    ("VC", "int",   4,  10),
]

COLUMN_ORDER = ["R", "A", "CC", "VC"]


# ---------------------------------------------------------------------------
# Constraint propagation
# ---------------------------------------------------------------------------

def propagate_domains(OD: float, L: float, G: float = 0.0) -> dict:
    """
    Tighten parameter domains based on constraint propagation.

    Constraints
    -----------
    C1:  CC < (π·OD) / (√3·R)
    C2:  VC < [L - 2(G+R)] / |2R - √3·π·OD/(6·CC)|  + 1
             (numerator positive requires R < (L-2G)/2)
    C3:  VC < [L - 2(R+G)] / R  + 1
             (numerator positive requires R < (L-2G)/2)

    Pass 1 — C1 alone:
        CC_hi <- min(CC_hi, floor(π·OD / (√3·R_lo)))
        R_hi  <- min(R_hi,  π·OD / (√3·CC_lo))

    Pass 2 — C3 alone:
        R_hi  <- min(R_hi, (L-2G) / (VC_lo+1))
        VC_hi <- min(VC_hi, floor((L-2(R_lo+G))/R_lo + 1))

    Pass 3 — numerator positivity guard for C2 and C3:
        R_hi  <- min(R_hi, (L-2G)/2)

    Pass 4 — C2 upper-bound on R (at VC=VC_lo, CC=CC_hi):
        R_hi  <- min(R_hi, [L-2G + √3·π·OD·(VC_lo-1)/(6·CC_hi)] / (2·VC_lo))

    Pass 5 — tighten VC_hi via C3 with final R_hi.
    """
    sqrt3 = math.sqrt(3)
    pi    = math.pi

    R_lo,  R_hi  = 2.0,  10.0
    CC_lo, CC_hi = 4,    22
    VC_lo, VC_hi = 4,    10

    # ------------------------------------------------------------------
    # Pass 1: C1 tightening
    # ------------------------------------------------------------------
    CC_hi = min(CC_hi, math.floor(pi * OD / (sqrt3 * R_lo) - 1e-9))
    R_hi  = min(R_hi,  pi * OD / (sqrt3 * CC_lo) - 1e-9)

    # ------------------------------------------------------------------
    # Pass 2: C3 tightening  VC < (L-2(R+G))/R + 1
    # ------------------------------------------------------------------
    if VC_lo > 1:
        R_hi = min(R_hi, (L - 2 * G) / (VC_lo + 1) - 1e-9)

    if L - 2 * (R_lo + G) <= 0:
        raise ValueError(
            f"L - 2(R_lo + G) = {L - 2*(R_lo+G):.4f} <= 0 at R_lo={R_lo}. "
            "No valid VC exists anywhere in the R domain."
        )

    VC_hi = min(VC_hi, math.floor(
        (L - 2 * (R_lo + G)) / R_lo + 1 - 1e-9
    ))

    # ------------------------------------------------------------------
    # Pass 3: numerator positivity guard for C2 and C3
    #         L - 2(G+R) > 0  =>  R < (L-2G)/2
    #         No CC_lo tightening needed — abs() handles sign flip.
    # ------------------------------------------------------------------
    R_num_zero = (L - 2 * G) / 2
    if R_hi >= R_num_zero:
        print(f"  Note: C2/C3 numerator goes negative at R={R_num_zero:.4f}. Clamping R_hi.")
        R_hi = min(R_hi, R_num_zero - 1e-9)

    # # ------------------------------------------------------------------
    # # Pass 4: C2 upper bound on R
    # #   (VC_lo-1) * |2R - √3·π·OD/(6·CC)| < L-2G-2R
    # #   In the valid region (denom > 0) this gives:
    # #   => R(2·VC_lo) < L-2G + √3·π·OD·(VC_lo-1)/(6·CC_hi)
    # # ------------------------------------------------------------------
    # if VC_lo > 1 and CC_hi > 0:
    #     R_hi_c2 = (
    #         (L - 2 * G) + (sqrt3 * pi * OD * (VC_lo - 1)) / (6 * CC_hi)
    #     ) / (2 * VC_lo) - 1e-9
    #     R_hi = min(R_hi, R_hi_c2)

    # ------------------------------------------------------------------
    # Pass 5: tighten VC_hi via C3 with final R_hi
    # ------------------------------------------------------------------
    VC_hi = min(VC_hi, math.floor(
        (L - 2 * (R_lo + G)) / R_lo + 1 - 1e-9
    ))

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if R_lo >= R_hi:
        raise ValueError(f"Empty R domain [{R_lo}, {R_hi:.4f}].")
    if CC_lo > CC_hi:
        raise ValueError(
            f"Propagation produced empty CC domain [{CC_lo}, {CC_hi}]. "
            "Constraints may be infeasible for these OD/L/G values."
        )
    if VC_lo > VC_hi:
        raise ValueError(
            f"Propagation produced empty VC domain [{VC_lo}, {VC_hi}]. "
            "Constraints may be infeasible for these OD/L/G values."
        )

    domains = {
        "R":  (R_lo,  round(R_hi,  4)),
        "A":  (30.0,  90.0),
        "CC": (CC_lo, CC_hi),
        "VC": (VC_lo, VC_hi),
        "OD": OD,
        "L":  L,
        "G":  G,
    }

    print(f"  Domain propagation results (OD={OD}, L={L}, G={G}):")
    print(f"    R  : [2.0,  10.0] -> [{domains['R'][0]},  {domains['R'][1]}]")
    print(f"    A  : [30.0, 90.0] -> [30.0, 90.0]  (unconstrained)")
    print(f"    CC : [4,    22  ] -> [{domains['CC'][0]},     {domains['CC'][1]}]")
    print(f"    VC : [4,    10  ] -> [{domains['VC'][0]},      {domains['VC'][1]}]")

    return domains

def domains_to_params(domains: dict) -> list:
    """Convert a propagated domains dict back to a PARAMS-style list for sampling."""
    return [
        ("R",  "float", domains["R"][0],  domains["R"][1]),
        ("A",  "float", domains["A"][0],  domains["A"][1]),
        ("CC", "int",   domains["CC"][0], domains["CC"][1]),
        ("VC", "int",   domains["VC"][0], domains["VC"][1]),
    ]


# ---------------------------------------------------------------------------
# LHS generation
# ---------------------------------------------------------------------------

def generate_lhs(k: int, seed: int | None = None, params: list | None = None) -> pd.DataFrame:
    """Generate k LHS points over the parameter space (no constraints).

    All continuous/integer parameters use PPF mapping:
      float : scipy.stats.uniform.ppf  ->  continuous uniform on [lo, hi]
      int   : scipy.stats.randint.ppf  ->  discrete uniform on {lo, ..., hi}
    """
    params = params or PARAMS
    d = len(params)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    raw = sampler.random(n=k)  # (k, d) in [0, 1)

    df = pd.DataFrame(index=range(k), columns=[p[0] for p in params])

    for i, (name, dtype, lo, hi) in enumerate(params):
        col = raw[:, i]
        if dtype == "float":
            df[name] = sp_uniform(loc=lo, scale=hi - lo).ppf(col)
        elif dtype == "int":
            df[name] = sp_randint(lo, hi + 1).ppf(col).astype(int)

    return df[COLUMN_ORDER]


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def apply_constraints(df: pd.DataFrame, OD: float, L: float, G: float = 0.0) -> pd.DataFrame:
    sqrt3 = math.sqrt(3)
    pi    = math.pi

    # C1
    c1 = df["CC"] < (pi * OD) / (sqrt3 * df["R"])

    # C2 — abs() on denominator is the true constraint.
    # Reject where numerator <= 0 (no valid VC can exist).
    numerator = L - 2 * (G + df["R"])
    denom_abs = (2 * df["R"] - (sqrt3 * pi * OD) / (6 * df["CC"])).abs()
    c2_rhs = numerator / denom_abs + 1
    c2 = (numerator > 0) & (df["VC"] < c2_rhs)

    # C3
    numerator_c3 = L - 2 * (df["R"] + G)
    c3 = (numerator_c3 > 0) & (df["VC"] < numerator_c3 / df["R"] + 1)

    mask = c1 & c2 & c3
    n_removed = (~mask).sum()
    if n_removed:
        print(f"  Removed {n_removed} point(s): "
              f"C1={(~c1).sum()}, C2={(~c2).sum()}, C3={(~c3).sum()} violations.")

    return df[mask].reset_index(drop=True)

def sample_with_constraints(
    k: int,
    OD: float,
    L: float,
    G: float,
    seed: int | None = None,
    max_attempts: int = 50,
    params: list | None = None,
) -> pd.DataFrame:
    """
    Collect k valid LHS points under constraints using repeated batches.
    """
    params = params or PARAMS
    collected: list[pd.DataFrame] = []
    total_valid = 0
    rng = np.random.default_rng(seed)

    for attempt in range(1, max_attempts + 1):
        batch_seed = int(rng.integers(0, 2**31))
        batch = generate_lhs(k, seed=batch_seed, params=params)
        valid = apply_constraints(batch, OD, L, G)
        collected.append(valid)
        total_valid += len(valid)
        print(f"  Attempt {attempt}: {len(valid)}/{k} passed (total so far: {total_valid})")
        if total_valid >= k:
            break
    else:
        print(f"  Warning: only {total_valid}/{k} valid points found after {max_attempts} attempts.")

    combined = pd.concat(collected, ignore_index=True)
    return combined.head(k)[COLUMN_ORDER]


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------

# Persistent instance files:
#   <instance>.sobol_state.json  ->  engine state + full sampling metadata
#   <instance>.csv               ->  accumulated sample rows

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _sobol_state_path(instance: str) -> str:
    return f"{instance}.sobol_state.json"


def _load_sobol_state(instance: str) -> dict:
    """Load persisted Sobol state from disk."""
    import json, os
    path = _sobol_state_path(instance)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"num_generated": 0, "seed": None}


def _save_sobol_state(
    instance: str,
    num_generated: int,
    seed: int | None,
    OD: float | None = None,
    L: float | None = None,
    domains: dict | None = None,
    propagated: bool = False,
) -> None:
    """
    Persist Sobol engine state with full sampling metadata.

    JSON structure
    --------------
    {
      "num_generated": <int>,     # total raw Sobol draws (used by fast_forward on resume)
      "seed":          <int|null>, # original engine seed
      "OD":            <float>,   # outer diameter constant used in constraints
      "L":             <float>,   # length constant used in constraints
      "propagated":    <bool>,    # whether --propagate was used
      "domains": {                # exact parameter bounds used for sampling
        "R":  [lo, hi],
        "A":  [lo, hi],
        "CC": [lo, hi],
        "VC": [lo, hi]
      }
    }
    """
    import json

    # Domains actually sampled: propagated bounds if provided, else original PARAMS
    if domains is not None:
        sampled_domains = {
            "R":  list(domains["R"]),
            "A":  list(domains["A"]),
            "CC": list(domains["CC"]),
            "VC": list(domains["VC"]),
        }
    else:
        sampled_domains = {
            name: [lo, hi]
            for name, _, lo, hi in PARAMS
        }

    state = {
        "num_generated": num_generated,
        "seed":          seed,
        "OD":            OD,
        "L":             L,
        "propagated":    propagated,
        "domains":       sampled_domains,
    }

    with open(_sobol_state_path(instance), "w") as f:
        json.dump(state, f, indent=2)


def _raw_sobol_to_df(raw: np.ndarray, params: list | None = None) -> pd.DataFrame:
    """Map a (n, d) Sobol array in [0, 1) to the PARAMS space."""
    params = params or PARAMS
    df = pd.DataFrame(index=range(len(raw)), columns=[p[0] for p in params])
    for i, (name, dtype, lo, hi) in enumerate(params):
        col = raw[:, i]
        if dtype == "float":
            df[name] = np.round(sp_uniform(loc=lo, scale=hi - lo).ppf(col), 2)
        elif dtype == "int":
            df[name] = sp_randint(lo, hi + 1).ppf(col).astype(int)
    return df[COLUMN_ORDER]


def generate_sobol(
    k: int,
    seed: int | None = None,
    instance: str | None = None,
    OD: float | None = None,
    L: float | None = None,
    max_attempts: int = 50,
    domains: dict | None = None,
) -> pd.DataFrame:
    """
    Generate k Sobol points, optionally appending to a named persistent instance.

    Parameters
    ----------
    k : int
        Number of NEW valid points requested. Must be a power of 2; if not,
        rounded up to the next power of 2 with a warning.
    seed : int or None
        Seed for the Sobol engine. Ignored when resuming an existing instance
        (original seed is reused and the engine is fast-forwarded to maintain
        sequence continuity).
    instance : str or None
        Name of a persistent instance (no file extension). State is stored in
        <instance>.sobol_state.json and the sample in <instance>.csv.
        If the instance already exists, new points are appended and the Sobol
        sequence continues without repetition via fast_forward().
    OD, L : float or None
        Constraint constants. Both must be provided to enable constraint filtering.
    max_attempts : int
        Maximum over-sampling passes when constraints knock out points.
    domains : dict or None
        Propagated domains from propagate_domains(). If provided, sampling is
        restricted to the tightened bounds before constraint filtering.

    Returns
    -------
    pd.DataFrame
        Full accumulated sample (existing + new) when using an instance,
        otherwise just the newly generated points.
    """
    import os

    # --- Power-of-2 check ---
    if not _is_power_of_2(k):
        k_orig = k
        k = _next_power_of_2(k)
        print(f"  Warning: {k_orig} is not a power of 2. Rounding up to {k}.")

    use_constraints = OD is not None and L is not None
    propagated      = domains is not None
    params          = domains_to_params(domains) if propagated else PARAMS
    d               = len(params)

    # --- Load existing instance state if resuming ---
    num_previously_generated = 0
    existing_df: pd.DataFrame | None = None
    effective_seed = seed

    if instance is not None:
        state = _load_sobol_state(instance)
        num_previously_generated = state["num_generated"]
        if num_previously_generated > 0:
            effective_seed = state.get("seed", seed)

        instance_csv = f"{instance}.csv"
        if os.path.exists(instance_csv):
            existing_df = pd.read_csv(instance_csv)
            print(f"  Instance '{instance}': resuming — {len(existing_df)} existing points, "
                  f"engine at position {num_previously_generated}.")

    # --- Build engine, fast-forward past already-generated points ---
    engine = qmc.Sobol(d=d, scramble=True, seed=effective_seed)
    if num_previously_generated > 0:
        engine.fast_forward(num_previously_generated)

    # --- Collect k valid new points ---
    collected: list[pd.DataFrame] = []
    total_valid = 0
    total_drawn = 0

    for attempt in range(1, max_attempts + 1):
        raw = engine.random(k)
        total_drawn += k
        batch = _raw_sobol_to_df(raw, params=params)

        valid = apply_constraints(batch, OD, L, domains["G"] if domains else 0.0) if use_constraints else batch
        collected.append(valid)
        total_valid += len(valid)
        print(f"  Attempt {attempt}: {len(valid)}/{k} passed "
              f"(total valid so far: {total_valid})")
        if total_valid >= k:
            break
    else:
        print(f"  Warning: only {total_valid}/{k} valid points "
              f"after {max_attempts} attempts.")

    new_df = pd.concat(collected, ignore_index=True).head(k)[COLUMN_ORDER]

    # --- Persist instance state and merged CSV ---
    if instance is not None:
        updated_num_generated = num_previously_generated + total_drawn
        _save_sobol_state(
            instance,
            num_generated=updated_num_generated,
            seed=effective_seed,
            OD=OD,
            L=L,
            domains=domains,
            propagated=propagated,
        )

        full_df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if existing_df is not None else new_df
        )
        full_df.to_csv(f"{instance}.csv", index=False)
        print(f"  Instance '{instance}': {len(full_df)} total points "
              f"saved to '{instance}.csv'.")
        return full_df

    return new_df


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    strategy: str = "random",
    sort_col: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Draw n points from an existing LHS sample using stratified sampling,
    preserving the space-filling spread of the original design.
    """
    K = len(df)
    if n > K:
        raise ValueError(f"Cannot select {n} points from a sample of {K}.")
    if n == K:
        return df.reset_index(drop=True)

    if strategy == "random":
        return _stratified_random(df, n, sort_col=sort_col, seed=seed)
    elif strategy == "maxmin":
        return _stratified_maxmin(df, n)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'random' or 'maxmin'.")


def _stratified_random(df, n, sort_col, seed):
    rng = np.random.default_rng(seed)
    if sort_col is not None:
        if sort_col not in df.columns:
            raise ValueError(f"sort_col '{sort_col}' not found in DataFrame.")
        order = df[sort_col].values
    else:
        normed = (df.values - df.values.min(axis=0)) / (df.values.ptp(axis=0) + 1e-12)
        order = np.linalg.norm(normed, axis=1)
    sorted_idx = np.argsort(order)
    strata = np.array_split(sorted_idx, n)
    chosen = [rng.choice(stratum) for stratum in strata]
    return df.iloc[chosen].reset_index(drop=True)


def _stratified_maxmin(df, n):
    values = df.values.astype(float)
    col_min = values.min(axis=0)
    col_range = values.ptp(axis=0) + 1e-12
    normed = (values - col_min) / col_range
    centroid = normed.mean(axis=0)
    seed_idx = int(np.argmin(np.linalg.norm(normed - centroid, axis=1)))
    selected = [seed_idx]
    min_dist = np.linalg.norm(normed - normed[seed_idx], axis=1)
    min_dist[seed_idx] = -1.0
    for _ in range(n - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dists = np.linalg.norm(normed - normed[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
        min_dist[next_idx] = -1.0
    return df.iloc[selected].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latin Hypercube Sampler with optional constraints and stratified subsampling."
    )
    # --- LHS ---
    parser.add_argument("--k", type=int, default=None,
                        help="Number of LHS points to generate.")
    parser.add_argument("--output", type=str, default="lhs_samples.csv",
                        help="Output CSV for the full LHS sample (default: lhs_samples.csv).")
    parser.add_argument("--OD", type=float, default=None,
                        help="Constant OD used in constraints (outer diameter).")
    parser.add_argument("--L", type=float, default=None,
                        help="Constant L used in constraints (length).")
    parser.add_argument("--G", "--gap", type=float, default=0.0,
                    help="Gap constant G used in C2 and C3 (default: 0.0).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--propagate", action="store_true",
                        help="Tighten parameter domains analytically via constraint propagation "
                             "before sampling. Requires --OD and --L. Significantly reduces "
                             "wasted samples when constraints are tight.")
    # --- Stratified subsampling ---
    parser.add_argument("--subsample", type=int, default=None, metavar="N",
                        help="If set, draw N points from the LHS via stratified subsampling.")
    parser.add_argument("--subsample-strategy", type=str, default="random",
                        choices=["random", "maxmin", "both"], dest="subsample_strategy",
                        help="Subsampling strategy: 'random', 'maxmin', or 'both'.")
    parser.add_argument("--subsample-output", type=str, default=None,
                        dest="subsample_output",
                        help="Output CSV for the subsampled points. Ignored when --subsample-strategy=both.")
    parser.add_argument("--sort-col", type=str, default=None, dest="sort_col",
                        help="Column to sort by when using the 'random' stratified strategy.")
    # --- Sobol ---
    parser.add_argument("--sobol", type=int, default=None, metavar="K",
                        help="Generate K Sobol points (rounded up to next power of 2 if needed).")
    parser.add_argument("--sobol-output", type=str, default="sobol_samples.csv",
                        dest="sobol_output",
                        help="Output CSV for a standalone Sobol sample (default: sobol_samples.csv). "
                             "Ignored when --sobol-instance is set.")
    parser.add_argument("--sobol-instance", type=str, default=None, dest="sobol_instance",
                        metavar="NAME",
                        help="Named persistent Sobol instance. Saves/resumes NAME.csv and "
                             "NAME.sobol_state.json. New points are appended each run.")
    
    return parser.parse_args()


def main():
    args = parse_args()

    use_constraints = args.OD is not None and args.L is not None
    if (args.OD is None) != (args.L is None):
        print("Warning: both --OD and --L must be provided to enable constraints. Ignoring.")
        use_constraints = False

    # --- Domain propagation ---
    domains: dict | None = None
    if args.propagate:
        if not use_constraints:
            print("Warning: --propagate requires --OD and --L. Skipping propagation.")
        else:
            print(f"\nRunning domain propagation...")
            domains = propagate_domains(args.OD, args.L, args.G)
            print()

    # ------------------------------------------------------------------ LHS
    if args.k is not None:
        params = domains_to_params(domains) if domains is not None else None
        print(f"\nGenerating {args.k} LHS points...")
        print(f"Constraints : {'ENABLED  (OD=' + str(args.OD) + ', L=' + str(args.L) + ')' if use_constraints else 'DISABLED'}")
        print(f"Propagation : {'ENABLED' if domains is not None else 'DISABLED'}")
        print(f"Random seed : {args.seed if args.seed is not None else 'not set'}\n")

        if use_constraints:
            df = sample_with_constraints(args.k, args.OD, args.L, args.G, seed=args.seed, params=params)
        else:
            df = generate_lhs(args.k, seed=args.seed, params=params)

        df.to_csv(args.output, index=False)
        print(f"\nFull LHS: {len(df)} point(s) written to '{args.output}'.")
        print("\nSample preview (first 5 rows):")
        print(df.head().to_string(index=False))

        if use_constraints and len(df) < args.k:
            print(f"\nNote: only {len(df)} valid points found. "
                  "Consider relaxing constraints or increasing --k.")

        if args.subsample is not None:
            n = args.subsample
            print(f"\n--- Stratified subsampling: selecting {n} points from {len(df)} ---")
            strategies = (["random", "maxmin"] if args.subsample_strategy == "both"
                          else [args.subsample_strategy])
            for strat in strategies:
                print(f"\n  Strategy: {strat}")
                subset = stratified_subsample(
                    df, n, strategy=strat, sort_col=args.sort_col, seed=args.seed
                )
                if args.subsample_strategy == "both" or args.subsample_output is None:
                    out_path = f"subsample_{strat}.csv"
                else:
                    out_path = args.subsample_output
                subset.to_csv(out_path, index=False)
                print(f"  {len(subset)} point(s) written to '{out_path}'.")
                print(f"  Preview:\n{subset.to_string(index=False)}")

    # ----------------------------------------------------------------- Sobol
    if args.sobol is not None:
        print(f"\n--- Sobol sampling: requesting {args.sobol} points ---")
        print(f"Constraints : {'ENABLED  (OD=' + str(args.OD) + ', L=' + str(args.L) + ')' if use_constraints else 'DISABLED'}")
        print(f"Propagation : {'ENABLED' if domains is not None else 'DISABLED'}")

        sobol_df = generate_sobol(
            k=args.sobol,
            seed=args.seed,
            instance=args.sobol_instance,
            OD=args.OD if use_constraints else None,
            L=args.L if use_constraints else None,
            domains=domains,
        )

        if args.sobol_instance is None:
            sobol_df.to_csv(args.sobol_output, index=False)
            print(f"  {len(sobol_df)} point(s) written to '{args.sobol_output}'.")

        print(f"\nSobol preview (first 5 rows):\n{sobol_df.head().to_string(index=False)}")


if __name__ == "__main__":
    main()