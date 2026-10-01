"""
Latin Hypercube / Sobol sampler for the (N, T)-configured design space.

Every design point is  [R, A, CC, VC, T, N]:
    R, A, CC, VC : geometry parameters (bounds in constraints.PARAM_BOUNDS)
    N            : 6 = hexagonal cells, 0 = ellipsoidal holes
    T            : 0 = untwisted part,  1 = twist-angle part

T and N are NOT extra Sobol/LHS dimensions.  Each (N, T) configuration gets its own
4-D stream, its own geometric constraints (src/constraints.py) and its own domain
propagation.  K always means "K points per configuration".  Configurations without
derived constraints (T = 1 for now) cannot be sampled.

Usage:
    # Basic usage (no constraints), all sampleable configurations:
    python sample.py --k 50 --output samples.csv

    # With constraints (provide OD and L constants):
    python sample.py --k 50 --output samples.csv --OD 12.0 --L 40.0

    # With a fixed random seed for reproducibility:
    python sample.py --k 50 --output samples.csv --OD 12.0 --L 40.0 --seed 42

    # Only some configurations  (N:T pairs, NxTy names, 'all' or 'legacy' = 6:0):
    python sample.py --k 50 --OD 40.0 --L 50.0 --G 3.0 --configs 6:0,0:0

    # Generate a full LHS of K points, then draw n=10 test points via stratified sampling:
    python sample.py --k 50 --subsample 10 --output samples.csv --subsample-output test_points.csv

    # Stratified subsampling strategies: 'random' (default), 'maxmin', or 'both'
    python sample.py --k 50 --subsample 10 --subsample-strategy maxmin --output samples.csv

    # Gold
    python sample.py --k 300 --subsample 10 --OD 40.0 --L 50.0 --seed 42 --subsample-strategy maxmin --output lhs_full.csv --subsample-output test_points.csv

    # Generate 64 Sobol points per configuration (no constraints):
    python sample.py --sobol 64 --sobol-output sobol.csv

    # Generate 64 Sobol points per configuration with constraints:
    python sample.py --sobol 64 --OD 40.0 --L 50.0 --G 3.0 --sobol-output sobol.csv

    # Generate 64 Sobol points with constraints AND domain propagation:
    python sample.py --sobol 64 --OD 40.0 --L 50.0 --G 3.0 --propagate --sobol-output sobol.csv

    # Create a named instance (saves myrun.csv + myrun.sobol_state.json):
    python sample.py --sobol 1024 --OD 42.0 --L 50.0 --sobol-instance Sample_OD42L50

    # Create a named instance with domain propagation (legacy family only):
    python sample.py --sobol 1024 --OD 40.0 --L 50.0 --G 3.0 --seed 42 --propagate --configs 6:0 --sobol-instance Sample_OD40L50G3_New

    # Append 64 more sequential points per configuration to the same instance:
    python sample.py --sobol 64 --OD 40.0 --L 50.0 --G 3.0 --propagate --configs 6:0 --sobol-instance Sample_OD40L50G3_New
    """

import argparse
import json
import math
import os
import secrets

import numpy as np
import pandas as pd
from scipy.stats import qmc, randint as sp_randint, uniform as sp_uniform

from constraints import (
    ALL_CONFIGS,
    CONFIG_COLS,
    LEGACY_CONFIG,
    PARAM_BOUNDS,
    PARAM_COLS,
    ConstraintsNotDefinedError,
    config_label,
    constraint_names,
    constraints_for,
    evaluate,
    parse_configs,
)


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------
# Each entry: (name, dtype, low, high)
PARAMS = [
    ("R",  "float", *PARAM_BOUNDS["R"]),
    ("A",  "float", *PARAM_BOUNDS["A"]),
    ("CC", "int",   *PARAM_BOUNDS["CC"]),
    ("VC", "int",   *PARAM_BOUNDS["VC"]),
]

COLUMN_ORDER = PARAM_COLS + CONFIG_COLS          # R, A, CC, VC, T, N

# Fixed per-configuration seed offsets: a configuration keeps its stream whichever
# subset of configurations a run selects.  (6, 0) has offset 0, so a (6, 0)-only run
# reproduces the stream of the pre-(N, T) sampler.
_CONFIG_SEED_OFFSET = {cfg: i for i, cfg in enumerate(ALL_CONFIGS)}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _resolve_configs(configs) -> list[tuple[int, int]]:
    """
    None -> the legacy single configuration (6, 0)  (keeps old callers unchanged).
    Otherwise a spec string or a sequence of (N, T) pairs; placeholder configurations
    raise ConstraintsNotDefinedError.
    """
    if configs is None:
        return [LEGACY_CONFIG]
    return parse_configs(configs if isinstance(configs, str) else list(configs))


def _config_seed(seed: int | None, config: tuple[int, int]) -> int | None:
    return None if seed is None else int(seed) + _CONFIG_SEED_OFFSET[tuple(config)]


def _attach_config(df: pd.DataFrame, config: tuple[int, int]) -> pd.DataFrame:
    """Return the four parameter columns plus constant T and N columns for `config`."""
    out = df[PARAM_COLS].copy()
    out["T"] = int(config[1])
    out["N"] = int(config[0])
    return out[COLUMN_ORDER]


def _ensure_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add T=0 / N=6 to frames written before the (N, T) extension."""
    df = df.copy()
    if "T" not in df.columns:
        df["T"] = LEGACY_CONFIG[1]
    if "N" not in df.columns:
        df["N"] = LEGACY_CONFIG[0]
    extra = [c for c in df.columns if c not in COLUMN_ORDER]
    return df[COLUMN_ORDER + extra]


def _domains_for(domains, config, n_configs: int):
    """Pick the domains dict for `config` from a flat legacy dict or a {(N, T): dict} mapping."""
    if domains is None:
        return None
    if "R" in domains:                       # flat dict (legacy callers)
        if n_configs > 1:
            raise ValueError("A single domains dict cannot be shared by several configurations; "
                             "pass a {(N, T): domains} mapping.")
        return domains
    return domains.get(tuple(config))


def _params_for(params, config, n_configs: int) -> list:
    """Pick the PARAMS-style list for `config` from a list or a {(N, T): list} mapping."""
    if params is None:
        return PARAMS
    if isinstance(params, dict):
        return params.get(tuple(config), PARAMS)
    if n_configs > 1:
        raise ValueError("A single params list cannot be shared by several configurations; "
                         "pass a {(N, T): params} mapping.")
    return params


def _resolve_G(G: float | None, domains) -> float:
    """Explicit G wins; otherwise take it from the domains dict(s); otherwise 0."""
    if G is not None:
        return float(G)
    if domains:
        if "G" in domains:
            return float(domains["G"])
        for dom in domains.values():
            if isinstance(dom, dict) and "G" in dom:
                return float(dom["G"])
    return 0.0


# ---------------------------------------------------------------------------
# Constraint propagation
# ---------------------------------------------------------------------------

def propagate_domains(OD: float, L: float, G: float = 0.0, config: tuple[int, int] = LEGACY_CONFIG) -> dict:
    """
    Tighten parameter domains for one configuration by constraint propagation.

    (6, 0) - hexagonal cells (constraints C1..C3 in constraints.py)
        Pass 1 - C1:  CC_hi <- min(CC_hi, floor(pi*OD / (sqrt(3)*R_lo)))
                      R_hi  <- min(R_hi,  pi*OD / (sqrt(3)*CC_lo))
        Pass 2 - C3:  R_hi  <- min(R_hi, (L-2G) / (VC_lo+1))
                      VC_hi <- min(VC_hi, floor((L-2(R_lo+G))/R_lo + 1))
        Pass 3 - numerator positivity:  R_hi <- min(R_hi, (L-2G)/2)
        Pass 5 - tighten VC_hi via C3.
        (C2 is not used for tightening; the sampler still enforces it row by row.)

    (0, 0) - ellipsoidal holes (constraints E1..E3)
        E2:  CC_hi <- min(CC_hi, floor(pi*OD / (2*R_lo)));   R_hi <- min(R_hi, pi*OD / (2*CC_lo))
        E1:  R_hi  <- min(R_hi, (L-2G) / (VC_lo+1));         VC_hi <- min(VC_hi, floor((L-2G)/R_lo - 1))
        (E3 couples R to the row spacing implicitly, so it is left to row filtering.)

    Other configurations with derived constraints but no propagator keep the full box.
    Placeholder configurations raise ConstraintsNotDefinedError.
    """
    config = (int(config[0]), int(config[1]))
    constraints_for(config)          # raises for placeholders / unknown configs

    if config == (6, 0):
        return _propagate_domains_hex(OD, L, G)
    if config == (0, 0):
        return _propagate_domains_ellipsoid(OD, L, G)

    domains = {name: tuple(PARAM_BOUNDS[name]) for name in PARAM_COLS}
    domains.update({"OD": OD, "L": L, "G": G})
    print(f"  No propagator for {config_label(config)}: keeping the full parameter box.")
    return domains


def _propagate_domains_hex(OD: float, L: float, G: float = 0.0) -> dict:
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
    # ------------------------------------------------------------------
    R_num_zero = (L - 2 * G) / 2
    if R_hi >= R_num_zero:
        print(f"  Note: C2/C3 numerator goes negative at R={R_num_zero:.4f}. Clamping R_hi.")
        R_hi = min(R_hi, R_num_zero - 1e-9)

    # ------------------------------------------------------------------
    # Pass 5: tighten VC_hi via C3 with final R_hi
    # ------------------------------------------------------------------
    VC_hi = min(VC_hi, math.floor(
        (L - 2 * (R_lo + G)) / R_lo + 1 - 1e-9
    ))

    _check_domains(R_lo, R_hi, CC_lo, CC_hi, VC_lo, VC_hi)
    return _finish_domains("N6_T0", OD, L, G, R_lo, R_hi, CC_lo, CC_hi, VC_lo, VC_hi)


def _propagate_domains_ellipsoid(OD: float, L: float, G: float = 0.0) -> dict:
    pi = math.pi

    R_lo,  R_hi  = 2.0,  10.0
    CC_lo, CC_hi = 4,    22
    VC_lo, VC_hi = 4,    10

    # E2: R < pi*OD / (2*CC)
    CC_hi = min(CC_hi, math.floor(pi * OD / (2 * R_lo) - 1e-9))
    R_hi  = min(R_hi,  pi * OD / (2 * CC_lo) - 1e-9)

    # E1: R < (L-2G) / (VC+1)
    R_hi  = min(R_hi, (L - 2 * G) / (VC_lo + 1) - 1e-9)
    VC_hi = min(VC_hi, math.floor((L - 2 * G) / R_lo - 1 - 1e-9))

    _check_domains(R_lo, R_hi, CC_lo, CC_hi, VC_lo, VC_hi)
    return _finish_domains("N0_T0", OD, L, G, R_lo, R_hi, CC_lo, CC_hi, VC_lo, VC_hi)


def _check_domains(R_lo, R_hi, CC_lo, CC_hi, VC_lo, VC_hi) -> None:
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


def _finish_domains(label, OD, L, G, R_lo, R_hi, CC_lo, CC_hi, VC_lo, VC_hi) -> dict:
    domains = {
        "R":  (R_lo,  round(R_hi,  4)),
        "A":  (30.0,  90.0),
        "CC": (CC_lo, CC_hi),
        "VC": (VC_lo, VC_hi),
        "OD": OD,
        "L":  L,
        "G":  G,
    }

    print(f"  Domain propagation results [{label}] (OD={OD}, L={L}, G={G}):")
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
    """Generate k LHS points over the 4-D parameter space (no constraints, no T/N columns).

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

    return df[PARAM_COLS]


def generate_lhs_configs(
    k: int,
    seed: int | None = None,
    params=None,
    configs=None,
) -> pd.DataFrame:
    """K unconstrained LHS points per configuration, with T and N columns attached."""
    cfgs = _resolve_configs(configs)
    frames = [
        _attach_config(
            generate_lhs(k, seed=_config_seed(seed, cfg), params=_params_for(params, cfg, len(cfgs))),
            cfg,
        )
        for cfg in cfgs
    ]
    return pd.concat(frames, ignore_index=True)[COLUMN_ORDER]


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def apply_constraints(df: pd.DataFrame, OD: float, L: float, G: float = 0.0, verbose: bool = True) -> pd.DataFrame:
    """
    Keep the rows that satisfy the constraints of their own (N, T) configuration.

    Frames without T / N columns are treated as the legacy (6, 0) configuration and
    come back without them.  Rows of a placeholder configuration raise
    ConstraintsNotDefinedError - they are never silently passed or dropped.
    """
    if len(df) == 0:
        return df.reset_index(drop=True)

    report = evaluate(df, OD, L, G)

    undefined = ~report["defined"]
    if undefined.any():
        cfgs = sorted({(int(n), int(t)) for n, t in zip(report.loc[undefined, "N"], report.loc[undefined, "T"])})
        raise ConstraintsNotDefinedError(
            f"Rows from configuration(s) (N, T) = {cfgs} have no derived constraints (placeholder)."
        )

    mask = report["feasible"].to_numpy()
    if verbose and not mask.all():
        parts = []
        for name in constraint_names():
            n_bad = int((report[name] == False).sum())          # noqa: E712  (NA rows excluded)
            if n_bad:
                parts.append(f"{name}={n_bad}")
        print(f"  Removed {int((~mask).sum())} point(s): {', '.join(parts)} violations.")

    return df[mask].reset_index(drop=True)


def sample_with_constraints(
    k: int,
    OD: float,
    L: float,
    G: float,
    seed: int | None = None,
    max_attempts: int = 500,
    params=None,
    configs=None,
) -> pd.DataFrame:
    """
    Collect k valid LHS points *per configuration* under constraints using repeated batches.

    configs=None keeps the legacy behaviour (one configuration, (6, 0)).
    params may be one PARAMS-style list (single configuration) or {(N, T): list}.
    """
    cfgs = _resolve_configs(configs)
    frames: list[pd.DataFrame] = []

    for cfg in cfgs:
        cparams = _params_for(params, cfg, len(cfgs))
        rng = np.random.default_rng(_config_seed(seed, cfg))
        collected: list[pd.DataFrame] = []
        total_valid = 0
        attempts = 0

        for attempts in range(1, max_attempts + 1):
            batch_seed = int(rng.integers(0, 2**31))
            batch = _attach_config(generate_lhs(k, seed=batch_seed, params=cparams), cfg)
            valid = apply_constraints(batch, OD, L, G, verbose=False)
            collected.append(valid)
            total_valid += len(valid)
            if total_valid >= k:
                break
        else:
            print(f"  Warning [{config_label(cfg)}]: only {total_valid}/{k} valid points "
                  f"found after {max_attempts} attempts.")

        print(f"  [{config_label(cfg)}] {min(total_valid, k)}/{k} valid points "
              f"({attempts} batch(es) of {k})")
        frames.append(pd.concat(collected, ignore_index=True).head(k))

    return pd.concat(frames, ignore_index=True)[COLUMN_ORDER]


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------

# Persistent instance files:
#   <instance>.sobol_state.json  ->  per-configuration engine state + sampling metadata
#   <instance>.csv               ->  accumulated sample rows (R, A, CC, VC, T, N)

_STATE_VERSION = 2


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
    """
    Load persisted Sobol state from disk, always in the v2 layout.

    v1 files (a single stream, written before the (N, T) extension) are read as the
    legacy configuration (6, 0); the file is rewritten as v2 on the next save.
    """
    path = _sobol_state_path(instance)
    if not os.path.exists(path):
        return {"version": _STATE_VERSION, "seed": None, "configs": {}}

    with open(path) as f:
        raw = json.load(f)

    if "configs" in raw:                                   # v2
        raw.setdefault("version", _STATE_VERSION)
        return raw

    return {                                               # v1 -> (6, 0)
        "version": _STATE_VERSION,
        "seed": raw.get("seed"),
        "OD": raw.get("OD"),
        "L": raw.get("L"),
        "G": raw.get("G"),
        "propagated": raw.get("propagated", False),
        "configs": {
            config_label(LEGACY_CONFIG): {
                "num_generated": raw.get("num_generated", 0),
                "domains": raw.get("domains"),
            }
        },
    }


def _save_sobol_state(instance: str, state: dict) -> None:
    """
    Persist Sobol engine state.

    JSON structure
    --------------
    {
      "version":    2,
      "seed":       <int|null>,     # base seed; configuration streams use seed + fixed offset
      "OD", "L", "G": <float|null>, # constraint constants used
      "propagated": <bool>,
      "configs": {
        "N6_T0": {
          "num_generated": <int>,   # raw Sobol draws so far (used by fast_forward on resume)
          "domains": {"R": [lo, hi], "A": [lo, hi], "CC": [lo, hi], "VC": [lo, hi]}
        },
        "N0_T0": { ... }
      }
    }
    """
    with open(_sobol_state_path(instance), "w") as f:
        json.dump(state, f, indent=2)


def _domains_record(domains: dict | None) -> dict:
    """Domains actually sampled: propagated bounds if provided, else the original PARAMS box."""
    if domains is not None:
        return {name: list(domains[name]) for name in PARAM_COLS}
    return {name: [lo, hi] for name, _, lo, hi in PARAMS}


def _raw_sobol_to_df(raw: np.ndarray, params: list | None = None) -> pd.DataFrame:
    """Map a (n, d) Sobol array in [0, 1) to the parameter space (four columns)."""
    params = params or PARAMS
    df = pd.DataFrame(index=range(len(raw)), columns=[p[0] for p in params])
    for i, (name, dtype, lo, hi) in enumerate(params):
        col = raw[:, i]
        if dtype == "float":
            df[name] = np.round(sp_uniform(loc=lo, scale=hi - lo).ppf(col), 2)
        elif dtype == "int":
            df[name] = sp_randint(lo, hi + 1).ppf(col).astype(int)
    return df[PARAM_COLS]


def generate_sobol(
    k: int,
    seed: int | None = None,
    instance: str | None = None,
    OD: float | None = None,
    L: float | None = None,
    max_attempts: int = 500,
    domains: dict | None = None,
    G: float | None = None,
    configs=None,
) -> pd.DataFrame:
    """
    Generate k Sobol points per configuration, optionally appending to a named persistent instance.

    Parameters
    ----------
    k : int
        Number of NEW valid points requested for EACH configuration. Must be a power of 2;
        if not, rounded up to the next power of 2 with a warning.
    seed : int or None
        Base seed for the Sobol engines (configuration streams use seed + a fixed offset;
        (6, 0) uses the seed itself). Ignored when resuming an existing instance (the
        original seed is reused and each engine is fast-forwarded to maintain sequence
        continuity). An unseeded new instance gets a random seed that is persisted, so it
        can be resumed.
    instance : str or None
        Name of a persistent instance (no file extension). State is stored in
        <instance>.sobol_state.json and the sample in <instance>.csv.
        If the instance already exists, new points are appended and each configuration's
        Sobol sequence continues without repetition via fast_forward(). Files written
        before the (N, T) extension are read as (6, 0) and migrated.
    OD, L : float or None
        Constraint constants. Both must be provided to enable constraint filtering.
    max_attempts : int
        Maximum blocks of k draws per configuration when constraints knock out points
        (ellipsoidal holes accept only ~6 % of the box, so this is generous).
    domains : dict or None
        Propagated domains from propagate_domains(). A flat dict is only valid for a
        single configuration; for several pass {(N, T): domains}. If provided, sampling
        is restricted to the tightened bounds before constraint filtering.
    G : float or None
        Gap constant for the constraints. None -> domains["G"] if present, else 0.
    configs : sequence of (N, T), spec string, or None
        None = legacy single configuration (6, 0). The CLI passes every sampleable one.

    Returns
    -------
    pd.DataFrame with columns R, A, CC, VC, T, N.
        Full accumulated sample (existing + new) when using an instance,
        otherwise just the newly generated points.
    """
    # --- Power-of-2 check ---
    if not _is_power_of_2(k):
        k_orig = k
        k = _next_power_of_2(k)
        print(f"  Warning: {k_orig} is not a power of 2. Rounding up to {k}.")

    cfgs            = _resolve_configs(configs)
    use_constraints = OD is not None and L is not None
    propagated      = domains is not None
    G_eff           = _resolve_G(G, domains)

    # --- Load existing instance state if resuming ---
    state: dict = {"version": _STATE_VERSION, "seed": None, "configs": {}}
    existing_df: pd.DataFrame | None = None
    effective_seed = seed

    if instance is not None:
        state = _load_sobol_state(instance)
        any_previous = any(c.get("num_generated", 0) > 0 for c in state["configs"].values())
        if any_previous:
            effective_seed = state.get("seed")
        elif effective_seed is None:
            effective_seed = secrets.randbits(31)   # persisted below, so resume is a real continuation

        instance_csv = f"{instance}.csv"
        if os.path.exists(instance_csv):
            existing_df = _ensure_config_columns(pd.read_csv(instance_csv))
            positions = {key: c.get("num_generated", 0) for key, c in state["configs"].items()}
            print(f"  Instance '{instance}': resuming — {len(existing_df)} existing points, "
                  f"engine positions {positions}.")

    # --- Draw k valid new points for every configuration ---
    new_frames: list[pd.DataFrame] = []
    for cfg in cfgs:
        key    = config_label(cfg)
        cstate = state["configs"].get(key, {"num_generated": 0, "domains": None})
        n_prev = int(cstate.get("num_generated", 0))
        dom    = _domains_for(domains, cfg, len(cfgs))
        params = domains_to_params(dom) if dom is not None else PARAMS

        engine = qmc.Sobol(d=len(params), scramble=True, seed=_config_seed(effective_seed, cfg))
        if n_prev > 0:
            engine.fast_forward(n_prev)

        collected: list[pd.DataFrame] = []
        total_valid = 0
        total_drawn = 0
        attempts    = 0

        for attempts in range(1, max_attempts + 1):
            raw = engine.random(k)
            total_drawn += k
            batch = _attach_config(_raw_sobol_to_df(raw, params=params), cfg)
            valid = apply_constraints(batch, OD, L, G_eff, verbose=False) if use_constraints else batch
            collected.append(valid)
            total_valid += len(valid)
            if total_valid >= k:
                break
        else:
            print(f"  Warning [{key}]: only {total_valid}/{k} valid points "
                  f"after {max_attempts} attempts.")

        accept = 100.0 * total_valid / total_drawn if total_drawn else 0.0
        print(f"  [{key}] {min(total_valid, k)}/{k} valid points "
              f"({total_drawn} drawn in {attempts} block(s), {accept:.1f}% accepted)")

        new_frames.append(pd.concat(collected, ignore_index=True).head(k))
        state["configs"][key] = {
            "num_generated": n_prev + total_drawn,
            "domains":       _domains_record(dom),
        }

    new_df = pd.concat(new_frames, ignore_index=True)[COLUMN_ORDER]

    # --- Persist instance state and merged CSV ---
    if instance is not None:
        state.update({
            "version":    _STATE_VERSION,
            "seed":       effective_seed,
            "OD":         OD,
            "L":          L,
            "G":          G_eff if use_constraints else None,
            "propagated": propagated,
        })
        _save_sobol_state(instance, state)

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
    Draw n points from an existing sample using stratified sampling,
    preserving the space-filling spread of the original design.

    When the sample contains several (N, T) configurations, n is first allocated across
    them in proportion to their sizes (at least one each when n allows) and the strategy
    is applied inside each configuration. Distances use the R, A, CC, VC columns only.
    """
    K = len(df)
    if n > K:
        raise ValueError(f"Cannot select {n} points from a sample of {K}.")
    if n == K:
        return df.reset_index(drop=True)

    if strategy not in ("random", "maxmin"):
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'random' or 'maxmin'.")

    if all(c in df.columns for c in CONFIG_COLS):
        labels = df["N"].astype(int).astype(str) + "_" + df["T"].astype(int).astype(str)
        if labels.nunique() > 1:
            sizes = labels.value_counts().sort_index().to_dict()
            alloc = _allocate(n, sizes)
            parts = []
            for label, n_g in alloc.items():
                if n_g == 0:
                    continue
                group = df[labels == label].reset_index(drop=True)
                parts.append(_subsample_one(group, n_g, strategy, sort_col, seed))
            return pd.concat(parts, ignore_index=True)

    return _subsample_one(df, n, strategy, sort_col, seed)


def _subsample_one(df, n, strategy, sort_col, seed):
    if n >= len(df):
        return df.reset_index(drop=True)
    if strategy == "random":
        return _stratified_random(df, n, sort_col=sort_col, seed=seed)
    return _stratified_maxmin(df, n)


def _allocate(n: int, sizes: dict) -> dict:
    """Largest-remainder allocation of n draws over groups, each capped at its size."""
    labels = list(sizes)
    total = sum(sizes.values())
    quota = {l: n * sizes[l] / total for l in labels}
    alloc = {l: min(sizes[l], int(math.floor(quota[l]))) for l in labels}
    floor_each = 1 if n >= len(labels) else 0
    for l in labels:
        alloc[l] = max(alloc[l], min(floor_each, sizes[l]))
    while sum(alloc.values()) < n:
        room = [l for l in labels if alloc[l] < sizes[l]]
        l = max(room, key=lambda x: quota[x] - alloc[x])
        alloc[l] += 1
    while sum(alloc.values()) > n:
        over = [l for l in labels if alloc[l] > floor_each]
        l = min(over, key=lambda x: quota[x] - alloc[x])
        alloc[l] -= 1
    return alloc


def _distance_values(df: pd.DataFrame) -> np.ndarray:
    """Numeric matrix used for distances: everything except the T / N configuration columns."""
    cols = [c for c in df.columns if c not in CONFIG_COLS] or list(df.columns)
    return df[cols].values.astype(float)


def _stratified_random(df, n, sort_col, seed):
    rng = np.random.default_rng(seed)
    if sort_col is not None:
        if sort_col not in df.columns:
            raise ValueError(f"sort_col '{sort_col}' not found in DataFrame.")
        order = df[sort_col].values
    else:
        values = _distance_values(df)
        normed = (values - values.min(axis=0)) / (np.ptp(values, axis=0) + 1e-12)
        order = np.linalg.norm(normed, axis=1)
    sorted_idx = np.argsort(order)
    strata = np.array_split(sorted_idx, n)
    chosen = [rng.choice(stratum) for stratum in strata]
    return df.iloc[chosen].reset_index(drop=True)


def _stratified_maxmin(df, n):
    values = _distance_values(df)
    col_min = values.min(axis=0)
    col_range = np.ptp(values, axis=0) + 1e-12
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
        description="Latin Hypercube / Sobol sampler for the (N, T) configurations, "
                    "with optional constraints and stratified subsampling."
    )
    # --- Configurations ---
    parser.add_argument("--configs", type=str, default="all",
                        help="Configurations to sample: comma separated N:T pairs (6:0,0:0) or "
                             "NxTy names (N6T0), 'all' = every configuration with derived "
                             "constraints (default), 'legacy' = 6:0. K points are drawn PER "
                             "configuration. Placeholder configurations (T=1) are rejected.")
    # --- LHS ---
    parser.add_argument("--k", type=int, default=None,
                        help="Number of LHS points to generate per configuration.")
    parser.add_argument("--output", type=str, default="lhs_samples.csv",
                        help="Output CSV for the full LHS sample (default: lhs_samples.csv).")
    parser.add_argument("--OD", type=float, default=None,
                        help="Constant OD used in constraints (outer diameter).")
    parser.add_argument("--L", type=float, default=None,
                        help="Constant L used in constraints (length).")
    parser.add_argument("--G", "--gap", type=float, default=0.0,
                    help="Gap constant G used in the constraints (default: 0.0).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--propagate", action="store_true",
                        help="Tighten parameter domains analytically via constraint propagation "
                             "before sampling. Requires --OD and --L. Significantly reduces "
                             "wasted samples when constraints are tight.")
    # --- Stratified subsampling ---
    parser.add_argument("--subsample", type=int, default=None, metavar="N",
                        help="If set, draw N points from the LHS via stratified subsampling "
                             "(allocated across configurations).")
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
                        help="Generate K Sobol points per configuration (rounded up to next power of 2 if needed).")
    parser.add_argument("--sobol-output", type=str, default="sobol_samples.csv",
                        dest="sobol_output",
                        help="Output CSV for a standalone Sobol sample (default: sobol_samples.csv). "
                             "Ignored when --sobol-instance is set.")
    parser.add_argument("--sobol-instance", type=str, default=None, dest="sobol_instance",
                        metavar="NAME",
                        help="Named persistent Sobol instance. Saves/resumes NAME.csv and "
                             "NAME.sobol_state.json. New points are appended each run.")

    return parser.parse_args()


def _print_config_counts(df: pd.DataFrame) -> None:
    counts = df.groupby(["N", "T"]).size()
    summary = ", ".join(f"(N={n}, T={t}): {c}" for (n, t), c in counts.items())
    print(f"Points per configuration -> {summary}")


def main():
    args = parse_args()

    try:
        configs = parse_configs(args.configs)
    except (ValueError, ConstraintsNotDefinedError) as exc:
        raise SystemExit(f"error: --configs: {exc}")

    use_constraints = args.OD is not None and args.L is not None
    if (args.OD is None) != (args.L is None):
        print("Warning: both --OD and --L must be provided to enable constraints. Ignoring.")
        use_constraints = False

    cfg_names = ", ".join(config_label(c) for c in configs)

    # --- Domain propagation (per configuration) ---
    domains: dict | None = None
    if args.propagate:
        if not use_constraints:
            print("Warning: --propagate requires --OD and --L. Skipping propagation.")
        else:
            print(f"\nRunning domain propagation...")
            domains = {}
            for cfg in configs:
                domains[cfg] = propagate_domains(args.OD, args.L, args.G, cfg)
            print()

    # ------------------------------------------------------------------ LHS
    if args.k is not None:
        params = {cfg: domains_to_params(dom) for cfg, dom in domains.items()} if domains is not None else None
        print(f"\nGenerating {args.k} LHS points per configuration...")
        print(f"Configs     : {cfg_names}")
        print(f"Constraints : {'ENABLED  (OD=' + str(args.OD) + ', L=' + str(args.L) + ', G=' + str(args.G) + ')' if use_constraints else 'DISABLED'}")
        print(f"Propagation : {'ENABLED' if domains is not None else 'DISABLED'}")
        print(f"Random seed : {args.seed if args.seed is not None else 'not set'}\n")

        if use_constraints:
            df = sample_with_constraints(args.k, args.OD, args.L, args.G, seed=args.seed,
                                         params=params, configs=configs)
        else:
            df = generate_lhs_configs(args.k, seed=args.seed, params=params, configs=configs)

        df.to_csv(args.output, index=False)
        print(f"\nFull LHS: {len(df)} point(s) written to '{args.output}'.")
        _print_config_counts(df)
        print("\nSample preview (first 5 rows):")
        print(df.head().to_string(index=False))

        if use_constraints and len(df) < args.k * len(configs):
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
        print(f"\n--- Sobol sampling: requesting {args.sobol} points per configuration ---")
        print(f"Configs     : {cfg_names}")
        print(f"Constraints : {'ENABLED  (OD=' + str(args.OD) + ', L=' + str(args.L) + ', G=' + str(args.G) + ')' if use_constraints else 'DISABLED'}")
        print(f"Propagation : {'ENABLED' if domains is not None else 'DISABLED'}")

        sobol_df = generate_sobol(
            k=args.sobol,
            seed=args.seed,
            instance=args.sobol_instance,
            OD=args.OD if use_constraints else None,
            L=args.L if use_constraints else None,
            domains=domains,
            G=args.G,
            configs=configs,
        )

        if args.sobol_instance is None:
            sobol_df.to_csv(args.sobol_output, index=False)
            print(f"  {len(sobol_df)} point(s) written to '{args.sobol_output}'.")

        _print_config_counts(sobol_df)
        print(f"\nSobol preview (first 5 rows):\n{sobol_df.head().to_string(index=False)}")


if __name__ == "__main__":
    main()
