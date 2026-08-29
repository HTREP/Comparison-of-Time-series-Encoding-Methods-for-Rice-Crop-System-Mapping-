import argparse
import os

import numpy as np
import pandas as pd

from preprocess import preprocess_daily

CLASSES = {"SRC": 2, "DRC": 4, "HRC": 5, "TRC": 6}
T_DAYS = 726
REVISIT = 5
CLOUD_FRACTION = 0.25


def synth_profile(n_cycles, rng):
    t = np.arange(T_DAYS)
    period = T_DAYS / n_cycles
    phase = rng.uniform(0, period)
    base = 0.5 * (1 + np.sin(2 * np.pi * (t - phase) / period))
    base = base ** rng.uniform(1.4, 2.2)
    amp = rng.uniform(0.45, 0.70)
    floor = rng.uniform(0.04, 0.12)
    drift = rng.normal(0, 0.02, T_DAYS).cumsum() * 0.01
    return np.clip(floor + amp * base + drift, 0.0, 1.0)


def sample_observations(profile, rng):
    days = np.arange(0, T_DAYS, REVISIT, dtype=np.float64)
    vals = profile[days.astype(int)] + rng.normal(0, 0.03, len(days))
    n_cloud = int(len(days) * CLOUD_FRACTION)
    cloudy = rng.choice(len(days), n_cloud, replace=False)
    vals[cloudy] = np.nan
    keep = np.isfinite(vals)
    return days[keep], vals[keep]


def build_split(n_per_class, seed):
    rng = np.random.default_rng(seed)
    target = np.arange(T_DAYS, dtype=np.float64)
    rows = []
    for cls, n_cycles in CLASSES.items():
        for i in range(n_per_class):
            profile = synth_profile(n_cycles, rng)
            days, vals = sample_observations(profile, rng)
            daily = preprocess_daily(vals, days, target)[0]
            rows.append({"point_id": f"{cls}_{i:03d}", "class": cls,
                         "x": float(rng.uniform(580000, 635000)),
                         "y": float(rng.uniform(1562000, 1653000)),
                         **{f"t{d}": float(v) for d, v in enumerate(daily)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--n-per-class", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for name, seed in [("train", 1), ("eval", 2)]:
        df = build_split(args.n_per_class, seed)
        path = os.path.join(args.outdir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{path}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
