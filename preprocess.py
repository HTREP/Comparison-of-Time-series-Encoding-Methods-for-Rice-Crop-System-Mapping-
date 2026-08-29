import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

VAL_RANGE = (0.0, 1.0)
SG_WINDOW = 31
SG_ORDER = 3
MIN_VALID = 4


def preprocess_daily(values, obs_days, target_days):
    vals = np.asarray(values, dtype=np.float64).copy()
    if vals.ndim == 1:
        vals = vals[None, :]

    vals[(vals < VAL_RANGE[0]) | (vals > VAL_RANGE[1])] = np.nan

    keep = np.isfinite(vals).sum(axis=1) >= MIN_VALID
    if not keep.all():
        raise ValueError(f"{int((~keep).sum())} series have fewer than "
                         f"{MIN_VALID} valid observations")

    df = pd.DataFrame(vals.T, index=np.asarray(obs_days, dtype=np.float64))
    df = df.reindex(np.asarray(target_days, dtype=np.float64))
    df = df.interpolate(method="index", limit_direction="both")
    daily = df.to_numpy().T

    win = SG_WINDOW if SG_WINDOW % 2 == 1 else SG_WINDOW - 1
    if daily.shape[1] >= win and win > SG_ORDER:
        daily = savgol_filter(daily, window_length=win, polyorder=SG_ORDER, axis=1)

    return daily.astype(np.float32)
