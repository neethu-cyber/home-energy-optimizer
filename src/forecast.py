"""Day-ahead PV and load forecasts made at 13:00 (when tomorrow's spot prices are published).

perfect      the actual values (upper bound, not achievable)
persistence  same hour on the most recent day that is already observed at decision time
ml           gradient boosting on information available at 13:00: calendar, recent PV, a clear-sky proxy
             (best PV at this hour over the last 14 days) and recent cloud cover. No weather forecast is used,
             because historical weather forecasts are not freely available; that is the main limitation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def persistence(series: np.ndarray, t0: int, hours: np.ndarray) -> np.ndarray:
    """For each absolute hour j >= t0: value 24 h earlier if already observed, otherwise 48 h earlier."""
    lag = np.where(hours - 24 < t0, 24, 48)
    return series[hours - lag]


def ml_features(df: pd.DataFrame, t0: int, hours: np.ndarray) -> pd.DataFrame:
    pv, cloud = df["pv"].to_numpy(), df["cloud"].to_numpy()
    loc = df["local"].iloc[hours]
    past = slice(max(0, t0 - 24), t0)
    clear = np.array([pv[[j - 24 * d for d in range(1, 15) if j - 24 * d < t0 and j - 24 * d >= 0]].max()
                      if t0 >= 24 else 0.0 for j in hours])
    doy = loc.dt.dayofyear.to_numpy()
    return pd.DataFrame({
        "hour": loc.dt.hour.to_numpy(),
        "doy_sin": np.sin(2 * np.pi * doy / 365), "doy_cos": np.cos(2 * np.pi * doy / 365),
        "lead": (hours - t0) // 24,
        "pv_persist": persistence(pv, t0, hours),
        "pv_last24": pv[past].sum(),
        "cloud_last24": cloud[past].mean(),
        "cloud_last6": cloud[max(0, t0 - 6):t0].mean(),
        "clear_sky_proxy": clear,
    })


class PVModel:
    def __init__(self):
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=40, random_state=0)

    def fit(self, df, decisions, horizons):
        X = pd.concat([ml_features(df, t0, h) for t0, h in zip(decisions, horizons)], ignore_index=True)
        y = np.concatenate([df["pv"].to_numpy()[h] for h in horizons])
        self.model.fit(X, y)
        return self

    def predict(self, df, t0, hours):
        return np.clip(self.model.predict(ml_features(df, t0, hours)), 0, None)
