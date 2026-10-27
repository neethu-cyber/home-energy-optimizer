"""Real market and weather data, plus a model of one Nürnberg household.

Real data (in data/):
  - Day-ahead electricity prices for Germany/Luxembourg from SMARD (Bundesnetzagentur), hourly, German local time.
  - Hourly weather for Nürnberg from Open-Meteo (temperature, global horizontal irradiance, cloud cover).
    Times are fixed UTC+1 and each value is the mean of the PRECEDING hour.

Modelled from the real weather (no public household meter data exists):
  - PV output of a 10 kWp roof system
  - base household load (3,500 kWh/year)
  - heat pump electricity for space heating and hot water
  - EV charging needs
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TZ = "Europe/Berlin"


@dataclass
class House:
    pv_kwp: float = 10.0
    pv_factor: float = 0.90          # orientation gain x performance ratio (~1,100 kWh/kWp in sunny 2025)
    base_load_kwh: float = 3500.0    # per year, household appliances
    space_heat_kwh: float = 11000.0  # per year at the weather of 2025
    hot_water_kwh: float = 2500.0    # per year
    heating_limit_c: float = 15.0    # no space heating above this outdoor temperature
    batt_kwh: float = 10.0
    batt_kw: float = 5.0
    batt_eff: float = 0.95           # one-way efficiency (about 90% round trip)
    batt_wear: float = 0.02          # EUR per kWh discharged (battery degradation)
    ev_kw: float = 11.0              # wallbox
    grid_kw: float = 30.0            # house connection


@dataclass
class Tariff:
    fixed_price: float = 0.33        # EUR/kWh, typical fixed household tariff 2025
    dyn_markup: float = 0.17         # EUR/kWh on top of spot: grid fees, taxes, levies, supplier margin (before VAT)
    vat: float = 0.19
    feed_in: float = 0.079           # EUR/kWh EEG feed-in tariff, rooftop PV <= 10 kWp (2025)
    # Solarspitzengesetz (Feb 2025): new PV systems get no feed-in payment in hours with negative spot prices.
    no_feed_in_when_negative: bool = True


def load_market_weather() -> pd.DataFrame:
    """Hourly frame indexed by interval START in UTC, with local time, spot price and weather."""
    p = pd.read_csv(DATA / "smard_day_ahead_prices_hourly.csv", sep=";", encoding="utf-8-sig", na_values="-")
    start = pd.to_datetime(p["Start date"], format="%b %d, %Y %I:%M %p")
    start = start.dt.tz_localize(TZ, ambiguous="infer").dt.tz_convert("UTC")
    price = pd.Series(pd.to_numeric(p.iloc[:, 2], errors="coerce").to_numpy() / 1000.0, index=start, name="spot")

    w = pd.read_csv(DATA / "open_meteo_nuernberg_hourly.csv", skiprows=3)
    w.columns = ["time", "temp", "ghi", "cloud"]
    stamp = pd.to_datetime(w["time"]) - pd.Timedelta(hours=1)          # fixed UTC+1 -> UTC
    w.index = (stamp - pd.Timedelta(hours=1)).dt.tz_localize("UTC")      # preceding-hour mean -> interval start
    df = w[["temp", "ghi", "cloud"]].join(price, how="inner").dropna()
    df["local"] = df.index.tz_convert(TZ)
    return df


def household(df: pd.DataFrame, h: House, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    loc = out["local"]
    hour, dow, doy = loc.dt.hour.to_numpy(), loc.dt.dayofweek.to_numpy(), loc.dt.dayofyear.to_numpy()

    # PV: irradiance x size x factor, with -0.4%/degC above 25 degC cell temperature
    t_cell = out["temp"] + 0.025 * out["ghi"]
    out["pv"] = h.pv_kwp * out["ghi"] / 1000 * h.pv_factor * (1 - 0.004 * np.clip(t_cell - 25, 0, None))

    # Base load: daily shape (weekday/weekend), winter +15%, random variation
    wk = np.array([.30, .26, .24, .24, .25, .32, .55, .70, .55, .45, .42, .45,
                   .55, .48, .42, .42, .48, .62, .78, .85, .80, .68, .52, .38])
    we = np.array([.32, .28, .25, .24, .24, .26, .32, .45, .62, .70, .66, .65,
                   .70, .60, .52, .50, .55, .66, .80, .86, .80, .68, .55, .40])
    shape = np.where(dow[:, None] >= 5, we[None, :], wk[None, :])[np.arange(len(hour)), hour]
    base = shape * (1 + 0.15 * np.cos(2 * np.pi * (doy - 15) / 365)) * rng.lognormal(-0.02, 0.2, len(hour))
    year1 = (loc < loc.iloc[0] + pd.Timedelta(days=365)).to_numpy()
    out["base_load"] = base * h.base_load_kwh / base[year1].sum()

    # Heat pump: heat demand from degree-hours; COP = 45% of Carnot with a 40 degC supply temperature
    dh = np.clip(h.heating_limit_c - out["temp"].to_numpy(), 0, None)
    heat = dh * h.space_heat_kwh / dh[year1].sum()
    dhw_shape = np.where((hour >= 6) & (hour <= 8) | (hour >= 18) & (hour <= 21), 2.2, 0.6)
    heat = heat + dhw_shape * h.hot_water_kwh / dhw_shape[year1].sum()
    cop = np.clip(0.45 * (40 + 273.15) / (40 - out["temp"].to_numpy()), 2.0, 5.5)
    out["heat_pump"] = heat / cop
    out["load"] = out["base_load"] + out["heat_pump"]

    # EV: home 18:00-07:00; energy needed by 07:00 depends on the day's driving
    out["ev_plugged"] = ((hour >= 18) | (hour < 7)).astype(int)
    out["ev_session"] = -1
    out["ev_need"] = 0.0
    session_start = np.where(hour == 18)[0]
    for k, i in enumerate(session_start):
        end = min(i + 13, len(out))                                   # 18:00 -> 07:00
        weekend = dow[i] >= 5
        need = rng.choice([0.0, 12.0], p=[0.6, 0.4]) if weekend else float(np.clip(rng.normal(8, 2.5), 3, 14))
        out.iloc[i:end, out.columns.get_loc("ev_session")] = k
        out.iloc[i, out.columns.get_loc("ev_need")] = need
    return out


def prices(df: pd.DataFrame, t: Tariff):
    """Buy and sell price per hour for the fixed and the dynamic tariff (EUR/kWh)."""
    spot = df["spot"].to_numpy()
    sell = np.where((spot < 0) & t.no_feed_in_when_negative, 0.0, t.feed_in)
    fixed_buy = np.full(len(spot), t.fixed_price)
    dyn_buy = (spot + t.dyn_markup) * (1 + t.vat)
    return fixed_buy, dyn_buy, sell
