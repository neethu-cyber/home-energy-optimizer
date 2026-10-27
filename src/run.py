"""Experiments on real 2025-2026 German day-ahead prices and Nürnberg weather.

Part 1 (12 months, 2 Jan 2025 -> 2 Jan 2026): what do PV, a battery, a dynamic tariff and smart control each save?
Part 2 (2 Jan 2026 -> 30 Sep 2026, out of sample): how much is a better PV forecast worth in euros?

Usage:  python src/run.py     (from the project root). Outputs go to results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import House, Tariff, household, load_market_weather, prices
from forecast import PVModel
from simulate import decision_hours, horizon_end, run_mpc, run_rule

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)


def idx(df, stamp):
    return int(np.searchsorted(df["local"].dt.tz_localize(None).to_numpy(), np.datetime64(stamp)))


def table(results):
    rows = {k: {m: v for m, v in r.items() if not m.startswith("_")} for k, r in results.items()}
    return pd.DataFrame(rows).T


def main():
    house, tariff = House(), Tariff()
    df = household(load_market_weather(), house)
    fixed, dyn, sell = prices(df, tariff)
    no_pv = df.copy()
    no_pv["pv"] = 0.0

    # ---------------- Part 1: one full year ----------------
    s1, e1 = idx(df, "2025-01-02T13:00"), idx(df, "2026-01-02T13:00")
    print("Part 1: simulating 12 months (2 Jan 2025 - 2 Jan 2026) ...")
    p1 = {
        "1 No PV, no battery (fixed tariff)": run_rule(no_pv, house, fixed, sell, s1, e1, battery=False),
        "2 PV only (fixed tariff)": run_rule(df, house, fixed, sell, s1, e1, battery=False),
        "3 PV + battery, standard rule (fixed tariff)": run_rule(df, house, fixed, sell, s1, e1),
        "4 PV + battery, standard rule (dynamic tariff)": run_rule(df, house, dyn, sell, s1, e1),
        "5 Smart MILP control (dynamic tariff, simple forecast)": run_mpc(df, house, dyn, sell, s1, e1, "persistence"),
        "6 Smart MILP control (dynamic tariff, perfect forecast)": run_mpc(df, house, dyn, sell, s1, e1, "perfect"),
    }
    t1 = table(p1)
    base = t1["annual_cost_eur"].iloc[0]
    t1["saving_vs_no_pv_eur"] = base - t1["annual_cost_eur"]
    t1.to_csv(OUT / "part1_strategies_2025.csv")

    # ---------------- Part 2: value of forecasting (out of sample) ----------------
    print("Part 2: training the ML PV forecast on 2025 and testing on 2026 ...")
    train_dec = decision_hours(df, s1, e1)
    model = PVModel().fit(df, train_dec, [np.arange(t0, horizon_end(df, t0)) for t0 in train_dec])
    s2, e2 = idx(df, "2026-01-02T13:00"), idx(df, "2026-09-29T13:00")
    p2 = {
        "Standard rule (dynamic tariff)": run_rule(df, house, dyn, sell, s2, e2),
        "Smart MILP, persistence forecast": run_mpc(df, house, dyn, sell, s2, e2, "persistence"),
        "Smart MILP, ML forecast (gradient boosting)": run_mpc(df, house, dyn, sell, s2, e2, "ml", pv_model=model),
        "Smart MILP, perfect forecast": run_mpc(df, house, dyn, sell, s2, e2, "perfect"),
    }
    t2 = table(p2)
    t2.to_csv(OUT / "part2_forecast_value_2026.csv")

    # ---------------- Real price facts ----------------
    y25 = df[(df["local"] >= "2025-01-01") & (df["local"] < "2026-01-01")]
    facts = {
        "mean_spot_ct": 100 * y25["spot"].mean(),
        "negative_price_hours": int((y25["spot"] < 0).sum()),
        "min_spot_ct": 100 * y25["spot"].min(),
        "max_spot_ct": 100 * y25["spot"].max(),
        "pv_yield_kwh_per_kwp": y25["pv"].sum() / house.pv_kwp,
        "household_load_kwh": y25["load"].sum(),
        "ev_kwh": y25["ev_need"].sum(),
        "mean_dynamic_buy_ct": 100 * dyn[(df["local"] >= "2025-01-01").to_numpy() & (df["local"] < "2026-01-01").to_numpy()].mean(),
    }
    pd.Series(facts).to_csv(OUT / "data_facts_2025.csv", header=["value"])

    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60)
    cols = ["annual_cost_eur", "saving_vs_no_pv_eur", "self_sufficiency", "grid_import_kwh", "grid_export_kwh",
            "avg_import_price_ct", "battery_cycles", "unpaid_export_negative_price_kwh", "ev_charged_at_night_share"]
    fmt = lambda v: f"{v:,.2f}"
    print("\n=== Data facts 2025 ===")
    print(pd.Series(facts).to_string(float_format=fmt))
    print("\n=== Part 1: 12 months, 2 Jan 2025 - 2 Jan 2026 ===")
    print(t1[cols].to_string(float_format=fmt))
    print("\n=== Part 2: 2 Jan - 30 Sep 2026 (out of sample) ===")
    print(t2[["annual_cost_eur", "pv_forecast_mae_kwh", "self_sufficiency", "avg_import_price_ct", "battery_cycles"]]
          .rename(columns={"annual_cost_eur": "cost_eur"}).to_string(float_format=fmt))

    plot_price_heatmap(y25)
    plot_strategies(t1)
    plot_week(df, p1, dyn)
    plot_forecast_value(t2)
    print(f"\nResults written to {OUT}")


def plot_price_heatmap(y25):
    piv = y25.assign(month=y25["local"].dt.month, hour=y25["local"].dt.hour).pivot_table(
        index="hour", columns="month", values="spot", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv, aspect="auto", cmap="RdYlGn_r", origin="lower")
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(0, 24, 3))
    ax.set_ylabel("Hour of day (German time)")
    ax.set_title("Average German day-ahead price 2025 (ct/kWh), real SMARD data")
    fig.colorbar(im, ax=ax, label="ct/kWh")
    fig.tight_layout()
    fig.savefig(OUT / "price_heatmap_2025.png", dpi=150)
    plt.close(fig)


def plot_strategies(t1):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    names = [n[2:] for n in t1.index]
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, t1["annual_cost_eur"], color=["#999999", "#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#6c757d"])
    for yi, (c, s) in zip(y, zip(t1["annual_cost_eur"], t1["self_sufficiency"])):
        ax.text(c + 20, yi, f"{c:,.0f} EUR  |  {s:.0%} self-sufficient", va="center", fontsize=8)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("Annual electricity cost incl. EV and heat pump [EUR]")
    ax.set_title("What each step saves: real 2025 prices and Nürnberg weather")
    ax.set_xlim(0, t1["annual_cost_eur"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(OUT / "annual_cost_by_strategy.png", dpi=150)
    plt.close(fig)


def plot_week(df, p1, dyn):
    """A spring week with negative prices: standard rule vs. smart control."""
    a = idx(df, "2025-05-05T00:00")
    b = a + 24 * 7
    loc = df["local"].iloc[a:b].dt.tz_localize(None)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.5, 1.5]})
    axes[0].step(loc, 100 * dyn[a:b], where="post", color="black", lw=1)
    axes[0].axhline(0, color="#999999", lw=0.6)
    axes[0].set_ylabel("Dynamic price\n[ct/kWh]")
    axes[0].set_title("Week of 5 May 2025: standard battery rule vs. smart MILP control (real prices)")
    for ax, key, label in [(axes[1], "4 PV + battery, standard rule (dynamic tariff)", "Standard rule"),
                           (axes[2], "6 Smart MILP control (dynamic tariff, perfect forecast)", "Smart MILP")]:
        s = p1[key]["_series"]
        o = a - s["start"]
        ax.fill_between(loc, df["pv"].iloc[a:b], step="post", color="#f4d35e", alpha=0.6, label="PV")
        ax.step(loc, s["ev"][o:o + 24 * 7], where="post", color="#3a86ff", lw=1.2, label="EV charging")
        ax.step(loc, s["buy"][o:o + 24 * 7], where="post", color="#d62828", lw=1, label="Grid import")
        ax2 = ax.twinx()
        ax2.plot(loc, s["soc"][o:o + 24 * 7], color="#2a9d8f", lw=1.6, label="Battery level")
        ax2.set_ylim(0, 10.5)
        ax2.set_ylabel("Battery [kWh]", color="#2a9d8f")
        ax.set_ylabel(f"{label}\n[kWh per hour]")
        ax.set_ylim(0, 12)
        ax.legend(loc="upper left", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "example_week_may_2025.png", dpi=150)
    plt.close(fig)


def plot_forecast_value(t2):
    sub = t2.iloc[1:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in sub.iterrows():
        ax.scatter(r["pv_forecast_mae_kwh"], r["annual_cost_eur"], s=70)
        ax.annotate(name.replace("Smart MILP, ", ""), (r["pv_forecast_mae_kwh"], r["annual_cost_eur"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(t2["annual_cost_eur"].iloc[0], color="#999999", ls="--", lw=1)
    ax.text(0, t2["annual_cost_eur"].iloc[0], " standard rule", va="bottom", fontsize=8, color="#666666")
    ax.set_xlabel("PV forecast error (MAE, kWh per hour)")
    ax.set_ylabel("Cost 2 Jan - 30 Sep 2026 [EUR]")
    ax.set_title("What a better PV forecast is worth (out of sample, 2026)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_value_2026.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
