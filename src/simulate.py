"""Hour-by-hour simulation of a household with PV, battery and EV under different control strategies.

Rule-based ("self-consumption", what most home batteries do today):
    EV charges at full power as soon as it is plugged in. PV covers the house first, then the EV,
    surplus charges the battery, the rest is exported. The battery discharges to cover the house (not the EV).

Optimized (model predictive control, what smart energy managers do):
    Every day at 13:00, when tomorrow's day-ahead prices are published, solve the MILP in optimizer.py from now
    until the end of tomorrow with forecasts of PV and load, then execute the battery and EV plan for 24 hours.
    The actual PV and load differ from the forecast; the battery follows its planned power (within its limits)
    and the grid absorbs the difference.
"""
import numpy as np
import pandas as pd

from forecast import persistence
from optimizer import schedule


def _settle(pv, load, ev, ch, dis, buy_price, sell_price, wear):
    grid = load + ev + ch - pv - dis
    buy, sell = np.clip(grid, 0, None), np.clip(-grid, 0, None)
    return buy, sell, buy * buy_price - sell * sell_price + wear * dis


def run_rule(df, house, buy_price, sell_price, start, end, battery=True):
    pv, load = df["pv"].to_numpy(), df["load"].to_numpy()
    sess, need0 = df["ev_session"].to_numpy(), df["ev_need"].to_numpy()
    cap = house.batt_kwh if battery else 0.0
    soc = cap / 2
    remaining = {}
    rec = {k: np.zeros(end - start) for k in ["ev", "ch", "dis", "soc"]}
    for i, t in enumerate(range(start, end)):
        s = sess[t]
        if need0[t] > 0 or (s >= 0 and s not in remaining):
            remaining[s] = remaining.get(s, 0.0) + need0[t]
        ev = min(house.ev_kw, remaining.get(s, 0.0)) if s >= 0 else 0.0
        if s >= 0:
            remaining[s] -= ev
        surplus = pv[t] - load[t]                       # PV covers the house first
        ch = dis = 0.0
        if surplus > 0:
            surplus_after_ev = surplus - ev
            if surplus_after_ev > 0:
                ch = min(surplus_after_ev, house.batt_kw, (cap - soc) / house.batt_eff)
        else:
            dis = min(-surplus, house.batt_kw, soc * house.batt_eff)
        soc += house.batt_eff * ch - dis / house.batt_eff
        rec["ev"][i], rec["ch"][i], rec["dis"][i], rec["soc"][i] = ev, ch, dis, soc
    return _result(df, house, buy_price, sell_price, start, end, rec)


def horizon_end(df, t0):
    """Index of 00:00 local time the day after tomorrow (end of the day-ahead price window)."""
    target = (df["local"].iloc[t0].normalize() + pd.Timedelta(days=2)).tz_localize(None)
    local_naive = df["local"].dt.tz_localize(None).to_numpy()
    return min(int(np.searchsorted(local_naive, np.datetime64(target))), len(df))


def decision_hours(df, start, end):
    loc = df["local"].iloc[start:end]
    return [start + i for i in np.where(loc.dt.hour.to_numpy() == 13)[0]]


def run_mpc(df, house, buy_price, sell_price, start, end, forecast="persistence", pv_model=None, v_end=0.15):
    pv, load = df["pv"].to_numpy(), df["load"].to_numpy()
    sess, need0 = df["ev_session"].to_numpy(), df["ev_need"].to_numpy()
    soc = house.batt_kwh / 2
    rec = {k: np.zeros(end - start) for k in ["ev", "ch", "dis", "soc"]}
    decisions = decision_hours(df, start, end)
    errors = []
    for d, t0 in enumerate(decisions):
        t_end = horizon_end(df, t0)
        hours = np.arange(t0, t_end)
        if forecast == "perfect":
            pv_hat, load_hat = pv[hours], load[hours]
        else:
            load_hat = persistence(load, t0, hours)
            pv_hat = pv_model.predict(df, t0, hours) if forecast == "ml" else persistence(pv, t0, hours)
        errors.append(np.abs(pv_hat[:24] - pv[hours[:24]]))
        s = sess[hours]
        needs = {k: need0[hours][s == k].sum() for k in set(s) - {-1}}
        last = s[-1]
        closed = {k for k in needs if k != last or t_end == len(df)}
        ch, dis, ev = schedule(pv_hat, load_hat, buy_price[hours], sell_price[hours], (s >= 0).astype(int), s,
                               needs, soc, house, v_end=v_end, closed=closed)
        stop = decisions[d + 1] if d + 1 < len(decisions) else min(end, t0 + 24)
        for k, t in enumerate(range(t0, min(stop, end))):
            net = ch[k] - dis[k]
            if net >= 0:
                c_, d_ = min(net, (house.batt_kwh - soc) / house.batt_eff), 0.0
            else:
                c_, d_ = 0.0, min(-net, soc * house.batt_eff)
            soc += house.batt_eff * c_ - d_ / house.batt_eff
            i = t - start
            rec["ev"][i], rec["ch"][i], rec["dis"][i], rec["soc"][i] = ev[k], c_, d_, soc
    out = _result(df, house, buy_price, sell_price, decisions[0], min(end, decisions[-1] + 24), rec,
                  offset=decisions[0] - start)
    out["pv_forecast_mae_kwh"] = float(np.concatenate(errors).mean())
    return out


def _result(df, house, buy_price, sell_price, start, end, rec, offset=0):
    sl = slice(start, end)
    r = {k: v[offset:offset + end - start] for k, v in rec.items()}
    pv, load, spot = df["pv"].to_numpy()[sl], df["load"].to_numpy()[sl], df["spot"].to_numpy()[sl]
    buy, sell, cost = _settle(pv, load, r["ev"], r["ch"], r["dis"], buy_price[sl], sell_price[sl], house.batt_wear)
    consumption = load.sum() + r["ev"].sum()
    return {
        "annual_cost_eur": float(cost.sum()),
        "grid_import_kwh": float(buy.sum()),
        "grid_export_kwh": float(sell.sum()),
        "self_sufficiency": float(1 - buy.sum() / consumption),
        "avg_import_price_ct": float(100 * (buy * buy_price[sl]).sum() / max(buy.sum(), 1e-9)),
        "battery_cycles": float(r["dis"].sum() / max(house.batt_kwh, 1e-9)),
        "unpaid_export_negative_price_kwh": float(sell[spot < 0].sum()),
        "ev_charged_at_night_share": float(r["ev"][(df["local"].dt.hour.to_numpy()[sl] < 7)].sum() / max(r["ev"].sum(), 1e-9)),
        "hours": end - start,
        "_series": {"buy": buy, "sell": sell, "ev": r["ev"], "ch": r["ch"], "dis": r["dis"], "soc": r["soc"],
                    "start": start},
    }
