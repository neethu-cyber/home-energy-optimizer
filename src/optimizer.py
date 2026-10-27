"""Mixed-integer schedule for one planning horizon (up to 36 hours).

For each hour t the optimizer chooses:
    buy[t], sell[t] >= 0        grid import / export (kWh)
    ch[t], dis[t]   >= 0        home battery charge / discharge (kWh)
    soc[t]          >= 0        battery state of charge at the end of hour t
    ev[t]           >= 0        EV charging (kWh), only while the car is plugged in
    curt[t]         >= 0        PV curtailment (only worth it when export would cost money)
    zb[t], zg[t] in {0,1}       battery direction and grid direction (no simultaneous charge/discharge,
                                no simultaneous buy/sell; both matter when prices turn negative)

min  sum_t  buy_price[t] * buy[t] - sell_price[t] * sell[t] + wear * dis[t]  -  v_end * soc[T]

s.t. pv[t] - curt[t] + dis[t] + buy[t] = load[t] + ev[t] + ch[t] + sell[t]     energy balance
     soc[t] = soc[t-1] + eta * ch[t] - dis[t] / eta,   soc[t] <= capacity        battery dynamics
     ch[t] <= P * zb[t],   dis[t] <= P * (1 - zb[t])                              battery power and direction
     buy[t] <= G * zg[t],  sell[t] <= G * (1 - zg[t])                             grid connection and direction
     ev[t] <= P_ev * plugged[t]
     sum_{t in session} ev[t] = need[session]   if the car leaves within the horizon
     sum_{t in session} ev[t] <= need[session]  otherwise

v_end values energy left in the battery at the end of the horizon, so the optimizer does not empty it
just because the horizon ends.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def schedule(pv, load, buy_price, sell_price, plugged, session, needs, soc0, house, v_end=0.15, closed=None):
    """Return (ch, dis, ev) for each hour of the horizon.

    session: session id per hour (-1 = not plugged); needs: dict session id -> kWh still needed
    closed:  set of session ids whose departure lies inside the horizon (need must be met exactly)
    """
    T = len(pv)
    nv = 9
    BUY, SELL, CH, DIS, SOC, EV, CURT, ZB, ZG = range(nv)
    idx = lambda k, t: k * T + t
    n = nv * T
    c = np.zeros(n)
    lb, ub = np.zeros(n), np.full(n, np.inf)
    integ = np.zeros(n)
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = [0]

    def add(coefs, l, u):
        for j, v in coefs:
            rows.append(r[0]); cols.append(j); vals.append(v)
        lo.append(l); hi.append(u); r[0] += 1

    eta, P, G = house.batt_eff, house.batt_kw, house.grid_kw
    for t in range(T):
        c[idx(BUY, t)] = buy_price[t]
        c[idx(SELL, t)] = -sell_price[t]
        c[idx(DIS, t)] = house.batt_wear
        ub[idx(SOC, t)] = house.batt_kwh
        ub[idx(EV, t)] = house.ev_kw * plugged[t]
        ub[idx(CURT, t)] = pv[t]
        for z in (ZB, ZG):
            ub[idx(z, t)] = 1
            integ[idx(z, t)] = 1
        add([(idx(BUY, t), 1), (idx(DIS, t), 1), (idx(CURT, t), -1), (idx(SELL, t), -1), (idx(EV, t), -1),
             (idx(CH, t), -1)], load[t] - pv[t], load[t] - pv[t])
        prev = [(idx(SOC, t - 1), -1)] if t > 0 else []
        rhs = soc0 if t == 0 else 0.0
        add([(idx(SOC, t), 1), (idx(CH, t), -eta), (idx(DIS, t), 1 / eta)] + prev, rhs, rhs)
        add([(idx(CH, t), 1), (idx(ZB, t), -P)], -np.inf, 0)
        add([(idx(DIS, t), 1), (idx(ZB, t), P)], -np.inf, P)
        add([(idx(BUY, t), 1), (idx(ZG, t), -G)], -np.inf, 0)
        add([(idx(SELL, t), 1), (idx(ZG, t), G)], -np.inf, G)
    c[idx(SOC, T - 1)] -= v_end

    closed = closed or set()
    for s in set(session) - {-1}:
        hours = [t for t in range(T) if session[t] == s]
        need = needs.get(s, 0.0)
        cap = house.ev_kw * len(hours)
        target = min(need, cap)
        add([(idx(EV, t), 1) for t in hours], target if s in closed else 0.0, target)

    A = coo_matrix((vals, (rows, cols)), shape=(r[0], n)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"mip_rel_gap": 1e-4, "time_limit": 30})
    if res.x is None:
        raise RuntimeError(res.message)
    x = res.x
    get = lambda k: x[k * T:(k + 1) * T]
    return get(CH), get(DIS), get(EV)
