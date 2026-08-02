# TIER: strong
"""Skill-tracking policy (the intended insight).

Instead of a fixed target storage level, size the DAILY target headroom
from two terms:
  unseen risk  = (1 - skill[t]) * flash_risk * capacity
                 -- how much surge VOLUME could be hiding behind an
                 untrustworthy forecast at this lead time (flash_risk is the
                 site's known worst-case surge, as a fraction of capacity --
                 a multi-day pulse can dump more water than any one day's
                 rate suggests, so the standing buffer must be sized off the
                 total volume, not a daily rate).
  visible risk = skill[t] * max(0, forecast[t] - climatology[t])
                 -- an above-baseline signal, discounted by how much the
                 forecast at this lead can actually be trusted.

Then, for each day t, look forward across a planning window and take the
tightest (smallest) storage level that STILL allows reaching every future
day's required low point by draining at the release-rate cap (a backward
reachability / scheduling relaxation): if day t+k needs headroom H(t+k),
today's storage must be <= C - H(t+k) + r_max*k, since we can shed at most
r_max*k over the next k days. Taking the min over k gives the tightest
storage level consistent with every upcoming risk day -- this is what lets
the policy sit close to full when skill stays good far out (there is
plenty of runway to react) and forces it to hold real headroom NOW when
skill decays fast (there won't be runway later)."""
import sys, json

HIGH_FRAC = 0.92
MIN_FRAC = 0.15
WINDOW = 60


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]; C = inst["capacity"]; Dmin = inst["dead_storage"]
    Rmax = inst["r_max"]; demand = inst["demand"]; forecast = inst["forecast"]
    skill = inst["skill"]; climatology = inst["climatology"]
    flash_risk = inst["flash_risk"]

    headroom_needed = []
    for t in range(T):
        unseen = (1.0 - skill[t]) * flash_risk * C
        visible = skill[t] * max(0.0, forecast[t] - climatology[t])
        headroom_needed.append(unseen + visible)

    target_storage = []
    for t in range(T):
        best = HIGH_FRAC * C
        w = min(WINDOW, T - t)
        for k in range(w):
            allowed_today = (C - headroom_needed[t + k]) + Rmax * k
            if allowed_today < best:
                best = allowed_today
        if best < MIN_FRAC * C:
            best = MIN_FRAC * C
        if best > HIGH_FRAC * C:
            best = HIGH_FRAC * C
        target_storage.append(best)

    S_est = inst["init_storage"]
    release = []
    for t in range(T):
        inflow_guess = forecast[t]
        proj = S_est + inflow_guess
        rel_demand = min(demand[t], max(0.0, proj - Dmin))
        proj_after_demand = proj - rel_demand
        rel = rel_demand
        tgt = target_storage[t]
        if proj_after_demand > tgt:
            rel += proj_after_demand - tgt
        rel = max(0.0, min(rel, Rmax, proj - Dmin))
        release.append(rel)
        S_est = proj - rel
        if S_est < Dmin: S_est = Dmin
        if S_est > C: S_est = C

    print(json.dumps({"release": release}))


if __name__ == "__main__":
    main()
