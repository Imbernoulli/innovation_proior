# TIER: strong
"""Anticipatory-drawdown insight, applied cascade-wide.

Key idea (backward "just-in-time" release scheduling): given a dam's full
inflow forecast, the cumulative amount it MUST have released by tick t to
avoid overflow is a deterministic, instance-only quantity:

    required_cum(t) = max(0, storage0 + cum_inflow(t) - capacity)

A schedule that releases as LATE as feasible while still meeting every one
of these deadlines, subject to the per-tick rate cap release_max, is found by
a backward recursion:

    CR(T-1) = required_cum(T-1)
    CR(t)   = max(required_cum(t), CR(t+1) - release_max)      for t = T-2..0
    release(t) = CR(t) - CR(t-1)

Because CR(t) is pulled up whenever the RATE CAP can't cover the gap to a
future deadline, this recursion automatically starts releasing water several
ticks BEFORE the inflow pulse that would otherwise cause an overflow -- the
exact "spill before the peak arrives" behaviour the theme calls for.

Dams are solved in cascade order (1 -> 2 -> 3) because dam i+1's forecast
includes dam i's ALREADY-DECIDED release, routed in after the travel delay --
this is the multi-dam coordination: dam 2's plan reacts to what dam 1 is
about to send it, and dam 3's plan reacts to what dam 2 is about to send it.

Global peak-shaving refinement on dam 3: avoiding dam-3 overflow is not the
same as avoiding a TOWN flood -- the town also has its own local inflow
pulse. So for dam 3 only, the backward pass is run against an ARTIFICIALLY
SHRUNK capacity in a window just before the town's forecast inflow peak. This
forces dam 3 to draw itself down earlier than strictly necessary (a locally
"wasteful" release while the reservoir isn't even close to full) purely to
buy absorption headroom for the moment the town's own pulse hits -- accepting
a local loss for a global (town-flow) gain.
"""
import sys, json


def backward_schedule(storage0, capacity, rmax, inflow, T, eff_capacity=None):
    if eff_capacity is None:
        eff_capacity = [capacity] * T
    cum_in = 0.0
    required = [0.0] * T
    for t in range(T):
        cum_in += inflow[t]
        required[t] = max(0.0, storage0 + cum_in - eff_capacity[t])

    CR = [0.0] * T
    CR[T - 1] = required[T - 1]
    for t in range(T - 2, -1, -1):
        CR[t] = max(required[t], CR[t + 1] - rmax)

    release = [0.0] * T
    prev = 0.0
    for t in range(T):
        r = CR[t] - prev
        if r < 0.0:
            r = 0.0
        if r > rmax:
            r = rmax
        release[t] = r
        prev = CR[t]
    return release


def simulate_dam(storage0, capacity, rmax, total_inflow, release, T):
    """Replay physics for one dam to get its ACTUAL release trace (needed to
    build the next dam's routed-in forecast, and to know true end storage)."""
    s = storage0
    actual = [0.0] * T
    for t in range(T):
        avail = s + total_inflow[t]
        req = min(max(release[t], 0.0), rmax)
        a = min(req, avail)
        left = avail - a
        if left > capacity:
            a += left - capacity
            s = capacity
        else:
            s = left
        actual[t] = a
    return actual


def main():
    inst = json.load(sys.stdin)
    T = inst["t_steps"]
    d1, d2, d3 = inst["dam1"], inst["dam2"], inst["dam3"]
    delay12, delay23 = inst["delay12"], inst["delay23"]
    town_in = inst["town_inflow"]

    # --- Dam 1: only its own inflow. ---
    rel1 = backward_schedule(d1["storage0"], d1["capacity"], d1["release_max"],
                              d1["inflow"], T)
    act1 = simulate_dam(d1["storage0"], d1["capacity"], d1["release_max"],
                         d1["inflow"], rel1, T)

    # --- Dam 2: own inflow + dam 1's routed release. ---
    routed2 = [act1[t - delay12] if t - delay12 >= 0 else 0.0 for t in range(T)]
    total2 = [d2["inflow"][t] + routed2[t] for t in range(T)]
    rel2 = backward_schedule(d2["storage0"], d2["capacity"], d2["release_max"],
                              total2, T)
    act2 = simulate_dam(d2["storage0"], d2["capacity"], d2["release_max"],
                         total2, rel2, T)

    # --- Dam 3: own inflow + dam 2's routed release, WITH the soft-cap
    #     anticipatory trick around the town's forecast peak. ---
    routed3 = [act2[t - delay23] if t - delay23 >= 0 else 0.0 for t in range(T)]
    total3 = [d3["inflow"][t] + routed3[t] for t in range(T)]

    town_peak_t = max(range(T), key=lambda t: town_in[t])
    window = max(delay12 + delay23, 6)
    shrink = 0.55
    eff_cap3 = []
    for t in range(T):
        if town_peak_t - window <= t <= town_peak_t:
            eff_cap3.append(d3["capacity"] * shrink)
        else:
            eff_cap3.append(d3["capacity"])

    rel3 = backward_schedule(d3["storage0"], d3["capacity"], d3["release_max"],
                              total3, T, eff_capacity=eff_cap3)

    print(json.dumps({"release1": rel1, "release2": rel2, "release3": rel3}))


if __name__ == "__main__":
    main()
