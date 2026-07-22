# TIER: strong
# Anticipatory-buffer target scheduling.
#
# INSIGHT: the inflow array mixes an untagged slow seasonal drift with fast storm bursts,
# and the outlet's own reaction speed (Rmax per day) is fixed. A fixed reactive target
# (the greedy tier) only reacts to a storm on the day it lands, when Rmax can no longer
# drain away more than Rmax -- so any burst bigger than that overtops the target's slack
# no matter what you release that day. Since the WHOLE record is known in advance, the
# only way to avoid this is to lower the level BEFORE the storm arrives, using exactly
# the amount of headroom the storm will need and no more (over-buffering trades flood
# risk for shortage risk against min_level).
#
# Concretely:
#   1. Estimate each day's local "slow" baseline with a rolling MEDIAN (robust to the
#      storm bursts themselves, which are sparse and brief): baseline[t] = median of
#      inflow in a +/-4-day window around t.
#   2. spike_excess[t] = max(0, inflow[t] - baseline[t]); only the part of that excess
#      ABOVE what a single day's Rmax can already absorb reactively needs a buffer:
#      req_buffer[t] = max(0, spike_excess[t] - Rmax).
#   3. needed[t] = the largest req_buffer within the next 10 days (a lookahead window):
#      "how much headroom must exist right now to survive whatever is coming soon".
#   4. Scale the aggressiveness of pre-draining by the instance's own flood_coef vs.
#      shortage_coef (given in the input): buffer harder when floods are punished more,
#      buffer more conservatively when shortages are punished more.
#   5. Track a dynamic target = (normal operating target) - needed[t]*scale, clipped to
#      stay above min_level + a safety margin (so pre-draining never itself causes a
#      shortage) and below a soft ceiling. React to THIS target exactly like greedy does
#      to its fixed one -- the only change is that the target itself already "knows" a
#      storm is coming and has moved out of its way in time.
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]; cap = inst["cap"]; Rmax = inst["Rmax"]
    min_level = inst["min_level"]; inflow = inst["inflow"]
    flood_coef = inst["flood_coef"]; shortage_coef = inst["shortage_coef"]

    W = 10      # lookahead window (days)
    HALF = 4    # rolling-median half-window

    baseline_local = []
    for t in range(T):
        lo = max(0, t - HALF); hi = min(T, t + HALF + 1)
        window = sorted(inflow[lo:hi])
        baseline_local.append(window[len(window) // 2])

    spike_excess = [max(0.0, inflow[t] - baseline_local[t]) for t in range(T)]
    req_buffer = [max(0.0, spike_excess[t] - Rmax) for t in range(T)]

    needed = [0.0] * T
    for t in range(T):
        hi = min(T, t + W)
        needed[t] = max(req_buffer[t:hi]) if t < hi else 0.0

    scale = 1.0 + 0.4 * (flood_coef - shortage_coef) / (flood_coef + shortage_coef)
    base_target = 0.75 * cap
    lo_bound = min_level + 0.05 * cap
    hi_bound = 0.85 * cap

    L = inst["L0"]
    out = []
    for t in range(T):
        tgt = base_target - needed[t] * scale
        tgt = max(lo_bound, min(hi_bound, tgt))
        avail = L + inflow[t]
        rel = max(0.0, avail - tgt)
        rel = min(rel, Rmax)
        out.append(rel)
        raw = avail - rel
        L = cap if raw > cap else raw

    print(json.dumps({"releases": out}))


if __name__ == "__main__":
    main()
