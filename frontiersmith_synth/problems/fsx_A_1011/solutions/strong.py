# TIER: strong
# Insight: cooling is not downtime, it is a scheduled resource. Reformulate each
# workpiece's next pending operation as a TIME INTERVAL [entry, exit] during
# which its own passive Newton-cooling trajectory sits inside the required
# window -- entry/exit follow in closed form from the cooling law, no simulation
# needed. The forge then just needs a single-resource interval scheduler: among
# pieces whose interval is open RIGHT NOW, run the one whose interval closes
# soonest (earliest-deadline-first), so nothing is left to expire. When nothing
# is ready, jump the clock forward to the next interval's opening instead of
# idling blindly -- that jump is exactly the "free" choreographed cooling the
# family is named for: whichever piece is not at the forge is still working.
# Only if every remaining piece has already missed its window do we spend a
# reheat, and even then we pick the most overdue piece so nothing is wasted.
import sys, math

Tamb = 20.0


def cool_time_to(T_from, T_to, k):
    if T_from <= T_to:
        return 0.0
    return math.log((T_from - Tamb) / (T_to - Tamb)) / k


def main():
    global Tamb
    toks = sys.stdin.read().split()
    it = iter(toks)
    W = int(next(it))
    Tamb = float(next(it))
    H_lo = float(next(it)); H_hi = float(next(it))
    M_lo = float(next(it)); M_hi = float(next(it))
    L_lo = float(next(it)); L_hi = float(next(it))
    REHEAT_TEMP = float(next(it))
    REHEAT_DUR = float(next(it))
    OP_DUR = float(next(it))
    PENALTY = float(next(it))
    HORIZON = float(next(it))
    bands = [(H_lo, H_hi), (M_lo, M_hi), (L_lo, L_hi)]

    state = []
    for pid in range(1, W + 1):
        T0 = float(next(it)); k = float(next(it))
        v1 = int(next(it)); v2 = int(next(it)); v3 = int(next(it))
        state.append(dict(pid=pid, idx=0, t_r=0.0, T_r=T0, k=k, v=[v1, v2, v3], done=False))

    t = 0.0
    actions = []
    remaining = W
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > 100000:
            break
        best_ready = None   # (exit_time, state)
        best_future = None  # (entry_time, state)
        for s in state:
            if s["done"]:
                continue
            lo, hi = bands[s["idx"]]
            k = s["k"]
            if s["T_r"] <= hi:
                entry_abs = s["t_r"]
            else:
                entry_abs = s["t_r"] + cool_time_to(s["T_r"], hi, k)
            exit_abs = s["t_r"] + cool_time_to(s["T_r"], lo, k)
            if entry_abs - 1e-9 <= t <= exit_abs + 1e-9:
                if best_ready is None or exit_abs < best_ready[0]:
                    best_ready = (exit_abs, s)
            elif t < entry_abs:
                if best_future is None or entry_abs < best_future[0]:
                    best_future = (entry_abs, s)

        if best_ready is not None:
            _, s = best_ready
            actions.append(("OP", s["pid"], t))
            t += OP_DUR
            s["idx"] += 1
            if s["idx"] == 3:
                s["done"] = True
                remaining -= 1
        elif best_future is not None:
            entry_abs, _ = best_future
            t = entry_abs
        else:
            # every remaining piece already missed its window -> reheat the
            # most overdue one (smallest exit time) and let it recover
            cand = None
            for s in state:
                if s["done"]:
                    continue
                lo, hi = bands[s["idx"]]
                exit_abs = s["t_r"] + cool_time_to(s["T_r"], lo, s["k"])
                if cand is None or exit_abs < cand[0]:
                    cand = (exit_abs, s)
            s = cand[1]
            actions.append(("RH", s["pid"], t))
            t += REHEAT_DUR
            s["t_r"], s["T_r"] = t, REHEAT_TEMP

    out = [str(len(actions))]
    for (typ, pid, st) in actions:
        out.append("%s %d %.9f" % (typ, pid, st))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
