import sys, random

# ---- fixed thermal physics constants (documented in statement.md; NOT hidden) ----
AMBIENT = 25          # ambient depot temperature
THETA_SAFE = 35       # degradation-free temperature threshold
RHO = 0.82            # per-tick Newton cooling retention factor (0<RHO<1)
HEAT = 2.0            # temperature added per unit of charging power, per tick


def fcfs_schedule(buses, D, C, power_level):
    """Immediate-start, first-come-first-served, capacity-respecting schedule at a FIXED
    power_level. buses = list of (a_i, E_i) in arrival order (index = bus id).
    Returns list of start times s_i, or None if some bus cannot be fit before D."""
    n = len(buses)
    occ = [0] * (D + 1)  # occ[t] = # buses active during tick t
    starts = [None] * n
    order = sorted(range(n), key=lambda i: (buses[i][0], i))
    for i in order:
        a_i, e_i = buses[i]
        L = -(-e_i // power_level)  # ceil
        if L == 0:
            L = 0
            starts[i] = a_i
            continue
        s = a_i
        placed = False
        while s + L <= D:
            if all(occ[t] < C for t in range(s, s + L)):
                placed = True
                break
            s += 1
        if not placed:
            return None
        starts[i] = s
        for t in range(s, s + L):
            occ[t] += 1
    return starts


def build_instance(rng, n, d_hint, c, pmax, bpow, hot_frac, slack_frac, tight):
    """Draw buses, then repair D upward until the (loose) baseline power-level FCFS
    schedule fits everyone -- this guarantees a feasible artifact always exists."""
    buses_raw = []
    for i in range(n):
        # arrival spread across the first ~60% of the (hinted) night
        a_i = rng.randint(0, max(1, int(d_hint * 0.55)))
        hot = rng.random() < hot_frac
        if hot:
            theta0 = rng.randint(THETA_SAFE + 8, THETA_SAFE + 32)
        else:
            theta0 = rng.randint(AMBIENT, THETA_SAFE + 4)
        e_i = rng.randint(pmax * 3, pmax * 9)
        c10 = rng.randint(5, 20)
        buses_raw.append([a_i, theta0, e_i, c10])

    D = d_hint
    for _attempt in range(30):
        pairs = [(b[0], b[2]) for b in buses_raw]
        ok_bpow = fcfs_schedule(pairs, D, c, bpow) is not None
        ok_pmax = fcfs_schedule(pairs, D, c, pmax) is not None
        if ok_bpow and ok_pmax:
            break
        D += max(4, D // 6)
    else:
        raise RuntimeError("could not repair feasibility")

    # optionally trim D back down close to the tight feasibility point to induce a
    # genuine berth-crunch trap (only when tight=True): shrink slack while keeping the
    # baseline (bpow) schedule feasible.
    if tight:
        lo, hi = max(b[0] for b in buses_raw), D
        while lo < hi:
            mid = (lo + hi) // 2
            pairs = [(b[0], b[2]) for b in buses_raw]
            feasible = (fcfs_schedule(pairs, mid, c, bpow) is not None and
                        fcfs_schedule(pairs, mid, c, pmax) is not None)
            if feasible:
                hi = mid
            else:
                lo = mid + 1
        D = hi + rng.randint(1, max(2, n // 3))

    buses_raw.sort(key=lambda b: b[0])
    return D, buses_raw


def main():
    tid = int(sys.argv[1])
    rng = random.Random(190000 + 37 * tid)

    # difficulty / trap ladder over the 10 testIds
    #  1-2 : small, generous slack & capacity           (sanity)
    #  3-4 : medium, several very hot buses w/ slack     (trap: greedy roasts them)
    #  5-7 : larger, tighter berths -> real crunch       (trap: patience must be rationed)
    #  8-10: large / adversarial mixes                   (trap: mixed tight+slack buses)
    plan = {
        1: dict(n=5,  d=26, c=4, pmax=4, hot=0.2, tight=False),
        2: dict(n=7,  d=30, c=4, pmax=4, hot=0.3, tight=False),
        3: dict(n=10, d=34, c=4, pmax=4, hot=0.6, tight=False),
        4: dict(n=12, d=34, c=3, pmax=4, hot=0.8, tight=True),
        5: dict(n=16, d=30, c=3, pmax=4, hot=0.65, tight=True),
        6: dict(n=18, d=30, c=3, pmax=4, hot=0.7,  tight=True),
        7: dict(n=22, d=34, c=4, pmax=4, hot=0.75, tight=True),
        8: dict(n=28, d=38, c=4, pmax=4, hot=0.5,  tight=True),
        9: dict(n=32, d=40, c=5, pmax=4, hot=0.8,  tight=True),
        10: dict(n=38, d=44, c=5, pmax=4, hot=0.6, tight=True),
    }[tid]

    n, d_hint, c, pmax = plan["n"], plan["d"], plan["c"], plan["pmax"]
    bpow = max(1, pmax // 2)
    D, buses = build_instance(rng, n, d_hint, c, pmax, bpow, plan["hot"], 0.0, plan["tight"])

    lines = [f"{n} {D} {c} {pmax}"]
    for a_i, theta0, e_i, c10 in buses:
        lines.append(f"{a_i} {theta0} {e_i} {c10}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
