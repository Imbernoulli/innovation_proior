import sys, math

AMBIENT = 25
THETA_SAFE = 35
RHO = 0.82
HEAT = 2.0


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def fcfs_schedule(buses, D, C, power_level):
    """Immediate-start, FCFS, capacity-respecting schedule at a fixed power level.
    buses = list of (a_i, E_i) in row order. Returns list of start times, or None."""
    n = len(buses)
    occ = [0] * (D + 1)
    starts = [None] * n
    order = sorted(range(n), key=lambda i: (buses[i][0], i))
    for i in order:
        a_i, e_i = buses[i]
        L = -(-e_i // power_level) if e_i > 0 else 0
        if L == 0:
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


def degrade_from_power(a_i, theta0, e_i, c10, D, power):
    """Simulate temperature/SoC forward from tick a_i to D-1 given a full power[0..D-1]
    array (power[t] used for t in [a_i, D)), return the accumulated degradation."""
    c = c10 / 10.0
    theta = float(theta0)
    energy_cum = 0.0
    deg = 0.0
    for t in range(a_i, D):
        soc = min(1.0, energy_cum / e_i) if e_i > 0 else 1.0
        excess = theta - THETA_SAFE
        if excess > 0:
            deg += c * excess * excess * soc
        p = power[t]
        theta = AMBIENT + (theta - AMBIENT) * RHO + HEAT * p
        energy_cum += p
    return deg


def degrade_from_start(a_i, theta0, e_i, c10, D, start, pmax):
    """Build the implied continuous-full-power-from-`start` power array on the fly and
    return its degradation (used only for the internal baseline; participant output is
    checked against the general degrade_from_power on its literal array)."""
    power = [0] * D
    t = start
    remaining = e_i
    while remaining > 0 and t < D:
        p = min(pmax, remaining)
        power[t] = p
        remaining -= p
        t += 1
    return degrade_from_power(a_i, theta0, e_i, c10, D, power)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        n = int(next(it)); D = int(next(it)); C = int(next(it)); pmax = int(next(it))
        buses = []
        for _ in range(n):
            a_i = int(next(it)); theta0 = int(next(it)); e_i = int(next(it)); c10 = int(next(it))
            buses.append((a_i, theta0, e_i, c10))
    except Exception:
        fail("bad input")

    # ---- internal baseline B: immediate FULL-POWER FCFS construction (the naive
    # "charge on arrival as fast as possible" instinct -- always feasible by
    # generator construction) ----
    pairs = [(b[0], b[2]) for b in buses]
    base_starts = fcfs_schedule(pairs, D, C, pmax)
    if base_starts is None:
        fail("instance has no feasible baseline (generator bug)")
    B = 0.0
    for i, (a_i, theta0, e_i, c10) in enumerate(buses):
        B += degrade_from_start(a_i, theta0, e_i, c10, D, base_starts[i], pmax)
    B = max(1e-6, B)

    # ---- parse participant output: n rows of D integers each ----
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    need = n * D
    if len(raw) != need:
        fail("wrong token count: need %d got %d" % (need, len(raw)))
    try:
        vals = [int(x) for x in raw[:need]]
    except Exception:
        fail("non-integer token in output")

    for v in vals:
        if v != v or math.isinf(v):
            fail("non-finite token")

    power_rows = [vals[i * D:(i + 1) * D] for i in range(n)]

    for i, (a_i, theta0, e_i, c10) in enumerate(buses):
        row = power_rows[i]
        for t, p in enumerate(row):
            if p < 0 or p > pmax:
                fail("power out of range bus=%d t=%d p=%d" % (i, t, p))
            if t < a_i and p != 0:
                fail("charging before arrival bus=%d t=%d" % (i, t))
        if sum(row[a_i:D]) < e_i:
            fail("insufficient energy delivered bus=%d" % i)

    # ---- shared berth capacity: concurrently-charging buses <= C at every tick ----
    occ = [0] * D
    for row in power_rows:
        for t in range(D):
            if row[t] > 0:
                occ[t] += 1
    for t in range(D):
        if occ[t] > C:
            fail("berth capacity exceeded at t=%d (%d>%d)" % (t, occ[t], C))

    # ---- objective: total battery degradation (minimize) ----
    F = 0.0
    for i, (a_i, theta0, e_i, c10) in enumerate(buses):
        F += degrade_from_power(a_i, theta0, e_i, c10, D, power_rows[i])
    F = max(1e-9, F)

    sc = min(1000.0, 100.0 * B / F)
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
