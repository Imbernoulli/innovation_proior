import sys, random

R = 12
D_MAX = 300
V = 260
F_PERMILLE = 25
NAMES = ["ZH","BE","LU","UR","SZ","OW","NW","GL","ZG","FR","SO","BS"]


def piecewise_rate(d, e):
    d0, d1, r1, d2, r2 = e['d0'], e['d1'], e['r1'], e['d2'], e['r2']
    if d <= d0:
        return 0
    if d <= d1:
        return (r1 * (d - d0)) // (d1 - d0)
    if d <= d2:
        return r1 + ((r2 - r1) * (d - d1)) // (d2 - d1)
    return r2


def region_value(d, e, V, Fpm):
    rate = piecewise_rate(d, e)
    ret = (e['pop'] * rate) // 1_000_000
    payout = ret * d
    fcost = (payout * Fpm) // 1000
    return ret * V - payout - fcost, ret


def standalone_optimum(e, V, Fpm, DMAX):
    best_d, best_v = 0, None
    for d in range(0, DMAX + 1):
        val, _ = region_value(d, e, V, Fpm)
        if best_v is None or val > best_v:
            best_v, best_d = val, d
    return best_d, best_v


def gen_region_econ(rng):
    # d0 = a "dead zone": nobody bothers returning bottles below this deposit. Making
    # d0 vary per canton (rather than always 0) is what keeps each canton's true
    # elasticity-optimal deposit genuinely region-specific instead of collapsing to a
    # single universal constant driven only by the global value/float-rate.
    for _ in range(300):
        pop = rng.randint(3000, 16000)
        d0 = rng.randint(0, 200)
        seg1 = rng.randint(15, 150)
        d1 = min(D_MAX - 5, d0 + seg1)
        r1 = rng.randint(100000, 650000)
        gap_d = rng.randint(30, 250)
        d2 = min(D_MAX, d1 + gap_d)
        if d2 <= d1 or d1 <= d0:
            continue
        max_r2 = r1 + (r1 * (d2 - d1)) // (d1 - d0)
        max_r2 = min(max_r2 - 1, 1_000_000)
        min_r2 = r1 + max(1, (d2 - d1) // 20)
        if min_r2 >= max_r2:
            continue
        r2 = rng.randint(min_r2, max_r2)
        return {'pop': pop, 'd0': d0, 'd1': d1, 'r1': r1, 'd2': d2, 'r2': r2}
    return {'pop': 8000, 'd0': 20, 'd1': 100, 'r1': 400000, 'd2': 200, 'r2': 700000}


def gen_graph(rng, n=R):
    # Base corridors are deliberately EXPENSIVE (90-170) so an incidental single-hop
    # differential from independent per-canton optimization is rarely profitable on its
    # own; only the deliberately planted trap corridors (see apply_trap) are cheap
    # enough to exploit. This keeps "benign" instances genuinely benign for a network-
    # blind solver while still requiring the strong solver to compute real shortest
    # paths (multi-hop can still occasionally undercut a single expensive direct edge).
    edges = []
    order = list(range(n))
    rng.shuffle(order)
    for idx in range(n):
        u = order[idx]
        v = order[(idx + 1) % n]
        cost = rng.randint(90, 150)
        edges.append((u, v, cost))
    extra = rng.randint(6, 10)
    seen_pairs = set((min(u, v), max(u, v)) for u, v, _ in edges)
    tries = 0
    added = 0
    while added < extra and tries < 300:
        tries += 1
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        p = (min(u, v), max(u, v))
        if p in seen_pairs:
            continue
        cost = rng.randint(90, 170)
        edges.append((u, v, cost))
        seen_pairs.add(p)
        added += 1
    return edges


def remove_direct_edge(edges, a, b):
    p = (min(a, b), max(a, b))
    return [e for e in edges if (min(e[0], e[1]), max(e[0], e[1])) != p]


def apply_trap(regions, edges, i_lo, i_hi, hub1, hub2, pop_lo=110000, pop_hi=55000):
    regions[i_lo] = dict(regions[i_lo])
    regions[i_hi] = dict(regions[i_hi])
    # i_lo: very HIGH population, zero dead-zone and an early-saturating curve -> a LOW
    # standalone optimum yet a LARGE genuine-returns base, so every diverted bottle
    # really hurts.
    regions[i_lo].update(pop=pop_lo, d0=0, d1=15, r1=700000, d2=30, r2=720000)
    # i_hi: a large dead-zone (past the checker's uniform reference deposit, so the
    # baseline gets ZERO credit here) plus a very late-saturating curve that keeps
    # rewarding higher deposits far out -> a HIGH standalone optimum, the destination
    # haulers relay towards.
    regions[i_hi].update(pop=pop_hi, d0=150, d1=270, r1=80000, d2=299, r2=980000)
    edges = remove_direct_edge(edges, i_lo, i_hi)
    opt_lo, _ = standalone_optimum(regions[i_lo], V, F_PERMILLE, D_MAX)
    opt_hi, _ = standalone_optimum(regions[i_hi], V, F_PERMILLE, D_MAX)
    gap = max(40, opt_hi - opt_lo)
    per_edge = max(1, (gap * 3) // 10)
    edges = edges + [(i_lo, hub1, per_edge), (hub1, hub2, per_edge), (hub2, i_hi, per_edge)]
    return regions, edges


TRAP_PLAN = {
    3: [(0, 6, 3, 9), (1, 7, 4, 10), (2, 8, 5, 11)],
    5: [(2, 8, 5, 11), (0, 4, 7, 10), (1, 6, 3, 9)],
    7: [(1, 9, 3, 6), (2, 10, 5, 8), (0, 11, 4, 7)],
    9: [(0, 5, 2, 11), (1, 6, 4, 9), (3, 8, 7, 10)],
    10: [(0, 6, 3, 9), (1, 7, 4, 10), (2, 8, 5, 11)],
}
# Lighter single-corridor traps on most of the otherwise "benign" tests: 9 of the 10
# tests carry at least one arbitrage-exploitable pair, so a network-blind solver rarely
# gets a completely free ride (test 2 is left as pure background elasticity tuning).
LIGHT_TRAP_PLAN = {
    1: [(0, 6, 3, 9)],
    4: [(2, 8, 5, 11)],
    6: [(0, 5, 2, 9)],
    8: [(1, 6, 4, 11)],
}
LIGHT_TRAP_POP_LO = 2900
LIGHT_TRAP_POP_HI = 2350


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20260 + 37 * testId)

    regions = [gen_region_econ(rng) for _ in range(R)]
    for i, e in enumerate(regions):
        e['name'] = NAMES[i]

    edges = gen_graph(rng, R)

    for (i_lo, i_hi, h1, h2) in TRAP_PLAN.get(testId, []):
        regions, edges = apply_trap(regions, edges, i_lo, i_hi, h1, h2)
    for (i_lo, i_hi, h1, h2) in LIGHT_TRAP_PLAN.get(testId, []):
        regions, edges = apply_trap(regions, edges, i_lo, i_hi, h1, h2,
                                     pop_lo=LIGHT_TRAP_POP_LO, pop_hi=LIGHT_TRAP_POP_HI)

    out = []
    out.append(f"{R} {V} {D_MAX} {F_PERMILLE}")
    for e in regions:
        out.append(f"{e['name']} {e['pop']} {e['d0']} {e['d1']} {e['r1']} {e['d2']} {e['r2']}")
    out.append(str(len(edges)))
    for (u, v, c) in edges:
        out.append(f"{u} {v} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
