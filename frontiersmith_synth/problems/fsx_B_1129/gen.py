import sys, random, math

# Difficulty ladder: testId 1..10, small -> large (up to N=600 SKUs).
# A = number of aisles, L = slots (depths) per aisle, N = A*L = number of SKUs.
# K = number of order-wave scenarios. orders = (lo,hi) orders per scenario.
#
# SKUs fall into three tiers:
#  - CORE: a small "evergreen" assortment, referenced by almost every regular
#    (non-spike) order -> very high global pick frequency.
#  - TREND: n_trend "fashion" pools (most of the catalog). Each is essentially
#    absent from regular scenarios but gets a batch of dedicated, large orders
#    in its ONE designated spike (tail) scenario -> low-to-moderate global
#    frequency, concentrated entirely in that scenario.
#  - FILLER: never referenced by any order (irrelevant dead stock).
# n_spike scenarios are designated as spike/tail scenarios (one trend pool
# each). osz = regular order size range. trend_osz = spike-order size range.
LADDER = {
    1:  dict(A=8,  L=2,  K=6,  orders=(10, 14),  n_trend=1, n_spike=1, trend_frac=0.65, core_frac=0.35, trend_loaded=4,  osz=(2, 3), trend_osz=(3, 4), W=3),
    2:  dict(A=10, L=3,  K=8,  orders=(12, 16),  n_trend=1, n_spike=1, trend_frac=0.68, core_frac=0.30, trend_loaded=5,  osz=(2, 3), trend_osz=(3, 4), W=3),
    3:  dict(A=12, L=3,  K=8,  orders=(14, 18),  n_trend=1, n_spike=1, trend_frac=0.70, core_frac=0.25, trend_loaded=6,  osz=(2, 3), trend_osz=(3, 5), W=3),
    4:  dict(A=16, L=3,  K=10, orders=(16, 22),  n_trend=2, n_spike=2, trend_frac=0.72, core_frac=0.20, trend_loaded=7,  osz=(2, 4), trend_osz=(3, 5), W=3),
    5:  dict(A=20, L=4,  K=12, orders=(20, 26),  n_trend=2, n_spike=2, trend_frac=0.72, core_frac=0.15, trend_loaded=9,  osz=(2, 4), trend_osz=(3, 5), W=4),
    6:  dict(A=24, L=4,  K=14, orders=(24, 30),  n_trend=2, n_spike=2, trend_frac=0.73, core_frac=0.12, trend_loaded=11, osz=(2, 4), trend_osz=(3, 5), W=4),
    7:  dict(A=25, L=6,  K=16, orders=(30, 40),  n_trend=3, n_spike=3, trend_frac=0.74, core_frac=0.10, trend_loaded=13, osz=(2, 4), trend_osz=(3, 5), W=4),
    8:  dict(A=27, L=8,  K=18, orders=(40, 55),  n_trend=3, n_spike=3, trend_frac=0.75, core_frac=0.08, trend_loaded=14, osz=(2, 4), trend_osz=(3, 5), W=4),
    9:  dict(A=30, L=12, K=20, orders=(60, 80),  n_trend=3, n_spike=3, trend_frac=0.75, core_frac=0.06, trend_loaded=15, osz=(2, 4), trend_osz=(3, 5), W=4),
    10: dict(A=30, L=20, K=20, orders=(90, 120), n_trend=3, n_spike=3, trend_frac=0.75, core_frac=0.05, trend_loaded=15, osz=(2, 4), trend_osz=(3, 5), W=4),
}

# candidate multipliers tried on cfg["trend_loaded"] during self-calibration
_CANDIDATE_MULT = [0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]
_TAIL_FRAC = 0.15   # must match solutions/strong.py
_BOOST = 6          # must match solutions/strong.py


def make_order(rng, N, pool, sz):
    if len(pool) >= sz:
        return list(rng.sample(pool, sz))
    skus = list(pool)
    rest = [x for x in range(N) if x not in pool]
    skus += rng.sample(rest, sz - len(pool))
    return skus


def _generate(test_id, cfg, trend_loaded, seed_salt, core_frac_mult=1.0):
    A, L, K = cfg["A"], cfg["L"], cfg["K"]
    n_trend, n_spike = cfg["n_trend"], cfg["n_spike"]
    olo, ohi = cfg["orders"]
    szlo, szhi = cfg["osz"]
    tszlo, tszhi = cfg["trend_osz"]
    W = cfg["W"]
    N = A * L

    rng = random.Random(20260726 + 1000 * test_id + 97 * seed_salt)
    ids = list(range(N))
    rng.shuffle(ids)

    trend_total = int(N * cfg["trend_frac"])
    trend_total = max(trend_total, n_trend * (tszhi + 2))
    trend_total = min(trend_total, N - (szhi + 2) - 2)  # leave room for core + filler
    per_trend = trend_total // n_trend
    trend_comms = []
    pos = 0
    for i in range(n_trend):
        trend_comms.append(ids[pos:pos + per_trend])
        pos += per_trend
    trend_used = pos

    remaining = N - trend_used
    # core must be big enough that a regular order never has to pad with
    # non-core (trend/filler) SKUs, or trend "leaks" into regular scenarios.
    core_size = max(szhi + 2, int(remaining * cfg["core_frac"] * core_frac_mult))
    core_size = min(core_size, remaining - 2)
    core = ids[pos:pos + core_size]
    pos += core_size
    filler = ids[pos:]  # dead stock: never referenced by any order

    spike_scen = set(rng.sample(range(1, K + 1), min(n_spike, K)))
    spike_comm_of = {}
    for i, s in enumerate(sorted(spike_scen)):
        spike_comm_of[s] = i % n_trend

    core_noise = 0.02

    def draw_normal(rng, sz):
        if filler and rng.random() < core_noise:
            return make_order(rng, N, filler, sz)
        return make_order(rng, N, core, sz)

    scenarios = []
    for s in range(1, K + 1):
        q = rng.randint(olo, ohi)
        orders = []
        if s in spike_scen:
            tc = spike_comm_of[s]
            for _ in range(min(trend_loaded, q)):
                sz = rng.randint(tszlo, tszhi)
                orders.append(make_order(rng, N, trend_comms[tc], sz))
            for _ in range(max(0, q - trend_loaded)):
                sz = rng.randint(szlo, szhi)
                orders.append(draw_normal(rng, sz))
        else:
            for _ in range(q):
                sz = rng.randint(szlo, szhi)
                orders.append(draw_normal(rng, sz))
        rng.shuffle(orders)
        scenarios.append(orders)

    return N, A, L, K, W, scenarios


def _scenario_totals(perm, L, W, scenarios):
    totals = []
    for orders in scenarios:
        tot = 0
        for skus in orders:
            aisles = set()
            max_depth = {}
            for sku in skus:
                slot = perm[sku]
                aisle = slot // L + 1
                depth = slot % L + 1
                aisles.add(aisle)
                if depth > max_depth.get(aisle, 0):
                    max_depth[aisle] = depth
            if not aisles:
                continue
            tot += 2 * W * max(aisles) + 2 * sum(max_depth.values())
        totals.append(tot)
    return totals


def _p90(values):
    v = sorted(values)
    K = len(v)
    idx = max(0, min(K - 1, math.ceil(0.9 * K) - 1))
    return v[idx]


def _sc_cost(perm, L, W, orders):
    tot = 0
    for skus in orders:
        aisles = set()
        max_depth = {}
        for sku in skus:
            slot = perm[sku]
            aisle = slot // L + 1
            depth = slot % L + 1
            aisles.add(aisle)
            if depth > max_depth.get(aisle, 0):
                max_depth[aisle] = depth
        if not aisles:
            continue
        tot += 2 * W * max(aisles) + 2 * sum(max_depth.values())
    return tot


def _strong_like(N, A, L, K, W, scenarios, freq):
    """Mirrors solutions/strong.py's bootstrap -> tail-affinity-graph ->
    boost algorithm EXACTLY, used only here to self-calibrate the
    generator's difficulty knobs against the real reference solution."""
    order = sorted(range(N), key=lambda s: (-freq[s], s))
    perm = [0] * N
    for slot, sku in enumerate(order):
        perm[sku] = slot
    costs = [(s, _sc_cost(perm, L, W, scenarios[s])) for s in range(K)]
    costs.sort(key=lambda t: -t[1])
    tail_n = max(1, math.ceil(_TAIL_FRAC * K))
    tail_idx = set(s for s, _ in costs[:tail_n])
    weighted_degree = [0] * N
    for s in tail_idx:
        for skus in scenarios[s]:
            uniq = list(dict.fromkeys(skus))
            m = len(uniq)
            if m < 2:
                continue
            for sku in uniq:
                weighted_degree[sku] += (m - 1)
    score = [freq[i] + _BOOST * weighted_degree[i] for i in range(N)]
    order2 = sorted(range(N), key=lambda s: (-score[s], -freq[s], s))
    perm2 = [0] * N
    for slot, sku in enumerate(order2):
        perm2[sku] = slot
    return perm2


def _evaluate(N, A, L, K, W, scenarios):
    freq = [0] * N
    for orders in scenarios:
        for skus in orders:
            for sku in skus:
                freq[sku] += 1
    ident = list(range(N))
    B = _p90(_scenario_totals(ident, L, W, scenarios))
    greedy_order = sorted(range(N), key=lambda s: (-freq[s], s))
    perm_g = [0] * N
    for slot, sku in enumerate(greedy_order):
        perm_g[sku] = slot
    Fg = _p90(_scenario_totals(perm_g, L, W, scenarios))
    perm_s = _strong_like(N, A, L, K, W, scenarios, freq)
    Fs = _p90(_scenario_totals(perm_s, L, W, scenarios))
    rg = min(1000.0, 100.0 * B / max(1e-9, Fg)) / 1000.0
    rs = min(1000.0, 100.0 * B / max(1e-9, Fs)) / 1000.0
    return rg, rs, rs - rg


def _serialize(N, A, L, K, W, scenarios):
    lines = [f"{N} {A} {L} {K} {W}"]
    for orders in scenarios:
        lines.append(str(len(orders)))
        for o in orders:
            lines.append(str(len(o)) + " " + " ".join(map(str, o)))
    return "\n".join(lines) + "\n"


_CORE_FRAC_MULT = [0.5, 0.75, 1.0, 1.5, 2.0]


def build(test_id):
    cfg = LADDER.get(test_id, LADDER[1])
    base_loaded = cfg["trend_loaded"]
    olo = cfg["orders"][0]

    best = None
    best_gap = None
    salt = 0
    for mult in _CANDIDATE_MULT:
        cand = max(2, min(olo - 1, round(base_loaded * mult)))
        for cfmult in _CORE_FRAC_MULT:
            salt += 1
            inst = _generate(test_id, cfg, cand, salt, core_frac_mult=cfmult)
            rg, rs, gap = _evaluate(*inst)
            # keep headroom: skip candidates where strong is (near-)saturated
            if rs > 0.90:
                continue
            if best_gap is None or gap > best_gap:
                best_gap = gap
                best = inst

    if best is None:  # fallback: no candidate had rs<=0.90, just take best gap
        best_gap = None
        salt = 0
        for mult in _CANDIDATE_MULT:
            cand = max(2, min(olo - 1, round(base_loaded * mult)))
            for cfmult in _CORE_FRAC_MULT:
                salt += 1
                inst = _generate(test_id, cfg, cand, salt, core_frac_mult=cfmult)
                rg, rs, gap = _evaluate(*inst)
                if best_gap is None or gap > best_gap:
                    best_gap = gap
                    best = inst

    return _serialize(*best)


def main():
    i = int(sys.argv[1])
    sys.stdout.write(build(i))


if __name__ == "__main__":
    main()
