import sys, json, math, random, heapq, isorun

# ==========================================================================
# fsx_B_0936 -- coupled-inventory-rebalancing (Format B, isolated candidate)
# Theme: "Stock stations before bikes run out"
#
# A bike-share-like network of n stations. Your program chooses an initial
# bike count at every station under a total-bike BUDGET. A frozen, seeded
# trip schedule is then replayed against your allocation: each trip needs an
# available bike at its origin (else it is LOST) and, when it arrives, a free
# dock at its destination (else the bike is STRANDED and leaves the system
# for good). Objective: minimize weighted lost+stranded trips.
#
# Mechanisms composed: coupled-flow-balancing (stations are linked by the
# trips that move bikes between them) + inventory-budget (one shared bike
# budget) + saturation-stockout (a station with 0 bikes loses ALL demand
# there until restocked; a station at full docks strands ALL arrivals).
# Innovation hook: the right buffer size follows each station's own
# stockout/overflow SENSITIVITY CURVE over time (how loss responds to its
# stock level, given WHEN its trips happen), not its aggregate net demand
# (trips-out minus trips-in) -- a station can have near-zero net demand yet
# a large transient deficit if its outflow and inflow are phase-separated.
# ==========================================================================

W_PICKUP = 1.0
W_STRAND = 8.0

ROLE_OUT_MULT = {
    'source': [3.0, 2.0, 0.15],
    'hub':    [4.0, 0.5, 0.03],
    'sink':   [0.3, 0.3, 0.3],
    'quiet':  [0.3, 0.3, 0.3],
}
ROLE_IN_MULT = {
    'source': [0.15, 0.15, 0.15],
    'hub':    [0.03, 0.5, 4.0],
    'sink':   [0.3, 1.5, 3.5],
    'quiet':  [0.3, 0.3, 0.3],
}
BASE_OUT = {'source': 9.0, 'hub': 18.0, 'sink': 0.7, 'quiet': 1.0}
BASE_IN  = {'source': 0.7, 'hub': 18.0, 'sink': 9.0, 'quiet': 1.0}
CAP_RANGE = {'source': (6, 10), 'hub': (9, 13), 'sink': (8, 13), 'quiet': (4, 6)}
PHASE_START = [0, 40, 80]

ROLE_PATTERNS = [
    ['source', 'hub', 'sink', 'quiet'],
    ['hub', 'hub', 'source', 'sink'],
    ['source', 'source', 'sink', 'hub'],
    ['hub', 'sink', 'source', 'quiet'],
]

# (n, role_pattern index, trips-per-station K, budget fraction of idealized need)
SPECS = [
    (9,  1, 14, 0.70),
    (10, 0, 18, 0.85),
    (8,  2, 14, 0.70),
    (11, 1, 18, 0.85),
    (9,  3, 14, 0.70),
    (12, 0, 18, 0.85),
    (10, 2, 14, 0.70),
    (13, 1, 18, 0.85),
    (10, 3, 14, 0.70),
    (14, 0, 18, 0.85),
]


def compute_deficit(n, trips):
    """Idealized (infinite-capacity, infinite-liquidity) prefix deficit per
    station: the minimum initial stock that would avoid ANY stockout, used
    only here to calibrate a genuinely scarce per-instance budget."""
    events = []
    for (t, o, d, dt) in trips:
        events.append((t, 1, o, -1))
        events.append((t + dt, 0, d, 1))
    events.sort(key=lambda e: (e[0], e[1]))
    running = [0] * n
    minrun = [0] * n
    for (t, typ, s, delta) in events:
        running[s] += delta
        if running[s] < minrun[s]:
            minrun[s] = running[s]
    return [max(0, -m) for m in minrun]


def gen_instance(seed, n, pattern_idx, k_trips, frac_budget, hub_cap_div=2):
    rng = random.Random(seed)
    pattern = ROLE_PATTERNS[pattern_idx]
    roles = [pattern[i % len(pattern)] for i in range(n)]
    rng.shuffle(roles)
    cap_hub = max(2, n // hub_cap_div)
    hub_idx = [i for i, r in enumerate(roles) if r == 'hub']
    if len(hub_idx) > cap_hub:
        rng.shuffle(hub_idx)
        for i in hub_idx[cap_hub:]:
            roles[i] = 'quiet'
    capacity = [rng.randint(*CAP_RANGE[roles[s]]) for s in range(n)]

    w_out = [[BASE_OUT[roles[s]] * ROLE_OUT_MULT[roles[s]][p] for p in range(3)] for s in range(n)]
    w_in  = [[BASE_IN[roles[s]]  * ROLE_IN_MULT[roles[s]][p]  for p in range(3)] for s in range(n)]

    n_trips = int(k_trips * n)
    trips = []
    for _ in range(n_trips):
        p = rng.randrange(3)
        wo = [w_out[s][p] for s in range(n)]
        o = rng.choices(range(n), weights=wo, k=1)[0]
        wi = [w_in[s][p] if s != o else 0.0 for s in range(n)]
        if sum(wi) <= 0:
            continue
        d = rng.choices(range(n), weights=wi, k=1)[0]
        t = PHASE_START[p] + rng.randint(0, 39)
        dt = rng.randint(1, 3)
        trips.append([t, o, d, dt])
    trips.sort(key=lambda x: x[0])

    deficit = compute_deficit(n, trips)
    sum_def = sum(deficit)
    budget = max(n, min(int(round(frac_budget * sum_def)), int(0.9 * sum(capacity))))
    return {
        "n": n, "capacity": capacity, "budget": budget, "trips": trips,
        "w_pickup": W_PICKUP, "w_strand": W_STRAND,
    }


def make_instances():
    out = []
    for si, (n, pat, k, frac) in enumerate(SPECS):
        pub = gen_instance(9000 + si, n, pat, k, frac)
        out.append({"public": pub, "hidden": {}})
    return out


def simulate_detail(n, capacity, trips, init):
    """Discrete-event replay: departures need a bike at the origin (else a
    LOST pickup); arrivals need a free dock at the destination (else the
    bike is STRANDED, permanently removed). At equal timestamps, pending
    arrivals are applied before that tick's departures."""
    bikes = list(init)
    order = sorted(range(len(trips)), key=lambda i: (trips[i][0], i))
    hp = []
    seq = 0
    lost = 0
    strand = 0
    for idx in order:
        t, o, d, dt = trips[idx]
        while hp and hp[0][0] <= t:
            at, sq, dest = heapq.heappop(hp)
            if bikes[dest] < capacity[dest]:
                bikes[dest] += 1
            else:
                strand += 1
        if bikes[o] > 0:
            bikes[o] -= 1
            heapq.heappush(hp, (t + dt, seq, d)); seq += 1
        else:
            lost += 1
    while hp:
        at, sq, dest = heapq.heappop(hp)
        if bikes[dest] < capacity[dest]:
            bikes[dest] += 1
        else:
            strand += 1
    return lost, strand


def sim_loss(n, capacity, trips, init, w_pickup, w_strand):
    lost, strand = simulate_detail(n, capacity, trips, init)
    return w_pickup * lost + w_strand * strand


def equal_split(n, capacity, budget):
    base = budget // n
    rem = budget - base * n
    alloc = [base] * n
    for i in range(rem):
        alloc[i] += 1
    return [min(a, c) for a, c in zip(alloc, capacity)]


def baseline(inst):
    pub = inst["public"]
    n, cap, budget, trips = pub["n"], pub["capacity"], pub["budget"], pub["trips"]
    a = equal_split(n, cap, budget)
    return sim_loss(n, cap, trips, a, pub["w_pickup"], pub["w_strand"])


def score(inst, ans):
    pub = inst["public"]
    n, cap, budget, trips = pub["n"], pub["capacity"], pub["budget"], pub["trips"]
    if not isinstance(ans, dict) or "init" not in ans:
        return False, 0.0
    init = ans["init"]
    if not isinstance(init, list) or len(init) != n:
        return False, 0.0
    clean = []
    for i, v in enumerate(init):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, 0.0
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        r = round(v)
        if abs(v - r) > 1e-6:
            return False, 0.0
        if r < 0 or r > cap[i]:
            return False, 0.0
        clean.append(int(r))
    if sum(clean) > budget + 1e-6:
        return False, 0.0
    loss = sim_loss(n, cap, trips, clean, pub["w_pickup"], pub["w_strand"])
    if loss != loss or loss < 0:
        return False, 0.0
    return True, loss


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
