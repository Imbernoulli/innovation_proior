"""
Shared deterministic instance builder for fsx_B_1168 (sensor-fault-disentangle).
Imported by BOTH gen.py (prints only the public fields) and verify.py (recomputes the
hidden ground truth from the same testId-seeded RNG stream, never shipped to the solver).
Pure Python stdlib only (random, math) -- no numpy, fully deterministic bit-for-bit.
"""
import random, math

SEED_BASE = 913471


def build_graph(n, rng, extra_frac):
    """Connected redundancy graph: random spanning tree + extra edges. Returns adj (list of
    sorted neighbour lists) and sorted edge list."""
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for idx in range(1, n):
        i = order[idx]
        j = order[rng.randrange(idx)]
        edges.add(tuple(sorted((i, j))))
    extra = max(0, int(round(extra_frac * n)))
    tries = 0
    while len(edges) < (n - 1) + extra and tries < 40 * n:
        a = rng.randrange(n)
        b = rng.randrange(n)
        tries += 1
        if a == b:
            continue
        edges.add(tuple(sorted((a, b))))
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    for a in range(n):
        adj[a].sort()
    return adj, sorted(edges)


def bfs_dist(adj, src, max_d=2):
    n = len(adj)
    dist = [-1] * n
    dist[src] = 0
    frontier = [src]
    d = 0
    while frontier and d < max_d:
        nxt = []
        d += 1
        for u in frontier:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


def smooth_levels(adj, rng, rounds=18, lo=-3.0, hi=3.0):
    n = len(adj)
    L = [rng.uniform(lo, hi) for _ in range(n)]
    for _ in range(rounds):
        newL = []
        for i in range(n):
            nb = adj[i]
            if nb:
                m = (L[i] + sum(L[j] for j in nb)) / (1 + len(nb))
            else:
                m = L[i]
            newL.append(m)
        L = newL
    return L


def difficulty_params(test_id):
    """Ladder: small/easy -> large/adversarial (trap) cases. testId 1..10."""
    table = {
        # id: (N,      T,  F_max, F_true, extra_edge_frac, event_mode,      noise_sigma)
        1:  (8,   30, 1, 1, 0.4, "none",       0.06),
        2:  (10,  35, 2, 1, 0.4, "far_small",  0.06),
        3:  (12,  40, 2, 2, 0.5, "far_mod",    0.07),
        4:  (14,  45, 2, 2, 0.5, "trap_nbr",   0.07),
        5:  (16,  50, 2, 2, 0.5, "trap_same",  0.08),
        6:  (18,  55, 3, 3, 0.5, "trap_nbr",   0.08),
        7:  (20,  60, 3, 3, 0.6, "trap_same",  0.08),
        8:  (22,  65, 3, 3, 0.6, "trap_nbr",   0.09),
        9:  (24,  70, 3, 2, 0.6, "trap_nbr",   0.09),
        10: (27,  80, 4, 3, 0.6, "trap_same",  0.10),
    }
    return table[test_id]


def build_instance(test_id):
    n, T, F_max, F_true, extra_frac, event_mode, sigma = difficulty_params(test_id)
    rng = random.Random(SEED_BASE + 97 * test_id)

    adj, edges = build_graph(n, rng, extra_frac)
    L = smooth_levels(adj, rng)

    Ag = rng.uniform(0.8, 1.6)
    Pg = rng.uniform(0.35, 0.55) * T
    phase = rng.uniform(0, 2 * math.pi)
    Gl = rng.uniform(-0.01, 0.01)

    def g(t):
        return Ag * math.sin(2 * math.pi * t / Pg + phase) + Gl * t

    # ---- faults -----------------------------------------------------
    fault_ids = rng.sample(range(n), F_true)
    fault = {}
    for fid in fault_ids:
        typ = rng.choice(["offset", "drift"])
        if typ == "offset":
            a = rng.choice([-1, 1]) * rng.uniform(1.4, 2.6)
            b = 0.0
        else:
            a = 0.0
            # scale so the total drift swing over the record matches the offset scale
            b = rng.choice([-1, 1]) * rng.uniform(1.4, 2.6) / T
        fault[fid] = (a, b)

    # ---- event (genuine local anomaly, part of the TRUE field) ------
    width = max(6, int(round(0.20 * T)))
    t0 = rng.randrange(0, max(1, T - width))
    Ae = rng.uniform(1.6, 2.6)
    decay = 0.5

    epicenter = None
    if event_mode == "none":
        pass
    elif event_mode == "far_small":
        # a small event far from any fault sensor
        Ae *= 0.55
        cands = [v for v in range(n) if v not in fault]
        epicenter = rng.choice(cands) if cands else None
    elif event_mode == "far_mod":
        cands = [v for v in range(n) if v not in fault]
        epicenter = rng.choice(cands) if cands else None
    elif event_mode == "trap_nbr":
        fsrc = rng.choice(fault_ids)
        cands = [v for v in adj[fsrc] if v not in fault] or [v for v in range(n) if v not in fault]
        epicenter = rng.choice(cands)
        Ae *= 1.9
    elif event_mode == "trap_same":
        epicenter = rng.choice(fault_ids)
        Ae *= 1.7
    else:
        raise ValueError(event_mode)

    event_amt = [0.0] * n  # per-node peak-amplitude multiplier (bump shape scaled by this)
    if epicenter is not None:
        dist = bfs_dist(adj, epicenter, max_d=2)
        for v in range(n):
            if dist[v] == 0:
                event_amt[v] = 1.0
            elif dist[v] == 1:
                event_amt[v] = decay
            elif dist[v] == 2:
                event_amt[v] = decay * decay

    def event_val(v, t):
        if event_amt[v] == 0.0:
            return 0.0
        if t0 <= t < t0 + width:
            phase_t = math.pi * (t - t0) / width
            return event_amt[v] * Ae * math.sin(phase_t)
        return 0.0

    # ---- assemble clean truth Y and noisy/faulted reading R --------
    Y = [[0.0] * T for _ in range(n)]
    R = [[0.0] * T for _ in range(n)]
    nrng = random.Random(SEED_BASE + 131 * test_id + 7)
    for i in range(n):
        a_i, b_i = fault.get(i, (0.0, 0.0))
        for t in range(T):
            y = L[i] + g(t) + event_val(i, t)
            Y[i][t] = y
            noise = nrng.gauss(0.0, sigma)
            R[i][t] = y + noise + a_i + b_i * t

    return {
        "test_id": test_id, "n": n, "T": T, "F_max": F_max,
        "adj": adj, "edges": edges, "R": R, "Y": Y,
        "fault": fault, "event_mode": event_mode, "epicenter": epicenter,
        "sigma": sigma,
    }
