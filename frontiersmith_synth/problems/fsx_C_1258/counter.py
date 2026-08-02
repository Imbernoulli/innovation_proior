#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for the Bloom-cascade
layer-size-allocation problem.

The instance is fully reproducible from the testId alone (see gen.py), so
this checker re-derives the member set / hash coefficients / hot-key table
*and* an independent held-out tail sample (never shown to the participant)
directly from testId, using the identical seeded formulas as gen.py. Scoring
is always computed against the held-out sample, never against the visible
tail the participant read from stdin.
"""
import sys
import random

L = 4
KMAX = 6
P = (1 << 31) - 1
COST = [1, 3, 9, 27]
DPEN = 2500
MIN_M = 8
MAX_TOKEN = 10 ** 18


def rng(test_id, role):
    return random.Random(test_id * 1_000_003 + role)


def build_instance(test_id):
    n = 500 + 100 * test_id
    universe = 25 * n
    bits_per_key = 20
    budget = bits_per_key * n

    r_mem = rng(test_id, 11)
    members = sorted(r_mem.sample(range(universe), n))
    mset = set(members)

    coeffs = []
    for i in range(L):
        r_c = rng(test_id, 100 + i)
        layer_coeffs = [(r_c.randint(1, P - 1), r_c.randint(0, P - 1)) for _ in range(KMAX)]
        coeffs.append(layer_coeffs)

    skewed = test_id >= 4
    hot = []
    if skewed:
        r_hot = rng(test_id, 21)
        h_count = 5
        chosen = set()
        while len(chosen) < h_count:
            k = r_hot.randrange(universe)
            if k not in mset:
                chosen.add(k)
        for k in sorted(chosen):
            w = r_hot.randint(30, 80)
            hot.append((k, w))

    def gen_tail(role_tail, count):
        r_t = rng(test_id, role_tail)
        tail = []
        got = 0
        while got < count:
            k = r_t.randrange(universe)
            if k not in mset:
                tail.append((k, 1))
                got += 1
        return tail

    tail_count = 400 + 80 * test_id
    tail_vis = gen_tail(31, tail_count)
    tail_hid = gen_tail(41, tail_count)

    return dict(test_id=test_id, universe=universe, n=n, budget=budget,
                members=members, mset=mset, coeffs=coeffs, hot=hot,
                tail_vis=tail_vis, tail_hid=tail_hid)


def h(x, a, b, m):
    return ((a * x + b) % P) % m


def build_layer_bits(members, coeffs_i, m_i, k_i):
    bits = bytearray(m_i)
    for x in members:
        for j in range(k_i):
            a, b = coeffs_i[j]
            bits[h(x, a, b, m_i)] = 1
    return bits


def layer_passes(bits, x, coeffs_i, k_i, m_i):
    for j in range(k_i):
        a, b = coeffs_i[j]
        if not bits[h(x, a, b, m_i)]:
            return False
    return True


def simulate_cost(members, mset, coeffs, layer_cfg, queries):
    bits_layers = [build_layer_bits(members, coeffs[i], layer_cfg[i][0], layer_cfg[i][1])
                   for i in range(L)]
    total = 0
    for (x, w) in queries:
        units = 0
        survived = True
        for i in range(L):
            units += COST[i]
            m_i, k_i = layer_cfg[i]
            if not layer_passes(bits_layers[i], x, coeffs[i], k_i, m_i):
                survived = False
                break
        if survived and x not in mset:
            units += DPEN
        total += w * units
    return total


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        first_line = f.readline().split()
    if not first_line:
        fail("empty instance file")
    try:
        test_id = int(first_line[0])
    except ValueError:
        fail("bad instance file")

    inst = build_instance(test_id)

    try:
        with open(out_path) as f:
            tokens = f.read().split()
    except OSError:
        fail("cannot read output")

    if len(tokens) != 2 * L:
        fail("expected %d integers (m_i k_i per layer), got %d" % (2 * L, len(tokens)))

    parsed = []
    for tok in tokens:
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer / nan / inf token '%s'" % tok)
        if abs(v) > MAX_TOKEN:
            fail("token out of range '%s'" % tok)
        parsed.append(v)

    layer_cfg = []
    for i in range(L):
        m_i, k_i = parsed[2 * i], parsed[2 * i + 1]
        if m_i < MIN_M:
            fail("layer %d size m=%d below minimum %d" % (i + 1, m_i, MIN_M))
        if not (1 <= k_i <= KMAX):
            fail("layer %d hash count k=%d out of [1,%d]" % (i + 1, k_i, KMAX))
        layer_cfg.append((m_i, k_i))

    total_m = sum(m for (m, k) in layer_cfg)
    if total_m > inst["budget"]:
        fail("total bits %d exceeds budget %d" % (total_m, inst["budget"]))

    score_queries = inst["hot"] + inst["tail_hid"]

    F = simulate_cost(inst["members"], inst["mset"], inst["coeffs"], layer_cfg, score_queries)

    ref_m = inst["budget"] // L
    ref_cfg = [(ref_m, 1) for _ in range(L - 1)]
    ref_cfg.append((inst["budget"] - ref_m * (L - 1), 1))
    F_ref = simulate_cost(inst["members"], inst["mset"], inst["coeffs"], ref_cfg, score_queries)

    # Cap at 900/1000 (=0.9), not 1000, so no submission -- however good --
    # can saturate the score to 1.0: this keeps headroom above the best
    # reference solution regardless of instance-specific variance.
    sc = min(900.0, 100.0 * F_ref / max(1e-9, float(F)))
    print("submitted_cost=%d reference_cost=%d Ratio: %.6f" % (F, F_ref, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
