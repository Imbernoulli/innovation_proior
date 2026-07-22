import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
        it = iter(toks)
        N = int(next(it)); K = int(next(it))
        ALPHA = float(next(it)); LAMBDA = float(next(it))
        pts = []
        for _ in range(N):
            x = int(next(it)); y = int(next(it))
            pts.append((x, y))
        bonus = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                bonus[i][j] = int(next(it))
    except Exception:
        fail("bad input")
    if N < 3 or K < 1 or K >= N:
        fail("bad input dims")
    return N, K, ALPHA, LAMBDA, pts, bonus


def simulate(merges, N, K, ALPHA, LAMBDA, pts, bonus, total_bonus_abs):
    """Replays an irreversible merge sequence. Returns the clamped objective F.
    Assumes `merges` is already validated as exactly N-K well-formed pairs of
    integers (the caller of the participant path validates BEFORE calling)."""
    T = N - K
    info = {}
    for i in range(1, N + 1):
        x, y = pts[i - 1]
        info[i] = {"cnt": 1, "sx": float(x), "sy": float(y), "members": [i]}
    active = set(range(1, N + 1))
    cum_cost = 0.0
    for t, (a, b) in enumerate(merges, start=1):
        ia, ib = info[a], info[b]
        sizeA, sizeB = ia["cnt"], ib["cnt"]
        cxA, cyA = ia["sx"] / sizeA, ia["sy"] / sizeA
        cxB, cyB = ib["sx"] / sizeB, ib["sy"] / sizeB
        d = math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)
        raw_cost = (sizeA * sizeB / (sizeA + sizeB)) * d
        remaining = T - t
        weight = 1.0 + ALPHA / (1.0 + remaining)
        cum_cost += raw_cost * weight
        new_id = N + t
        info[new_id] = {"cnt": sizeA + sizeB, "sx": ia["sx"] + ib["sx"], "sy": ia["sy"] + ib["sy"],
                         "members": ia["members"] + ib["members"]}
        active.discard(a); active.discard(b); active.add(new_id)
    affinity = 0.0
    for c in active:
        mem = info[c]["members"]
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                affinity += bonus[mem[i] - 1][mem[j] - 1]
    raw = affinity + total_bonus_abs - LAMBDA * cum_cost
    return max(0.0, raw)


def naive_baseline_merges(N, K):
    queue = list(range(1, N + 1))
    merges = []
    next_id = N + 1
    while len(queue) > K:
        a = queue.pop(0); b = queue.pop(0)
        merges.append((a, b))
        queue.append(next_id)
        next_id += 1
    return merges


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    N, K, ALPHA, LAMBDA, pts, bonus = read_instance(inf)
    total_bonus_abs = sum(abs(bonus[i][j]) for i in range(N) for j in range(i + 1, N))
    T = N - K

    # ---- internal baseline B: naive fixed index-pairing, never touches geometry/affinity ----
    B_raw = simulate(naive_baseline_merges(N, K), N, K, ALPHA, LAMBDA, pts, bonus, total_bonus_abs)
    B = max(B_raw, 1e-9)

    # ---- parse & strictly validate participant output ----
    try:
        data = open(outf).read().split()
    except Exception:
        fail("cannot read output")
    if not data:
        fail("empty output")
    try:
        T_claim = int(data[0])
    except (ValueError, OverflowError):
        fail("non-integer merge count")
    if T_claim != T:
        fail("merge count %d != expected %d" % (T_claim, T))
    rest = data[1:]
    if len(rest) != 2 * T_claim:
        fail("expected %d merge tokens, got %d" % (2 * T_claim, len(rest)))

    merges = []
    for k in range(T_claim):
        try:
            a = int(rest[2 * k]); b = int(rest[2 * k + 1])
        except (ValueError, OverflowError):
            fail("non-integer cluster id at merge %d" % (k + 1))
        merges.append((a, b))

    active = set(range(1, N + 1))
    for t, (a, b) in enumerate(merges, start=1):
        if a == b:
            fail("self-merge at step %d" % t)
        if a not in active or b not in active:
            fail("merge %d references inactive/unknown cluster (%d, %d)" % (t, a, b))
        active.discard(a); active.discard(b)
        active.add(N + t)
    if len(active) != K:
        fail("final cluster count %d != K=%d" % (len(active), K))

    F = simulate(merges, N, K, ALPHA, LAMBDA, pts, bonus, total_bonus_abs)

    sc = min(1000.0, 100.0 * F / B)
    print("N=%d K=%d F=%.4f B=%.4f Ratio: %.6f" % (N, K, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
