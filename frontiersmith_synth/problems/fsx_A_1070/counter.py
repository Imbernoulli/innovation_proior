import sys

MAXFIRINGS = 300


def simulate_scaled(N, firings, T):
    """firings: list of (pos, s, d, p), all non-negative ints.
    Exact integer diffusion: h'[i] = h[i]/2 + h[i-1]/4 + h[i+1]/4 (insulated /
    mirrored boundary). Returns (Hs, scale) with true heat[i] == Hs[i] / scale
    and scale == 4**T -- pure integer arithmetic, bit-for-bit deterministic."""
    Hs = [0] * N
    scale = 1
    for t in range(T):
        for (pos, s, d, p) in firings:
            if s <= t < s + d:
                Hs[pos] += p * scale
        nHs = [0] * N
        for i in range(N):
            left = Hs[i - 1] if i - 1 >= 0 else Hs[i]
            right = Hs[i + 1] if i + 1 < N else Hs[i]
            nHs[i] = 2 * Hs[i] + left + right
        Hs = nHs
        scale *= 4
    return Hs, scale


def nearest_burner_idx(pos, burners):
    best_i, best_d = 0, None
    for i, bp in enumerate(burners):
        dd = abs(pos - bp)
        if best_d is None or dd < best_d or (dd == best_d and i < best_i):
            best_d = dd
            best_i = i
    return best_i


def baseline_fuel(N, T, Pmax, F0, burners, targets):
    used = sorted({nearest_burner_idx(x, burners) for x in targets})
    return len(used) * (Pmax * T + F0)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tok = f.read().split()
    it = iter(in_tok)
    N = int(next(it)); T = int(next(it)); H = int(next(it))
    Pmax = int(next(it)); F0 = int(next(it))
    numB = int(next(it)); numT = int(next(it))
    burners = [int(next(it)) for _ in range(numB)]
    targets = [int(next(it)) for _ in range(numT)]

    try:
        out_txt = open(out_path).read()
    except Exception:
        print("Ratio: 0.0 (cannot read output)")
        return 0

    out_tok = out_txt.split()

    def bad(reason):
        print("Ratio: 0.0 (%s)" % reason)

    if not out_tok:
        bad("empty output")
        return 0

    # ---- parse K, then K firings of 4 ints each; reject non-finite/garbage/huge ----
    try:
        K = int(out_tok[0])
    except ValueError:
        bad("K not an integer")
        return 0
    if K < 0 or K > MAXFIRINGS:
        bad("K out of range [0,%d]" % MAXFIRINGS)
        return 0
    need_tokens = 1 + 4 * K
    if len(out_tok) < need_tokens:
        bad("too few tokens for declared K")
        return 0
    if len(out_tok) > need_tokens:
        bad("trailing garbage tokens")
        return 0

    firings_idx = []
    try:
        pos = 1
        for _ in range(K):
            b_idx = int(out_tok[pos]); s = int(out_tok[pos + 1])
            d = int(out_tok[pos + 2]); p = int(out_tok[pos + 3])
            pos += 4
            if not (0 <= b_idx < numB):
                bad("burner index out of range")
                return 0
            if not (0 <= s <= T - 1):
                bad("start out of range")
                return 0
            if not (1 <= d <= T - s):
                bad("duration out of range / firing crosses deadline T")
                return 0
            if not (1 <= p <= Pmax):
                bad("power out of range")
                return 0
            firings_idx.append((b_idx, s, d, p))
    except ValueError:
        bad("non-integer token (e.g. nan/inf/float) in firing list")
        return 0

    pos_firings = [(burners[b_idx], s, d, p) for (b_idx, s, d, p) in firings_idx]
    Hs, scale = simulate_scaled(N, pos_firings, T)
    need = H * scale
    for x in targets:
        if Hs[x] < need:
            bad("target cell %d below threshold at deadline T" % x)
            return 0

    fuel = sum(p * d + F0 for (_b, s, d, p) in firings_idx)
    if fuel <= 0:
        bad("zero/negative fuel with feasible (impossible) plan")
        return 0

    B = baseline_fuel(N, T, Pmax, F0, burners, targets)
    sc = min(1000.0, 100.0 * B / max(1e-9, fuel))
    print("fuel=%d baseline=%d Ratio: %.6f" % (fuel, B, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
