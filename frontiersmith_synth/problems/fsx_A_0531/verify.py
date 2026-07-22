import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Fold simulator (shared, deterministic).
#
# State: `slots`, a left-to-right list; slot j is the list of original cell ids
# currently stacked at footprint position j. A simple fold at line k (1<=k<=w-1)
# reflects the SHORTER side across the line and merges its stacks onto the other
# side (ties fold the left side). The fold's stress is:
#     S  = total thickness of the flipped (shorter) side, plus
#     W * (number of reinforced creases bent by this fold),
# where crease c is "bent" iff cells c and c+1 currently lie on opposite sides
# of line k. Returns total cost, or None if any fold is illegal / final width>T.
# ---------------------------------------------------------------------------
def simulate(N, T, W, h, reinforced, folds):
    slots = [[c] for c in range(N)]
    cost = 0
    for k in folds:
        w = len(slots)
        if not (1 <= k <= w - 1):
            return None
        # positions of every cell for bent-crease detection
        pos = [0] * N
        for si, sl in enumerate(slots):
            for c in sl:
                pos[c] = si
        bent = 0
        for c in reinforced:
            if (pos[c] < k) != (pos[c + 1] < k):
                bent += 1
        if k <= w - k:
            # flip left (slots 0..k-1) onto right (slots k..w-1)
            left = slots[:k]
            S = sum(h[c] for sl in left for c in sl)
            new = [list(sl) for sl in slots[k:]]
            for idx in range(k):
                tgt = k - 1 - idx  # (2k-1-idx) - k
                new[tgt].extend(left[idx])
            slots = new
        else:
            # flip right (slots k..w-1) onto left (slots 0..k-1)
            right = slots[k:]
            S = sum(h[c] for sl in right for c in sl)
            new = [list(sl) for sl in slots[:k]]
            for j in range(k, w):
                tgt = 2 * k - 1 - j
                new[tgt].extend(slots[j])
            slots = new
        cost += S + W * bent
    if len(slots) > T:
        return None
    return cost


def baseline_cost(N, T, W, h, reinforced):
    # naive "accordion from the left": fold one slot at a time (k=1) until width<=T.
    folds = [1] * (N - T)
    c = simulate(N, T, W, h, reinforced, folds)
    return c if c is not None else 1


def main():
    inp = open(sys.argv[1]).read().split()
    try:
        it = iter(inp)
        N = int(next(it)); T = int(next(it)); R = int(next(it)); W = int(next(it))
        h = [int(next(it)) for _ in range(N)]
        reinforced = [int(next(it)) for _ in range(R)]
    except Exception:
        fail("bad input")

    for c in reinforced:
        if not (0 <= c <= N - 2):
            fail("bad crease")

    out = open(sys.argv[2]).read().split()
    try:
        M = int(out[0])
    except Exception:
        fail("no fold count")
    if M < 0 or M > 4 * N + 5:
        fail("fold count out of range")
    if len(out) < 1 + M:
        fail("missing folds")
    folds = []
    for t in out[1:1 + M]:
        try:
            v = int(t)
        except Exception:
            fail("non-integer fold %r" % t)
        folds.append(v)

    F = simulate(N, T, W, h, reinforced, folds)
    if F is None:
        fail("infeasible schedule")
    if F <= 0:
        fail("nonpositive cost")

    B = baseline_cost(N, T, W, h, reinforced)
    # Normalise so the accordion baseline (F=B) scores ~0.075 and a ~13x-cheaper
    # schedule caps at 1.0 -- the reduced coefficient leaves headroom above the
    # reference strong solution (the ceiling stays open for an RL policy).
    ratio = min(1.0, 0.075 * B / max(1e-9, float(F)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
