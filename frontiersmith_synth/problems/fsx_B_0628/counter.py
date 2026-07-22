import sys, math

# Format D checker -- endurance-race pacing planner.
#   1) Parse the instance (L, k, coefficients, intensity grid) from <in>.
#   2) Parse the participant plan from <out>: exactly 2*L integer tokens, read as
#      L pairs (idx, pit) with idx in [0,k-1] and pit in {0,1}.  ANY schema/range
#      violation, wrong token count, or non-integer/non-finite token -> Ratio 0.0.
#   3) Objective (minimize) F = exact total race time of the plan under the model
#         lap_time = base + a*wear**p + b/x ;  wear += x**q ;  pit: +P, wear=0.
#   4) Internal baseline B = total time of the "coast" plan (minimum intensity
#      every lap, no pits) -- the checker builds it itself.
#      Ratio = min(1, 0.1 * B / F).   (fewer seconds is better)

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def simulate(L, base, a, p, b, q, P, grid, xs, pits):
    wear = 0.0
    total = 0.0
    for i in range(L):
        if pits[i]:
            total += P
            wear = 0.0
        x = grid[xs[i]]
        total += base + a * (wear ** p) + b / x
        wear += x ** q
    return total

def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read instance")
    it = iter(inp)
    try:
        L = int(next(it)); k = int(next(it))
        base = float(next(it)); a = float(next(it)); p = float(next(it))
        b = float(next(it)); q = float(next(it)); P = float(next(it))
        grid = [float(next(it)) for _ in range(k)]
    except Exception:
        fail("bad instance header")
    if not (1 <= L <= 100000 and 2 <= k <= 64):
        fail("bad dims")
    if any(g <= 0.0 for g in grid):
        fail("nonpositive intensity in grid")

    # ---- parse participant plan ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if len(out) != 2 * L:
        fail("wrong token count (got %d, need %d)" % (len(out), 2 * L))
    xs = [0] * L
    pits = [0] * L
    for i in range(L):
        # strict integer parse rejects nan/inf/floats/garbage
        try:
            idx = int(out[2 * i])
            pf = int(out[2 * i + 1])
        except Exception:
            fail("non-integer token at lap %d" % i)
        if not (0 <= idx < k):
            fail("intensity index out of range at lap %d" % i)
        if pf not in (0, 1):
            fail("pit flag not in {0,1} at lap %d" % i)
        xs[i] = idx
        pits[i] = pf

    F = simulate(L, base, a, p, b, q, P, grid, xs, pits)
    if not math.isfinite(F) or F <= 0.0:
        fail("non-finite/degenerate objective")

    # baseline: coast at minimum intensity, no pits (checker's own construction)
    B = simulate(L, base, a, p, b, q, P, grid, [0] * L, [0] * L)
    if not math.isfinite(B) or B <= 0.0:
        fail("degenerate baseline")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
