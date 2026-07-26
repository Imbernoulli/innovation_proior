import sys

# Format D checker -- paper-snowflake fold/punch instruction-tape op-count.
#
#   1) Parse target hole set (i,j) pairs on an N x N sheet from <in>.
#   2) Simulate the participant's instruction tape from <out> EXACTLY:
#        FOLD_X            -- fold the far half of the current width onto the near half
#        FOLD_Y            -- fold the far half of the current height onto the near half
#        PUNCH <x> <y>     -- pierces every original cell currently stacked at (x,y)
#        UNFOLD_ALL        -- must appear exactly once, as the LAST instruction
#   3) EXACT-equality gate: the union of all punched original cells must equal the
#      target hole set exactly (no missing hole, no extra hole).
#   4) Objective (minimize) = total instruction count F.
#      Baseline B = |target| + 1 (punch every hole individually, zero folds).
#      Ratio = min(1, 0.1 * B / F).

MAX_OPS = 5000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    it = iter(inp)
    try:
        N = int(next(it))
        H = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= N <= 4096) or (N & (N - 1)) != 0:
        fail("N must be a power of two")
    if H < 0:
        fail("bad H")

    target = [[False] * N for _ in range(N)]
    cnt = 0
    try:
        for _ in range(H):
            i = int(next(it)); j = int(next(it))
            if not (0 <= i < N and 0 <= j < N):
                fail("target cell out of range")
            if not target[i][j]:
                target[i][j] = True
                cnt += 1
    except Exception:
        fail("bad target list")
    if cnt == 0:
        fail("degenerate empty target")

    out_tokens = open(sys.argv[2]).read().split()
    ot = iter(out_tokens)

    StackX = [[i] for i in range(N)]
    StackY = [[j] for j in range(N)]
    Wc, Hc = N, N
    holes = [[False] * N for _ in range(N)]
    op_count = 0
    seen_unfold = False

    for tok in ot:
        if seen_unfold:
            fail("instruction after UNFOLD_ALL")
        if tok == "FOLD_X":
            if Wc < 2 or Wc % 2 != 0:
                fail("FOLD_X: current width %d not foldable" % Wc)
            newWc = Wc // 2
            StackX = [StackX[x] + StackX[Wc - 1 - x] for x in range(newWc)]
            Wc = newWc
            op_count += 1
        elif tok == "FOLD_Y":
            if Hc < 2 or Hc % 2 != 0:
                fail("FOLD_Y: current height %d not foldable" % Hc)
            newHc = Hc // 2
            StackY = [StackY[y] + StackY[Hc - 1 - y] for y in range(newHc)]
            Hc = newHc
            op_count += 1
        elif tok == "PUNCH":
            try:
                x = int(next(ot)); y = int(next(ot))
            except Exception:
                fail("bad PUNCH arguments")
            if not (0 <= x < Wc and 0 <= y < Hc):
                fail("PUNCH out of current sheet bounds")
            for i in StackX[x]:
                row = holes[i]
                for j in StackY[y]:
                    row[j] = True
            op_count += 1
        elif tok == "UNFOLD_ALL":
            seen_unfold = True
            op_count += 1
        else:
            fail("unknown instruction token %r" % tok)
        if op_count > MAX_OPS:
            fail("instruction budget exceeded (> %d ops)" % MAX_OPS)

    if not seen_unfold:
        fail("missing UNFOLD_ALL (program must end with it)")

    for i in range(N):
        ti = target[i]; hi = holes[i]
        for j in range(N):
            if ti[j] != hi[j]:
                fail("hole set mismatch at (%d,%d)" % (i, j))

    F = op_count
    B = H + 1
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Ops: %d  Baseline: %d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
