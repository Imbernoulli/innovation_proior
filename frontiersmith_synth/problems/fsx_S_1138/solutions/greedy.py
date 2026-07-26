# TIER: greedy
# The obvious "textbook" approach: fold along the pattern's GLOBAL bilateral
# symmetry.  Repeatedly fold the width in half as long as doing so is exactly
# safe for EVERY position on the whole sheet (i.e. the sheet really is a
# perfect mirror image at this level); then do the same for height; then
# punch every remaining position that must be a hole.  This assumes the
# pattern's symmetry is exact -- a single stray "defect" cell anywhere on the
# sheet makes the very first fold unsafe (its mirror partner disagrees), so
# this strategy silently gives up on folding altogether and degenerates to
# punching every hole individually.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it))
    target = [[False] * N for _ in range(N)]
    for _ in range(H):
        i = int(next(it)); j = int(next(it))
        target[i][j] = True

    val = target
    Wc, Hc = N, N
    ops = []

    foldsX = 0
    while Wc >= 2 and Wc % 2 == 0:
        newW = Wc // 2
        newval = [[False] * Hc for _ in range(newW)]
        safe = True
        for x in range(newW):
            row_a = val[x]; row_b = val[Wc - 1 - x]
            nr = newval[x]
            for y in range(Hc):
                a = row_a[y]; b = row_b[y]
                if a != b:
                    safe = False
                    break
                nr[y] = a
            if not safe:
                break
        if not safe:
            break
        val = newval; Wc = newW; foldsX += 1

    foldsY = 0
    while Hc >= 2 and Hc % 2 == 0:
        newH = Hc // 2
        newval = [[False] * newH for _ in range(Wc)]
        safe = True
        for x in range(Wc):
            row = val[x]; nr = newval[x]
            for y in range(newH):
                a = row[y]; b = row[Hc - 1 - y]
                if a != b:
                    safe = False
                    break
                nr[y] = a
            if not safe:
                break
        if not safe:
            break
        val = newval; Hc = newH; foldsY += 1

    for _ in range(foldsX):
        ops.append("FOLD_X")
    for _ in range(foldsY):
        ops.append("FOLD_Y")
    for x in range(Wc):
        row = val[x]
        for y in range(Hc):
            if row[y]:
                ops.append("PUNCH %d %d" % (x, y))
    ops.append("UNFOLD_ALL")
    sys.stdout.write("\n".join(ops) + "\n")


if __name__ == "__main__":
    main()
