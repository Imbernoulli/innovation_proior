# TIER: greedy
# Gap-covering additive-basis greedy (Stohr / Mian-Chowla flavour): always keep the
# covered prefix [0, gap-1] contiguous and add the single depot slot that pushes the
# first uncovered slot 'gap' as far forward as possible (ties -> smallest slot).
import sys

def main():
    data = sys.stdin.read().split()
    n, M = int(data[0]), int(data[1])

    A = [0]
    covset = {0}
    gap = 1
    Aset = {0}
    while len(A) < n:
        g = gap
        cands = set()
        for a in A:
            for c in (g - a, g + a, a - g):
                if 1 <= c <= M and c not in Aset:
                    cands.add(c)
        if g % 2 == 0:
            c = g // 2
            if 1 <= c <= M and c not in Aset:
                cands.add(c)
        if not cands:                       # gap unreachable: add smallest free slot
            for c in range(1, M + 1):
                if c not in Aset:
                    cands.add(c); break
        best = None; bestr = -2
        for c in sorted(cands):             # deterministic: smallest slot wins ties
            newvals = {2 * c}
            for a in A:
                newvals.add(a + c); newvals.add(abs(a - c))
            gg = g
            while (gg in covset) or (gg in newvals):
                gg += 1
            if gg - 1 > bestr:
                bestr = gg - 1; best = c
        c = best
        for a in A:
            covset.add(a + c); covset.add(abs(a - c))
        covset.add(2 * c)
        A.append(c); Aset.add(c)
        while gap in covset:
            gap += 1

    out = [str(len(A))] + [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
