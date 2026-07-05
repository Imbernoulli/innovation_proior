# TIER: greedy
"""Lexicographic greedy cap growth over the entire catalogue: walk signatures in code
order, add each non-protected one iff it creates no resonance triple with the current
manifest. Beats the reference sweep."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    pw = [3 ** i for i in range(n)]
    forb = set()
    for _ in range(m):
        code = 0
        for i in range(n):
            code += int(next(it)) * pw[i]
        forb.add(code)

    N = 3 ** n

    def third(a, b):
        c = 0; aa = a; bb = b
        for i in range(n):
            c += ((-(aa % 3) - (bb % 3)) % 3) * pw[i]
            aa //= 3; bb //= 3
        return c

    S = set()
    order = []
    for v in range(N):
        if v in forb:
            continue
        ok = True
        for a in S:
            if third(a, v) in S:  # third != a,v automatically for distinct pts
                ok = False
                break
        if ok:
            S.add(v)
            order.append(v)

    out = []
    for v in order:
        digs = []
        x = v
        for _ in range(n):
            digs.append(x % 3); x //= 3
        out.append(" ".join(str(d) for d in digs))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
