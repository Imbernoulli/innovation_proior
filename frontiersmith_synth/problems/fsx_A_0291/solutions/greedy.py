# TIER: greedy
# Single lexicographic cap-set scaffold over F_3^n, excluding blocked routes.
# Maintains an incremental "forbidden completion" set F so each addition is O(|S|).
import sys

def main():
    tok = sys.stdin.read().split()
    idx = 0
    n = int(tok[idx]); idx += 1
    m = int(tok[idx]); idx += 1
    blocked = set()
    for _ in range(m):
        v = tuple(int(tok[idx + i]) for i in range(n)); idx += n
        blocked.add(v)
    space = 3 ** n

    def dec(e):
        d = []
        for _ in range(n):
            d.append(e % 3); e //= 3
        return tuple(d)

    S = []
    Sset = set()
    F = set()
    for e in range(space):
        p = dec(e)
        if p in blocked or p in Sset or p in F:
            continue
        for x in S:
            z = tuple((3 - ((p[i] + x[i]) % 3)) % 3 for i in range(n))
            F.add(z)
        S.append(p)
        Sset.add(p)

    out = [str(len(S))]
    for v in S:
        out.append(' '.join(map(str, v)))
    sys.stdout.write('\n'.join(out) + '\n')

main()
