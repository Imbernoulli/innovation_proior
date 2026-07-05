# TIER: trivial
# Reproduce the checker's internal baseline: the {0,1}^n sub-cube minus blocked routes.
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
    res = []
    for msk in range(2 ** (n - 1)):
        v = (0,) + tuple((msk >> i) & 1 for i in range(n - 1))
        if v not in blocked:
            res.append(v)
    out = [str(len(res))]
    for v in res:
        out.append(' '.join(map(str, v)))
    sys.stdout.write('\n'.join(out) + '\n')

main()
