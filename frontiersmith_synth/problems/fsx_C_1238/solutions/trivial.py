# TIER: trivial
"""Reproduces the checker's own weak baseline: enable only flags that have
NO requires and NO conflicts at all, in index order; pass otherwise."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    N = int(nxt())
    _values = [int(nxt()) for _ in range(N)]
    R = int(nxt())
    req_of = {i: [] for i in range(1, N + 1)}
    for _ in range(R):
        c = int(nxt())
        p = int(nxt())
        req_of[c].append(p)
    C = int(nxt())
    conf_of = {i: set() for i in range(1, N + 1)}
    for _ in range(C):
        a = int(nxt())
        b = int(nxt())
        conf_of[a].add(b)
        conf_of[b].add(a)

    isolated = set(
        i for i in range(1, N + 1)
        if len(req_of[i]) == 0 and len(conf_of[i]) == 0
    )
    out = []
    for i in range(1, N + 1):
        if i in isolated:
            out.append(f"E {i}")
        else:
            out.append("P")
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
