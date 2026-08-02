# TIER: trivial
"""Reproduces the checker's own baseline: a raw trie over the legal traces plus one
explicit self-looping, non-accepting sink for everything else. Always feasible, never
merges anything -- this is exactly the internal baseline B, so it scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = int(data[pos])
        pos += 1
        return v

    m = nxt()
    L = []
    n_l = nxt()
    for _ in range(n_l):
        ln = nxt()
        L.append(tuple(nxt() for _ in range(ln)))
    n_f = nxt()
    for _ in range(n_f):
        ln = nxt()
        for _ in range(ln):
            nxt()

    children = [dict()]
    accept = [False]
    for tr in L:
        cur = 0
        for c in tr:
            if c in children[cur]:
                cur = children[cur][c]
            else:
                children.append(dict())
                accept.append(False)
                nid = len(children) - 1
                children[cur][c] = nid
                cur = nid
        accept[cur] = True

    n_trie = len(children)
    sink = n_trie
    N = n_trie + 1
    delta = [[0] * m for _ in range(N)]
    for u in range(n_trie):
        for c in range(m):
            delta[u][c] = children[u].get(c, sink)
    for c in range(m):
        delta[sink][c] = sink
    acc = accept + [False]

    out = [str(N), " ".join("1" if a else "0" for a in acc)]
    for row in delta:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
