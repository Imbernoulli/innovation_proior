# TIER: strong
# The insight: the moves conserve a potential -- the token count M, the total
# position sum S = sum of token cells (each scatter/gather shifts +i-1 and +i+1
# against -2i, a net zero), and with them the whole reachable class. Local
# gains are bait: the best arrangement reachable from the start can sit behind
# moves that temporarily LOSE collected weight (a scatter that empties a
# weighted centre) or through zero-gain plateaus.
#
# So instead of grabbing local gains, we compute the reachable class itself
# (BFS over arrangements; every move preserves the conserved potential, so the
# search never leaves the class), evaluate collected weight over the WHOLE
# class, pick the best reachable arrangement, and output the routing move
# sequence that BFS parent pointers certify. The far-away heavy "bait" cells
# are automatically ignored: no reachable arrangement covers them.
import sys
from collections import deque

def coverage(state, w):
    return sum(w[c - 1] for c in set(state))

def moves_from(state, N):
    res = []
    occ = set(state)
    from collections import Counter
    cnt = Counter(state)
    for i in range(2, N):
        if cnt.get(i, 0) >= 2:
            lst = list(state)
            lst.remove(i); lst.remove(i)
            lst.append(i - 1); lst.append(i + 1)
            res.append((('s', i), tuple(sorted(lst))))
        if (i - 1) in occ and (i + 1) in occ:
            lst = list(state)
            lst.remove(i - 1); lst.remove(i + 1)
            lst.append(i); lst.append(i)
            res.append((('g', i), tuple(sorted(lst))))
    return res

def main():
    d = sys.stdin.read().split()
    N, M = int(d[0]), int(d[1])
    w = list(map(int, d[2:2 + N]))
    init = tuple(sorted(map(int, d[2 + N:2 + N + M])))

    par = {init: (None, None)}        # state -> (parent, move taken)
    dq = deque([init])
    best, bestcov = init, coverage(init, w)
    while dq:
        s = dq.popleft()
        for mv, ns in moves_from(s, N):
            if ns not in par:
                par[ns] = (s, mv)
                dq.append(ns)
                cv = coverage(ns, w)
                if cv > bestcov:
                    best, bestcov = ns, cv

    path = []
    x = best
    while par[x][0] is not None:
        path.append(par[x][1])
        x = par[x][0]
    path.reverse()

    print(len(path))
    for t, i in path:
        print(t, i)

if __name__ == "__main__":
    main()
