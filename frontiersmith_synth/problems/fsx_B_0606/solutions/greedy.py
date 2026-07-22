# TIER: greedy
# The obvious approach: treat it as an ATSP over the changeover matrix (nearest-neighbour
# tour that minimizes each successive transition) and clean ONLY when the residue cap would
# otherwise be exceeded (reactive cleaning at the last feasible moment). This ignores that
# clean placement -- not the tour -- carries most of the objective.
import sys

def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); R = int(next(it)); C = int(next(it))
    r = [int(next(it)) for _ in range(N)]
    s = [int(next(it)) for _ in range(N)]
    w = [[int(next(it)) for _ in range(N)] for _ in range(N)]
    return N, R, C, r, s, w

def main():
    N, R, C, r, s, w = read_instance()
    # nearest-neighbour tour starting from the cheapest-startup color
    start = min(range(N), key=lambda j: s[j])
    used = [False] * N; used[start] = True
    order = [start]; cur = start
    for _ in range(N - 1):
        nxt = min((j for j in range(N) if not used[j]), key=lambda j: w[cur][j])
        used[nxt] = True; order.append(nxt); cur = nxt
    # reactive cleaning: clean only when the next job would overflow
    out = []; A = 0
    for j in order:
        if A + r[j] > R:
            out.append("C"); A = 0
        out.append(str(j)); A += r[j]
    sys.stdout.write(" ".join(out) + "\n")

main()
