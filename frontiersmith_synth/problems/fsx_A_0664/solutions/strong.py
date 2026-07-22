# TIER: strong
"""Insight: SimRank-style iterative structural similarity (Jeh & Widom) -- two states
are similar if the states they transition to (under the SAME symbol) are themselves
similar -- propagated over several rounds from the trivial seed sim(u,u)=1 else 0. This
recovers behavioral adjacency that only shows up several hops downstream (which a
single-symbol / single-hop heuristic like 'greedy' cannot see), without ever leaking the
generator's hidden bit structure or state identity as a feature. A greedy nearest-
neighbour chain over the resulting similarity matrix then linearizes the states so that
behaviorally-similar states are next to each other, and a reflected-binary Gray code
turns that linear order into Hamming-distance-1-adjacent codewords -- collapsing cubes
in the minimized next-state logic."""
import sys


def gray_code(p, B):
    g = p ^ (p >> 1)
    return format(g, f"0{B}b")


def simrank(N, M, trans, rounds=6, decay=0.8):
    sim = [[1.0 if u == v else 0.0 for v in range(N)] for u in range(N)]
    for _ in range(rounds):
        new_sim = [[0.0] * N for _ in range(N)]
        for u in range(N):
            row_u = trans[u]
            for v in range(N):
                if u == v:
                    new_sim[u][v] = 1.0
                    continue
                row_v = trans[v]
                total = 0.0
                for s in range(M):
                    total += sim[row_u[s]][row_v[s]]
                new_sim[u][v] = decay * total / M
        sim = new_sim
    return sim


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    M = 1 << K
    B = N.bit_length() - 1
    trans = []
    for _u in range(N):
        row = [int(next(it)) for _ in range(M)]
        trans.append(row)

    sim = simrank(N, M, trans)

    order = [0]
    used = {0}
    while len(order) < N:
        last = order[-1]
        best_v, best_key = None, None
        for v in range(N):
            if v in used:
                continue
            key = (sim[last][v], -v)
            if best_key is None or key > best_key:
                best_key = key
                best_v = v
        order.append(best_v)
        used.add(best_v)

    codeword = [None] * N
    for pos, u in enumerate(order):
        codeword[u] = gray_code(pos, B)

    sys.stdout.write("\n".join(codeword) + "\n")


if __name__ == "__main__":
    main()
