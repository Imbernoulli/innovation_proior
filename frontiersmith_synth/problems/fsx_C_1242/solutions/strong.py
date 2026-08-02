# TIER: strong
"""The insight: this is a shortest-path problem over a tiny state graph, not a
single global choice. State = which of the 6 codes the array is currently
wired for. Layer i contributes layer_cost(i, code) at every state, plus a
SWITCH*P*Q edge weight whenever the code changes between layer i-1 and i. A
straightforward per-layer DP (Bellman/Viterbi over 6 states) finds the exact
minimum-cost code sequence -- switching dataflow on the layers where the
reload/idle-cell savings outweigh the reconfiguration cost, and staying put
where they don't. This is not "greedy plus more search": the fixed-mode
heuristic corresponds to forcing a single state across the whole path, which
this DP shows is provably suboptimal whenever the sequence mixes shapes."""
import sys

CODES = ["MKN", "MNK", "KMN", "KNM", "NMK", "NKM"]


def read_instance():
    toks = sys.stdin.read().split()
    i = 0

    def nxt():
        nonlocal i
        v = toks[i]
        i += 1
        return v

    P = int(nxt()); Q = int(nxt()); L = int(nxt())
    RELOAD = int(nxt()); SWITCH = int(nxt())
    layers = []
    for _ in range(L):
        m = int(nxt()); k = int(nxt()); n = int(nxt())
        layers.append({'M': m, 'K': k, 'N': n})
    return P, Q, L, RELOAD, SWITCH, layers


def layer_cost(P, Q, RELOAD, dims, code):
    d1, d2, s = code[0], code[1], code[2]
    D1, D2, S = dims[d1], dims[d2], dims[s]
    tp = -(-D1 // P)
    tq = -(-D2 // Q)
    pipe = P + Q - 1
    return tp * tq * P * Q * (RELOAD + S + pipe)


def main():
    P, Q, L, RELOAD, SWITCH, layers = read_instance()
    switch_edge = SWITCH * P * Q

    own = [layer_cost(P, Q, RELOAD, layers[0], c) for c in CODES]
    dp = own[:]                      # dp[j] = min cost of layers[0..0] ending in code j
    back = [[None] * 6 for _ in range(L)]  # back[i][j] = previous code idx chosen at layer i

    for i in range(1, L):
        own_i = [layer_cost(P, Q, RELOAD, layers[i], c) for c in CODES]
        ndp = [None] * 6
        for j in range(6):
            best_prev, best_val = None, None
            for pj in range(6):
                val = dp[pj] + (0 if pj == j else switch_edge)
                if best_val is None or val < best_val:
                    best_val, best_prev = val, pj
            ndp[j] = best_val + own_i[j]
            back[i][j] = best_prev
        dp = ndp

    end = min(range(6), key=lambda j: dp[j])
    seq = [None] * L
    seq[L - 1] = end
    for i in range(L - 1, 0, -1):
        seq[i - 1] = back[i][seq[i]]

    out = [str(L)] + [CODES[j] for j in seq]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
