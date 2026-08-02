import sys

# ---- shared tree helpers (heap-indexed full binary tree of depth T) ----
def depth(idx):
    return (idx + 1).bit_length() - 1

def path_bits(idx):
    bits = []
    j = idx
    while j > 0:
        parent = (j - 1) // 2
        bits.append(1 if j == 2 * parent + 2 else 0)
        j = parent
    bits.reverse()
    return bits

def build(T, cost, S0, SK, TL0, TK):
    """S[idx] = settlement value offered at node idx (any depth).
       L[leaf] = trial payoff at a depth-T leaf.
       Both depend only on the net number of favorable signals seen so far
       (f - u), which models the opponent's reservation price (S) and the
       trial's realized merits (L) both drifting with revealed information,
       at different sensitivities -- SK for the settlement track, TK for the
       trial track."""
    M = 2 ** (T + 1) - 1
    NLEAF = 2 ** T
    S = [0] * M
    L = [0] * NLEAF
    for idx in range(M):
        d = depth(idx)
        bits = path_bits(idx)
        f = sum(bits)
        u = d - f
        S[idx] = S0 + SK * (f - u)
        if d == T:
            leaf_j = idx - (2 ** T - 1)
            L[leaf_j] = TL0 + TK * (f - u)
    return S, L

# Hand-tuned ladder of 10 cases (testId -> parameters). testId 1-2 are small
# "warm-up" instances where a single fixed strategy already happens to be
# optimal (settling immediately, or always going to trial). testId 3-10 are
# discovery trees of growing depth where NEITHER fixed strategy is optimal:
# a staged, node-by-node policy dominates both, because it can bail out
# cheaply down unfavorable branches while riding favorable ones toward the
# (larger) trial payoff -- the option value that a single global choice
# cannot capture.
#   T     = number of discovery rounds (tree depth)
#   cost  = cost[k] = cost of completing round k+1 (accrues along every path
#           that reaches depth k+1) -- the information-arrival schedule
#   S0,SK = settlement track: offer at a node = S0 + SK*(favorable-unfavorable)
#   TL0,TK= trial track: payoff at a leaf   = TL0 + TK*(favorable-unfavorable)
CASES = {
    1:  dict(T=1, cost=[5],            S0=20,  SK=0,  TL0=40,  TK=20),
    2:  dict(T=1, cost=[40],           S0=100, SK=0,  TL0=20,  TK=15),
    3:  dict(T=2, cost=[2, 2],         S0=80,  SK=32, TL0=140, TK=320),
    4:  dict(T=2, cost=[1, 6],         S0=100, SK=40, TL0=150, TK=400),
    5:  dict(T=3, cost=[1, 1, 1],      S0=80,  SK=24, TL0=140, TK=239),
    6:  dict(T=3, cost=[1, 5, 1],      S0=80,  SK=24, TL0=82,  TK=239),
    7:  dict(T=4, cost=[1, 1, 1, 1],   S0=80,  SK=24, TL0=150, TK=240),
    8:  dict(T=4, cost=[1, 1, 1, 3],   S0=130, SK=39, TL0=190, TK=390),
    9:  dict(T=5, cost=[1, 1, 1, 1, 1],S0=130, SK=39, TL0=230, TK=390),
    10: dict(T=5, cost=[1, 1, 1, 1, 5],S0=130, SK=39, TL0=160, TK=390),
}

def main():
    i = int(sys.argv[1])
    if i not in CASES:
        i = ((i - 1) % 10) + 1
    p = CASES[i]
    T = p["T"]; cost = p["cost"]
    S, L = build(T, cost, p["S0"], p["SK"], p["TL0"], p["TK"])

    out = [str(T)]
    out.append(" ".join(str(c) for c in cost))
    out.append(" ".join(str(x) for x in S))
    out.append(" ".join(str(x) for x in L))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
