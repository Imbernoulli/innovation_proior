# TIER: strong
# The insight: this is an optimal-stopping problem. Value the OPTION to keep
# gathering information rather than pre-committing to a single global choice.
# Backward-induct from the terminal (trial) nodes: at every node, compare
# settling right there against the *expected value of continuing*, which is
# itself the average of the two children's already-computed optimal values.
# A node settles only if that beats what waiting (net of the next round's
# cost) is worth -- so the policy naturally rides favorable branches deeper
# (toward the bigger trial payoff) while cutting losses immediately down
# branches that turned unfavorable, capturing option value neither fixed
# global strategy can reach.
import sys
from fractions import Fraction as Fr

def depth(idx):
    return (idx + 1).bit_length() - 1

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    cost = [int(next(it)) for _ in range(T)]
    M = 2 ** (T + 1) - 1
    S = [int(next(it)) for _ in range(M)]
    NLEAF = 2 ** T
    L = [int(next(it)) for _ in range(NLEAF)]

    accrued = [Fr(0)] * M
    for idx in range(1, M):
        parent = (idx - 1) // 2
        accrued[idx] = accrued[parent] + cost[depth(parent)]

    value = [None] * M
    policy = ["S"] * M
    for idx in reversed(range(M)):
        d = depth(idx)
        settle_val = Fr(S[idx]) - accrued[idx]
        if d == T:
            leaf_j = idx - (2 ** T - 1)
            cont_val = Fr(L[leaf_j]) - accrued[idx]
        else:
            cont_val = Fr(1, 2) * (value[2 * idx + 1] + value[2 * idx + 2])
        if cont_val > settle_val:
            value[idx] = cont_val
            policy[idx] = "C"
        else:
            value[idx] = settle_val
            policy[idx] = "S"

    print(M)
    print(" ".join(policy))

if __name__ == "__main__":
    main()
