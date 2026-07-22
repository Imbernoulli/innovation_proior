#!/usr/bin/env python3
# gen.py <testId>  ->  prints ONE synchronizing-DFA instance to stdout.
#
# Family: reset-word-cerny.  We plant a Cerny-family automaton:
#   * one letter is a full n-cycle PERMUTATION  R  (i -> next in a hidden cyclic order),
#   * one letter is a single CONTRACTION        C  that merges exactly one R-adjacent pair
#     (state x and its R-successor y both map to y; C is identity elsewhere).
# The state labels are randomly relabelled and the two letters are shuffled, so the
# cyclic order and the location of the contraction are hidden.  This is exactly the
# Cerny automaton up to isomorphism (shortest reset word length (n-1)^2), but the
# planted permutation/contraction algebra must be RECOVERED from the transition table.
#
# Deterministic: everything is seeded from testId only.
import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(100000 + 977 * tid)

    # difficulty ladder: n grows with testId (all sizes are "trap" sizes for pairwise merging)
    ladder = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    n = ladder[(tid - 1) % len(ladder)]

    # hidden cyclic order:  pos i  ->  state label perm[i]
    perm = list(range(n))
    rng.shuffle(perm)

    # hidden location of the contraction inside the cycle
    j = rng.randrange(n)          # C merges pos j and pos j+1 (i.e. x=perm[j], y=perm[j+1])

    # build the two transition functions in state-label coordinates
    R = [0] * n                   # rotation (permutation)
    C = [0] * n                   # contraction
    for i in range(n):
        R[perm[i]] = perm[(i + 1) % n]
        # C is identity on every position except position j, which is pulled onto j+1
        src = perm[i]
        C[src] = perm[(i + 1) % n] if i == j else perm[i]

    # shuffle which symbol index is R vs C
    syms = [R, C]
    order = [0, 1]
    rng.shuffle(order)
    delta = [syms[o] for o in order]
    m = len(delta)

    out = ["%d %d" % (n, m)]
    for s in range(m):
        out.append(" ".join(str(delta[s][i]) for i in range(n)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
