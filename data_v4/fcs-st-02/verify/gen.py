import sys, random

# Random small-case generator for fcs-st-02 (prefix-function reconstruction).
# param: int seed (argv[1]).
# Emits: n on the first line, then n integers pi[0..n-1].
#
# We deliberately mix THREE regimes so the differential test hits both the
# feasible and the -1 branches hard:
#   (A) a genuine prefix function of a random small-alphabet string  -> feasible
#   (B) such a prefix function with one entry perturbed              -> usually -1
#   (C) a fully random integer array within [0, i]                   -> mostly -1
# Sizes stay tiny so the backtracking brute is fast.

def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        k = pi[i - 1]
        while k > 0 and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 12)
    regime = rng.randint(0, 3)  # 0,1 -> A ; 2 -> B ; 3 -> C  (bias toward feasible)

    if n == 0:
        print(0)
        print()
        return

    if regime <= 1:
        # (A) real prefix function of a random tiny-alphabet string
        alpha = rng.randint(1, 3)
        s = [rng.randint(0, alpha - 1) for _ in range(n)]
        pi = prefix_function(s)
    elif regime == 2:
        # (B) real prefix function, then corrupt one position to a legal-looking value
        alpha = rng.randint(1, 3)
        s = [rng.randint(0, alpha - 1) for _ in range(n)]
        pi = prefix_function(s)
        j = rng.randrange(n)
        pi[j] = rng.randint(0, j)  # stays within [0,i] so structural check won't trivially catch it
    else:
        # (C) fully random array, each entry in [0, i]
        pi = [rng.randint(0, i) for i in range(n)]

    print(n)
    print(' '.join(map(str, pi)))

main()
