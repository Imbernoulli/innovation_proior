# TIER: trivial
# Baseline: just emit every short word (length <= 3) in shortlex order, capped at
# Nmax. Reproduces the checker's reference density (~0.1).
import sys, itertools

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    k = int(next(it)); L = int(next(it)); Nmax = int(next(it))
    al = [str(i) for i in range(k)]
    words = []
    for ln in range(1, min(L, 3) + 1):
        for t in itertools.product(al, repeat=ln):
            words.append("".join(t))
    words = words[:Nmax]
    sys.stdout.write("\n".join(words) + "\n")

if __name__ == "__main__":
    main()
