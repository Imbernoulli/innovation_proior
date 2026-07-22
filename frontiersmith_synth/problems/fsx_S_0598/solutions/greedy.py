# TIER: greedy
# The obvious approach: enumerate words and de-duplicate by RAW STRING equality,
# taking the Nmax shortlex-smallest distinct words. This ignores the rewriting
# congruence entirely -- many of these short words share a normal form, so they
# collapse to far fewer distinct classes than the raw count suggests (the trap).
import sys, itertools

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    k = int(next(it)); L = int(next(it)); Nmax = int(next(it))
    al = [str(i) for i in range(k)]
    words = []
    for ln in range(1, L + 1):
        for t in itertools.product(al, repeat=ln):
            words.append("".join(t))
            if len(words) >= Nmax:
                sys.stdout.write("\n".join(words) + "\n")
                return
    sys.stdout.write("\n".join(words) + "\n")

if __name__ == "__main__":
    main()
