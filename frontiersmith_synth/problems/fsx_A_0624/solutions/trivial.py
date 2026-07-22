# TIER: trivial
# Uniform capacities: split T as evenly as possible. This reproduces the checker's
# own baseline construction, so it scores ~0.1 by design.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); M = int(toks[1]); T = int(toks[2])
    base = [T // M] * M
    for j in range(T % M):
        base[j] += 1
    sys.stdout.write("\n".join(map(str, base)) + "\n")

if __name__ == "__main__":
    main()
