# TIER: invalid
# Emits offsets OUT OF RANGE (all above M) -> checker feasibility gate rejects
# it and scores 0.
import sys

def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    A = [M + 1 + i for i in range(n)]   # every value exceeds M
    sys.stdout.write(" ".join(map(str, A)) + "\n")

if __name__ == "__main__":
    main()
