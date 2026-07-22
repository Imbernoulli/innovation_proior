# TIER: invalid
# Emits an infeasible artifact: assigns every project to bundle id K+1,
# which is out of the allowed [1,K] range, so the checker's feasibility gate
# must reject it -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); V = int(next(it)); K = int(next(it))
    print(P)
    print(" ".join(str(K + 1) for _ in range(P)))


if __name__ == "__main__":
    main()
