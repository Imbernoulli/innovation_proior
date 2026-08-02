# TIER: trivial
import sys

# Reproduces the simplest always-feasible construction: run at the top
# DVFS level for the entire horizon, never idle. No thought about
# deadlines, transitions, or convexity at all -- just brute force.


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0]); m = int(toks[1])
    print(" ".join(str(m - 1) for _ in range(T)))


if __name__ == "__main__":
    main()
