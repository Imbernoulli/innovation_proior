# TIER: greedy
# Structured difference-dominant construction: place stations with an irregular
# repeating gap pattern (1,2,3,1,2,3,...). This keeps many distinct differences while
# forcing sums to collide, giving R ~ 0.83 with no search. A fixed heuristic.
import sys


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    A = []
    x = 0
    i = 0
    seen = set()
    while len(A) < n and x <= M:
        if x not in seen:
            A.append(x)
            seen.add(x)
        i += 1
        x += 1 + (i % 3)
    # pad (should not trigger for M = 5n) with any remaining free slots
    v = 0
    while len(A) < n:
        if v not in seen and v <= M:
            A.append(v)
            seen.add(v)
        v += 1
    sys.stdout.write(" ".join(map(str, A[:n])) + "\n")


if __name__ == "__main__":
    main()
