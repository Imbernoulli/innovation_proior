# TIER: invalid
"""Deliberately infeasible: overspends the budget (and throws in a negative dose for
good measure) so the checker must reject it with Ratio: 0.0."""
import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    budget = int(data[4])
    doses = [0] * N
    if N > 0:
        doses[0] = budget + 10**6
    if N > 1:
        doses[1] = -5
    print(" ".join(str(d) for d in doses))

if __name__ == "__main__":
    main()
