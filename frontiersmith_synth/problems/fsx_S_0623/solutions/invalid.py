# TIER: invalid
"""Emits a structurally garbage / infeasible artifact (an out-of-range vertex
id in G), which must score 0 under the checker's strict feasibility gate."""
import sys


def main():
    n, m = map(int, sys.stdin.read().split())
    # Deliberately reference vertex n+1, which is out of the valid [1,n] range.
    print(n, 1)
    print(1, n + 1)
    print(n, 0)


if __name__ == "__main__":
    main()
