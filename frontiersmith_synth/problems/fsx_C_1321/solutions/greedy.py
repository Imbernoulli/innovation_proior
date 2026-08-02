# TIER: greedy
"""Obvious first-instinct heuristic: reactant enters at the two reservoirs,
so put the sites as close to the supply as possible -- split the budget
between both ends and pack each half as a solid block touching its
reservoir. This maximizes nominal proximity to supply and uses the full
budget, but ignores the crowding penalty between adjacent sites and the
diffusion-limited self-starvation of a dense cluster's inner cells."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0
    L = int(data[pos]); pos += 1
    B = int(data[pos]); pos += 1
    active = [0] * L
    left_n = (B + 1) // 2
    right_n = B - left_n
    for i in range(min(left_n, L)):
        active[i] = 1
    for i in range(right_n):
        active[L - 1 - i] = 1
    sys.stdout.write(" ".join(str(x) for x in active) + "\n")


if __name__ == "__main__":
    main()
