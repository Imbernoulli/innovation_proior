# TIER: greedy
"""The obvious first idea for "differential adhesion sorting": the classical
Steinberg intuition that alike cells should stick together. Diagonal-
dominant matrix -- J[a][a] = -Jmax (maximally sticky same-type contacts),
J[a][b] = +Jmax for a != b (maximally costly cross-type contacts) --
applied UNCONDITIONALLY, without even looking at target_type. This reliably
drives the ring toward full segregation (the engulfment/blob topology) and
scores well whenever that IS the target. But when the target is
"interleaved" (target_type=1) this is exactly backwards: maximizing
same-type clustering is the opposite of what mixing needs, so this
solution lands far from a frustration-aware strategy on every interleaved
case -- the trap."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[1])
    Jmax = int(data[2])
    # target_type intentionally ignored -- that is the trap.

    J = [[-Jmax if a == b else Jmax for b in range(T)] for a in range(T)]
    for row in J:
        print(" ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
