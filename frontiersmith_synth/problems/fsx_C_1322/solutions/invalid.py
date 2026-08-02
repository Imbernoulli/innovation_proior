# TIER: invalid
"""Deliberately infeasible artifact: an out-of-range seed index (K+999) paired
with a temperature schedule that INCREASES first (violates the monotone-cooling
requirement) before ever reaching T_min. Either defect alone must be rejected;
this trips both. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    T_start = float(data[1])
    T_min = float(data[2])

    # We don't need K precisely for an invalid answer -- just pick something
    # guaranteed out of range no matter K.
    bad_seed = 999999

    sched = []
    for i in range(N):
        if i < N // 2:
            sched.append(T_start + 5.0 + i)  # rises above T_start: invalid
        else:
            sched.append(T_min)

    out = [str(bad_seed)] + [str(x) for x in sched]
    print(" ".join(out))


if __name__ == "__main__":
    main()
