# TIER: greedy
"""Textbook approach: keep the even split, but pick each layer's hash count
via the classic optimal-false-positive-rate formula k = round((m/n) ln2),
clipped to [1, kmax]. This is what an average strong coder writes first --
it optimizes the *aggregate* false-positive rate as if every query and every
layer were interchangeable. It ignores layer-cost asymmetry and the fact
that a handful of known hot keys dominate real traffic."""
import sys
import math

L = 4


def main():
    data = sys.stdin.read().split()
    ptr = 0
    ptr += 1  # testId
    n = int(data[ptr]); ptr += 1
    universe = int(data[ptr]); ptr += 1
    ll = int(data[ptr]); ptr += 1
    kmax = int(data[ptr]); ptr += 1
    budget = int(data[ptr]); ptr += 1
    assert ll == L

    ref_m = budget // L
    m_list = [ref_m] * (L - 1) + [budget - ref_m * (L - 1)]

    out = []
    for m in m_list:
        k = round((m / n) * math.log(2))
        k = max(1, min(kmax, k))
        out.append(f"{m} {k}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
