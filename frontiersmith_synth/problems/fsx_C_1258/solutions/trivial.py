# TIER: trivial
"""Split the bit budget evenly across the 4 layers, k=1 everywhere.
Reproduces the checker's own reference construction (~0.1)."""
import sys

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
    # we don't need the rest of the instance at all for this tier
    assert ll == L

    ref_m = budget // L
    layers = [(ref_m, 1) for _ in range(L - 1)]
    layers.append((budget - ref_m * (L - 1), 1))

    out = []
    for (m, k) in layers:
        out.append(f"{m} {k}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
