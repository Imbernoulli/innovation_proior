# TIER: invalid
"""Deliberately infeasible: prints 24 prices far outside the allowed
[P_MIN, P_MAX] band, which the checker must reject with Ratio: 0.0."""
T = 24


def main():
    print(" ".join("999999.0" for _ in range(T)))


if __name__ == "__main__":
    main()
