# TIER: invalid
"""Emits a syntactically-valid expression that predicts a NEGATIVE corrosion
rate (physically infeasible) -- the checker must reject it with Ratio 0.0."""


def main():
    print("-1.0 * ( Cl + T + pH + tex )")


if __name__ == "__main__":
    main()
