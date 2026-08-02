# TIER: strong
"""The insight: the winter branch never approaches the inverter's ceiling,
so no amount of curve-fitting on it can locate the ceiling's exact level --
but that does not mean no ceiling exists. This solution (1) fits the
irradiance-to-power slope through the physically correct temperature-scaled
feature X = G*(1 - 0.004*(T-25)) (a through-origin least-squares fit, since
power is ~0 at zero irradiance), which is what genuinely explains the
winter data, and (2) still emits a HARD ceiling, sized from the *stated*
plausible clip-fraction range (58%-88% of nameplate) at its midpoint, even
though this array's own exact fraction was never observed. Reasoning about
the existence and rough level of an unseen bound beats pretending it is not
there."""
import sys

CAP_FRAC = 0.73  # midpoint of the stated [0.58, 0.88] clip-fraction range


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = data[3:]
    num = den = 0.0
    for i in range(n):
        G = float(rows[3 * i])
        T = float(rows[3 * i + 1])
        P = float(rows[3 * i + 2])
        X = G * (1.0 - 0.004 * (T - 25.0))
        num += X * P
        den += X * X
    a = num / den if den > 1e-9 else 0.0
    print("min(%.8f * N, %.8f * (G * (1 - 0.004 * (T - 25))))" % (CAP_FRAC, a))


if __name__ == "__main__":
    main()
