# TIER: strong
"""Insight: the capacity-queue shape is L = L0 + g*O/(Cap-O). Over the calm
training window O barely moves, so L(O) itself looks almost flat/linear and a
quadratic-in-O curvature fit would be numerically hopeless. But apply the
RECIPROCAL transform x = 1/O, y = 1/(L-L0): the SAME relationship becomes
EXACTLY LINEAR,

    y = (Cap/g) * x - 1/g

(no approximation -- this is the exact algebraic inverse). Even small stable-
regime fluctuations in O trace out this line cleanly, so an ordinary linear
regression in (x, y) space pins down g = -1/intercept robustly, and the
resulting formula extrapolates correctly through the held-out shock where O
climbs well above anything seen in training."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    L0 = float(header[3])
    Os, Ls = [], []
    for i in range(1, n + 1):
        parts = data[i].split()
        Os.append(float(parts[1]))
        Ls.append(float(parts[2]))

    xs = [1.0 / o for o in Os]
    ys = [1.0 / max(l - L0, 1e-9) for l in Ls]

    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx

    g_hat = -1.0 / intercept if abs(intercept) > 1e-9 else 1.0
    if g_hat <= 0:
        g_hat = abs(g_hat) + 1e-6

    print("L0 + %.10g*O/(Cap-O)" % g_hat)


if __name__ == "__main__":
    main()
