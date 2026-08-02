# TIER: greedy
"""The obvious textbook move: the training cloud (O, L) looks almost linear (the
window is calm, so the congestion feedback is dormant), so fit an ordinary
least-squares straight line L = a + b*O and extrapolate it. No pole, no
curvature -- this is exactly the trap: it fits the calm window well and
underpredicts badly once the held-out shock pushes O well past the training
range."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    Os, Ls = [], []
    for i in range(1, n + 1):
        parts = data[i].split()
        Os.append(float(parts[1]))
        Ls.append(float(parts[2]))

    mO = sum(Os) / n
    mL = sum(Ls) / n
    sxx = sum((o - mO) ** 2 for o in Os)
    sxy = sum((o - mO) * (l - mL) for o, l in zip(Os, Ls))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = mL - b * mO

    print("%.10g + %.10g*O" % (a, b))


if __name__ == "__main__":
    main()
