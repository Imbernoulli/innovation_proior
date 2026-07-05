# TIER: strong
# Scaling law WITH an irreducible floor:  y = E + A * x**(-al).
# For a fixed exponent al, (E, A) are recovered by a linear least-squares fit of
# y against u = x**(-al); we grid-search al and keep the best train fit. The floor
# term makes extrapolation into the large-throughput regime accurate. Only the
# irreducible measurement scatter limits the held-out error -> leaves headroom.
import sys, math


def linfit(us, ys):
    n = len(us)
    su = sum(us); sy = sum(ys)
    suu = sum(v * v for v in us); suy = sum(a * b for a, b in zip(us, ys))
    d = n * suu - su * su
    A = (n * suy - su * sy) / d
    E = (sy - A * su) / n
    return A, E


def rmse(pred, ys):
    return math.sqrt(sum((p - y) ** 2 for p, y in zip(pred, ys)) / len(ys))


def main():
    toks = sys.stdin.read().split()
    m = int(toks[1])
    xs = []; ys = []
    idx = 2
    for _ in range(m):
        xs.append(float(toks[idx])); ys.append(float(toks[idx + 1]))
        idx += 2

    best = None
    ai = 1
    while ai < 200:
        al = 0.2 + ai * 0.01
        us = [x ** (-al) for x in xs]
        A, E = linfit(us, ys)
        pred = [E + A * u for u in us]
        f = rmse(pred, ys)
        if best is None or f < best[0]:
            best = (f, al, A, E)
        ai += 1

    _, al, A, E = best
    sys.stdout.write("%.8g + %.8g * x ** %.8g\n" % (E, A, -al))


if __name__ == "__main__":
    main()
