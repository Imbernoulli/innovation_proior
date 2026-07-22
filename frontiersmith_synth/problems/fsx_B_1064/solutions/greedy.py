# TIER: greedy
# Obvious first approach: optimize EACH canton's own deposit against its OWN elasticity
# curve in isolation, exactly as if it were a single independent market. Never looks at
# the hauler network at all -- so it is blind to arbitrage corridors between regions.
import sys


def piecewise_rate(d, d0, d1, r1, d2, r2):
    if d <= d0:
        return 0
    if d <= d1:
        return (r1 * (d - d0)) // (d1 - d0)
    if d <= d2:
        return r1 + ((r2 - r1) * (d - d1)) // (d2 - d1)
    return r2


def region_value(d, pop, d0, d1, r1, d2, r2, V, Fpm):
    rate = piecewise_rate(d, d0, d1, r1, d2, r2)
    ret = (pop * rate) // 1_000_000
    payout = ret * d
    fcost = (payout * Fpm) // 1000
    return ret * V - payout - fcost


def standalone_optimum(pop, d0, d1, r1, d2, r2, V, Fpm, D_MAX):
    best_d, best_v = 0, None
    for d in range(0, D_MAX + 1):
        val = region_value(d, pop, d0, d1, r1, d2, r2, V, Fpm)
        if best_v is None or val > best_v:
            best_v, best_d = val, d
    return best_d


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    V = int(next(it))
    D_MAX = int(next(it))
    Fpm = int(next(it))
    deposits = []
    for _ in range(n):
        next(it)  # name
        pop = int(next(it))
        d0 = int(next(it))
        d1 = int(next(it))
        r1 = int(next(it))
        d2 = int(next(it))
        r2 = int(next(it))
        d_opt = standalone_optimum(pop, d0, d1, r1, d2, r2, V, Fpm, D_MAX)
        deposits.append(d_opt)
    # edges are read but never used -- greedy ignores the transport network
    print(" ".join(str(d) for d in deposits))


if __name__ == "__main__":
    main()
