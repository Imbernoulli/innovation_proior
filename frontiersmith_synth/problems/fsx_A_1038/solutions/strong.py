# TIER: strong
# The insight: the array response is literally a polynomial A(z) = sum_i x_i z^i
# evaluated at the N-th roots of unity (bearing bin j <-> z_j = e^{2pi i j/N}).
# Split the emitter budget into a ZEROS budget and a GAIN budget instead of
# tweaking phases for destructive interference:
#   - pick a divisor d of N (d <= K) that does not divide any protected bin but
#     divides every harbor bin. A "comb" of d equally-spaced emitters (any common
#     phase, any starting offset) is EXACTLY zero at every non-multiple-of-d bin
#     (a finite geometric series), so it hard-nulls all protected bearings for free
#     and only ever contributes at harbor bins -- this is root placement, not
#     cancellation search.
#   - spend the remaining L-1 = K/d - 1 combs (same d, different offsets/phases)
#     purely on GAIN: brute-force the offsets and per-comb phases to maximize the
#     worst-illuminated harbor, since each comb stays exactly zero at every
#     protected bin regardless of its phase or offset.
import sys, math, cmath, itertools
from math import comb as ncomb


def divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def intensity(N, P, chosen, j):
    s = 0j
    for (i, p) in chosen:
        s += cmath.exp(2j * math.pi * (p / P + i * j / N))
    return abs(s) ** 2


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it)); P = int(next(it))
    T = int(next(it))
    targets = [int(next(it)) for _ in range(T)]
    Q = int(next(it))
    protected = []
    for _ in range(Q):
        q = int(next(it)); next(it)
        protected.append(q)

    phase_factors = [cmath.exp(2j * math.pi * p / P) for p in range(P)]
    best_val = -1.0
    best_chosen = None

    for dd in divisors(N):
        if dd > K:
            continue
        L = K // dd
        g = N // dd
        if L < 1 or L > g or L > 5:
            continue
        if any(t % dd != 0 for t in targets):
            continue
        if any(q % dd == 0 for q in protected):
            continue
        if ncomb(g, L) * (P ** L) > 400000:
            continue

        # per-offset phase factor toward each harbor bin (comb magnitude dd baked in)
        of_tab = [[dd * cmath.exp(2j * math.pi * a * j / N) for j in targets] for a in range(g)]

        for offs in itertools.combinations(range(g), L):
            for phases in itertools.product(range(P), repeat=L):
                minval = None
                for ti in range(T):
                    s = 0j
                    for li, a in enumerate(offs):
                        s += phase_factors[phases[li]] * of_tab[a][ti]
                    v = s.real * s.real + s.imag * s.imag
                    if minval is None or v < minval:
                        minval = v
                if minval > best_val:
                    best_val = minval
                    chosen = []
                    for a, c in zip(offs, phases):
                        for m in range(dd):
                            chosen.append((a + m * g, c))
                    best_chosen = chosen

    if best_chosen is None:
        # should not happen on well-formed instances; fall back to a single emitter
        best_chosen = [(0, 0)]

    print(len(best_chosen))
    for (i, p) in best_chosen:
        print(i, p)


main()
