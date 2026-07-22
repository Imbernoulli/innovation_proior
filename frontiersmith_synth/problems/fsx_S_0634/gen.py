import sys, math

# Named real constants used to build target ratios (all deterministic, no RNG needed for
# their values -- reproducibility comes from Python's fixed math library implementation).
CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "sqrt2": math.sqrt(2.0),
    "sqrt3": math.sqrt(3.0),
    "sqrt5": math.sqrt(5.0),
    "sqrt7": math.sqrt(7.0),
    "sqrt10": math.sqrt(10.0),
    "phi": (1.0 + math.sqrt(5.0)) / 2.0,
    "cbrt2": 2.0 ** (1.0 / 3.0),
    "ln2": math.log(2.0),
    "ln5": math.log(5.0),
    "inv_pi": 1.0 / math.pi,
}


def best_convergent(x, max_den):
    """Continued-fraction convergents of x; return the (h, k) pair with the largest
    denominator k <= max_den (the best rational approximation with bounded denominator)."""
    a0 = math.floor(x)
    h_prev2, h_prev1 = 1, a0
    k_prev2, k_prev1 = 0, 1
    best = (h_prev1, k_prev1)
    frac = x - a0
    for _ in range(60):
        if abs(frac) < 1e-15:
            break
        frac = 1.0 / frac
        a = math.floor(frac)
        if a <= 0:
            break
        h = a * h_prev1 + h_prev2
        k = a * k_prev1 + k_prev2
        if k > max_den or h <= 0:
            break
        best = (h, k)
        h_prev2, h_prev1 = h_prev1, h
        k_prev2, k_prev1 = k_prev1, k
        frac -= a
    return best


def target_frac(name, max_den):
    h, k = best_convergent(CONSTS[name], max_den)
    g = math.gcd(h, k)
    return h // g, k // g


def emit(G, Tmin, Tmax, entries):
    # entries: list of (const_name, max_den, lambda)
    lines = ["%d %d %d %d" % (G, Tmin, Tmax, len(entries))]
    for name, max_den, lam in entries:
        P, Q = target_frac(name, max_den)
        lines.append("%d %d %.6f" % (P, Q, lam))
    sys.stdout.write("\n".join(lines) + "\n")


def main():
    i = int(sys.argv[1])
    G = 3

    # difficulty ladder, testId 1..10. TRAP cases (marked) are engineered so that
    # sequential per-stage rounding (greedy) lands far from the jointly-optimal
    # combination (strong): either via a heavy cost weight it ignores entirely, or
    # via a very tight teeth range that makes its remainder unreachable.
    if i == 1:
        emit(G, 8, 15, [("sqrt2", 50, 0.02), ("phi", 50, 0.02)])
    elif i == 2:
        emit(G, 8, 17, [("sqrt3", 200, 0.05), ("ln2", 200, 0.05)])
    elif i == 3:  # TRAP: heavy cost weight, greedy ignores lambda entirely
        emit(G, 8, 18, [("pi", 2000, 0.6), ("e", 2000, 0.6), ("sqrt5", 2000, 0.6)])
    elif i == 4:
        emit(G, 10, 22, [("sqrt7", 1000, 0.1), ("cbrt2", 1000, 0.1), ("ln5", 1000, 0.1)])
    elif i == 5:  # TRAP: very tight teeth range, big-denominator target
        emit(G, 8, 13, [("pi", 500000, 0.01), ("sqrt10", 500000, 0.01)])
    elif i == 6:
        emit(G, 10, 26, [("sqrt2", 5000, 0.15), ("phi", 5000, 0.15),
                          ("sqrt3", 5000, 0.15), ("ln2", 5000, 0.15)])
    elif i == 7:  # TRAP: lambda alternates hard/soft within one case
        emit(G, 9, 20, [("pi", 100000, 0.5), ("e", 100000, 0.02),
                         ("sqrt5", 100000, 0.5), ("inv_pi", 100000, 0.02)])
    elif i == 8:
        emit(G, 12, 28, [("sqrt7", 20000, 0.08), ("cbrt2", 20000, 0.08),
                          ("sqrt10", 20000, 0.08), ("ln5", 20000, 0.08)])
    elif i == 9:
        emit(G, 14, 32, [("pi", 1000000, 0.05), ("e", 1000000, 0.05), ("sqrt2", 1000000, 0.05),
                          ("phi", 1000000, 0.05), ("sqrt3", 1000000, 0.05)])
    else:  # i == 10, TRAP: widest range + big K + mixed heavy/light lambda + huge denominators
        emit(G, 8, 27, [("pi", 2000000, 0.5), ("e", 2000000, 0.01), ("sqrt5", 2000000, 0.5),
                         ("sqrt10", 2000000, 0.01), ("phi", 2000000, 0.5), ("sqrt7", 2000000, 0.01)])


if __name__ == "__main__":
    main()
