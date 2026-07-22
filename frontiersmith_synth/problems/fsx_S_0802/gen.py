import sys, random

# gen.py <testId>  -- prints ONE palindromic-polynomial instance to stdout.
#
# Line 1: d               (even degree, 4 <= d <= 32)
# Line 2: a_0 a_1 ... a_d  (d+1 integer coefficients, guaranteed a_i == a_{d-i})
#
# Difficulty ladder (small -> large/adversarial): larger d makes Horner's linear
# multiplicative depth exponentially more expensive under the leveled cost, while
# a log-depth restructuring stays cheap -- every test is a "trap" case for the
# textbook minimal-mult-count evaluator.

DEGREES = {
    1: 4,   2: 6,   3: 8,   4: 10,  5: 12,
    6: 16,  7: 20,  8: 24,  9: 28,  10: 32,
}

def main():
    tid = int(sys.argv[1])
    d = DEGREES[tid]
    rng = random.Random(70705 + 1000 * tid)

    m = d // 2
    coeffs = [0] * (d + 1)
    # a_0 (= a_d) must be nonzero so the polynomial has exact degree d.
    coeffs[0] = rng.choice([v for v in range(-40, 41) if v != 0])
    for k in range(1, m + 1):
        coeffs[k] = rng.randint(-40, 40)
    for k in range(0, m + 1):
        coeffs[d - k] = coeffs[k]

    out = [str(d), " ".join(str(c) for c in coeffs)]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
