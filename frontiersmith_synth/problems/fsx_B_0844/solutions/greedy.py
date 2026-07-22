# TIER: greedy
"""Greedy / textbook approach: recover the modulus via gcd of the determinant-style
combinations z_k = y_{k+1}^2 - y_{k+2}*y_k (this part is correct and free), then solve
for the multiplier/increment with ONE global modular equation over the FULL recovered
modulus, and fast-forward with a single affine binary-exponentiation. It never checks
that the gcd it took IS m (rather than a multiple of m), and it never decomposes by
prime factor -- so it silently trusts a corrupted modulus whenever the trap is present."""
import sys
import math


def fast_forward(a, c, mod, x0, k):
    e = k - 1
    res_a, res_c = 1, 0
    cur_a, cur_c = a % mod, c % mod
    while e > 0:
        if e & 1:
            res_a, res_c = (cur_a * res_a) % mod, (cur_a * res_c + cur_c) % mod
        cur_a, cur_c = (cur_a * cur_a) % mod, (cur_a * cur_c + cur_c) % mod
        e >>= 1
    return (res_a * x0 + res_c) % mod


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].strip())
    xs = list(map(int, data[1].split()))
    q = int(data[2].strip())
    ks = list(map(int, data[3].split()))

    ys = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    zs = [ys[i + 1] ** 2 - ys[i + 2] * ys[i] for i in range(len(ys) - 2)]
    g = 0
    for z in zs:
        g = math.gcd(g, abs(z))
    m_guess = g if g > 0 else 1  # trusted blindly as "the" modulus

    a = c = None
    if m_guess > 1:
        for i in range(min(60, len(ys) - 1)):
            yi = ys[i] % m_guess
            if yi != 0 and math.gcd(yi, m_guess) == 1:
                a = (ys[i + 1] * pow(yi, -1, m_guess)) % m_guess
                c = (xs[i + 1] - a * xs[i]) % m_guess
                break

    if a is None:
        # every consecutive difference is non-invertible mod the recovered modulus:
        # give up and repeat the last logged draw.
        print(" ".join(str(xs[-1]) for _ in ks))
        return

    preds = [fast_forward(a, c, m_guess, xs[0], k) for k in ks]
    print(" ".join(map(str, preds)))


if __name__ == "__main__":
    main()
