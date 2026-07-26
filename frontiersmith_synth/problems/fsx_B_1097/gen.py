#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the subsequence-antichain layering problem.
Deterministic: the instance is a pure function of testId (no RNG at all)."""
import sys


def make_case(t):
    a = 3 if t <= 2 else 4
    Lmax = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6, 7: 6, 8: 7, 9: 7, 10: 8}[t]
    # tuned constants: peak length gets weight Wp on a cap deliberately << a**Lmax, while
    # the "middle" lengths get moderate weight/cap so a wiser mix beats one committed length.
    W1_base, Wo_base, Wp_mult, cap_mid_base, cap_peak_base = 8, 10, 1.8, 8, 6
    W1 = W1_base + t
    Wo = Wo_base + (t % 3)
    Wp = int(Wp_mult * W1)
    weight = [W1] + [Wo] * (Lmax - 2) + [Wp]
    cap_mid = cap_mid_base + (t % 3)
    cap_peak = cap_peak_base + (t % 3)
    cap = [a] + [cap_mid] * (Lmax - 2) + [cap_peak]
    T = sum(cap)
    return a, Lmax, T, weight, cap


def main():
    t = int(sys.argv[1])
    a, Lmax, T, weight, cap = make_case(t)
    print(a, Lmax, T)
    print(" ".join(map(str, weight)))
    print(" ".join(map(str, cap)))


if __name__ == "__main__":
    main()
