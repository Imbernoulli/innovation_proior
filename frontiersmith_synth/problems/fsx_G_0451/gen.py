#!/usr/bin/env python3
"""gen.py <testId> -- print ONE GF(2^k) multiplier-synthesis instance to stdout.

testId 1..7 -> k = 3..9 (small -> large / harder). Each instance fixes a standard
irreducible degree-k modulus f(x) over GF(2) (as used in ECC / finite-field hardware).
Deterministic: the instance depends ONLY on testId. No randomness in scoring.
"""
import sys

# (k -> modulus coefficient bits m_0..m_k, with m_k == 1). All irreducible over GF(2).
MODULI = {
    3: [1, 1, 0, 1],                       # x^3 + x + 1
    4: [1, 1, 0, 0, 1],                    # x^4 + x + 1
    5: [1, 0, 1, 0, 0, 1],                 # x^5 + x^2 + 1
    6: [1, 1, 0, 0, 0, 0, 1],              # x^6 + x + 1
    7: [1, 1, 0, 0, 0, 0, 0, 1],           # x^7 + x + 1
    8: [1, 1, 0, 1, 1, 0, 0, 0, 1],        # x^8 + x^4 + x^3 + x + 1  (AES field)
    9: [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],     # x^9 + x + 1
}


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > 7:
        t = 7
    k = t + 2
    m = MODULI[k]
    out = [str(k), " ".join(str(x) for x in m)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
