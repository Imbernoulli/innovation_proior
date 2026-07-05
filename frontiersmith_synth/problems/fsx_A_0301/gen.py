#!/usr/bin/env python3
# gen.py <testId> : print ONE instance (n + weights over F_3^n) to stdout.
# testId 1..8 = difficulty ladder over n; seed varies per testId so equal-n
# cases still get distinct weights (deterministic, seed depends ONLY on testId).
import sys

MASK = (1 << 64) - 1

def weight(i, s):
    # deterministic splitmix64-style hash -> integer weight in [1, 998002]
    z = (i * 0x9E3779B97F4A7C15 + s * 0xD1B54A32D192ED03 + 0x2545F4914F6CDD1D) & MASK
    z ^= z >> 30; z = (z * 0xBF58476D1CE4E5B9) & MASK
    z ^= z >> 27; z = (z * 0x94D049BB133111EB) & MASK
    z ^= z >> 31
    r = z % 1000
    return 1 + r * r

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ladder = [4, 4, 5, 5, 6, 6, 7, 7]
    n = ladder[(t - 1) % len(ladder)]
    seed = 12345 + 6791 * t
    N = 3 ** n
    out = [str(n)]
    W = [str(weight(i, seed)) for i in range(N)]
    # 40 weights per line
    for j in range(0, N, 40):
        out.append(" ".join(W[j:j + 40]))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
