# TIER: strong
# Algebraic construction: the multiplicative inverse over GF(2^n) with 0 -> 0
# (the AES-style S-box core).  Known to have very low linearity / high
# nonlinearity, far beyond what random search finds -- but for most n it is
# still not proven optimal, leaving headroom.
import sys

POLY = {2: 7, 3: 11, 4: 19, 5: 37, 6: 67, 7: 131, 8: 283,
        9: 529, 10: 1033, 11: 2053, 12: 4179}

def gfmul(a, b, red, n):
    p = 0
    top = 1 << (n - 1)
    mask = (1 << n) - 1
    for _ in range(n):
        if b & 1:
            p ^= a
        b >>= 1
        hi = a & top
        a = (a << 1) & mask
        if hi:
            a ^= red
    return p

def inverse_table(n):
    N = 1 << n
    red = POLY[n] ^ (1 << n)   # reduction polynomial minus its leading term
    inv = [0] * N
    # brute-force multiplicative inverse (N is small)
    for x in range(1, N):
        if inv[x]:
            continue
        for y in range(1, N):
            if gfmul(x, y, red, n) == 1:
                inv[x] = y
                inv[y] = x
                break
    return inv

def main():
    n = int(sys.stdin.read().split()[0])
    inv = inverse_table(n)
    sys.stdout.write("\n".join(str(v) for v in inv) + "\n")

if __name__ == "__main__":
    main()
