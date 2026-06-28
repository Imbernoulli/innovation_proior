import sys
from math import gcd

def main():
    data = sys.stdin.read().split()
    idx = 0
    k = int(data[idx]); idx += 1
    cong = []
    for _ in range(k):
        ri = int(data[idx]); mi = int(data[idx + 1]); idx += 2
        cong.append((ri % mi, mi))

    # lcm of all moduli (Python bignum: exact)
    L = 1
    for _, mi in cong:
        L = L // gcd(L, mi) * mi

    # Brute: scan x in [0, L) and return the first x satisfying every congruence.
    # Intended only for small L (the generator keeps L small).
    for x in range(L):
        good = True
        for ri, mi in cong:
            if x % mi != ri:
                good = False
                break
        if good:
            print(x)
            return
    print(-1)

main()
