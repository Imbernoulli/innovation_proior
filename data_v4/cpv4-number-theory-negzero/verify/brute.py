import sys
from math import gcd

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1
    out = []
    for _ in range(q):
        l = int(data[idx]); idx += 1
        r = int(data[idx]); idx += 1
        # 1-indexed inclusive -> python slice
        g = 0
        for i in range(l - 1, r):
            g = gcd(g, abs(a[i]))   # gcd seeded with 0; abs strips sign; gcd(0,0)=0
        out.append(str(g))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
