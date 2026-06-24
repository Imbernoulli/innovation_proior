import sys
from math import gcd

def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        L = int(data[idx]); idx += 1
        R = int(data[idx]); idx += 1
        # Exhaustive: count x in [L, R] inclusive with gcd(x, m) == 1.
        cnt = 0
        for x in range(L, R + 1):
            if gcd(x, m) == 1:
                cnt += 1
        out.append(str(cnt))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
