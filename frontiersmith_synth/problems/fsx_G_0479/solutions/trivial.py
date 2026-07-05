# TIER: trivial
# Power-of-two guard band: {1,2,4,8,...}<=n. Trivially Sidon (unique binary sums).
# Reproduces the checker baseline exactly -> Ratio ~ 0.1.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); k = int(d[1])
    forb = set(int(x) for x in d[2:2 + k])
    out = []
    v = 1
    while v <= n:
        if v not in forb:
            out.append(v)
        v *= 2
    print(" ".join(map(str, out)))

if __name__ == "__main__":
    main()
