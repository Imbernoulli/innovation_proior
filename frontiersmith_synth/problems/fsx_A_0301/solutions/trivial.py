# TIER: trivial
# Emit the binary sub-cube {0,1}^n (all coords in {0,1}) -- always line-free,
# and exactly the checker's internal baseline -> Ratio ~= 0.10.
import sys

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    N = 3 ** n
    out = []
    for i in range(N):
        x = i; ok = True
        for _ in range(n):
            if x % 3 == 2:
                ok = False; break
            x //= 3
        if ok:
            out.append(i)
    sys.stdout.write(str(len(out)) + "\n")
    sys.stdout.write(" ".join(map(str, out)) + "\n")

if __name__ == "__main__":
    main()
