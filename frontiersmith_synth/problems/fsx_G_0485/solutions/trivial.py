# TIER: trivial
# Disjoint GC-block partition: floor(n/w) blocks of w consecutive strong bases.
# Reproduces the checker's internal baseline B -> Ratio ~ 0.1.
import sys

def main():
    p = sys.stdin.read().split()
    n, w, d = int(p[0]), int(p[1]), int(p[2])
    k = n // w
    out = []
    for i in range(k):
        s = ['0'] * n
        for j in range(i * w, (i + 1) * w):
            s[j] = '1'
        out.append(''.join(s))
    sys.stdout.write('\n'.join(out) + ('\n' if out else ''))

if __name__ == "__main__":
    main()
