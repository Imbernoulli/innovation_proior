# TIER: greedy
# Halton (2,3) low-discrepancy sequence: reliably beats the diagonal baseline for any N,
# but is denser-clustered than a good rank-1 lattice.
import sys

t = sys.stdin.read().split()
N = int(t[0])


def halton(i, b):
    f = 1.0
    r = 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


print(N)
out = []
for k in range(N):
    x = halton(k + 1, 2)
    y = halton(k + 1, 3)
    out.append("%.12f %.12f" % (x, y))
sys.stdout.write("\n".join(out) + "\n")
