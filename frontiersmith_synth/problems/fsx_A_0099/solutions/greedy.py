# TIER: greedy
# Hammersley set in 3D: first coordinate = (i+0.5)/m, remaining two = radical inverses
# in bases 2 and 3. A single fixed low-discrepancy construction, no search.
import sys

def radinv(i, b):
    f = 1.0
    r = 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r

def main():
    tok = sys.stdin.read().split()
    m = int(tok[0])
    out = []
    for i in range(m):
        x = (i + 0.5) / m
        y = radinv(i, 2)
        z = radinv(i, 3)
        # nudge exact 0 off the boundary so the anchored box does not degenerate
        y = min(1.0 - 1e-9, max(1e-9, y))
        z = min(1.0 - 1e-9, max(1e-9, z))
        out.append("%.10f %.10f %.10f" % (x, y, z))
    sys.stdout.write("\n".join(out) + "\n")

main()
