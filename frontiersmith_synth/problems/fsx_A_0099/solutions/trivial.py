# TIER: trivial
# Baseline: all relays strung along the main diagonal of the cube. This reproduces the
# checker's internal reference construction, so it scores ~0.1.
import sys

def main():
    tok = sys.stdin.read().split()
    m = int(tok[0])
    out = []
    for i in range(m):
        c = (i + 0.5) / m
        out.append("%.10f %.10f %.10f" % (c, c, c))
    sys.stdout.write("\n".join(out) + "\n")

main()
