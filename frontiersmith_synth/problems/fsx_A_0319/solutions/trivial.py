# TIER: trivial
# Pile all probes on the main diagonal.  This is exactly the checker's baseline
# construction, so its star discrepancy equals B and the ratio is ~0.1.
import sys


def main():
    t = sys.stdin.read().split()
    d = int(t[0]); m = int(t[1])
    out = []
    for i in range(m):
        v = (i + 0.5) / m
        out.append("%.10f %.10f" % (v, v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
