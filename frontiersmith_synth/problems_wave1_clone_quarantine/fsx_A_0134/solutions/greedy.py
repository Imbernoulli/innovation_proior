# TIER: greedy
# Greedy strategy: spread sensors onto a *rounder* elliptical ring (semi-axis
# 0.25 in y) so triples are fatter than the flat baseline arc.
import sys, math

def main():
    n = int(sys.stdin.read().split()[0])
    b = 0.25
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        out.append("%.15g %.15g" % (0.5 + 0.5 * math.cos(t), 0.5 + b * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
