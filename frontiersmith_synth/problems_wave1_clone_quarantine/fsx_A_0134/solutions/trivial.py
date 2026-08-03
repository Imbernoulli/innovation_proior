# TIER: trivial
# Baseline strategy: drop every sensor onto a single flat elliptical arc
# (semi-axis 0.10 in y). This reproduces the checker's internal baseline.
import sys, math

def main():
    n = int(sys.stdin.read().split()[0])
    b = 0.10
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        out.append("%.15g %.15g" % (0.5 + 0.5 * math.cos(t), 0.5 + b * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
