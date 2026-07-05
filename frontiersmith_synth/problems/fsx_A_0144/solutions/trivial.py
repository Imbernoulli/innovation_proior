# TIER: trivial
# Baseline strategy: drop every gauge station onto a single thin inscribed
# elliptical arc (y semi-axis 0.028). This reproduces the checker's internal
# baseline construction exactly, so it scores about 0.1.
import sys, math

CX = 1.0 / (2.0 + math.sqrt(2.0))
R0 = 0.28
B = 0.028

def main():
    n = int(sys.stdin.read().split()[0])
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        out.append("%.15g %.15g" % (CX + R0 * math.cos(t), CX + B * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
