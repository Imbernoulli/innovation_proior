# TIER: greedy
# Greedy strategy: spread the stations onto a *rounder* inscribed ellipse
# (y semi-axis 0.07) so every triple is fatter than the thin baseline arc.
import sys, math

CX = 1.0 / (2.0 + math.sqrt(2.0))
R0 = 0.28
B = 0.07

def main():
    n = int(sys.stdin.read().split()[0])
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        out.append("%.15g %.15g" % (CX + R0 * math.cos(t), CX + B * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
