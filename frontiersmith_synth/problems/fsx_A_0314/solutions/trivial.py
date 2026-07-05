# TIER: trivial
# Ring layout: turbines equally spaced on the inscribed circle. This is exactly
# the checker's internal baseline, so it scores ~0.1.
import sys, math

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    cx, cy, r = 0.5, 0.5, 0.26
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        out.append("%.17g %.17g" % (cx + r * math.cos(t), cy + r * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
