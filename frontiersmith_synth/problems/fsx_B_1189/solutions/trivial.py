# TIER: trivial
"""Ignores every echo reading entirely. Outputs a generic room: W walls
whose image points sit at fixed radius R0=1.0 from the source, evenly
spread in angle. This is exactly the checker's own internal baseline
construction (see verify.py: baseline_image_points), so this solution's
score is calibrated to land near the 0.1 reference point."""
import sys
import math


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it)); K = int(next(it)); _tid = int(next(it))
    Sx = float(next(it)); Sy = float(next(it))

    R0 = 1.0
    OFFSET = 0.15
    d = R0 / 2.0

    out = [str(W)]
    for k in range(W):
        theta = 2 * math.pi * k / W + OFFSET
        nx, ny = math.cos(theta), math.sin(theta)
        px, py = Sx + d * nx, Sy + d * ny
        perp_x, perp_y = -ny, nx
        p1x, p1y = px + perp_x, py + perp_y
        p2x, p2y = px - perp_x, py - perp_y
        out.append("%.9f %.9f %.9f %.9f" % (p1x, p1y, p2x, p2y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
