# TIER: greedy
"""Tighter square grid with near-touching domes (radius ~ half the cell). Larger
domes than the baseline grid => higher sum of radii, but still square packing."""
import sys
import math

CX, CY = 0.5, 0.5


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    r0 = float(toks[1])
    G = int(math.ceil(math.sqrt(N))) + 1
    rg = 0.45 / G
    out = []
    for i in range(G):
        for j in range(G):
            if len(out) >= N:
                break
            x = (i + 0.5) / G
            y = (j + 0.5) / G
            if math.hypot(x - CX, y - CY) >= r0 + rg + 1e-9:
                out.append((x, y, rg))
        if len(out) >= N:
            break
    # Safety pad (should not trigger for the ladder): shrink onto a finer grid.
    while len(out) < N:
        G += 1
        rg = 0.45 / G
        out = []
        for i in range(G):
            for j in range(G):
                if len(out) >= N:
                    break
                x = (i + 0.5) / G
                y = (j + 0.5) / G
                if math.hypot(x - CX, y - CY) >= r0 + rg + 1e-9:
                    out.append((x, y, rg))
            if len(out) >= N:
                break
    for (x, y, r) in out:
        print("%.9f %.9f %.9f" % (x, y, r))


if __name__ == "__main__":
    main()
