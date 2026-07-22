# TIER: strong
import sys
from collections import defaultdict

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    W = int(next(it)); H = int(next(it))
    A = int(next(it))
    anchors = [int(next(it)) for _ in range(A)]
    L = int(next(it))
    loads = []
    for _ in range(L):
        r = int(next(it)); c = int(next(it)); f = int(next(it))
        loads.append((r, c, f))
    M = int(next(it)); K = int(next(it))

    anchor_set = set((0, a) for a in anchors)
    tree = set(anchor_set)
    material = set()
    flux = defaultdict(int)

    # Insight #1 (topology emergence): grow a SHARED tree by always attaching
    # the currently-nearest unconnected mound to whatever has already been
    # built (Steiner-style), instead of wiring every mound to the seabed
    # independently. Nearby mounds fuse into one trunk instead of paying for
    # their own redundant conduit. Connections are always face-respecting
    # (move one axis at a time) -- the insight here is exploiting geometry,
    # never diagonal shortcuts.
    remaining = sorted(loads, key=lambda t: min(abs(t[0] - 0) + abs(t[1] - a) for a in anchors))
    for (r_l, c_l, f_l) in remaining:
        best = None; bd = None
        for (tr, tc) in tree:
            dd = abs(tr - r_l) + abs(tc - c_l)
            if bd is None or dd < bd or (dd == bd and (tr, tc) < best):
                bd = dd; best = (tr, tc)
        sr, sc = best
        path = []
        step = 1 if r_l >= sr else -1
        r = sr
        while r != r_l:
            r += step
            path.append((r, sc))
        step = 1 if c_l >= sc else -1
        c = sc
        while c != c_l:
            c += step
            path.append((r_l, c))
        for cell in path:
            if cell not in tree:
                material.add(cell)
                tree.add(cell)
            flux[cell] += f_l

    # Insight #2 (compliance-minimization + tapering): the trunk-sharing above
    # already frees up budget versus paying for every mound independently.
    # Spend what's left where it carries the most combined load first
    # (highest accumulated flux = the merged trunk near the anchors), and
    # within a tie prefer the cell closer to the anchor -- i.e. widen (taper)
    # the base of a branch before its tip, mirroring how a real branching
    # structure thickens toward the root under compliance minimization.
    leftover = M - len(material)
    if leftover > 0:
        order = sorted(material, key=lambda c: (-flux[c], c[1], c[0]))
        for cell in order:
            if leftover <= 0:
                break
            r, c = cell
            for dc in (1, -1):
                nc = c + dc
                if leftover <= 0:
                    break
                if 0 <= nc <= W - 1 and 1 <= r <= H - 1 and (r, nc) not in material and (r, nc) not in anchor_set:
                    material.add((r, nc))
                    leftover -= 1

    cells = sorted(material)
    print(len(cells))
    out = []
    for (r, c) in cells:
        out.append(f"{r} {c}")
    if out:
        print("\n".join(out))

main()
