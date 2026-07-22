# TIER: greedy
import sys

def diagonal_connect(src, dst):
    """'Obvious' shortest-looking connector: walk diagonally toward the target
    while both coordinates still differ (this LOOKS efficient -- it minimises
    step count via a Chebyshev-style shortcut), then finish axis-aligned. Any
    diagonal step only touches its predecessor at a CORNER, not a shared face,
    so whenever the nearest existing point isn't already row- or
    column-aligned with the new mound this silently plants an unconnected
    (corner-only) branch."""
    r, c = src
    tr, tc = dst
    path = []
    while r != tr and c != tc:
        r += 1 if tr > r else -1
        c += 1 if tc > c else -1
        path.append((r, c))
    while r != tr:
        r += 1 if tr > r else -1
        path.append((r, c))
    while c != tc:
        c += 1 if tc > c else -1
        path.append((r, c))
    return path

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
    material = set()
    tree = set(anchor_set)
    first_load_cells = []

    for i, (r_l, c_l, f_l) in enumerate(loads):
        # connect to whichever built point is nearest (reuse structure -- looks smart)
        best = None; bd = None
        for (tr, tc) in tree:
            dd = abs(tr - r_l) + abs(tc - c_l)
            if bd is None or dd < bd or (dd == bd and (tr, tc) < best):
                bd = dd; best = (tr, tc)
        path = diagonal_connect(best, (r_l, c_l))
        for cell in path:
            if cell not in tree and 1 <= cell[0] <= H - 1 and 0 <= cell[1] <= W - 1:
                material.add(cell)
                tree.add(cell)
                if i == 0:
                    first_load_cells.append(cell)

    # "solid is strong": reinforce whatever was built for the first mound with
    # extra parallel material until the budget runs out.
    leftover = M - len(material)
    if leftover > 0:
        for cell in list(first_load_cells):
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
