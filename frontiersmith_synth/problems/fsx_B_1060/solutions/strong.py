# TIER: strong
"""The insight: a support voxel placed now does not just hold up the ONE part voxel
directly above it -- because of the 45-degree overhang-support rule, one filled cell at
layer z-1 gives a foothold to up to 5 cells at layer z (itself plus its 4 side neighbours).
So the future part voxels a layer will need should decide TODAY's support placement, not
the other way around.

We therefore plan top-down (the shape is fully known before we ever have to commit): start
from the requirement at the highest occupied layer (which is exactly its target voxels),
and walk down layer by layer. At each step we already know the requirement -- the set of
cells that MUST be filled at layer z -- and we choose a minimal-cost set of cells at layer
z-1 whose 45-degree "shadow" covers that requirement, using a greedy weighted set cover
(prefer candidates that cover more requirement per unit cost, and PREFER untrapped
candidates -- columns with no future part voxel above them -- when several candidates cover
equally well). Target voxels at z-1 are free (they must be printed anyway) and are counted
first. The resulting filled(z-1) becomes the requirement for the next layer down.

This is exactly the family's innovation hook: support voxels are placed for the FUTURE part
voxels they will enable, because the print is monotone (once a layer is built its overhangs
can never be revisited)."""
import sys

NBR = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def main():
    data = sys.stdin.read().split()
    idx = 0
    Lx = int(data[idx]); Ly = int(data[idx + 1]); Lz = int(data[idx + 2]); idx += 3
    T = int(data[idx]); idx += 1
    Tz = {}  # z -> set of (x,y)
    Tset = set()
    for _ in range(T):
        x = int(data[idx]); y = int(data[idx + 1]); z = int(data[idx + 2]); idx += 3
        Tset.add((x, y, z))
        Tz.setdefault(z, set()).add((x, y))

    col_maxz = {}
    for (x, y, z) in Tset:
        k = (x, y)
        if k not in col_maxz or z > col_maxz[k]:
            col_maxz[k] = z

    def cost_at(x, y, z):
        return 3 if col_maxz.get((x, y), -1) > z else 1

    def in_bounds(x, y):
        return 0 <= x < Lx and 0 <= y < Ly

    zmax = max(Tz.keys())
    filled_xy = {zmax: set(Tz.get(zmax, set()))}  # layer -> set of (x,y) filled there

    for z in range(zmax, 0, -1):
        need = set(filled_xy[z]) | Tz.get(z, set())
        filled_xy[z] = need

        forced = set(Tz.get(z - 1, set()))  # target cells at z-1: free, must be printed anyway
        covered = set()
        for (cx, cy) in forced:
            for (dx, dy) in NBR:
                covered.add((cx + dx, cy + dy))
        remaining = need - covered

        chosen = set(forced)
        while remaining:
            # candidate cells = neighbourhood of the still-uncovered requirement
            candidates = set()
            for (rx, ry) in remaining:
                for (dx, dy) in NBR:
                    cx, cy = rx + dx, ry + dy
                    if in_bounds(cx, cy):
                        candidates.add((cx, cy))
            best = None
            best_key = None
            for cand in candidates:
                cx, cy = cand
                cov = set()
                for (dx, dy) in NBR:
                    n = (cx + dx, cy + dy)
                    if n in remaining:
                        cov.add(n)
                if not cov:
                    continue
                c = cost_at(cx, cy, z - 1)
                gain_ratio = len(cov) / c
                # deterministic tie-break: highest gain/cost, then most cells covered,
                # then untrapped over trapped, then lexicographically smallest coordinate
                key = (gain_ratio, len(cov), -c, -cx, -cy)
                if best is None or key > best_key:
                    best = cand; best_key = key; best_cov = cov
            chosen.add(best)
            remaining -= best_cov

        filled_xy[z - 1] = chosen

    voxels = []
    for (x, y, z) in Tset:
        voxels.append((x, y, z, "P"))
    for z, cols in filled_xy.items():
        for (x, y) in cols:
            if (x, y, z) not in Tset:
                voxels.append((x, y, z, "S"))

    out = [str(len(voxels))]
    for (x, y, z, t) in voxels:
        out.append(f"{x} {y} {z} {t}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
