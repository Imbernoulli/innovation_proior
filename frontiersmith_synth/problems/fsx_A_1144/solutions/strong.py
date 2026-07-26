# TIER: strong
"""
The insight: treat the voxel set as a sparse B-ARY TREE (an octree
generalized to branching factor B) and CANONICALIZE nodes bottom-up (a la
BDD/SVDAG minimization) -- any two sub-cells with the SAME relative
occupancy pattern (up to translation, or up to a single-axis reflection)
are the SAME sub-assembly, built once and reused everywhere it recurs. This
directly builds a "shape addition chain": every op is either a brand-new
distinct sub-pattern, or a cheap reuse (translate / reflect) of an
already-built one. For a target that is genuinely self-similar with
recursion base b, the number of DISTINCT sub-patterns stays small even as V
grows like b^L, so total ops ~ O(L) = O(log_b V) instead of the O(V) that
placing (or even row/run-compressing) each voxel needs.

The algorithm does not know the generator's recursion base, depth, or
kept-offset table. It only looks at the raw voxel set, tries a handful of
plausible branching factors B (2..5), builds the canonicalized tree for
each, and keeps whichever build used the fewest ops -- exactly the kind of
"search over a small family of decompositions, keep the cheapest" a
compression-minded solver would do.
"""
import sys


def build_for_base(pts, B):
    minx = min(p[0] for p in pts)
    miny = min(p[1] for p in pts)
    minz = min(p[2] for p in pts)
    maxx = max(p[0] for p in pts)
    maxy = max(p[1] for p in pts)
    maxz = max(p[2] for p in pts)
    span = max(maxx - minx, maxy - miny, maxz - minz) + 1
    size = 1
    while size < span:
        size *= B

    shifted = [(x - minx, y - miny, z - minz) for (x, y, z) in pts]

    out = []

    def emit(line):
        out.append(line)
        return len(out) - 1

    cache = {}  # (size, pattern_frozenset) -> assembly idx (identity orientation)

    def reflect_pat(pat, sz, axis):
        if axis == 0:
            return frozenset((sz - 1 - x, y, z) for (x, y, z) in pat)
        elif axis == 1:
            return frozenset((x, sz - 1 - y, z) for (x, y, z) in pat)
        else:
            return frozenset((x, y, sz - 1 - z) for (x, y, z) in pat)

    def lookup(pat, sz):
        key = (sz, pat)
        if key in cache:
            return cache[key], None
        for axis in (0, 1, 2):
            rkey = (sz, reflect_pat(pat, sz, axis))
            if rkey in cache:
                return cache[rkey], axis
        return None, None

    def build(points, x0, y0, z0, sz):
        if not points:
            return None, frozenset()
        if sz == 1:
            pat = frozenset({(0, 0, 0)})
            hit, _ = lookup(pat, 1)
            if hit is not None:
                return hit, pat
            idx = emit("U")
            cache[(1, pat)] = idx
            return idx, pat

        step = sz // B
        buckets = {}
        for (x, y, z) in points:
            ox = min(B - 1, (x - x0) // step)
            oy = min(B - 1, (y - y0) // step)
            oz = min(B - 1, (z - z0) // step)
            buckets.setdefault((ox, oy, oz), []).append((x, y, z))

        children = []
        local_pat = set()
        for (ox, oy, oz), b in buckets.items():
            cidx, cpat = build(b, x0 + ox * step, y0 + oy * step, z0 + oz * step, step)
            if cidx is None:
                continue
            off = (ox * step, oy * step, oz * step)
            children.append((off, cidx))
            for (dx, dy, dz) in cpat:
                local_pat.add((dx + off[0], dy + off[1], dz + off[2]))
        local_pat = frozenset(local_pat)
        if not children:
            return None, frozenset()

        hit, axis = lookup(local_pat, sz)
        if hit is not None:
            if axis is None:
                return hit, local_pat
            ridx = emit("R %d %d" % (hit, axis))
            off3 = [0, 0, 0]
            off3[axis] = sz - 1
            ridx = emit("T %d %d %d %d" % (ridx, off3[0], off3[1], off3[2]))
            return ridx, local_pat

        if len(children) == 1 and children[0][0] == (0, 0, 0):
            idx = children[0][1]
        else:
            placed = []
            for (off, cidx) in children:
                if off == (0, 0, 0):
                    placed.append(cidx)
                else:
                    placed.append(emit("T %d %d %d %d" % (cidx, off[0], off[1], off[2])))
            level = placed
            while len(level) > 1:
                nxt = []
                i, n = 0, len(level)
                while i + 1 < n:
                    nxt.append(emit("M %d %d" % (level[i], level[i + 1])))
                    i += 2
                if i < n:
                    nxt.append(level[i])
                level = nxt
            idx = level[0]

        cache[(sz, local_pat)] = idx
        return idx, local_pat

    root_idx, root_pat = build(shifted, 0, 0, 0, size)
    if root_idx is None:
        return None
    if (minx, miny, minz) != (0, 0, 0):
        emit("T %d %d %d %d" % (root_idx, minx, miny, minz))
    return out


def main():
    sys.setrecursionlimit(10000)
    data = sys.stdin.buffer.read().split()
    v = int(data[0])
    if v == 0:
        return
    pts = [(int(data[1 + 3 * k]), int(data[2 + 3 * k]), int(data[3 + 3 * k])) for k in range(v)]

    best = None
    for B in (2, 3, 4, 5):
        prog = build_for_base(pts, B)
        if prog is not None and (best is None or len(prog) < len(best)):
            best = prog

    sys.stdout.write("\n".join(best) + "\n")


if __name__ == "__main__":
    main()
