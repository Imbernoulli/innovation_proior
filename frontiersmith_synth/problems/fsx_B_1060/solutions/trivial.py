# TIER: trivial
"""Trivial baseline: print a SOLID bounding box. Take the axis-aligned (x,y) bounding box
of the target and the target's max height, and fill every cell in that box (from the plate
up to the top) as support unless it's already a target voxel. Always feasible (a solid box
trivially satisfies "same column below" support), but hugely wasteful -- this literally
reproduces the checker's own reference baseline."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    Lx = int(data[idx]); Ly = int(data[idx + 1]); Lz = int(data[idx + 2]); idx += 3
    T = int(data[idx]); idx += 1
    Tset = set()
    for _ in range(T):
        x = int(data[idx]); y = int(data[idx + 1]); z = int(data[idx + 2]); idx += 3
        Tset.add((x, y, z))

    xs = [c[0] for c in Tset]; ys = [c[1] for c in Tset]; zs = [c[2] for c in Tset]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    zmax = max(zs)

    voxels = []
    for (x, y, z) in Tset:
        voxels.append((x, y, z, "P"))
    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            for z in range(0, zmax + 1):
                if (x, y, z) not in Tset:
                    voxels.append((x, y, z, "S"))

    out = [str(len(voxels))]
    for (x, y, z, t) in voxels:
        out.append(f"{x} {y} {z} {t}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
