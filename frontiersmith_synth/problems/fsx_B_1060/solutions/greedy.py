# TIER: greedy
"""The obvious first idea: support every overhanging target voxel with a straight column
directly underneath it, in its OWN (x,y) column only. For every column that contains at
least one target voxel, fill everything from the plate (z=0) up to the column's tallest
target voxel. This ignores two things a stronger solver would exploit: (1) a single
support voxel supports up to 5 cells (itself + 4 neighbours) in the layer above, so
neighbouring columns can share one support pillar instead of each growing their own; and
(2) whether a support voxel ends up permanently trapped under printed part material -- this
solution never tries to route around that."""
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

    col_maxz = {}
    for (x, y, z) in Tset:
        k = (x, y)
        if k not in col_maxz or z > col_maxz[k]:
            col_maxz[k] = z

    voxels = []
    for (x, y, z) in Tset:
        voxels.append((x, y, z, "P"))
    for (x, y), zmax in col_maxz.items():
        for z in range(0, zmax + 1):
            if (x, y, z) not in Tset:
                voxels.append((x, y, z, "S"))

    out = [str(len(voxels))]
    for (x, y, z, t) in voxels:
        out.append(f"{x} {y} {z} {t}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
