# TIER: invalid
"""Prints ONLY the target voxels, with zero support material. This is exactly the trap the
statement warns about: it fails feasibility everywhere the shape overhangs empty space,
since a printed layer with an unsupported overhang can never be fixed after the fact."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    Lx = int(data[idx]); Ly = int(data[idx + 1]); Lz = int(data[idx + 2]); idx += 3
    T = int(data[idx]); idx += 1
    rows = []
    for _ in range(T):
        x = int(data[idx]); y = int(data[idx + 1]); z = int(data[idx + 2]); idx += 3
        rows.append((x, y, z))

    out = [str(len(rows))]
    for (x, y, z) in rows:
        out.append(f"{x} {y} {z} P")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
