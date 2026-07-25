import sys
import random

# Random SMALL-case generator: python3 gen.py <seed>
# Small grids with a mix of open cells and boulders, a blast budget K, and
# distinct S / T. Tuned so the obstacle-elimination structure shows up:
# sometimes the only short routes go through boulders.

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    R = rng.randint(1, 6)
    C = rng.randint(1, 6)
    while R * C < 2:                 # need two distinct cells for S and T
        R = rng.randint(1, 6)
        C = rng.randint(1, 6)

    wall_p = rng.choice([0.0, 0.15, 0.3, 0.45, 0.6])

    grid = [['.' for _ in range(C)] for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if rng.random() < wall_p:
                grid[r][c] = '#'

    cells = [(r, c) for r in range(R) for c in range(C)]
    rng.shuffle(cells)
    sr, sc = cells[0]
    tr, tc = cells[1]
    grid[sr][sc] = 'S'
    grid[tr][tc] = 'T'

    # blast budget: 0..(a few). Keep small so layered state space stays tiny.
    K = rng.randint(0, min(R * C, 6))

    out = [f"{R} {C} {K}"]
    for r in range(R):
        out.append("".join(grid[r]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
