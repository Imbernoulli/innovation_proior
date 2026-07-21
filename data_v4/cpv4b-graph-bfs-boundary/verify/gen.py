import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    R = rng.randint(1, 6)
    C = rng.randint(1, 6)

    # weights: open '.', wall '#', source '*'
    wall_p = rng.choice([0.0, 0.1, 0.25, 0.4])
    src_p = rng.choice([0.1, 0.2, 0.35])

    grid = []
    has_src = False
    for r in range(R):
        row = []
        for c in range(C):
            x = rng.random()
            if x < wall_p:
                row.append('#')
            elif x < wall_p + src_p:
                row.append('*')
                has_src = True
            else:
                row.append('.')
        grid.append(''.join(row))

    # Occasionally force no sources at all to test that corner.
    if has_src and rng.random() < 0.15:
        grid = [row.replace('*', '.') for row in grid]
    # Occasionally force at least one source.
    if not any('*' in row for row in grid) and rng.random() < 0.5:
        r = rng.randrange(R); c = rng.randrange(C)
        row = list(grid[r]); row[c] = '*'; grid[r] = ''.join(row)

    maxd = R + C  # an upper bound on any reachable distance
    # Choose a band that often straddles the interesting boundaries.
    L = rng.randint(0, maxd)
    U = rng.randint(L, maxd + 2)

    print(R, C, L, U)
    for row in grid:
        print(row)

main()
