import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    H = random.randint(1, 7)
    W = random.randint(1, 7)
    # Random L, R within small range; allow L=0 to probe the station-inclusion boundary.
    R = random.randint(0, 6)
    L = random.randint(0, R)

    cells = []
    for i in range(H):
        row = []
        for j in range(W):
            r = random.random()
            if r < 0.25:
                row.append('#')
            elif r < 0.40:
                row.append('S')
            else:
                row.append('.')
        cells.append(row)

    # Ensure at least one station exists; if none, place one on a random open-ish cell.
    has_station = any('S' in row for row in cells)
    if not has_station:
        i = random.randint(0, H-1)
        j = random.randint(0, W-1)
        cells[i][j] = 'S'

    print(H, W)
    print(L, R)
    for row in cells:
        print(''.join(row))

main()
