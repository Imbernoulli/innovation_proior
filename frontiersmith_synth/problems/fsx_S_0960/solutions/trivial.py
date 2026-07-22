# TIER: trivial
"""Never discard. Scan the grid row-major from (0,0) for the FIRST legal spot
(y ascending, then x ascending). Never tries rotation. No lookahead at all --
just the dumbest legal thing a program can do."""
import sys, json


def legal(grid, W, H, k, x, y, dw, dh):
    if x < 0 or y < 0 or dw <= 0 or dh <= 0 or x + dw > W or y + dh > H:
        return False
    kx0 = max(0, x - k); kx1 = min(W, x + dw + k)
    ky0 = max(0, y - k); ky1 = min(H, y + dh + k)
    for yy in range(ky0, ky1):
        row = grid[yy]
        for xx in range(kx0, kx1):
            if row[xx]:
                return False
    return True


def commit(grid, W, H, k, x, y, dw, dh):
    kx0 = max(0, x - k); kx1 = min(W, x + dw + k)
    ky0 = max(0, y - k); ky1 = min(H, y + dh + k)
    for yy in range(ky0, ky1):
        row = grid[yy]
        for xx in range(kx0, kx1):
            row[xx] = True


def main():
    inst = json.load(sys.stdin)
    W, H, k = inst["W"], inst["H"], inst["k"]
    grid = [[False] * W for _ in range(H)]
    out = []
    for it in inst["items"]:
        w, h = it["w"], it["h"]
        placed = None
        for y in range(0, H - h + 1):
            for x in range(0, W - w + 1):
                if legal(grid, W, H, k, x, y, w, h):
                    placed = (x, y)
                    break
            if placed:
                break
        if placed:
            commit(grid, W, H, k, placed[0], placed[1], w, h)
            out.append({"action": "place", "x": placed[0], "y": placed[1], "rot": 0})
        else:
            out.append({"action": "discard"})
    print(json.dumps({"decisions": out}))


if __name__ == "__main__":
    main()
