# TIER: greedy
"""Classic bottom-left-fill: never discard, try both rotations, place every
arriving order at the lowest-y-then-lowest-x legal spot on the WHOLE sheet.
This is the obvious textbook recipe for online rectangle packing -- and it has
no notion that a rare, value-dense order might arrive later needing room."""
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


def find_best(grid, W, H, k, w, h):
    best = None
    for rot in (0, 1):
        dw, dh = (w, h) if rot == 0 else (h, w)
        if dw > W or dh > H:
            continue
        found = None
        for y in range(0, H - dh + 1):
            for x in range(0, W - dw + 1):
                if legal(grid, W, H, k, x, y, dw, dh):
                    found = (y, x)
                    break
            if found:
                break
        if found:
            cand = (found[0], found[1], rot, dw, dh)
            if best is None or cand[:2] < best[:2]:
                best = cand
    return best


def main():
    inst = json.load(sys.stdin)
    W, H, k = inst["W"], inst["H"], inst["k"]
    grid = [[False] * W for _ in range(H)]
    out = []
    for it in inst["items"]:
        best = find_best(grid, W, H, k, it["w"], it["h"])
        if best:
            y, x, rot, dw, dh = best
            commit(grid, W, H, k, x, y, dw, dh)
            out.append({"action": "place", "x": x, "y": y, "rot": rot})
        else:
            out.append({"action": "discard"})
    print(json.dumps({"decisions": out}))


if __name__ == "__main__":
    main()
