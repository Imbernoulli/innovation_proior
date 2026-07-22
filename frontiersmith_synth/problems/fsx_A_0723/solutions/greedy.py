# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); base = int(next(it))
    K = int(next(it))
    counter_at = {}
    for _ in range(K):
        r = int(next(it)); c = int(next(it)); d = int(next(it)); w = int(next(it))
        scope = next(it)
        counter_at[(r, c)] = (d, w, scope)
    H = int(next(it))
    hint_map = {}
    for _ in range(H):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        hint_map[(r, c)] = v

    def partner(r):
        return (r + R // 2) % R

    def scope_cells(r, scope):
        if scope == "ROW":
            return [(r, cc) for cc in range(C)]
        rp = partner(r)
        return [(r, cc) for cc in range(C)] + [(rp, cc) for cc in range(C)]

    # "obvious" recipe: one left-to-right pass, row-major, filling each counter cell
    # by literally counting the target digit *in the grid as currently written* --
    # i.e. "write counts, recount, rewrite" done once per cell, textbook style. Free
    # non-counter cells are left at 0 (never revisited). Because most of the scope
    # hasn't been visited yet when an early cell is filled (unvisited cells read as
    # their still-default 0), and PAIR scope drags in a whole partner row that may not
    # be touched until much later, the recounted digit is systematically stale --
    # exactly the "recount, rewrite" trap: no global fixpoint check, no going back.
    grid = [[0] * C for _ in range(R)]
    for (r, c), v in hint_map.items():
        grid[r][c] = v

    for r in range(R):
        for c in range(C):
            if (r, c) in hint_map:
                continue
            spec = counter_at.get((r, c))
            if spec is None:
                continue  # filler cell: left at the default 0
            d, w, scope = spec
            cells = scope_cells(r, scope)
            actual = 0
            for (rr, cc) in cells:
                if grid[rr][cc] == d:
                    actual += 1
            grid[r][c] = actual if actual < base else base - 1

    out_lines = [" ".join(map(str, row)) for row in grid]
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
