# TIER: greedy
"""The obvious single-pass "biggest tool first" tool-path: scan the pocket in raster order
(row-major); at the first still-uncovered cell, try the largest catalog tool with its
top-left corner AT that cell, falling back to smaller sizes, then finally size 1. Emit
stamps in the order they are placed (the natural scan order).

This is a reasonable first attempt -- but because it never looks ahead, it commits to a
tool size purely from local fit at one raster position, and the program interleaves tool
sizes every time the scan crosses a chamber boundary or a checkerboard barrier. Each such
crossing costs a full tool-change fee, and this happens dozens of times per test."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    H, W, K, C = map(int, data[0].split())
    catalog = list(map(int, data[1].split()))
    grid = [data[2 + r] for r in range(H)]

    sizes_desc = sorted(catalog, reverse=True)

    pre = [[0] * (W + 1) for _ in range(H + 1)]
    for r in range(H):
        rowsum = 0
        for c in range(W):
            rowsum += 1 if grid[r][c] == '#' else 0
            pre[r + 1][c + 1] = pre[r][c + 1] + rowsum

    def box_sum(r, c, s):
        return pre[r + s][c + s] - pre[r][c + s] - pre[r + s][c] + pre[r][c]

    covered = [[False] * W for _ in range(H)]
    program = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != '#' or covered[r][c]:
                continue
            placed = False
            for s in sizes_desc:
                if r + s <= H and c + s <= W and box_sum(r, c, s) == s * s:
                    for rr in range(r, r + s):
                        crow = covered[rr]
                        for cc in range(c, c + s):
                            crow[cc] = True
                    program.append((s, r, c))
                    placed = True
                    break
            if not placed:
                # size 1 is always in the catalog and always fits a pocket cell
                covered[r][c] = True
                program.append((1, r, c))

    out = [str(len(program))]
    for (s, r, c) in program:
        out.append(f"{s} {r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
