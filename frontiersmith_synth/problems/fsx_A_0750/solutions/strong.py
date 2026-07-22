# TIER: strong
"""The insight: separate WHERE a tool is used from WHEN it appears in the program.

Process catalog sizes from largest to smallest. For the CURRENT size s, repeatedly search
the whole grid (not just the raster frontier) for the valid s x s placement that covers the
most still-uncovered pocket cells, place it, and repeat until no s-sized placement can add
any new coverage; only then drop to the next smaller size. This correctly discovers each
chamber's matching tool radius wherever it lies in the grid (multi-radius-coverage), and
because it exhausts one size completely before touching the next, every stamp of size s is
already contiguous in program order for free -- the program only pays for a tool change
when it moves to a genuinely new size class, not every time geometry happens to change
locally. That is the fixed-changeover-cost trade the naive raster scan misses."""
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

    for s in sizes_desc:
        while True:
            # coverage prefix sum over currently-uncovered pocket cells
            cpre = [[0] * (W + 1) for _ in range(H + 1)]
            for r in range(H):
                rowsum = 0
                crow = covered[r]
                grow = grid[r]
                for c in range(W):
                    rowsum += 1 if (grow[c] == '#' and not crow[c]) else 0
                    cpre[r + 1][c + 1] = cpre[r][c + 1] + rowsum

            def gain(r, c, s=s, cpre=cpre):
                return cpre[r + s][c + s] - cpre[r][c + s] - cpre[r + s][c] + cpre[r][c]

            best = None
            best_gain = 0
            for r in range(H - s + 1):
                for c in range(W - s + 1):
                    if box_sum(r, c, s) != s * s:
                        continue
                    g = gain(r, c)
                    if g > best_gain:
                        best_gain = g
                        best = (r, c)
            if best is None:
                break
            r, c = best
            for rr in range(r, r + s):
                crow = covered[rr]
                for cc in range(c, c + s):
                    crow[cc] = True
            program.append((s, r, c))

    out = [str(len(program))]
    for (s, r, c) in program:
        out.append(f"{s} {r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
