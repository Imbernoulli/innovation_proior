# TIER: trivial
# Reserve every free cell in the single best time slot (row). A single row can never
# contain a conflict corner, so this is always feasible. Reproduces the checker baseline.
import sys


def read_instance():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); nb = int(next(it))
    blocked = set()
    for _ in range(nb):
        r = int(next(it)); c = int(next(it))
        blocked.add((r, c))
    return N, blocked


def main():
    N, blocked = read_instance()
    best_r, best_cells = 0, []
    for r in range(N):
        cells = [(r, c) for c in range(N) if (r, c) not in blocked]
        if len(cells) > len(best_cells):
            best_cells = cells
    out = [str(len(best_cells))]
    for (r, c) in best_cells:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


main()
