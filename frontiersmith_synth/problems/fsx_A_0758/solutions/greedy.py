# TIER: greedy
# The obvious recipe: repeatedly apply whichever single move (from the moves
# legal RIGHT NOW) most improves the objective immediately, stop when no move
# helps. It never considers "expand", because every expand rule in this
# family is individually cost-increasing -- so this recipe can never see the
# canyon collapses that only become available after an expand.
import sys


def main():
    toks = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = toks[idx[0]]
        idx[0] += 1
        return v

    n = int(nxt()); k = int(nxt()); r = int(nxt())
    s = [int(nxt()) for _ in range(n)]
    cost = [0] * (k + 1)
    for v in range(1, k + 1):
        cost[v] = int(nxt())
    me = int(nxt())
    expand = {}
    for _ in range(me):
        v = int(nxt()); x = int(nxt()); y = int(nxt())
        expand[v] = (x, y)
    mc = int(nxt())
    collapse = {}
    for _ in range(mc):
        x = int(nxt()); y = int(nxt()); z = int(nxt())
        collapse[(x, y)] = z

    cur = list(s)
    moves = []
    steps = 0

    while steps < r:
        best_delta = 0
        best_pos = -1
        for i in range(len(cur) - 1):
            x, y = cur[i], cur[i + 1]
            if (x, y) in collapse:
                z = collapse[(x, y)]
                delta = cost[z] - cost[x] - cost[y]
                if delta < best_delta:
                    best_delta = delta
                    best_pos = i
        if best_pos < 0:
            break
        i = best_pos
        x, y = cur[i], cur[i + 1]
        z = collapse[(x, y)]
        cur[i:i + 2] = [z]
        moves.append(("C", i + 1))
        steps += 1

    print(len(moves))
    out = []
    for op, pos in moves:
        out.append(f"{op} {pos}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
