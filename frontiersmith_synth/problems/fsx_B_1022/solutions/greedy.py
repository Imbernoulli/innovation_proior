# TIER: greedy
"""The obvious "competent coder" fix over trivial: re-sort the gates by column so
each boat's own route is shortest (no backtracking) -- straightforward single-agent
pathfinding. It still aims every boat at the exact same fixed mark (the center of
each gate's band) and launches the whole fleet at tick 0. It never reasons about
the fleet as a field: every boat still sails the identical "racing line" through
the identical cells at the identical times, so the wake keeps compounding on the
follow boats. This is the trap: efficient for one boat, ruinous for the fleet."""
import sys


def diag_moves(r, c, r2, c2):
    dr, dc = r2 - r, c2 - c
    steps = max(abs(dr), abs(dc))
    mv = []
    if steps == 0:
        return mv
    vsign = 1 if dr > 0 else -1
    hsign = 1 if dc > 0 else -1
    verr = herr = 0.0
    vstep, hstep = abs(dr) / steps, abs(dc) / steps
    rr, cc = r, c
    for _ in range(steps):
        verr += vstep
        herr += hstep
        moved = False
        if verr >= 0.5 and rr != r2:
            mv.append('D' if vsign > 0 else 'U'); rr += vsign; verr -= 1.0; moved = True
        if herr >= 0.5 and cc != c2:
            mv.append('R' if hsign > 0 else 'L'); cc += hsign; herr -= 1.0; moved = True
        if not moved:
            if rr != r2:
                mv.append('D' if vsign > 0 else 'U'); rr += vsign
            elif cc != c2:
                mv.append('R' if hsign > 0 else 'L'); cc += hsign
    while rr != r2:
        mv.append('D' if r2 > rr else 'U'); rr += 1 if r2 > rr else -1
    while cc != c2:
        mv.append('R' if c2 > cc else 'L'); cc += 1 if c2 > cc else -1
    return mv


def main():
    toks = sys.stdin.read().split()
    p = 0
    N, B, G, w, CAP, Smax, maxMoves = (int(toks[p + i]) for i in range(7))
    p += 7
    starts = [int(toks[p + i]) for i in range(B)]
    p += B
    gates = []
    for _ in range(G):
        c, lo, hi = int(toks[p]), int(toks[p + 1]), int(toks[p + 2])
        p += 3
        gates.append((c, lo, hi))

    gorder = sorted(gates, key=lambda g: g[0])    # shortest-path fix: re-sort by column

    out = []
    for i in range(B):
        r, c = starts[i], 0
        mv = []
        for (gc, lo, hi) in gorder:
            tr = (lo + hi) // 2                    # still the same fixed mark for everyone
            mv += diag_moves(r, c, tr, gc)
            r, c = tr, gc
        mv += diag_moves(r, c, r, N - 1)
        out.append(f"0 {''.join(mv)}")              # still launches the whole fleet at tick 0
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
