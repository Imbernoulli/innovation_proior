# TIER: trivial
"""Naive 'clear everything, then rebuild' mover. Every window it evacuates
ALL occupied stage cells (even props that did not need to move), fully
unloads both wings (placing needed props, parking the rest on spare
cells), then pushes the leftovers back. No precedence-awareness, no
cross-window lookahead: this is the checker's own internal baseline."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); S = int(next(it)); w = int(next(it))
    prec = [0] * (M + 1)
    for p in range(1, M + 1):
        prec[p] = int(next(it))
    lc = int(next(it))
    L0 = [int(next(it)) for _ in range(lc)]
    rc = int(next(it))
    R0 = [int(next(it)) for _ in range(rc)]
    scenes = []
    for _ in range(S):
        scenes.append([int(next(it)) for _ in range(N)])
    for _ in range(S - 1):
        next(it)  # budgets, unused by this naive strategy

    stage = scenes[0][:]
    wingL = L0[:]
    wingR = R0[:]

    out_lines = []
    for i in range(S - 1):
        target = scenes[i + 1]
        moves = []
        for c in range(N):
            if stage[c] != 0:
                pid = stage[c]
                wing = "L" if len(wingL) <= len(wingR) else "R"
                stk = wingL if wing == "L" else wingR
                moves.append(("PUSH", c, wing, pid))
                stk.append(pid)
                stage[c] = 0
        parking = [c for c in range(N) if target[c] == 0]
        pi = 0
        parked = []
        for wing, stk in (("L", wingL), ("R", wingR)):
            while stk:
                pid = stk.pop()
                dest = None
                for c in range(N):
                    if target[c] == pid and stage[c] != pid:
                        dest = c
                        break
                if dest is None:
                    dest = parking[pi]; pi += 1
                    parked.append((dest, pid))
                moves.append(("POP", wing, dest, pid))
                stage[dest] = pid
        for cell, pid in parked:
            wing = "L" if len(wingL) <= len(wingR) else "R"
            stk = wingL if wing == "L" else wingR
            moves.append(("PUSH", cell, wing, pid))
            stk.append(pid)
            stage[cell] = 0

        out_lines.append(str(len(moves)))
        for typ, a, b, pid in moves:
            out_lines.append("%s %s %s %d" % (typ, a, b, pid))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
