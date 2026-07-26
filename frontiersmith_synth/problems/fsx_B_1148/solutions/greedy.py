# TIER: greedy
"""Per-window mover: only evacuates cells that actually need to change
(unlike trivial's full clear), but plans ONE window at a time with no
regard for when an evicted prop will be needed again, and picks a wing
purely by remaining free capacity (ignores precedence order entirely).
This is the 'obvious' first approach: locally sensible, no lookahead."""
import sys


def retrieve(stage, wingL, wingR, target, N, moves):
    """Shared mechanical retrieval: pop tops that are immediately needed;
    when blocked, excavate the shallower wing; park unneeded excavated
    props on spare (target==0) cells; finally push leftovers back
    (capacity-only wing choice, matching this tier's style)."""
    target_cell_for = {}
    for c in range(N):
        if target[c] != 0:
            target_cell_for[target[c]] = c
    parking = [c for c in range(N) if target[c] == 0]
    pi = 0
    progress = True
    while progress:
        progress = False
        for wing, stk in (("L", wingL), ("R", wingR)):
            while stk and stk[-1] in target_cell_for and stage[target_cell_for[stk[-1]]] == 0:
                pid = stk.pop()
                c = target_cell_for[pid]
                moves.append(("POP", wing, c, pid))
                stage[c] = pid
                progress = True
        if progress:
            continue
        remaining_set = set(pid for pid, c in target_cell_for.items() if stage[c] != pid)
        if not remaining_set:
            break
        best = None
        for wing, stk in (("L", wingL), ("R", wingR)):
            depth = None
            for idx in range(len(stk) - 1, -1, -1):
                if stk[idx] in remaining_set:
                    depth = len(stk) - 1 - idx
                    break
            if depth is not None and (best is None or depth < best[0]):
                best = (depth, wing)
        if best is None:
            break
        wing = best[1]
        stk = wingL if wing == "L" else wingR
        top = stk.pop()
        dest = parking[pi]; pi += 1
        moves.append(("POP", wing, dest, top))
        stage[dest] = top
        progress = True
    for c in range(N):
        if target[c] == 0 and stage[c] != 0:
            pid = stage[c]
            wing = "L" if len(wingL) <= len(wingR) else "R"
            stk = wingL if wing == "L" else wingR
            moves.append(("PUSH", c, wing, pid))
            stk.append(pid)
            stage[c] = 0


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
        next(it)  # budgets unused: no lookahead planning here

    stage = scenes[0][:]
    wingL = L0[:]
    wingR = R0[:]

    out_lines = []
    for i in range(S - 1):
        target = scenes[i + 1]
        moves = []
        leaving_cells = [c for c in range(N) if stage[c] != 0 and stage[c] != target[c]]
        for c in leaving_cells:
            pid = stage[c]
            freeL = w - len(wingL)
            freeR = w - len(wingR)
            wing = "L" if freeL >= freeR else "R"
            stk = wingL if wing == "L" else wingR
            moves.append(("PUSH", c, wing, pid))
            stk.append(pid)
            stage[c] = 0
        retrieve(stage, wingL, wingR, target, N, moves)

        out_lines.append(str(len(moves)))
        for typ, a, b, pid in moves:
            out_lines.append("%s %s %s %d" % (typ, a, b, pid))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
