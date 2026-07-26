# TIER: strong
"""Lookahead prop-lifetime planner. Treats each wing as a register: when a
window must evacuate several props at once, it pushes them in DESCENDING
order of 'next time this prop is needed on stage' (over the WHOLE remaining
scene sequence, not just this window) -- so a prop needed again soon ends
up near the top of its wing (cheap to reload) instead of buried under
props that will not be needed for a long time. Wing choice additionally
prefers whichever wing keeps the push precedence-clean (incoming
precedence >= current top's precedence), which a per-window-only mover has
no reason to check. Retrieval mechanics themselves are the same simple
excavate-when-blocked routine as the greedy tier -- the entire edge comes
from planning WHEN to spill and WHERE, exploiting both the wing-buffer
capacity and the precedence-swap rule together across the whole timeline."""
import sys


def retrieve(stage, wingL, wingR, target, N, moves, prec, w, next_use_fn=None):
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
    leftover_cells = [c for c in range(N) if target[c] == 0 and stage[c] != 0]
    if next_use_fn is not None:
        leftover_cells.sort(key=lambda c: -next_use_fn(stage[c]))
    for c in leftover_cells:
        pid = stage[c]
        wing = choose_wing(wingL, wingR, pid, prec, w)
        stk = wingL if wing == "L" else wingR
        moves.append(("PUSH", c, wing, pid))
        stk.append(pid)
        stage[c] = 0


def choose_wing(wingL, wingR, pid, prec, w):
    cands = []
    for name, stk in (("L", wingL), ("R", wingR)):
        if len(stk) < w:
            viol = 0 if (not stk or prec[pid] >= prec[stk[-1]]) else 1
            cands.append((viol, -(w - len(stk)), name))
    cands.sort()
    return cands[0][2]


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
        next(it)  # budgets: this tier stays efficient enough to rarely need them

    stage = scenes[0][:]
    wingL = L0[:]
    wingR = R0[:]

    def next_use(pid, i):
        # i indexes the window being built (0-based; connects scenes[i] -> scenes[i+1]).
        if pid in scenes[i + 1]:
            return i + 1
        for s in range(i + 2, S):
            if pid in scenes[s]:
                return s
        return 10 ** 9

    out_lines = []
    for i in range(S - 1):
        target = scenes[i + 1]
        moves = []
        leaving_cells = [c for c in range(N) if stage[c] != 0 and stage[c] != target[c]]
        # descending next-use first (pushed deep); soonest-needed last (ends up on top)
        order = sorted(leaving_cells, key=lambda c: -next_use(stage[c], i))
        for c in order:
            pid = stage[c]
            wing = choose_wing(wingL, wingR, pid, prec, w)
            stk = wingL if wing == "L" else wingR
            moves.append(("PUSH", c, wing, pid))
            stk.append(pid)
            stage[c] = 0
        retrieve(stage, wingL, wingR, target, N, moves, prec, w,
                 next_use_fn=lambda pid: next_use(pid, i))

        out_lines.append(str(len(moves)))
        for typ, a, b, pid in moves:
            out_lines.append("%s %s %s %d" % (typ, a, b, pid))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
