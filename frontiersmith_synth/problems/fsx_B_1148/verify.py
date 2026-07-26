#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- checker for stage-blackout-scene-swap.

Simulates the participant's PUSH/POP move sequence against the instance,
validating feasibility strictly (Ratio: 0.0 on ANY violation), then scores
the objective (total ticks + exceeded-window penalty) against an internal
"clear-and-rebuild" baseline construction run through the SAME simulator.
"""
import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_input(path):
    toks = open(path).read().split()
    it = iter(toks)
    try:
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
        budgets = [int(next(it)) for _ in range(S - 1)]
    except StopIteration:
        raise ValueError("truncated input")
    return N, M, S, w, prec, L0, R0, scenes, budgets


PENALTY_MULT = 10  # P = PENALTY_MULT * N per exceeded window


def simulate(N, w, prec, L0, R0, scenes, budgets, windows_moves, strict):
    """Replay all windows. windows_moves[i] = list of (TYPE, a, b, pid).
    strict=True -> any structural violation raises ValueError(msg) (used for
    the untrusted participant output). strict=False is used only for the
    checker's own baseline, which is constructed to always be valid; any
    violation there indicates an internal bug, so we also raise (caller
    treats it as fatal, not a graceful Ratio:0)."""
    stage = scenes[0][:]
    wingL = L0[:]
    wingR = R0[:]
    P = PENALTY_MULT * N
    total_ticks = 0
    exceeded = 0
    S = len(scenes)
    if len(windows_moves) != S - 1:
        raise ValueError("expected %d windows, got %d" % (S - 1, len(windows_moves)))
    for i in range(S - 1):
        moves = windows_moves[i]
        ticks_i = 0
        for mv in moves:
            typ, a, b, pid = mv
            if pid < 1 or pid > len(prec) - 1:
                raise ValueError("bad prop id %r" % (pid,))
            if typ == "PUSH":
                src, wing = a, b
                if not (0 <= src < N):
                    raise ValueError("push src out of range")
                if wing not in ("L", "R"):
                    raise ValueError("bad wing %r" % (wing,))
                if stage[src] != pid:
                    raise ValueError("push: cell %d does not hold prop %d" % (src, pid))
                stk = wingL if wing == "L" else wingR
                if len(stk) >= w:
                    raise ValueError("push: wing %s is full" % wing)
                cost = 1 if (not stk or prec[pid] >= prec[stk[-1]]) else 2
                stk.append(pid)
                stage[src] = 0
                ticks_i += cost
            elif typ == "POP":
                wing, dest = a, b
                if wing not in ("L", "R"):
                    raise ValueError("bad wing %r" % (wing,))
                if not (0 <= dest < N):
                    raise ValueError("pop dest out of range")
                stk = wingL if wing == "L" else wingR
                if not stk:
                    raise ValueError("pop: wing %s is empty" % wing)
                if stk[-1] != pid:
                    raise ValueError("pop: top of wing %s is not prop %d" % (wing, pid))
                if stage[dest] != 0:
                    raise ValueError("pop: dest cell %d not empty" % dest)
                stk.pop()
                stage[dest] = pid
                ticks_i += 1
            else:
                raise ValueError("unknown move type %r" % (typ,))
        target = scenes[i + 1]
        if stage != target:
            raise ValueError("window %d: stage layout does not match scene %d" % (i + 1, i + 2))
        total_ticks += ticks_i
        if ticks_i > budgets[i]:
            exceeded += 1
    return total_ticks, exceeded, P


def build_baseline_moves(N, M, w, prec, L0, R0, scenes):
    """Always-feasible, deliberately naive 'clear everything, then rebuild'
    construction: every window, evacuate ALL occupied cells (even ones that
    do not need to change), fully unload both wings (placing needed props,
    parking the rest on spare cells), then push the parked leftovers back.
    Requires N >= M (guaranteed by the generator)."""
    stage = scenes[0][:]
    wingL = L0[:]
    wingR = R0[:]
    S = len(scenes)
    windows_moves = []
    for i in range(S - 1):
        target = scenes[i + 1]
        moves = []
        # Phase 1: evacuate every occupied cell, cell-ascending, naive wing pick.
        for c in range(N):
            if stage[c] != 0:
                pid = stage[c]
                wing = "L" if len(wingL) <= len(wingR) else "R"
                stk = wingL if wing == "L" else wingR
                moves.append(("PUSH", c, wing, pid))
                stk.append(pid)
                stage[c] = 0
        # Phase 2: fully unload both wings; place needed props, park the rest.
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
        # Phase 3: push parked leftovers back into the wings.
        for cell, pid in parked:
            wing = "L" if len(wingL) <= len(wingR) else "R"
            stk = wingL if wing == "L" else wingR
            moves.append(("PUSH", cell, wing, pid))
            stk.append(pid)
            stage[cell] = 0
        windows_moves.append(moves)
    return windows_moves


def parse_output(path, S):
    toks = open(path).read().split()
    it = iter(toks)
    windows_moves = []
    try:
        for _ in range(S - 1):
            k = int(next(it))
            if k < 0 or k > 200000:
                raise ValueError("bad move count")
            moves = []
            for _ in range(k):
                typ = next(it)
                if typ not in ("PUSH", "POP"):
                    raise ValueError("bad move type %r" % (typ,))
                a_raw = next(it)
                b_raw = next(it)
                pid_raw = next(it)
                pid = int(pid_raw)
                if typ == "PUSH":
                    a = int(a_raw)
                    b = b_raw
                else:
                    a = a_raw
                    b = int(b_raw)
                moves.append((typ, a, b, pid))
            windows_moves.append(moves)
    except (StopIteration, ValueError) as e:
        raise ValueError("parse error: %s" % e)
    return windows_moves


def main():
    N, M, S, w, prec, L0, R0, scenes, budgets = parse_input(sys.argv[1])

    try:
        windows_moves = parse_output(sys.argv[2], S)
    except ValueError as e:
        fail(str(e))

    try:
        F_ticks, F_exceeded, P = simulate(N, w, prec, L0, R0, scenes, budgets, windows_moves, strict=True)
    except ValueError as e:
        fail(str(e))
    F = F_ticks + P * F_exceeded

    base_moves = build_baseline_moves(N, M, w, prec, L0, R0, scenes)
    B_ticks, B_exceeded, _ = simulate(N, w, prec, L0, R0, scenes, budgets, base_moves, strict=False)
    B = max(1, B_ticks + P * B_exceeded)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d(ticks=%d,exceeded=%d) B=%d Ratio: %.6f" % (F, F_ticks, F_exceeded, B, sc / 1000.0))


if __name__ == "__main__":
    main()
