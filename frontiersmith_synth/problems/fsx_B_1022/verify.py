#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the regatta wake-fanout problem.

Reads the fleet instance from <in> and the participant's per-boat schedule from <out>.
Replays every boat through a shared, tick-ordered discrete-event simulation of the
decaying wake field (see simulate() below), enforces feasibility strictly, and scores
the (minimized) sum of finish ticks against the checker's own naive single-lane
construction. Prints `Ratio: <float in [0,1]>` on the last line and exits 0.
"""
import sys, heapq

MOVE_DELTA = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def fail(reason):
    print("Reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
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
    return dict(N=N, B=B, G=G, w=w, CAP=CAP, Smax=Smax, maxMoves=maxMoves,
                starts=starts, gates=gates)


def simulate(N, w, CAP, starts, gates, start_ticks, moves_list):
    """Event-driven replay of the fleet through the decaying wake field.

    Cost of a move that ENTERS a new cell = 1 + (# entries of that cell, by any
    boat, with entry tick e satisfying 0 <= ta - e <= w, i.e. entered at or
    before the current arrival candidate ta and not yet decayed); capped at
    CAP. The lower bound (e <= ta) is what keeps this causal: a deposit is only
    "recent" once it has actually happened, never before. Waiting in place
    ('S') always costs exactly 1 tick but still refreshes the cell's memory.
    Ties at the same tick are broken by ascending boat index (deterministic):
    a lower-indexed boat's entry, decided at an identical instant, is visible
    to a higher-indexed boat's move into the same cell at that same instant.
    Returns (feasible, reason, finish_ticks, F) where F = sum(finish_ticks).
    """
    B = len(starts)
    G = len(gates)
    recent = {}                      # cell -> list of recent entry ticks (pruned)
    row = list(starts)
    col = [0] * B
    cur_tick = list(start_ticks)
    mi = [0] * B
    mask = [0] * B
    born = [False] * B
    finish = [None] * B
    full_mask = (1 << G) - 1

    def check_gates(i):
        for gi in range(G):
            gc, glo, ghi = gates[gi]
            if col[i] == gc and glo <= row[i] <= ghi:
                mask[i] |= (1 << gi)

    def deposit(cell, t):
        recent.setdefault(cell, []).append(t)

    def recent_count(cell, ta):
        lst = recent.get(cell)
        if not lst:
            return 0
        # Queries against a given cell arrive in non-decreasing `ta` order (the
        # whole simulation is processed in non-decreasing global tick order),
        # so an entry more than w ticks older than THIS ta is too old for every
        # later query too -- safe to prune permanently. An entry with a tick
        # AHEAD of `ta` hasn't happened yet from this query's viewpoint (it's
        # a later-arriving move that was already costed): it must NOT be
        # dropped, only excluded from this count, since a future query (larger
        # ta) may legitimately see it once it becomes "recent" or "just now".
        lst[:] = [t for t in lst if ta - t <= w]
        return sum(1 for t in lst if t <= ta)

    heap = [(start_ticks[i], i) for i in range(B)]
    heapq.heapify(heap)

    while heap:
        t, i = heapq.heappop(heap)
        if t != cur_tick[i]:
            continue
        if not born[i]:
            born[i] = True
            deposit((row[i], col[i]), cur_tick[i])
            check_gates(i)
        if mi[i] >= len(moves_list[i]):
            finish[i] = cur_tick[i]
            continue
        mv = moves_list[i][mi[i]]
        mi[i] += 1
        r, c = row[i], col[i]
        if mv == 'S':
            arrival = cur_tick[i] + 1
            deposit((r, c), arrival)
        else:
            dr, dc = MOVE_DELTA[mv]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                return False, f"boat {i} left the grid", None, None
            ta = cur_tick[i] + 1
            k = recent_count((nr, nc), ta)
            cost = min(CAP, 1 + k)
            arrival = cur_tick[i] + cost
            deposit((nr, nc), arrival)
            r, c = nr, nc
        row[i], col[i] = r, c
        cur_tick[i] = arrival
        check_gates(i)
        heapq.heappush(heap, (cur_tick[i], i))

    for i in range(B):
        if finish[i] is None:
            finish[i] = cur_tick[i]
        if col[i] != N - 1:
            return False, f"boat {i} never reached the finish column", None, None
        if mask[i] != full_mask:
            return False, f"boat {i} missed gate(s), mask={mask[i]:b}", None, None

    return True, "ok", finish, sum(finish)


def diag_moves(r, c, r2, c2):
    """Interleave vertical/horizontal steps (Bresenham-style) from (r,c) to (r2,c2)."""
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


def naive_baseline_plan(inst):
    """Checker's own trivial feasible construction: every boat departs at tick 0 and
    sails the SAME single racing line -- straight to the CENTER of each gate's band,
    visiting gates in the exact order the input lists them (no re-sorting, no lane
    choice, no staggering). This is deliberately the worst-case, fully-converged fleet."""
    N, B = inst['N'], inst['B']
    starts_t = [0] * B
    moves_list = []
    for i in range(B):
        r, c = inst['starts'][i], 0
        mv = []
        for (gc, lo, hi) in inst['gates']:
            tr = (lo + hi) // 2
            mv += diag_moves(r, c, tr, gc)
            r, c = tr, gc
        mv += diag_moves(r, c, r, N - 1)
        moves_list.append(mv)
    return starts_t, moves_list


def parse_output(path, B, Smax, maxMoves):
    lines = [ln for ln in open(path).read().splitlines()]
    # drop trailing blank lines only
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) != B:
        fail(f"expected {B} boat lines, got {len(lines)}")
    starts_t, moves_list = [], []
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) == 1:
            parts = parts + [""]
        if len(parts) != 2:
            fail(f"boat {i}: expected 2 tokens, got {len(parts)}")
        try:
            st = int(parts[0])
        except ValueError:
            fail(f"boat {i}: non-integer start tick")
        if not (0 <= st <= Smax):
            fail(f"boat {i}: start tick {st} outside [0,{Smax}]")
        mv = parts[1]
        if len(mv) > maxMoves:
            fail(f"boat {i}: {len(mv)} moves exceeds cap {maxMoves}")
        if any(ch not in "UDLRS" for ch in mv):
            fail(f"boat {i}: illegal move character")
        starts_t.append(st)
        moves_list.append(mv)
    return starts_t, moves_list


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = read_instance(in_path)
    N, B, w, CAP = inst['N'], inst['B'], inst['w'], inst['CAP']

    starts_t, moves_list = parse_output(out_path, B, inst['Smax'], inst['maxMoves'])

    feas, reason, finish, F = simulate(N, w, CAP, inst['starts'], inst['gates'], starts_t, moves_list)
    if not feas:
        fail(reason)
    if F <= 0:
        fail("degenerate zero total time")

    _, _, _, B0 = simulate(N, w, CAP, inst['starts'], inst['gates'], *naive_baseline_plan(inst))
    if B0 is None or B0 <= 0:
        fail("internal baseline degenerate")

    sc = 100.0 * B0 / max(1e-9, F)
    if sc > 1000.0:
        sc = 1000.0
    print("F=%d baseline=%d" % (F, B0))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
