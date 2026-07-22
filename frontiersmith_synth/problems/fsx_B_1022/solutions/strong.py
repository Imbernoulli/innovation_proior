# TIER: strong
"""Mean-field reformulation via sequential exchange scheduling.

The objective couples boats ONLY through the shared decaying wake field, so
the right unit of decision isn't "each boat's own shortest path" -- it's the
fleet's joint SPATIAL DENSITY (which lane of the gate band a boat sails) and
PHASE (when it departs, relative to the decay window w). Neither ingredient
alone is always enough: sometimes the band has room for everyone (spreading
suffices, staggering would only add pointless delay); sometimes it doesn't
(a lane must be shared, and only a phase gap > w actually clears the wake).

We therefore schedule the fleet ONE BOAT AT A TIME, in index order. Boat i
is committed only after replaying the SAME wake-field physics the checker
uses (a small self-contained copy of it) against every already-committed
boat, for every candidate (lane, start-tick) pair, and picking whichever
candidate minimizes boat i's OWN finish tick given that fixed history. This
always weakly dominates "pick the fixed natural lane, launch at tick 0" and
"pick a fresh lane but never stagger" -- both are literally inside the
candidate set -- because it exploits the wake field directly instead of a
static formula tuned in the abstract."""
import sys, heapq

MOVE_DELTA = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


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


def simulate(N, w, CAP, starts, gates, start_ticks, moves_list):
    """Exact replica of the checker's wake-field replay (see verify.py)."""
    B = len(starts)
    G = len(gates)
    recent = {}
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
                return False, None, None
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
        if col[i] != N - 1 or mask[i] != full_mask:
            return False, None, None
    return True, finish, sum(finish)


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

    gorder = sorted(gates, key=lambda g: g[0])          # efficient (shortest-path) gate order
    lane_budget = min(hi - lo + 1 for (_, lo, hi) in gorder)  # narrowest band = # usable lanes

    def build_moves(i, lane):
        r, c = starts[i], 0
        mv = []
        for (gc, lo, hi) in gorder:
            tr = min(hi, lo + lane)
            mv += diag_moves(r, c, tr, gc)
            r, c = tr, gc
        mv += diag_moves(r, c, r, N - 1)
        return mv

    # Candidate departure phases: launch immediately, or wait out whole
    # multiples of (decay window + 1) so a shared lane's wake has fully faded.
    cand_starts = sorted(set([0] + [k * (w + 1) for k in range(1, 4) if k * (w + 1) <= Smax]))

    committed_starts, committed_moves = [], []
    for i in range(B):
        best = None  # (finish_i, lane, start, moves)
        for lane in range(lane_budget):
            mv = build_moves(i, lane)
            for st in cand_starts:
                feas, fin, _ = simulate(N, w, CAP, starts[:i + 1], gates,
                                         committed_starts + [st], committed_moves + [mv])
                if not feas:
                    continue
                f_i = fin[i]
                key = (f_i, st, lane)
                if best is None or key < (best[0], best[2], best[1]):
                    best = (f_i, lane, st, mv)
        committed_starts.append(best[2])
        committed_moves.append(best[3])

    out = [f"{committed_starts[i]} {''.join(committed_moves[i])}" for i in range(B)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
