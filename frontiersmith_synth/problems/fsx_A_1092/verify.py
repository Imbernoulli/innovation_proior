#!/usr/bin/env python3
"""
Deterministic checker for fsx_A_1092 -- "Humpyard Radix Resort".

Reads:
  <in>  : N T L Y a b s cap  then N lines "train_id slot_id" (arrival order,
          line 0 = top / immediately-accessible car of the inbound track).
  <out> : M  then M lines "engine t src dst k" -- a timed schedule of cut moves.

Track indices:
  0            = the working "current-sequence" track (starts with all N cars,
                 cap = N; may also be refilled later -- it is a normal LIFO stack).
  1 .. Y       = classification ("bucket") tracks, cap = cap each.
  Y+1 .. Y+T   = final tracks, one per outbound train; final track for train i
                 is Y+1+i, hard cap = L (exactly the train length).

Move semantics: pop the top k cars of src as an ordered cut (nearest-top first),
push them onto dst preserving that relative order (the popped-top car ends up
nearest dst's new top). This is what physically happens when a locomotive shoves
a coupled cut of cars from one track onto another.

Feasibility (ANY violation -> Ratio: 0.0):
  - well-formed tokens, engine in {1,2}, t finite >=0, src!=dst, k>=1 integer,
    0<=src,dst<=Y+T, src not a final track (cars never leave a final track),
    dst capacity respected, src has >= k cars at execution time.
  - GLOBAL lead mutex: sort moves by start time t; consecutive (by t) move
    intervals [t, t+dur) must not overlap, dur computed from a,b,k and a
    per-engine mode-switch penalty s (mode = 'D' if dst in [1,Y] else 'F').
  - at the end every final track holds exactly that train's L cars in the
    correct order (top-to-bottom = slot 0,1,...,L-1).

Objective (MINIMIZE): makespan F = time the last move finishes (t + dur).
Since the lead is a strict global mutex, minimizing F is equivalent to
minimizing total scheduled cost a*M + b*sum(k) + s*(#mode switches); optimal
schedules always pack moves back-to-back with zero idle time.

Internal baseline B: the checker's own reference construction -- dump ALL cars
onto ONE classification track (ignoring the other Y-1 buckets and the radix
structure entirely), then for every needed car dig it out one car at a time
(scratch-and-restore), an always-correct but asymptotically wasteful O(N^2)
single-engine strategy. Score (minimization): sc = min(1000, 100*B/F),
Ratio = sc/1000.
"""
import sys

MAX_MOVES_FACTOR = 60


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(in_path):
    with open(in_path) as fh:
        toks = fh.read().split()
    if len(toks) < 8:
        fail("truncated instance")
    N, T, L, Y, a, b, s, cap = (int(x) for x in toks[:8])
    rest = toks[8:]
    if len(rest) != 2 * N:
        fail("instance/car-count mismatch")
    cars = []
    for i in range(N):
        cars.append((int(rest[2 * i]), int(rest[2 * i + 1])))
    return N, T, L, Y, a, b, s, cap, cars


def mode_of(dst, Y):
    return 'D' if 1 <= dst <= Y else 'F'


def simulate_schedule(moves, N, T, L, Y, a, b, s, cap, cars, max_moves):
    """Replay a submitted move list. Returns (F, tracks_at_end) or calls fail()."""
    if len(moves) == 0:
        fail("no moves")
    if len(moves) > max_moves:
        fail("too many moves (%d > %d)" % (len(moves), max_moves))

    ntracks = Y + T + 1
    tracks = {i: [] for i in range(ntracks)}
    tracks[0] = list(reversed(cars))  # list[-1] = top (immediately accessible)

    def cap_of(idx):
        if idx == 0:
            return N
        if 1 <= idx <= Y:
            return cap
        return L  # final track for some train

    # stable order by start time; ties are illegal (checked below) but sort must
    # still be deterministic, so break ties by original submission order.
    order = sorted(range(len(moves)), key=lambda i: (moves[i][1], i))

    engine_last_mode = {}
    prev_end = 0.0
    F = 0.0
    for oi in order:
        engine, t, src, dst, k = moves[oi]
        if engine not in (1, 2):
            fail("bad engine id at move %d" % oi)
        if not (t == t and abs(t) != float("inf")):
            fail("non-finite start time at move %d" % oi)
        if t < 0:
            fail("negative start time at move %d" % oi)
        if not (0 <= src <= ntracks - 1) or not (0 <= dst <= ntracks - 1):
            fail("track index out of range at move %d" % oi)
        if src == dst:
            fail("src == dst at move %d" % oi)
        if src >= Y + 1:
            fail("move pulls cars OUT of a final track at move %d" % oi)
        if k < 1:
            fail("non-positive cut size at move %d" % oi)
        if t < prev_end - 1e-6:
            fail("lead-mutex overlap at move %d (t=%.6f < prev_end=%.6f)" % (oi, t, prev_end))
        if len(tracks[src]) < k:
            fail("source track %d has only %d cars, need %d (move %d)" % (src, len(tracks[src]), k, oi))
        if len(tracks[dst]) + k > cap_of(dst):
            fail("destination track %d capacity exceeded (move %d)" % (dst, oi))

        mode = mode_of(dst, Y)
        switch = engine in engine_last_mode and engine_last_mode[engine] != mode
        dur = a + b * k + (s if switch else 0)
        engine_last_mode[engine] = mode

        cut = tracks[src][-k:]
        del tracks[src][-k:]
        # preserve relative order: cut[-1] was nearest top of src -> ends nearest
        # top of dst, i.e. pushed LAST.
        tracks[dst].extend(cut)

        end = t + dur
        prev_end = end
        if end > F:
            F = end

    # final correctness
    for i in range(T):
        idx = Y + 1 + i
        blk = tracks[idx]
        if len(blk) != L:
            fail("train %d final track has %d cars, need %d" % (i, len(blk), L))
        # top-to-bottom reading = reversed(list) since list[-1] = top
        top_to_bottom = list(reversed(blk))
        for slot, (tr, sl) in enumerate(top_to_bottom):
            if tr != i or sl != slot:
                fail("train %d out of order at slot %d (found train=%d slot=%d)" % (i, slot, tr, sl))

    return F


def parse_output(out_path, max_moves):
    with open(out_path) as fh:
        toks = fh.read().split()
    if len(toks) == 0:
        fail("empty output")
    try:
        m = int(toks[0])
    except ValueError:
        fail("first token must be integer move count")
    if m < 0 or m > max_moves:
        fail("move count out of range")
    rest = toks[1:]
    if len(rest) != 5 * m:
        fail("expected %d move records, token count mismatch" % m)
    moves = []
    for i in range(m):
        chunk = rest[5 * i:5 * i + 5]
        try:
            engine = int(chunk[0])
            t = float(chunk[1])
            src = int(chunk[2])
            dst = int(chunk[3])
            k = int(chunk[4])
        except ValueError:
            fail("non-numeric move record %d" % i)
        for v in (t,):
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite value in move record %d" % i)
        moves.append((engine, t, src, dst, k))
    return moves


# ---------------------------------------------------------------------------
# internal baseline: dump-everything-into-one-bucket, dig-and-restore
# ---------------------------------------------------------------------------
def baseline_cost(N, T, L, Y, a, b, s, cap):
    """Always-correct, deliberately naive single-engine construction. Returns
    its own makespan cost, computed independent of any submitted output."""
    if Y < 2:
        fail("generator invariant violated: Y must be >= 2")

    # NOTE: we need the SAME instance's car list to build a baseline; caller
    # passes cars in.
    raise RuntimeError("call baseline_cost_with_cars instead")


def baseline_cost_with_cars(N, T, L, Y, a, b, s, cap, cars):
    track0 = list(reversed(cars))
    track1 = []
    track2 = []
    engine_last_mode = {}
    t = 0.0

    def do_move(src_list, dst_list, k, dst_is_bucket):
        nonlocal t
        mode = 'D' if dst_is_bucket else 'F'
        switch = 1 in engine_last_mode and engine_last_mode[1] != mode
        dur = a + b * k + (s if switch else 0)
        engine_last_mode[1] = mode
        t += dur
        cut = src_list[-k:]
        del src_list[-k:]
        dst_list.extend(cut)

    while track0:
        do_move(track0, track1, 1, True)

    finals = {tr: [] for tr in range(T)}
    for tr in range(T):
        for slot in range(L - 1, -1, -1):
            target = (tr, slot)
            pos = track1.index(target)
            depth = len(track1) - 1 - pos
            for _ in range(depth):
                do_move(track1, track2, 1, True)
            do_move(track1, finals[tr], 1, False)
            for _ in range(depth):
                do_move(track2, track1, 1, True)

    return t


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, T, L, Y, a, b, s, cap, cars = read_instance(in_path)
    max_moves = MAX_MOVES_FACTOR * (N + Y + T) + 200

    moves = parse_output(out_path, max_moves)
    F = simulate_schedule(moves, N, T, L, Y, a, b, s, cap, cars, max_moves)
    B = baseline_cost_with_cars(N, T, L, Y, a, b, s, cap, cars)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
