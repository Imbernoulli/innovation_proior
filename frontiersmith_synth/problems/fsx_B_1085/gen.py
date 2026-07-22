import sys, random

# ---------------------------------------------------------------------------
# seek-window-arrangement  (format C, minimise total pickup-arm travel)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.
#   Deterministic in testId only.
#
# Instance (stdout):
#   line 1:  N T cap w M
#   line 2:  M integers -- the cue sheet Q[0..M-1], each a cell id in [0,N)
#
# Story: a record press must assign each of N audio cells to one of T grooves
# (each groove holds at most `cap` cells).  A listener's cue sheet plays M
# cells in a fixed order.  The pickup arm does NOT play strictly in cue
# order: it keeps a look-ahead buffer of the oldest `w` not-yet-played cues,
# and always plays whichever buffered cue sits on the groove nearest the
# arm's current groove (ties -> earliest cue).  This is a bounded reorder
# window (depth w) over a fixed cue order.
#
# Hidden planted structure: cells are secretly partitioned into K "sides"
# (clusters) of size ~w.  Each cue-sheet "visit" plays one side's cells (a
# permutation / resampling of that side).  A MANDATORY coverage visit per
# side guarantees every cell is cued at least once; extra visits are drawn
# either uniformly (TRAP cases: cue frequency carries no signal about which
# side a cell belongs to) or Zipf-skewed (benign cases: frequency partially
# correlates with side).  Visits are globally shuffled, so a cell's FIRST
# cue time is scattered across the whole timeline regardless of its side --
# defeating "place cells in first-use order".  The generator never reveals
# the side partition; a solver must recover locality from which cell ids
# are cued within `w` positions of each other.
# ---------------------------------------------------------------------------

TRAP_IDS = {2, 3, 5, 7, 9}


def build_instance(t):
    rng = random.Random(20260721 + 977 * t)

    # small->larger ladder. N is always an exact multiple of w so EVERY side
    # has exactly w members -- no short "remainder" side. A remainder side
    # would get the same w samples-per-visit as every other side crammed
    # onto fewer distinct cells, inflating those cells' raw frequency far
    # above a normal side member's and leaking side identity through
    # frequency alone (defeating the "frequency carries no signal" trap).
    w_list = [6, 7, 7, 8, 8, 9, 9, 10, 10, 11]
    w = w_list[t - 1]
    N_list = [72, 91, 112, 136, 152, 171, 189, 210, 230, 264]
    N = N_list[t - 1]
    assert N % w == 0
    cap = max(2, w // 3)

    # ---- partition N cells into K equal-size sides of exactly w members ----
    ids = list(range(N))
    rng.shuffle(ids)          # side membership is unrelated to cell id order
    sides = [ids[i:i + w] for i in range(0, N, w)]
    K = len(sides)

    base_tracks = -(-N // cap)              # ceil(N/cap)
    T = base_tracks + K + 5                 # generous slack for any solver's own clustering

    trap = t in TRAP_IDS
    E = 3 + (t - 1) // 3                    # extra-visit multiplier grows with the ladder

    # ---- schedule of "visits"; each visit plays one side ------------------
    mandatory = list(range(K))              # guarantees full coverage of every side
    extra_count = K * E
    if trap:
        extra = [rng.randrange(K) for _ in range(extra_count)]
    else:
        order = list(range(K))
        rng.shuffle(order)                  # which side is "hot" varies by seed
        weights = [1.0 / (rank + 1) ** 1.5 for rank in range(K)]
        extra = rng.choices(order, weights=weights, k=extra_count)

    visits = [(s, True) for s in mandatory] + [(s, False) for s in extra]
    rng.shuffle(visits)                     # scrambles first-occurrence time globally

    Q = []
    for side_id, mandatory_flag in visits:
        members = sides[side_id]
        if mandatory_flag:
            chunk = list(members)
            rng.shuffle(chunk)
            while len(chunk) < w:
                chunk.append(rng.choice(members))
            chunk = chunk[:w]
        else:
            chunk = [rng.choice(members) for _ in range(w)]
        Q.extend(chunk)

    M = len(Q)
    return N, T, cap, w, M, Q


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    N, T, cap, w, M, Q = build_instance(t)
    out = []
    out.append("%d %d %d %d %d" % (N, T, cap, w, M))
    out.append(" ".join(str(x) for x in Q))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
