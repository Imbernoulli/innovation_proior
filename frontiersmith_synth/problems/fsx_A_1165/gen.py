import random
import sys

# Difficulty ladder: (N decrees, C fire-drills).
# One arrival per tick (a_i = i), so TMAX is derived from N plus slack.
#
# Most decrees are ROUTINE (large deadline slack, can sit pending a long
# time). Occasionally a short BURST of URGENT decrees arrives (several
# consecutive ticks, each with tiny slack) -- these force the pending queue
# to flush soon after the burst starts, so the burst's decrees end up
# bunched together near the TAIL of one sealed page (arrival/id order
# always places whatever just arrived last). We simulate the exact same
# deadline-driven group-commit batcher the reference solutions use, so we
# know precisely which page each burst landed in.
#
# A fire-drill's needed set is either (a) several decrees from ONE such
# burst page -- under plain arrival-order sealing every one of them sits
# near the very end of a page that may otherwise be mostly routine filler,
# so recovery must walk almost the entire page to restore them all; or (b)
# the single triggering decree from each of several consecutive pages --
# recovery must then walk back through that whole chain of pages. Neither
# form can be fixed by choosing WHEN to seal; only reordering the records
# INSIDE a page (crash-relevance first) shortens the scan.
SIZES = [
    (150, 12),
    (220, 16),
    (300, 20),
    (420, 24),
    (550, 28),
    (700, 32),
    (900, 36),
    (1150, 40),
    (1450, 44),
    (1800, 48),
]

HOT_LO, HOT_HI = 3, 6
CALM_LO, CALM_HI = 40, 70
P_BURST_START = 0.03
BURST_LEN_LO, BURST_LEN_HI = 4, 7


def simulate_pages(N, A, deadline):
    """Same deadline-driven group-commit batcher as solutions/greedy.py."""
    pages = []
    pending = []
    for i in range(1, N + 1):
        pending.append(i)
        min_dl = min(deadline[j] for j in pending)
        if min_dl <= A[i]:
            pages.append((min_dl, list(pending)))
            pending = []
    if pending:
        pages.append((min(deadline[j] for j in pending), list(pending)))
    return pages


def page_trigger(order, deadline, tick):
    for rid in order:
        if deadline[rid] == tick:
            return rid
    return order[-1]  # defensive fallback, should not happen


def build(testId):
    N, C = SIZES[(testId - 1) % len(SIZES)]
    rnd = random.Random(9001 + 17 * testId)

    A = [0] + list(range(1, N + 1))  # 1-indexed: A[i] = i
    D = [0] * (N + 1)
    burst_id = [-1] * (N + 1)
    cur_burst = -1
    remaining_burst = 0
    next_burst_id = 0
    for i in range(1, N + 1):
        if remaining_burst == 0:
            if rnd.random() < P_BURST_START:
                remaining_burst = rnd.randint(BURST_LEN_LO, BURST_LEN_HI)
                cur_burst = next_burst_id
                next_burst_id += 1
        if remaining_burst > 0:
            D[i] = rnd.randint(HOT_LO, HOT_HI)
            burst_id[i] = cur_burst
            remaining_burst -= 1
        else:
            D[i] = rnd.randint(CALM_LO, CALM_HI)
    deadline = [0] * (N + 1)
    for i in range(1, N + 1):
        deadline[i] = A[i] + D[i]

    TMAX = N + CALM_HI + 40

    pages = simulate_pages(N, A, deadline)
    triggers = [page_trigger(order, deadline, tick) for tick, order in pages]
    P = len(pages)

    # For each burst, find which single page holds the MAJORITY of its
    # members (a short burst can occasionally straddle a flush boundary) --
    # that page, with >=3 of the burst's members, is a "same-page" multi-need
    # candidate: several decrees that all landed close together near the
    # tail of one page.
    page_of_id = [-1] * (N + 1)
    pos_of_id = [-1] * (N + 1)
    for pidx, (tick, order) in enumerate(pages):
        for pos, rid in enumerate(order):
            page_of_id[rid] = pidx
            pos_of_id[rid] = pos

    burst_members = {}
    for i in range(1, N + 1):
        b = burst_id[i]
        if b >= 0:
            burst_members.setdefault(b, []).append(i)

    burst_pages = []
    for b, ids in burst_members.items():
        by_page = {}
        for rid in ids:
            by_page.setdefault(page_of_id[rid], []).append(rid)
        pidx, members = max(by_page.items(), key=lambda kv: len(kv[1]))
        if len(members) >= 3:
            burst_pages.append((pidx, members))

    crashes = []
    for j in range(C):
        use_burst = burst_pages and (j % 3 != 0 or P == 1)
        if use_burst:
            pidx, members = burst_pages[j % len(burst_pages)]
            m = min(len(members), rnd.randint(3, min(7, len(members))))
            needed = sorted(rnd.sample(members, m))
            next_tick = pages[pidx + 1][0] if pidx + 1 < P else TMAX
            max_dl = max(deadline[i] for i in needed)
            slack = max(0, min(5, next_tick - max_dl - 1))
            c_tick = max_dl + rnd.randint(0, slack) if next_tick - max_dl - 1 >= 0 else max_dl
            c_tick = max(c_tick, max_dl)
            c_tick = min(c_tick, TMAX)
        else:
            k = rnd.randint(3, min(7, P))
            p = rnd.randint(k - 1, P - 1)
            needed = sorted(triggers[p - k + 1: p + 1])
            page_tick = pages[p][0]
            next_tick = pages[p + 1][0] if p + 1 < P else TMAX
            slack = max(0, min(5, next_tick - page_tick - 1))
            c_tick = page_tick + rnd.randint(0, slack)
            c_tick = min(c_tick, TMAX)
        crashes.append((c_tick, needed))

    out = []
    out.append(f"{N} {C} {TMAX}")
    for i in range(1, N + 1):
        out.append(f"{A[i]} {D[i]}")
    for c_tick, needed in crashes:
        out.append(f"{c_tick} {len(needed)}")
        out.append(" ".join(str(x) for x in needed))
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(build(testId))


if __name__ == "__main__":
    main()
