import sys
import heapq
from fractions import Fraction


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_input(path):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("cannot read input")
    it = iter(toks)
    try:
        N = int(next(it)); M = int(next(it)); K = int(next(it))
        if N <= 0 or M <= 0 or K < 0:
            fail("bad header")
        dur = [0] * (N + 1)
        preds = [[] for _ in range(N + 1)]
        succs = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            d = int(next(it)); p = int(next(it))
            if d <= 0 or p < 0:
                fail("bad job record")
            plist = []
            for _ in range(p):
                pj = int(next(it))
                if not (1 <= pj < i):
                    fail("predecessor out of range / not topological")
                plist.append(pj)
            dur[i] = d
            preds[i] = plist
            for pj in plist:
                succs[pj].append(i)
        subsets = []
        for _ in range(K):
            num = int(next(it)); den = int(next(it)); s = int(next(it))
            if num <= 0 or den <= 0 or num <= den or s <= 0:
                fail("bad perturbation record")
            ids = []
            for _ in range(s):
                jid = int(next(it))
                if not (1 <= jid <= N):
                    fail("perturbation job id out of range")
                ids.append(jid)
            subsets.append((Fraction(num, den), ids))
    except StopIteration:
        fail("truncated input")
    except ValueError:
        fail("non-integer token in input")
    extra = list(it)
    if extra:
        fail("trailing tokens in input")
    return N, M, K, dur, preds, succs, subsets


def simulate(order, N, M, durs_scenario):
    """Discrete-event list scheduling on M identical machines under precedence
    (encoded via `preds_count`/`succs`, closed over via globals set by caller).
    order: list of job ids (priority: order[0] is highest priority).
    durs_scenario: dict/list job id -> duration to use in THIS scenario.
    Returns makespan (int)."""
    rank = [0] * (N + 1)
    for pos, jid in enumerate(order):
        rank[jid] = pos
    pred_count = [len(P) for P in PREDS]
    ready = []  # heap of (rank, job)
    for j in range(1, N + 1):
        if pred_count[j] == 0:
            heapq.heappush(ready, (rank[j], j))
    free_machines = M
    running = []  # heap of (finish_time, job)
    cur_time = 0
    started = [False] * (N + 1)
    finished_count = 0
    while True:
        while free_machines > 0 and ready:
            _, j = heapq.heappop(ready)
            started[j] = True
            heapq.heappush(running, (cur_time + durs_scenario[j], j))
            free_machines -= 1
        if not running:
            break
        # Batch EVERY job finishing at this earliest timestamp before doing any
        # new dispatch: a successor with several predecessors must see ALL of
        # them completed (not just whichever one this heap happened to pop
        # first) before it can be considered ready, and machines freed by
        # same-instant completions must all be available to the next dispatch
        # round together (not doled out one at a time between completions).
        cur_time = running[0][0]
        while running and running[0][0] == cur_time:
            _, j = heapq.heappop(running)
            free_machines += 1
            finished_count += 1
            for s in SUCCS[j]:
                pred_count[s] -= 1
                if pred_count[s] == 0:
                    heapq.heappush(ready, (rank[s], s))
        if finished_count == N and not ready and free_machines == M:
            break
    return cur_time


def inflate(dur, factor, ids):
    scen = list(dur)  # copy, index 0 unused
    for jid in ids:
        base = dur[jid]
        infl = base * factor
        # ceil to keep integer durations while remaining exact (Fraction math)
        scen[jid] = -(-infl.numerator // infl.denominator)
    return scen


def worst_case_makespan(order, N, M, dur, subsets):
    best = 0
    for factor, ids in subsets:
        scen = inflate(dur, factor, ids)
        mk = simulate(order, N, M, scen)
        if mk > best:
            best = mk
    if not subsets:
        # no perturbations published: worst case is simply the nominal run
        best = simulate(order, N, M, dur)
    return best


def zero_parallelism_bound(N, M, dur, subsets):
    """The checker's own trivial construction: run every job back-to-back on a
    single resource (as if you never bothered to exploit the M available
    machines at all), then take the worst of the K published scenarios. This
    is a valid, always-feasible degenerate schedule (sequential execution
    trivially respects every precedence constraint) and a legitimate upper
    bound for "no scheduling reasoning whatsoever"."""
    best = 0
    scenarios = [inflate(dur, factor, ids) for factor, ids in subsets] or [dur]
    for scen in scenarios:
        total = sum(scen[1:N + 1])
        if total > best:
            best = total
    return best


def main():
    global PREDS, SUCCS
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, M, K, dur, preds, succs, subsets = parse_input(in_path)
    PREDS, SUCCS = preds, succs

    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("cannot read output")

    if len(out_toks) != N:
        fail("expected %d integers, got %d" % (N, len(out_toks)))

    order = []
    seen = set()
    for tok in out_toks:
        try:
            v = int(tok)
        except (ValueError, OverflowError):
            fail("non-integer token %r" % tok)
        if not (1 <= v <= N):
            fail("job id out of range: %r" % tok)
        if v in seen:
            fail("duplicate job id %d" % v)
        seen.add(v)
        order.append(v)
    if len(seen) != N:
        fail("not a permutation of 1..N")

    F = worst_case_makespan(order, N, M, dur, subsets)
    if F <= 0:
        fail("non-positive makespan")

    # ---- internal baseline B: zero-parallelism (no reasoning about scheduling
    # or machines at all) worst case over the published sweep ----
    B = zero_parallelism_bound(N, M, dur, subsets)
    B = max(1, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Subsystem-Robust Release Schedule: F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
