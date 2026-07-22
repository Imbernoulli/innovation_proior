#!/usr/bin/env python3
"""gen.py <testId> -- pawn-shop shelf (offline generalized caching / admission) generator.

Prints ONE instance to stdout:
    N K C
    size_1 ... size_K
    id_1 ... id_N

All randomness is seeded ONLY from testId -> fully deterministic and reproducible.
testId 1..10 is a difficulty ladder: 1-2 sanity/small, 3/5/6/7/8/10 are deliberately
adversarial "trap" traces (one-hit wonders, competing loops, a huge rarely-reused
object) that punish a plain recency-based admit-always policy.
"""
import sys
import heapq
import random


def build(rng, streams, N):
    """Event-driven interleave of several visit streams into one length-N timeline.
    Each stream: dict(period:int, jitter:int, start:int, gen:callable(rng)->id or None,
                       remaining:int|None). gen returning None means the stream is
                       exhausted; it is dropped from the heap."""
    heap = []
    for si, st in enumerate(streams):
        heap.append((st["start"], si))
    heapq.heapify(heap)
    seq = []
    fallback = None
    while len(seq) < N and heap:
        tm, si = heapq.heappop(heap)
        st = streams[si]
        if st.get("remaining") is not None:
            if st["remaining"] <= 0:
                continue
            st["remaining"] -= 1
        idv = st["gen"](rng)
        if idv is None:
            continue
        seq.append(idv)
        if fallback is None:
            fallback = idv
        jit = st.get("jitter", 0)
        nxt = tm + st["period"] + (rng.randint(0, jit) if jit > 0 else 0)
        heapq.heappush(heap, (nxt, si))
    while len(seq) < N:
        seq.append(fallback if fallback is not None else 1)
    return seq[:N]


class IdAlloc:
    def __init__(self):
        self.next_id = 1
        self.sizes = {}

    def new_ids(self, count, size_lo, size_hi, rng):
        ids = []
        for _ in range(count):
            i = self.next_id
            self.next_id += 1
            self.sizes[i] = rng.randint(size_lo, size_hi)
            ids.append(i)
        return ids


def loop_stream(ids, period, jitter, start):
    """Cycles through `ids` in a freshly shuffled order each full lap (keeps every id
    equally hot; avoids a fixed round-robin phase that would leak info about order)."""
    state = {"queue": []}

    def gen(rng):
        if not state["queue"]:
            order = list(ids)
            rng.shuffle(order)
            state["queue"] = order
        return state["queue"].pop()

    return {"period": period, "jitter": jitter, "start": start, "gen": gen}


def wonder_stream(alloc, count, size_lo, size_hi, period, jitter, start, rng_seedless=True):
    pool = {"left": count, "lo": size_lo, "hi": size_hi}

    def gen(rng):
        if pool["left"] <= 0:
            return None
        pool["left"] -= 1
        i = alloc.next_id
        alloc.next_id += 1
        alloc.sizes[i] = rng.randint(pool["lo"], pool["hi"])
        return i

    return {"period": period, "jitter": jitter, "start": start, "gen": gen, "remaining": count}


def finalize(alloc, seq):
    """Remap the ids that actually appear in seq to a dense 1..K range (drops any
    allocated-but-unused id) and returns (K, sizes_list, remapped_seq)."""
    used = sorted(set(seq))
    remap = {old: new for new, old in enumerate(used, start=1)}
    K = len(used)
    sizes = [0] * (K + 1)
    for old, new in remap.items():
        sizes[new] = alloc.sizes[old]
    new_seq = [remap[x] for x in seq]
    return K, sizes[1:], new_seq


def emit(N, K, C, sizes, seq):
    out = [f"{N} {K} {C}", " ".join(str(s) for s in sizes), " ".join(str(x) for x in seq)]
    sys.stdout.write("\n".join(out) + "\n")


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260 + tid)
    alloc = IdAlloc()

    if tid == 1:
        # sanity: tiny, capacity generous, nothing ever needs eviction.
        ids = alloc.new_ids(5, 40, 200, rng)
        total = sum(alloc.sizes[i] for i in ids)
        C = total + 150
        N = 16
        seq = [rng.choice(ids) for _ in range(N)]

    elif tid == 2:
        # small warmup: mild popularity skew, some eviction pressure but no deep trap.
        ids = alloc.new_ids(10, 100, 2000, rng)
        total = sum(alloc.sizes[i] for i in ids)
        C = max(300, int(0.55 * total))
        N = 50
        weights = [1.0 / r for r in range(1, len(ids) + 1)]
        order = list(ids)
        rng.shuffle(order)
        seq = rng.choices(order, weights=weights, k=N)

    elif tid == 3:
        # TRAP: small hot loop + a burst of one-hit wonders each round.
        loop_ids = alloc.new_ids(5, 300, 900, rng)
        loop_sum = sum(alloc.sizes[i] for i in loop_ids)
        C = int(1.15 * loop_sum)
        streams = [
            loop_stream(loop_ids, period=len(loop_ids), jitter=0, start=0),
            wonder_stream(alloc, count=26, size_lo=200, size_hi=3000, period=len(loop_ids), jitter=1, start=1),
        ]
        N = 8 * len(loop_ids) + 26 + 20
        seq = build(rng, streams, N)

    elif tid == 4:
        # generic zipf-ish mixed trace (mostly locality, some noise); not a deliberate trap.
        ids = alloc.new_ids(60, 100, 5000, rng)
        total = sum(alloc.sizes[i] for i in ids)
        C = max(500, int(0.4 * total))
        N = 300
        order = list(ids)
        rng.shuffle(order)
        weights = [1.0 / r for r in range(1, len(order) + 1)]
        seq = rng.choices(order, weights=weights, k=N)

    elif tid == 5:
        # TRAP: bigger loop + heavier one-hit-wonder pressure.
        loop_ids = alloc.new_ids(10, 500, 1500, rng)
        loop_sum = sum(alloc.sizes[i] for i in loop_ids)
        C = int(1.2 * loop_sum)
        streams = [
            loop_stream(loop_ids, period=len(loop_ids), jitter=0, start=0),
            wonder_stream(alloc, count=80, size_lo=300, size_hi=4000, period=max(2, len(loop_ids) // 2), jitter=1, start=2),
        ]
        N = 14 * len(loop_ids) + 80 + 60
        seq = build(rng, streams, N)

    elif tid == 6:
        # TRAP: competing loops, one fast-cycling (high value density), one slow-cycling
        # (low value density) but with a bigger footprint; capacity fits only one well.
        loopA = alloc.new_ids(6, 700, 1200, rng)      # fast, small footprint
        loopB = alloc.new_ids(10, 700, 1200, rng)     # slow, bigger footprint
        sumA = sum(alloc.sizes[i] for i in loopA)
        sumB = sum(alloc.sizes[i] for i in loopB)
        C = int(1.1 * max(sumA, sumB))
        assert C < sumA + sumB
        streams = [
            loop_stream(loopA, period=len(loopA), jitter=0, start=0),
            loop_stream(loopB, period=4 * len(loopB), jitter=1, start=3),
        ]
        N = 15 * (len(loopA) + 1) + 20
        seq = build(rng, streams, N)

    elif tid == 7:
        # TRAP: small hot set + a rarely-reused huge object that eats most of capacity.
        loop_ids = alloc.new_ids(8, 300, 900, rng)
        loop_sum = sum(alloc.sizes[i] for i in loop_ids)
        C = int(1.25 * loop_sum)
        huge_size_lo = int(0.75 * C)
        huge_size_hi = int(0.9 * C)
        streams = [
            loop_stream(loop_ids, period=len(loop_ids), jitter=0, start=0),
            wonder_stream(alloc, count=4, size_lo=huge_size_lo, size_hi=huge_size_hi,
                          period=9 * len(loop_ids), jitter=2, start=5 * len(loop_ids)),
        ]
        N = 20 * len(loop_ids) + 4 + 20
        seq = build(rng, streams, N)

    elif tid == 8:
        # large combined trap: loop+wonders, competing loops, and a rare huge object.
        loop_ids = alloc.new_ids(12, 400, 1200, rng)
        loop_sum = sum(alloc.sizes[i] for i in loop_ids)
        loopB = alloc.new_ids(14, 400, 1200, rng)
        sumB = sum(alloc.sizes[i] for i in loopB)
        C = int(1.15 * max(loop_sum, sumB))
        huge_lo, huge_hi = int(0.7 * C), int(0.85 * C)
        streams = [
            loop_stream(loop_ids, period=len(loop_ids), jitter=0, start=0),
            loop_stream(loopB, period=5 * len(loopB), jitter=1, start=7),
            wonder_stream(alloc, count=120, size_lo=150, size_hi=4000, period=max(2, len(loop_ids) // 2), jitter=1, start=2),
            wonder_stream(alloc, count=5, size_lo=huge_lo, size_hi=huge_hi, period=13 * len(loop_ids), jitter=3, start=6 * len(loop_ids)),
        ]
        N = 1600
        seq = build(rng, streams, N)

    elif tid == 9:
        # large generic zipf (general-quality check, not a deliberate single trap).
        ids = alloc.new_ids(180, 80, 6000, rng)
        total = sum(alloc.sizes[i] for i in ids)
        C = max(2000, int(0.35 * total))
        N = 2200
        order = list(ids)
        rng.shuffle(order)
        weights = [1.0 / (r ** 1.1) for r in range(1, len(order) + 1)]
        seq = rng.choices(order, weights=weights, k=N)

    else:  # tid == 10: largest, most adversarial combined trace.
        loop_ids = alloc.new_ids(16, 400, 1300, rng)
        loop_sum = sum(alloc.sizes[i] for i in loop_ids)
        loopB = alloc.new_ids(20, 400, 1300, rng)
        sumB = sum(alloc.sizes[i] for i in loopB)
        C = int(1.15 * max(loop_sum, sumB))
        huge_lo, huge_hi = int(0.7 * C), int(0.85 * C)
        streams = [
            loop_stream(loop_ids, period=len(loop_ids), jitter=0, start=0),
            loop_stream(loopB, period=5 * len(loopB), jitter=1, start=9),
            wonder_stream(alloc, count=220, size_lo=150, size_hi=4500, period=max(2, len(loop_ids) // 2), jitter=1, start=3),
            wonder_stream(alloc, count=8, size_lo=huge_lo, size_hi=huge_hi, period=17 * len(loop_ids), jitter=3, start=8 * len(loop_ids)),
        ]
        N = 3000
        seq = build(rng, streams, N)

    K, sizes, seq = finalize(alloc, seq)
    N = len(seq)
    emit(N, K, C, sizes, seq)


if __name__ == "__main__":
    main()
