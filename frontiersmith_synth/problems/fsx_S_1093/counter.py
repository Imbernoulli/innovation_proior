#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic checker for "The Foresighted Referee".

Verifies:
  1. the claimed cut sequence exactly matches the true one (ground-truth simulation),
  2. every roster ('M') entry appears exactly once,
  3. every claimed cut is CERTIFIED: the transitive closure of the filed comparisons
     ('C' lines, oriented by the true ratings) proves the claimed player is below
     every other player registered-and-not-yet-cut at that moment.
Then scores cost = (#C) + 0.25*(#M) against an internal baseline B (cost of blind,
re-searched binary-insertion of the whole roster in registration order) and prints
`Ratio: <float in [0,1]>` on the LAST line. Any feasibility violation -> Ratio: 0.0.
"""
import sys


def fail(msg):
    sys.stdout.write("INFEASIBLE: %s\n" % msg)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        in_toks = open(in_path).read().split()
    except Exception:
        fail("cannot read input")

    ip = [0]

    def inext():
        if ip[0] >= len(in_toks):
            raise IndexError("truncated input")
        v = in_toks[ip[0]]
        ip[0] += 1
        return v

    try:
        n = int(inext())
        q = int(inext())
        if n <= 0 or q < 0 or q > n:
            fail("bad n/q")
        ratings = [int(inext()) for _ in range(n)]
        if len(set(ratings)) != n:
            fail("input ratings not distinct (generator bug)")
        T = n + q
        events = [inext() for _ in range(T)]
        if events.count('E') != n or events.count('C') != q:
            fail("input event counts inconsistent (generator bug)")
    except Exception as e:
        fail("malformed input: %r" % (e,))

    # ---- ground-truth simulation ----
    import heapq
    heap = []
    live_true = set()
    true_cuts = []
    live_masks = []  # bitmask of live set BEFORE each cut (includes the cut player)
    cur_mask = 0
    enroll_count = 0
    for ev in events:
        if ev == 'E':
            enroll_count += 1
            pid = enroll_count
            heapq.heappush(heap, (ratings[pid - 1], pid))
            live_true.add(pid)
            cur_mask |= (1 << pid)
        elif ev == 'C':
            while heap and heap[0][1] not in live_true:
                heapq.heappop(heap)
            r, pid = heapq.heappop(heap)
            live_true.discard(pid)
            true_cuts.append(pid)
            live_masks.append(cur_mask)
            cur_mask &= ~(1 << pid)
        else:
            fail("bad event token in input (generator bug)")

    # ---- read participant output ----
    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("cannot read output")

    op = [0]

    def onext():
        if op[0] >= len(out_toks):
            raise IndexError("truncated output")
        v = out_toks[op[0]]
        op[0] += 1
        return v

    try:
        claimed = [int(onext()) for _ in range(q)]
    except Exception:
        fail("cannot parse the %d claimed cuts" % q)

    if claimed != true_cuts:
        fail("claimed cut sequence does not match the true cut sequence")

    # ---- instruction stream: 'C i j' or 'M i' ----
    MAX_INSTR_TOKENS = 200 * max(1, n)  # generous DoS guard, well above any sane solution
    M_seen = set()
    edges_by_node = [[] for _ in range(n + 1)]
    seen_edge = set()
    cmp_count = 0
    move_count = 0
    consumed = 0
    while op[0] < len(out_toks):
        consumed += 1
        if consumed > MAX_INSTR_TOKENS:
            fail("too many instruction tokens (cap exceeded)")
        try:
            tok = onext()
        except Exception:
            fail("truncated instruction stream")
        if tok == 'M':
            try:
                i = int(onext())
            except Exception:
                fail("malformed M instruction")
            if not (1 <= i <= n):
                fail("M index out of range")
            if i in M_seen:
                fail("duplicate M for player %d" % i)
            M_seen.add(i)
            move_count += 1
        elif tok == 'C':
            try:
                i = int(onext())
                j = int(onext())
            except Exception:
                fail("malformed C instruction")
            if not (1 <= i <= n and 1 <= j <= n):
                fail("C index out of range")
            if i == j:
                fail("self-comparison")
            cmp_count += 1
            if cmp_count > 60 * n:
                fail("too many comparisons (cap exceeded)")
            ri, rj = ratings[i - 1], ratings[j - 1]
            lo, hi = (i, j) if ri < rj else (j, i)
            if (lo, hi) not in seen_edge:
                seen_edge.add((lo, hi))
                edges_by_node[lo].append(hi)
        else:
            fail("unknown instruction token %r" % (tok,))

    if len(M_seen) != n:
        fail("every player must get exactly one M entry (got %d of %d)" % (len(M_seen), n))

    # ---- transitive-closure reachability via bitset DP, processed high-rating -> low-rating ----
    order_by_rating = sorted(range(1, n + 1), key=lambda p: ratings[p - 1])
    reach = [0] * (n + 1)
    for p in reversed(order_by_rating):
        m = 0
        for c in edges_by_node[p]:
            m |= (1 << c) | reach[c]
        reach[p] = m

    # ---- verify certification of every cut ----
    for k in range(q):
        a = true_cuts[k]
        mask = live_masks[k]
        rivals = mask & ~(1 << a)
        if rivals & ~reach[a]:
            fail("cut #%d (player %d) is not certified by the filed comparisons" % (k + 1, a))

    cost = float(cmp_count) + 0.25 * float(move_count)

    # ---- baseline B: blind, re-searched binary-insertion of all n players (registration order) ----
    arr = []
    b_cmp = 0
    for pid in range(1, n + 1):
        r = ratings[pid - 1]
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            b_cmp += 1
            if arr[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        arr.insert(lo, r)
    B = 0.25 * n + b_cmp

    F = max(1e-9, cost)
    sc = min(1000.0, 100.0 * B / F)
    sys.stdout.write("cost=%.3f baseline=%.3f cmp=%d move=%d\n" % (cost, B, cmp_count, move_count))
    sys.stdout.write("Ratio: %.6f\n" % (sc / 1000.0))
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        sys.stdout.write("INFEASIBLE: unexpected error %r\n" % (e,))
        sys.stdout.write("Ratio: 0.0\n")
        sys.exit(0)
