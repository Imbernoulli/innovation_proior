import sys

# Format D checker -- "Boot Arena: Scheduling a Firmware Heap".
#
# The participant submits ONE linear order of 2N alloc/free events for N buffers whose
# precedence is given by a DAG (edge p->c means alloc(p) < alloc(c) < free(p)).
#   1) Parse the instance (sizes + precedence edges) from <in>.
#   2) Parse the participant's order from <out>: exactly 2N nonzero integers, |v| in [1,N];
#      +i = alloc buffer i, -i = free buffer i.  EXACT feasibility gate: every buffer
#      allocated exactly once and freed exactly once, after its own alloc, and every DAG
#      edge's alloc(p) < alloc(c) < free(p) constraint holds.  Any violation -> Ratio 0.0.
#   3) Replay the order through a FIXED deterministic first-fit-with-coalescing allocator
#      (a classic "wilderness chunk" bump allocator: new blocks reuse the lowest first-fit
#      free hole, or extend the top; adjacent free blocks coalesce; a coalesced block that
#      touches the current top retracts the top). Objective (minimize) = the historical
#      high-water mark F = the largest address ever touched.
#   4) Baseline B = the "allocate everything, free nothing until the very end" construction
#      (always feasible, reuses no hole ever) = sum of all sizes.
#      Ratio = min(1, 0.1 * B / F).  Trivial ~ 0.1; 10x tighter packing caps at 1.0.

MAXN = 2000        # matches statement.md's stated N bound
MAXM_PER_N = 4      # matches statement.md's stated M <= 4*N bound
MAXSIZE = 10 ** 6   # matches statement.md's stated 1 <= s_i <= 10^6 bound


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        data = open(sys.argv[1]).read().split()
    except Exception:
        fail("no input")
    it = iter(data)
    try:
        N = int(next(it))
        sizes = [0] * (N + 1)
        for i in range(1, N + 1):
            sizes[i] = int(next(it))
        M = int(next(it))
        edges = []
        for _ in range(M):
            p = int(next(it))
            c = int(next(it))
            edges.append((p, c))
    except Exception:
        fail("bad instance")
    if N < 1 or N > MAXN or any(not (1 <= sizes[i] <= MAXSIZE) for i in range(1, N + 1)):
        fail("bad sizes")
    if M < 0 or M > MAXM_PER_N * N:
        fail("bad M")
    for p, c in edges:
        if not (1 <= p < c <= N):
            fail("bad edge (must satisfy 1<=p<c<=N)")

    try:
        out_txt = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(out_txt) != 2 * N:
        fail("expected %d tokens, got %d" % (2 * N, len(out_txt)))

    ops = []
    try:
        for tok in out_txt:
            v = int(tok)
            if v == 0 or abs(v) > N:
                fail("token out of range: %s" % tok)
            ops.append(v)
    except ValueError:
        fail("non-integer token")

    alloc_pos = [None] * (N + 1)
    free_pos = [None] * (N + 1)
    seen_alloc = [False] * (N + 1)
    seen_free = [False] * (N + 1)
    for pos, v in enumerate(ops):
        i = abs(v)
        if v > 0:
            if seen_alloc[i]:
                fail("duplicate alloc %d" % i)
            seen_alloc[i] = True
            alloc_pos[i] = pos
        else:
            if not seen_alloc[i]:
                fail("free before alloc: %d" % i)
            if seen_free[i]:
                fail("duplicate free %d" % i)
            seen_free[i] = True
            free_pos[i] = pos
    for i in range(1, N + 1):
        if not seen_alloc[i] or not seen_free[i]:
            fail("buffer %d missing alloc/free" % i)

    for p, c in edges:
        if not (alloc_pos[p] < alloc_pos[c] < free_pos[p]):
            fail("precedence violated for edge %d->%d" % (p, c))

    # --- deterministic first-fit allocator with coalescing (wilderness reclaim) ---
    free_list = []  # sorted list of (start, size), maximal (non-adjacent) disjoint holes
    top = 0
    peak = 0
    block = [0] * (N + 1)

    def do_alloc(i):
        nonlocal top, peak
        sz = sizes[i]
        for k in range(len(free_list)):
            fs, fsz = free_list[k]
            if fsz >= sz:
                block[i] = fs
                if fsz == sz:
                    free_list.pop(k)
                else:
                    free_list[k] = (fs + sz, fsz - sz)
                if fs + sz > peak:
                    peak = fs + sz
                return
        block[i] = top
        top += sz
        if top > peak:
            peak = top

    def do_free(i):
        nonlocal top
        s = block[i]
        sz = sizes[i]
        idx = 0
        while idx < len(free_list) and free_list[idx][0] < s:
            idx += 1
        free_list.insert(idx, (s, sz))
        if idx + 1 < len(free_list):
            ns, nsz = free_list[idx]
            rs, rsz = free_list[idx + 1]
            if ns + nsz == rs:
                free_list[idx] = (ns, nsz + rsz)
                free_list.pop(idx + 1)
        if idx - 1 >= 0:
            ls, lsz = free_list[idx - 1]
            ns, nsz = free_list[idx]
            if ls + lsz == ns:
                free_list[idx - 1] = (ls, lsz + nsz)
                free_list.pop(idx)
                idx -= 1
        ns, nsz = free_list[idx]
        if ns + nsz == top:
            top = ns
            free_list.pop(idx)

    for v in ops:
        i = abs(v)
        if v > 0:
            do_alloc(i)
        else:
            do_free(i)

    F = peak if peak > 0 else 1
    B = sum(sizes[1:N + 1])
    if B < 1:
        B = 1

    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("N=%d peak=%d baseline=%d Ratio: %.6f" % (N, F, B, ratio))


if __name__ == '__main__':
    main()
