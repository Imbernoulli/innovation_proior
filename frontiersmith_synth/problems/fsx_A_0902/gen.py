import sys, random

# gen.py <testId> -- prints ONE "Boot Arena" instance (firmware heap scheduling).
#
# The instance is a forest of W INDEPENDENT "sliding-window" CHAIN components. Component g
# is a sequence of m_g buffers b_1..b_{m_g}; buffer b_k depends on its own recent history via
# a WINDOW of width w_g: edges b_k -> b_{k+1}, ..., b_k -> b_{k+min(w_g, m_g-k)}, meaning
# alloc(b_k) < alloc(b_{k+d}) < free(b_k) for d=1..w_g. So b_k must stay allocated until the
# next w_g buffers in its OWN chain have been allocated -- a rolling window of forced
# concurrency WITHIN each chain (no per-buffer freedom about allocation order: a chain's own
# buffers must be allocated in their chain-relative order). Components never share an edge.
#
# This plants TWO layered traps:
#  1) Cross-component: buffer INDICES are assigned round-robin ACROSS the W components (not
#     one component fully before the next), so a scheduler that just processes indices in
#     the given order is forced to keep pieces of every chain resident at once, multiplying
#     the peak by (roughly) the number of components.
#  2) Within a single chain (even fully isolated from the other components), sizes are drawn
#     random per buffer, so on MANY (not all -- this depends on the random draw, same as any
#     real fragmentation workload) instances freeing a buffer the INSTANT it is legally
#     allowed to (the standard liveness heuristic) lets first-fit wedge a smaller, later
#     buffer into part of that hole -- leaving an unusable remainder that is too small for a
#     same-chain buffer that needs it soon after, forcing the arena to extend when it did not
#     have to. Delaying a handful of frees by a few steps (so several holes coalesce into one,
#     or so a mismatched small buffer is never offered the hole at all) avoids this without
#     ever violating any precedence constraint -- a genuine first-fit hole-shape effect that
#     plain "process each chain to completion, freeing eagerly" (an obvious DFS-by-component
#     recipe) does not resolve on the instances where it occurs, since that recipe still
#     frees eagerly inside each chain. The cross-component effect (1) alone already separates
#     greedy from strong on every generated case; effect (2) adds extra headroom on top of
#     that on the instances where the random sizes happen to create a mismatch.
#
# All randomness is seeded from testId only -> fully deterministic, reproducible.

SPECS = {
    1:  dict(W=2, mlo=8,  mhi=10, wlo=3, whi=4, lo=15, hi=90),
    2:  dict(W=2, mlo=10, mhi=12, wlo=3, whi=4, lo=18, hi=100),
    3:  dict(W=3, mlo=10, mhi=12, wlo=4, whi=5, lo=18, hi=110),
    4:  dict(W=3, mlo=11, mhi=13, wlo=4, whi=5, lo=20, hi=120),
    5:  dict(W=3, mlo=12, mhi=14, wlo=4, whi=5, lo=22, hi=130),
    6:  dict(W=3, mlo=13, mhi=15, wlo=4, whi=5, lo=24, hi=140),
    7:  dict(W=3, mlo=14, mhi=16, wlo=4, whi=5, lo=26, hi=150),
    8:  dict(W=3, mlo=15, mhi=17, wlo=4, whi=5, lo=28, hi=160),
    9:  dict(W=3, mlo=16, mhi=18, wlo=4, whi=5, lo=30, hi=170),
    10: dict(W=3, mlo=17, mhi=19, wlo=4, whi=5, lo=32, hi=180),
}


def main():
    tid = int(sys.argv[1])
    spec = SPECS[tid]
    rng = random.Random(910001 + 7919 * tid)
    W = spec['W']

    comp_lens = [rng.randint(spec['mlo'], spec['mhi']) for _ in range(W)]
    comp_windows = [rng.randint(spec['wlo'], spec['whi']) for _ in range(W)]

    sizes = {}
    comp_idx_lists = [[] for _ in range(W)]
    idx_counter = 0
    maxlen = max(comp_lens)
    for pos in range(maxlen):
        for g in range(W):
            if pos < comp_lens[g]:
                idx_counter += 1
                sizes[idx_counter] = rng.randint(spec['lo'], spec['hi'])
                comp_idx_lists[g].append(idx_counter)
    N = idx_counter

    edges = []
    for g in range(W):
        idxs = comp_idx_lists[g]
        w = comp_windows[g]
        m = len(idxs)
        for k in range(m):
            for d in range(1, w + 1):
                if k + d < m:
                    edges.append((idxs[k], idxs[k + d]))

    out = [str(N), ' '.join(str(sizes[i]) for i in range(1, N + 1)), str(len(edges))]
    for p, c in edges:
        out.append("%d %d" % (p, c))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
