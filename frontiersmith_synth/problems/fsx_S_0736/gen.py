#!/usr/bin/env python3
"""
gen.py <testId> -- deterministic instance generator for "Stall Menu DAG" (egraph-shared
extraction).  Prints ONE instance to stdout.  All randomness is seeded purely from testId.

Instance = a layered acyclic "class graph" (an e-graph): class 0 is the market root, every
other class i in 1..N-1 is a "stall" (equivalence class) offering M_i candidate "dishes".
Each dish has an own prep-cost and references zero or more STRICTLY HIGHER-INDEXED stalls as
required ingredient-stalls (children).  Any assignment of one dish per stall therefore induces
an acyclic dependency graph by construction (child index > parent index always), so no cycle
can ever arise regardless of which dishes are picked -- the checker still verifies this
defensively.

Two kinds of structure are planted among the "entry" stalls hanging directly off the root:

  * GENERIC stalls: candidates differ only in cost, not in which ingredient-stalls they need
    (different recipes for the same dependency set) -- picking the cheapest is always correct,
    no sharing subtlety.  Minor background texture.

  * FUNNEL / DECOY groups: g "top" stalls, each offering THREE dishes:
      0: WASTEFUL  -- own cost pW, no ingredients.  Always the worst choice (pW is set far
         above every other option), included purely to punish a "reproduce the checker's
         index-0 baseline" strategy without ever tempting a real optimizer.
      1: PRIVATE    -- own cost pA, one PRIVATE ingredient-stall (cost pl) used by nobody else.
      2: SHARED     -- own cost pB, one common "kernel" ingredient-stall (cost K) shared by the
         whole group.
    Constants satisfy pB + K > pA + pl (STRICTLY, by a wide margin) -- so evaluated stall by
    stall, PRIVATE always looks better than SHARED.  But the kernel is paid for only ONCE if
    the group jointly adopts SHARED, so for a group of size g the true joint costs are
    g*(pA+pl) [all-private] vs g*pB + K [all-shared], and for large-enough g the coalition wins
    even though every single member, judged alone, prefers PRIVATE.  "Beneficial" groups are
    sized comfortably above this break-even; "decoy" groups look identical but are sized
    comfortably below it (private really is better there) -- a solver must actually compute
    the trade-off, not pattern-match "shared kernel = always switch".
"""
import sys, random


def _group_economics(rnd):
    pA = rnd.randint(8, 12)
    pl = rnd.randint(9, 11)
    delta = rnd.randint(0, 3)
    pB = pA + delta                        # shared option's own cost is comparable to pA
    denom = pl - delta                     # 6 <= denom <= 11 given the ranges above
    K = denom + rnd.randint(15, 40)        # guarantees pB + K > pA + pl by 15..40 (local trap)
    pW = (pA + pl) * 3 + rnd.randint(0, 10)  # always the worst of the three, by a wide margin
    g_min = K // denom + 1                 # smallest g where the coalition strictly wins
    return dict(pA=pA, pl=pl, pB=pB, denom=denom, K=K, pW=pW, g_min=g_min)


def build(test_id: int):
    rnd = random.Random(900000 + test_id * 104729 + 17)

    plan = {
        1:  dict(n_generic=6,  groups=[]),
        2:  dict(n_generic=8,  groups=[False, False]),
        3:  dict(n_generic=8,  groups=[True, False]),
        4:  dict(n_generic=10, groups=[True, True, False]),
        5:  dict(n_generic=12, groups=[True, True, True, False, False]),
        6:  dict(n_generic=16, groups=[True, True, True, True, False, False]),
        7:  dict(n_generic=20, groups=[True, True, True, True, True, False, False, False]),
        8:  dict(n_generic=24, groups=[True] * 6 + [False] * 3),
        9:  dict(n_generic=28, groups=[True] * 8 + [False] * 4),
        10: dict(n_generic=32, groups=[True] * 10 + [False] * 5),
    }
    cfg = plan[max(1, min(10, test_id))]
    n_generic = cfg["n_generic"]
    group_flags = cfg["groups"]
    C = 4                                   # multiple of g_min for beneficial groups (fixed:
                                             # keeps the per-group savings fraction moderate so
                                             # `strong` never saturates, however many groups
                                             # a harder test case adds)

    groups = []
    for beneficial in group_flags:
        e = _group_economics(rnd)
        if beneficial:
            size = C * e["g_min"] + rnd.randint(0, 3)
        else:
            size = max(1, e["g_min"] - 1)
        e["size"] = size
        e["beneficial"] = beneficial
        groups.append(e)

    n_tops = sum(g["size"] for g in groups)
    n_kernels = len(groups)
    n_leaves_priv = n_tops
    n_leaf_pool = max(6, n_generic // 2 + 4)

    # ---- index layout (root=0, generic entries, top entries, kernels, private leaves,
    #      generic leaf pool) -- every block only ever referenced by an EARLIER block, so
    #      child index > parent index holds everywhere. ----
    idx = 0
    root = idx; idx += 1
    generic_idx = list(range(idx, idx + n_generic)); idx += n_generic
    top_idx = list(range(idx, idx + n_tops)); idx += n_tops
    kernel_idx = list(range(idx, idx + n_kernels)); idx += n_kernels
    leaf_priv_idx = list(range(idx, idx + n_leaves_priv)); idx += n_leaves_priv
    leaf_pool_idx = list(range(idx, idx + n_leaf_pool)); idx += n_leaf_pool
    N = idx

    classes = [None] * N

    entry_ids = generic_idx + top_idx
    rnd.shuffle(entry_ids)
    classes[root] = [(0, entry_ids)]

    for g in generic_idx:
        k = rnd.randint(1, 2)
        children = rnd.sample(leaf_pool_idx, k)
        m = rnd.randint(2, 3)
        costs = rnd.sample(range(4, 26), m)
        cands = [(c, list(children)) for c in costs]
        rnd.shuffle(cands)
        classes[g] = cands

    for lp in leaf_pool_idx:
        classes[lp] = [(rnd.randint(1, 6), [])]

    tp = 0
    lv = 0
    for gi, g in enumerate(groups):
        kernel = kernel_idx[gi]
        classes[kernel] = [(g["K"], [])]
        for _ in range(g["size"]):
            top = top_idx[tp]; tp += 1
            leaf = leaf_priv_idx[lv]; lv += 1
            pl = max(1, g["pl"] + rnd.randint(-1, 1))
            classes[leaf] = [(pl, [])]
            classes[top] = [
                (g["pW"], []),            # 0: wasteful
                (g["pA"], [leaf]),        # 1: private
                (g["pB"], [kernel]),      # 2: shared
            ]

    assert all(c is not None for c in classes)
    return N, [root], classes


def main():
    test_id = int(sys.argv[1])
    N, roots, classes = build(test_id)
    out = [f"{N} {len(roots)}", " ".join(map(str, roots))]
    for cands in classes:
        out.append(str(len(cands)))
        for cost, children in cands:
            out.append(f"{cost} {len(children)}" + ("" if not children else " " + " ".join(map(str, children))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
