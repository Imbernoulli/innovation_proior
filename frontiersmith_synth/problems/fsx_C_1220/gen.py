#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of the let-generalization inference task.

Instance = a set of DEFINITION constraints (in generation order) over local type
variables t0..t(k-1) of a single let-bound definition, plus a batch of held-out
USE sites, each asserting equations that a correctly-instantiated occurrence of
the definition's type scheme must satisfy.

Deterministic: all randomness seeded purely from testId.
"""
import random
import sys

PALETTE = [
    ["int"], ["bool"], ["L", "int"], ["L", "bool"],
    ["P", "int", "bool"], ["P", "bool", "int"],
    ["F", "int", "bool"], ["F", "bool", "int"],
]


def render(tok_list):
    return " ".join(tok_list)


def link_eq(a, b, rng):
    """One definition equation that forces t_a and t_b into the same group."""
    style = rng.randrange(3)
    if style == 0:
        return "t%d = t%d" % (a, b)
    if style == 1:
        return "L t%d = L t%d" % (a, b)
    return "P t%d int = P t%d int" % (a, b)


def params(tid):
    rng = random.Random(900000 + tid * 7919)
    n_pinned = 2 if tid <= 2 else (2 if tid <= 4 else 1)
    n_free = 1 if tid == 1 else (2 if tid == 2 else min(2 + (tid - 2), 6))
    uses_per_free_base = 2 if tid <= 2 else min(2 + (tid - 2), 6)
    volatility = 0.0 if tid <= 2 else min(0.18 * (tid - 2), 1.0)
    conflict_p = 0.0 if tid <= 3 else min(0.10 * (tid - 3), 0.40)
    size_weights = [1] if tid <= 2 else [1, 2, 3]
    return rng, n_pinned, n_free, uses_per_free_base, volatility, conflict_p, size_weights


def main():
    tid = int(sys.argv[1])
    rng, n_pinned, n_free, uses_per_free, volatility, conflict_p, size_weights = params(tid)

    defs = []
    uses = []  # list of (list_of_eq_strings)
    var_ctr = [0]

    def alloc(sz):
        idxs = list(range(var_ctr[0], var_ctr[0] + sz))
        var_ctr[0] += sz
        return idxs

    pinned_groups = []  # (indices, ground_tokens)
    for _ in range(n_pinned):
        idxs = alloc(1)
        gt = PALETTE[rng.randrange(len(PALETTE))]
        pinned_groups.append((idxs, gt))
        defs.append("t%d = %s" % (idxs[0], render(gt)))

    free_groups = []  # (indices, anchor_tokens, volatility)
    for gi in range(n_free):
        sz = rng.choice(size_weights)
        idxs = alloc(sz)
        for j in range(1, sz):
            defs.append(link_eq(idxs[j - 1], idxs[j], rng))
        anchor = PALETTE[rng.randrange(len(PALETTE))]
        # first free group always low-volatility so small testIds have a clean warm-up
        gvol = 0.0 if (tid <= 2 or gi == 0 and tid <= 3) else volatility
        free_groups.append((idxs, anchor, gvol))

    # ---- uses: 1 sanity use per pinned group (always satisfiable) ----
    for idxs, gt in pinned_groups:
        uses.append(["t%d = %s" % (idxs[0], render(gt))])

    # ---- uses: several per free group, mixing anchor-consistent / varied / conflicting ----
    for idxs, anchor, gvol in free_groups:
        n_use = uses_per_free
        for _ in range(n_use):
            if len(idxs) >= 2 and rng.random() < conflict_p:
                a = rng.choice(idxs)
                b_choices = [x for x in idxs if x != a]
                b = rng.choice(b_choices)
                ta = PALETTE[rng.randrange(len(PALETTE))]
                tb = PALETTE[rng.randrange(len(PALETTE))]
                # only counts as a genuine trap-of-nature (unsatisfiable-by-anyone) if they differ
                if ta == tb:
                    tb = PALETTE[(PALETTE.index(tb) + 1) % len(PALETTE)]
                uses.append(["t%d = %s" % (a, render(ta)), "t%d = %s" % (b, render(tb))])
            else:
                v = rng.choice(idxs)
                if rng.random() < gvol:
                    req = PALETTE[rng.randrange(len(PALETTE))]
                else:
                    req = anchor
                uses.append(["t%d = %s" % (v, render(req))])

    # planted: one use that NO valid scheme can satisfy (a pinned slot asked for
    # the wrong ground type) -- guards every test case, including the easy
    # warm-ups, against saturating a sound solution's score at a perfect 1.0.
    pg_idxs, pg_gt = pinned_groups[0]
    wrong = PALETTE[(PALETTE.index(pg_gt) + 3) % len(PALETTE)]
    uses.append(["t%d = %s" % (pg_idxs[0], render(wrong))])

    k = var_ctr[0]
    m = len(defs)
    u = len(uses)

    out = []
    out.append("%d %d %d" % (k, m, u))
    out.extend(defs)
    for i, eqs in enumerate(uses, 1):
        out.append("USE %d %d" % (i, len(eqs)))
        out.extend(eqs)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
