# TIER: strong
"""The insight: the checker's own cost function IS the clustering criterion,
so use it to drive the partition instead of a fixed key like "first token".

Start from ONE cluster holding every line (no assumption about which
position is a stable tag). Recursively: for the current cluster, compare the
cost of keeping it as a single template, W + n*v (v = #positions that vary
within the cluster), against splitting it on each candidate position p --
partition by the literal VALUE at p and sum W + n_g*v_g over the resulting
sub-groups (one level of lookahead). Take the position that reduces total
cost the most; if none helps, the cluster becomes a leaf template. Recurse
into the chosen split.

This directly defeats the position-0-collision trap: when two families share
a tag, splitting on position 0 does not help (it produces one group, no
reduction), but splitting on whatever position DOES separate the two
families (their diverging constant literal) collapses each side's cost
sharply -- the algorithm finds that position because it is scored by the
same objective the checker uses, not because it assumed position 0 mattered.
Positions that are genuinely variable *within* a true family are never worth
splitting on: partitioning by an int/hex value shatters the group into many
near-singleton templates, each paying a fresh W header, which costs far more
than just wildcarding that one position -- so the recursion stops exactly at
the true family boundaries."""
import sys
from collections import defaultdict


def variable_positions(idxs, lines, W):
    v = 0
    for p in range(W):
        first = lines[idxs[0]][p]
        for i in idxs[1:]:
            if lines[i][p] != first:
                v += 1
                break
    return v


def keep_cost(idxs, lines, W):
    return W + len(idxs) * variable_positions(idxs, lines, W)


def best_split(idxs, lines, W):
    """Return (cost, [group_idx_lists]) for the best single-position split,
    or None if no position actually separates the cluster."""
    best = None
    for p in range(W):
        groups = defaultdict(list)
        for i in idxs:
            groups[lines[i][p]].append(i)
        if len(groups) < 2:
            continue
        cost = sum(keep_cost(g, lines, W) for g in groups.values())
        if best is None or cost < best[0]:
            best = (cost, list(groups.values()))
    return best


def build(idxs, lines, W, out_templates, assign):
    if len(idxs) == 1:
        i = idxs[0]
        out_templates.append(list(lines[i]))
        assign[i] = len(out_templates)
        return
    c_keep = keep_cost(idxs, lines, W)
    sp = best_split(idxs, lines, W)
    if sp is None or sp[0] >= c_keep:
        tmpl = []
        for p in range(W):
            first = lines[idxs[0]][p]
            same = all(lines[i][p] == first for i in idxs[1:])
            tmpl.append(first if same else "*")
        out_templates.append(tmpl)
        tid = len(out_templates)
        for i in idxs:
            assign[i] = tid
        return
    for g in sp[1]:
        build(g, lines, W, out_templates, assign)


def main():
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); W = int(next(it))
    lines = [[next(it) for _ in range(W)] for _ in range(N)]

    templates = []
    assign = [0] * N
    build(list(range(N)), lines, W, templates, assign)

    out = [str(len(templates))]
    for tmpl in templates:
        out.append(" ".join(tmpl))
    out.append(" ".join(str(a) for a in assign))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
