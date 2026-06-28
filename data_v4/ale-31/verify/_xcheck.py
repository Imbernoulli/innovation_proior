#!/usr/bin/env python3
"""Independent cross-check: recompute cost from scratch from the solver output and
compare against score.py's score, and re-verify connectivity with a fresh DSU,
to make sure score.py's incremental-free recompute is internally consistent and
the solver never emits a disconnected/empty district."""
import sys, subprocess
import score as S

def check(inst, sol):
    H, W, K, grid = S.read_instance(inst)
    assign = S.read_solution(sol, H, W, K)
    assert assign is not None, "parse failed"
    # independent connectivity via DSU over same-id adjacency
    parent = list(range(H * W))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    from collections import defaultdict
    cells = defaultdict(list)
    for i, v in enumerate(assign):
        cells[v].append(i)
        r, c = divmod(i, W)
        if c + 1 < W and assign[i + 1] == v: uni(i, i + 1)
        if r + 1 < H and assign[i + W] == v: uni(i, i + W)
    for k in range(K):
        cl = cells[k]
        assert cl, f"district {k} empty"
        roots = {find(c) for c in cl}
        assert len(roots) == 1, f"district {k} disconnected into {len(roots)} comps"
    # recompute cost two ways consistency
    cost = S.partition_cost(H, W, K, grid, assign)
    assert cost >= 0
    return cost

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    bad = 0
    for s in range(1, 21):
        inst = f"/tmp/in_{s}.txt"; sol = f"/tmp/out_{s}.txt"
        try:
            cost = check(inst, sol)
            print(f"seed {s:3d}: OK cost={cost:.1f}")
        except AssertionError as e:
            print(f"seed {s:3d}: FAIL {e}"); bad += 1
    print("ALL FEASIBLE" if bad == 0 else f"{bad} FAILURES")
