#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>  -- deterministic scorer for "Stall Menu DAG".

<in>  : the instance (trusted, generator-produced).
<out> : the participant artifact -- exactly N whitespace-separated integers, line i (0-indexed)
        is the chosen candidate-dish index for stall i.
<ans> : unused empty placeholder.

Validates strictly (any violation -> "Ratio: 0.0"), then computes:
  F = total prep-cost of the induced term-DAG reachable from the roots under the participant's
      choices, where each REACHABLE stall's chosen dish own-cost is counted EXACTLY ONCE no
      matter how many other stalls reference it (this is the whole point: cost is a property of
      the selected SET, not of the individual per-stall choices).
  B = the same quantity for the checker's own trivial baseline (candidate index 0 everywhere).
Prints  Ratio: min(1, 0.1 * B / F)   (minimization: fewer total prep-cost units is better).
"""
import sys


def fail(reason):
    sys.stderr.write(f"# {reason}\n")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---------------- parse the (trusted) instance ----------------
    try:
        toks = open(in_path).read().split()
        pos = 0

        def nxt():
            nonlocal pos
            v = toks[pos]; pos += 1
            return v

        N = int(nxt()); R = int(nxt())
        roots = [int(nxt()) for _ in range(R)]
        classes = []  # classes[i] = [(cost:int, children:list[int]), ...]
        for i in range(N):
            M = int(nxt())
            cands = []
            for _k in range(M):
                cost = int(nxt())
                L = int(nxt())
                children = [int(nxt()) for _ in range(L)]
                cands.append((cost, children))
            classes.append(cands)
    except Exception as e:
        fail(f"malformed instance: {e}")

    # ---------------- parse the (untrusted) participant output ----------------
    try:
        data = open(out_path, "r", errors="replace").read()
    except Exception:
        fail("cannot read output")

    if len(data) > 50_000_000:
        fail("output too large")

    out_toks = data.split()
    if len(out_toks) != N:
        fail(f"expected {N} tokens, got {len(out_toks)}")

    choice = []
    for t in out_toks:
        if len(t) > 32:
            fail("token too long")
        try:
            v = int(t)
        except Exception:
            fail(f"non-integer token {t!r}")
        choice.append(v)

    for i in range(N):
        M = len(classes[i])
        if choice[i] < 0 or choice[i] >= M:
            fail(f"choice out of range at stall {i}")

    # ---------------- reachability + cycle-guard + cost, iterative DFS ----------------
    WHITE, GRAY, BLACK = 0, 1, 2

    def total_cost(sel, get_cands):
        color = [WHITE] * N

        def dfs(src):
            if color[src] != WHITE:
                return
            color[src] = GRAY
            stack = [(src, 0, get_cands(src)[sel(src)][1])]
            while stack:
                node, i2, children = stack[-1]
                if i2 < len(children):
                    stack[-1] = (node, i2 + 1, children)
                    c = children[i2]
                    if color[c] == WHITE:
                        color[c] = GRAY
                        stack.append((c, 0, get_cands(c)[sel(c)][1]))
                    elif color[c] == GRAY:
                        fail("cycle detected in induced term DAG")
                else:
                    color[node] = BLACK
                    stack.pop()

        for r in roots:
            dfs(r)
        return sum(get_cands(i)[sel(i)][0] for i in range(N) if color[i] == BLACK)

    F = total_cost(lambda i: choice[i], lambda i: classes[i])
    if F <= 0:
        fail("nonpositive total cost")

    B = total_cost(lambda i: 0, lambda i: classes[i])
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
