#!/usr/bin/env python3
"""gen.py <testId> -- prints one Hydra-Combinator instance to stdout.

Deterministic in testId only (no external randomness). Builds a closed SKI term
containing:
  (a) an "attack" gadget: H nested S-duplicators wrapping a short I-chain -- the
      classic duplication trap (mechanism: duplication-cost-avoidance).
  (b) a "junk" gadget behind a K -- a term that is expensive to reduce but is
      *supposed* to be discarded untouched (mechanism: reduction-strategy-planning).
Both gadgets are combined as siblings under an inert (never-reducible) leaf head so
the whole term is a single connected instance.
"""
import sys

sys.setrecursionlimit(200000)


def mk_leaf(i):
    return ('L', i)


def is_app(n):
    return type(n) is tuple and n[0] == 'A'


def size(n):
    total = 0
    stack = [n]
    while stack:
        cur = stack.pop()
        total += 1
        if is_app(cur):
            stack.append(cur[1]); stack.append(cur[2])
    return total


def match_redex(n):
    if not is_app(n):
        return None
    f, a = n[1], n[2]
    if f == 'I':
        return ('I', a)
    if is_app(f) and f[1] == 'K':
        return ('K', f[2], a)
    if is_app(f) and is_app(f[1]) and f[1][1] == 'S':
        return ('S', f[1][2], f[2], a)
    return None


def fire(m):
    kind = m[0]
    if kind in ('K', 'I'):
        return m[1], 1
    X, Y, Z = m[1], m[2], m[3]
    return ('A', ('A', X, Z), ('A', Y, Z)), 1 + 2 * size(Z)


def leftmost_outermost_cost(term, step_cap=200000):
    total = 0
    steps = 0

    def find_and_fire(n):
        nonlocal total, steps
        m = match_redex(n)
        if m is not None:
            newn, c = fire(m)
            total += c
            steps += 1
            return newn, True
        if is_app(n):
            f2, ch = find_and_fire(n[1])
            if ch:
                return ('A', f2, n[2]), True
            a2, ch2 = find_and_fire(n[2])
            if ch2:
                return ('A', n[1], a2), True
            return n, False
        return n, False

    cur = term
    while True:
        if steps > step_cap:
            raise RuntimeError("step cap exceeded")
        cur, changed = find_and_fire(cur)
        if not changed:
            return cur, total


def cbv_cost(term, step_cap=200000):
    total = 0
    steps = 0

    def norm(n):
        nonlocal total, steps
        if not is_app(n):
            return n
        f2 = norm(n[1])
        a2 = norm(n[2])
        cur = ('A', f2, a2)
        while True:
            m = match_redex(cur)
            if m is None:
                return cur
            newn, c = fire(m)
            total += c
            steps += 1
            if steps > step_cap:
                raise RuntimeError("step cap exceeded")
            if not is_app(newn):
                return newn
            cur = ('A', norm(newn[1]), norm(newn[2]))

    nf = norm(term)
    return nf, total


def serialize(node):
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if is_app(n):
            out.append('@')
            stack.append(n[2])
            stack.append(n[1])
        elif type(n) is tuple and n[0] == 'L':
            out.append('L')
            out.append(str(n[1]))
        else:
            out.append(n)
    return ' '.join(out)


def i_chain(d, sub):
    n = sub
    for _ in range(d):
        n = ('A', 'I', n)
    return n


def s_dup_head(x, y, z):
    return ('A', ('A', ('A', 'S', x), y), z)


def hydra(H, d, leaf_id):
    """H nested duplicators wrapping an I-chain of length d over a fresh leaf."""
    n = i_chain(d, mk_leaf(leaf_id))
    for k in range(H):
        n = s_dup_head(mk_leaf(leaf_id + 1 + 2 * k), mk_leaf(leaf_id + 2 + 2 * k), n)
    return n


def build_junk(target_cost, leaf_start):
    """Return a term whose full (eager) reduction costs >= target_cost, built as a
    sibling-combination of small hydra chunks (greedy 'coin' decomposition over a
    doubling cost table) so every redex stays at SHALLOW tree depth -- unlike a long
    I-chain, this keeps checker replay near-linear even though the cost is large."""
    d_junk = 2
    table = []
    lid = leaf_start
    for H in range(1, 18):
        hy = hydra(H, d_junk, lid)
        lid += 2 * H + 1
        _, c = cbv_cost(hy)
        table.append((H, c))
    remaining = target_cost
    chunks = []
    lid2 = leaf_start + 10000
    for H, c in reversed(table):
        while c <= remaining:
            chunks.append(hydra(H, d_junk, lid2))
            lid2 += 2 * H + 1
            remaining -= c
    if not chunks:
        chunks.append(hydra(1, d_junk, lid2))
        lid2 += 3
    combined = ('A', mk_leaf(lid2 + 777), chunks[0])
    for ch in chunks[1:]:
        combined = ('A', combined, ch)
    return combined


def build_instance(t):
    H = 3 + (t - 1) // 2            # 3,3,4,4,5,5,6,6,7,7
    d = [3, 4, 5][(t - 1) % 3]      # cycles for variety
    C2 = 1.3

    attack = hydra(H, d, 1)
    _, cost_g0 = leftmost_outermost_cost(attack)
    target = C2 * cost_g0

    junk = build_junk(target, leaf_start=5000 + 37 * t)
    kjunk_gadget = ('A', ('A', 'K', mk_leaf(3)), junk)

    combiner = mk_leaf(999999)
    term = ('A', ('A', combiner, attack), kjunk_gadget)
    return term


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    term = build_instance(t)
    print(serialize(term))


if __name__ == "__main__":
    main()
