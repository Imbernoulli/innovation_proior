#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance of the Morphogen-Golf problem.

A "creature" is the turtle drawing of a string produced by a HIDDEN uniform
L-system (a length-preserving morphism phi over the drawing alphabet {F,+,-},
prolongable at the axiom F) iterated n times:  T = phi^n(F).

The instance we PRINT is the developed transcript T (space-separated tokens),
NOT the rule.  The solver must find a short gene (L-system) that grows to the
same FORM.  Everything is seeded by testId only -> fully deterministic.
"""
import sys, random

TERMS = ['F', '+', '-']
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# testId -> (modulus m, iterations n).  small -> large; >=5 large "trap" cases.
LADDER = {
    1: (2, 8),
    2: (2, 9),
    3: (3, 5),
    4: (2, 10),
    5: (2, 11),
    6: (3, 6),
    7: (2, 12),
    8: (2, 13),
    9: (3, 7),
    10: (2, 14),
}


def expand(axiom, rules, n, cap=4_000_000):
    cur = list(axiom)
    for _ in range(n):
        nxt = []
        for s in cur:
            r = rules.get(s)
            if r is None:
                nxt.append(s)
            else:
                nxt.extend(r)
            if len(nxt) > cap:
                return None
        cur = nxt
    return cur


def draw_edges(tokens):
    x = y = 0
    h = 0
    stack = []
    edges = set()
    for t in tokens:
        if t == 'F':
            dx, dy = DIRS[h]
            nx, ny = x + dx, y + dy
            edges.add(frozenset(((x, y), (nx, ny))))
            x, y = nx, ny
        elif t == '+':
            h = (h + 1) % 4
        elif t == '-':
            h = (h + 3) % 4
        elif t == '[':
            stack.append((x, y, h))
        elif t == ']':
            if stack:
                x, y, h = stack.pop()
    return edges


def recover(T):
    """General automatic-sequence recovery of a uniform morphism from T=phi^n(a).
    Returns (axiom, rules, n, m) or None. Used to VALIDATE recoverability in gen."""
    L = len(T)
    for m in range(2, 6):
        n = 0
        p = 1
        while p < L:
            p *= m
            n += 1
        if p != L or n == 0:
            continue
        a = T[0]
        first = {}
        for i, s in enumerate(T):
            if s not in first:
                first[s] = i
        rules = {}
        okflag = True
        for s, k in first.items():
            if (k + 1) * m > L:
                okflag = False
                break
            rules[s] = list(T[k * m:(k + 1) * m])
        if not okflag:
            continue
        if expand([a], rules, n) == list(T):
            return (a, rules, n, m)
    return None


def make_instance(test_id):
    m, n = LADDER[test_id]
    for attempt in range(20000):
        rng = random.Random(test_id * 7919 + attempt * 131 + 17)
        rules = {}
        for s in TERMS:
            rules[s] = [rng.choice(TERMS) for _ in range(m)]
        rules['F'][0] = 'F'  # prolongable at axiom F
        T = expand(['F'], rules, n)
        if T is None:
            continue
        if len(T) != m ** n:
            continue
        nF = sum(1 for t in T if t == 'F')
        nturn = sum(1 for t in T if t in ('+', '-'))
        if nF < 8 or nturn < 4:
            continue
        edges = draw_edges(T)
        if len(edges) < 16:
            continue
        xs = [p[0] for e in edges for p in e]
        ys = [p[1] for e in edges for p in e]
        if (max(xs) - min(xs)) < 3 or (max(ys) - min(ys)) < 3:
            continue
        # the whole POINT: the hidden rule must be recoverable by a general method
        rec = recover(T)
        if rec is None or rec[3] != m:
            continue
        return T
    raise RuntimeError("gen: no valid instance for test %d" % test_id)


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if test_id not in LADDER:
        test_id = ((test_id - 1) % 10) + 1
    T = make_instance(test_id)
    out = sys.stdout
    out.write("%d\n" % len(T))
    out.write(" ".join(T))
    out.write("\n")


if __name__ == "__main__":
    main()
