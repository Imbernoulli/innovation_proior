#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for the confluent term-rewriting
step-minimization problem.

<in>  : line1 = node count (informational), line2 = the term as an S-expression
        over {u, drop, dup, pair}.
<out> : line1 = M (number of rewrite steps), then M lines, each the POSITION of the
        node rewritten at that step: a string over {0,1} (path of child indices from
        the root), or "." for the root itself.

The checker replays the M steps against the term (rejecting any step that does not
address a live drop/dup redex), then requires the final term to contain NO drop/dup
node (i.e. it really is a normal form). Because the rewrite system is confluent and
terminating by construction (see gen.py docstring), reaching ANY redex-free term via
legal steps from the same start is proof enough that it is THE unique normal form --
the checker never needs to know or recompute that normal form itself.

Score (minimization): B = the checker's own reference step count under the naive
leftmost-outermost strategy (always contract the shallowest live redex, root first).
F = the participant's step count M. Ratio = min(1000, 100*B/F)/1000.
"""
import sys

MAX_STEPS = 200000
MAX_NODES_IN = 2_000_000


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def tokenize(text):
    return text.replace('(', ' ( ').replace(')', ' ) ').split()


class Parser:
    __slots__ = ('toks', 'i')

    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def next(self):
        if self.i >= len(self.toks):
            raise ValueError('unexpected EOF')
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self):
        tok = self.next()
        if tok == 'u':
            return ('u',)
        if tok != '(':
            raise ValueError('bad token %r' % tok)
        head = self.next()
        if head == 'drop':
            c = self.parse()
            if self.next() != ')':
                raise ValueError('expected )')
            return ('drop', c)
        if head == 'dup':
            c = self.parse()
            if self.next() != ')':
                raise ValueError('expected )')
            return ('dup', c)
        if head == 'pair':
            l = self.parse()
            r = self.parse()
            if self.next() != ')':
                raise ValueError('expected )')
            return ('pair', l, r)
        raise ValueError('bad head %r' % head)


def parse_term(text):
    p = Parser(tokenize(text))
    t = p.parse()
    if p.i != len(p.toks):
        raise ValueError('trailing tokens')
    return t


def count_nodes(t):
    if t[0] == 'u':
        return 1
    if t[0] in ('drop', 'dup'):
        return 1 + count_nodes(t[1])
    return 1 + count_nodes(t[1]) + count_nodes(t[2])


def apply_at(t, path):
    """Apply the rule at `path` (a string over '0'/'1', '' = root). Raises ValueError
    on any illegal step (bad path / not a redex)."""
    if path == '':
        if t[0] == 'drop':
            return t[1]
        if t[0] == 'dup':
            return ('pair', t[1], t[1])
        raise ValueError('no redex at this position (root is %s)' % t[0])
    c = path[0]
    if t[0] == 'pair':
        if c == '0':
            return ('pair', apply_at(t[1], path[1:]), t[2])
        elif c == '1':
            return ('pair', t[1], apply_at(t[2], path[1:]))
        raise ValueError('bad path char %r' % c)
    if t[0] in ('drop', 'dup'):
        if c == '0':
            return (t[0], apply_at(t[1], path[1:]))
        raise ValueError('bad path char %r for unary node' % c)
    raise ValueError('path runs past a leaf')


def find_outermost(t):
    """Leftmost-outermost redex search: the shallowest drop/dup, ties broken left.
    Used ONLY to compute the checker's own reference baseline B."""
    if t[0] in ('drop', 'dup'):
        return ''
    if t[0] == 'pair':
        r = find_outermost(t[1])
        if r is not None:
            return '0' + r
        r = find_outermost(t[2])
        if r is not None:
            return '1' + r
        return None
    return None


def outermost_cost(t):
    steps = 0
    while True:
        p = find_outermost(t)
        if p is None:
            return steps
        t = apply_at(t, p)
        steps += 1
        if steps > MAX_STEPS:
            # should never happen on generated instances; defensive only
            return steps


def has_redex(t, cache):
    key = id(t)
    v = cache.get(key)
    if v is not None:
        return v
    if t[0] in ('drop', 'dup'):
        res = True
    elif t[0] == 'u':
        res = False
    else:
        res = has_redex(t[1], cache) or has_redex(t[2], cache)
    cache[key] = res
    return res


def main():
    try:
        in_text = open(sys.argv[1]).read()
    except Exception:
        fail('cannot read input')

    lines = in_text.split('\n', 1)
    if len(lines) < 2:
        fail('malformed input')
    try:
        term = parse_term(lines[1].strip())
    except Exception as e:
        fail('unparsable input term (%s)' % e)

    if count_nodes(term) > MAX_NODES_IN:
        fail('input too large')

    # ---- internal baseline B: naive leftmost-outermost strategy on the SAME term ----
    B = outermost_cost(term)
    B = max(1, B)

    # ---- parse participant output ----
    try:
        out_text = open(sys.argv[2]).read()
    except Exception:
        fail('cannot read output')

    out_lines = out_text.split('\n')
    if not out_lines or out_lines[0].strip() == '':
        fail('empty output')
    try:
        m = int(out_lines[0].strip())
    except Exception:
        fail('bad step count header')
    if m < 0 or m > MAX_STEPS:
        fail('step count %d out of bounds' % m)
    if len(out_lines) - 1 < m:
        fail('fewer step lines than declared (%d < %d)' % (len(out_lines) - 1, m))

    steps = []
    for i in range(m):
        raw = out_lines[1 + i].strip()
        if raw == '.':
            path = ''
        else:
            if raw == '' or any(ch not in '01' for ch in raw):
                fail('bad position token %r at step %d' % (raw, i + 1))
            path = raw
        steps.append(path)

    # ---- replay ----
    cur = term
    for i, path in enumerate(steps):
        try:
            cur = apply_at(cur, path)
        except Exception as e:
            fail('illegal step %d at position %r (%s)' % (i + 1, path, e))

    # ---- feasibility: must be a genuine normal form (no drop/dup left anywhere) ----
    cache = {}
    if has_redex(cur, cache):
        fail('did not reach normal form (redex remains after %d steps)' % m)

    F = m
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("B=%d F=%d Ratio: %.6f" % (B, F, sc / 1000.0))


if __name__ == '__main__':
    main()
