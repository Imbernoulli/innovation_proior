# TIER: greedy
"""
The "obviously smarter" recipe: eagerly collapse every `drop` you can find first
(scan the whole tree, leftmost `drop` wins, repeat), since collapsing only ever
shrinks the term and can't hurt. Only once no `drop` remains anywhere does it fall
back to naive leftmost-outermost order among the remaining `dup` nodes.

This fixes HALF the problem (it never duplicates un-collapsed `drop`-work) but
misses the other half: it still fires `dup` nodes shallowest-first, so a `dup`
whose argument still contains an un-fired INNER `dup` duplicates that inner `dup`
too -- forcing it to fire twice, then that copy's own nested dup's fire twice each,
etc. Collapsing early is necessary but not sufficient; the dup/dup nesting order
matters just as much as the drop/dup order.
"""
import sys


def tokenize(text):
    return text.replace('(', ' ( ').replace(')', ' ) ').split()


class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self):
        tok = self.next()
        if tok == 'u':
            return ('u',)
        head = self.next()
        if head == 'drop':
            c = self.parse(); self.next(); return ('drop', c)
        if head == 'dup':
            c = self.parse(); self.next(); return ('dup', c)
        if head == 'pair':
            l = self.parse(); r = self.parse(); self.next(); return ('pair', l, r)
        raise ValueError('bad term')


def apply_at(t, path):
    if path == '':
        if t[0] == 'drop':
            return t[1]
        return ('pair', t[1], t[1])
    c = path[0]
    if t[0] == 'pair':
        if c == '0':
            return ('pair', apply_at(t[1], path[1:]), t[2])
        return ('pair', t[1], apply_at(t[2], path[1:]))
    return (t[0], apply_at(t[1], path[1:]))


_drop_cache = {}


def has_drop(t):
    key = id(t)
    v = _drop_cache.get(key)
    if v is not None:
        return v
    if t[0] == 'drop':
        res = True
    elif t[0] == 'u':
        res = False
    elif t[0] == 'dup':
        res = has_drop(t[1])
    else:
        res = has_drop(t[1]) or has_drop(t[2])
    _drop_cache[key] = res
    return res


def find_drop_anywhere(t):
    if t[0] == 'drop':
        return ''
    if t[0] == 'dup':
        if has_drop(t[1]):
            return '0' + find_drop_anywhere(t[1])
        return None
    if t[0] == 'pair':
        if has_drop(t[1]):
            return '0' + find_drop_anywhere(t[1])
        if has_drop(t[2]):
            return '1' + find_drop_anywhere(t[2])
        return None
    return None


def find_outermost_dup(t):
    if t[0] == 'dup':
        return ''
    if t[0] == 'pair':
        r = find_outermost_dup(t[1])
        if r is not None:
            return '0' + r
        r = find_outermost_dup(t[2])
        if r is not None:
            return '1' + r
    return None


def main():
    data = sys.stdin.read()
    _, text = data.split('\n', 1)
    term = Parser(tokenize(text.strip())).parse()
    # keep every intermediate root alive so id()-keyed memoization never sees a
    # recycled address from a garbage-collected tuple
    history = [term]

    steps = []
    while True:
        p = find_drop_anywhere(term)
        if p is None:
            p = find_outermost_dup(term)
        if p is None:
            break
        steps.append(p)
        term = apply_at(term, p)
        history.append(term)

    out = [str(len(steps))]
    out.extend('.' if p == '' else p for p in steps)
    sys.stdout.write('\n'.join(out) + '\n')


main()
