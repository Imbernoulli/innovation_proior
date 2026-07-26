# TIER: strong
"""
The insight: confluence already guarantees WHICH normal form we land on, no matter
what order we pick -- so the only thing left to optimize is total step count, and
that is entirely governed by one exchange argument: never fire a redex whose
argument still contains a redex. Always contract the DEEPEST available redex first
(leftmost-innermost). Concretely this means, for every `dup`/`drop` node: fully
finish its argument before touching it.

Why this is optimal, not just "innermost, the other textbook order": every unit of
work (each `drop` removed, each `dup` fired) is unavoidable somewhere in any
complete reduction. The only way a strategy can pay for a unit of work MORE than
once is if it duplicates that unit before performing it (fires an enclosing `dup`
while the work still sits, unfinished, inside its argument). Innermost-first order
guarantees a `dup` never fires until its argument is already fully normalized --
so every duplicated copy is copied for free. That is the "shrink before you
duplicate" rule the confluence guarantee sets you free to apply in any order you
like.
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


_redex_cache = {}


def has_redex(t):
    key = id(t)
    v = _redex_cache.get(key)
    if v is not None:
        return v
    if t[0] in ('drop', 'dup'):
        res = True
    elif t[0] == 'u':
        res = False
    else:
        res = has_redex(t[1]) or has_redex(t[2])
    _redex_cache[key] = res
    return res


def find_innermost(t):
    if t[0] == 'pair':
        if has_redex(t[1]):
            return '0' + find_innermost(t[1])
        if has_redex(t[2]):
            return '1' + find_innermost(t[2])
        return None
    if t[0] in ('drop', 'dup'):
        if has_redex(t[1]):
            return '0' + find_innermost(t[1])
        return ''
    return None


def main():
    data = sys.stdin.read()
    _, text = data.split('\n', 1)
    term = Parser(tokenize(text.strip())).parse()
    history = [term]  # keepalive so id()-keyed memoization stays valid

    steps = []
    while True:
        p = find_innermost(term)
        if p is None:
            break
        steps.append(p)
        term = apply_at(term, p)
        history.append(term)

    out = [str(len(steps))]
    out.extend('.' if p == '' else p for p in steps)
    sys.stdout.write('\n'.join(out) + '\n')


main()
