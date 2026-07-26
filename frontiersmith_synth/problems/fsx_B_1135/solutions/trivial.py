# TIER: trivial
"""
Reproduces the checker's own reference baseline: leftmost-outermost reduction.
Always contract the shallowest live redex (root first, ties broken left). This is
the textbook "normal order" default -- it is guaranteed to reach the normal form
(the system is confluent+terminating so ANY legal order does), but every time it
fires a `dup` before that dup's argument has been cleaned up, it duplicates
whatever un-reduced `drop`-work still lives inside -- and then has to redo that
work once per copy, forever after.
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


def find_outermost(t):
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


def main():
    data = sys.stdin.read()
    _, text = data.split('\n', 1)
    term = Parser(tokenize(text.strip())).parse()

    steps = []
    while True:
        p = find_outermost(term)
        if p is None:
            break
        steps.append(p)
        term = apply_at(term, p)

    out = [str(len(steps))]
    out.extend('.' if p == '' else p for p in steps)
    sys.stdout.write('\n'.join(out) + '\n')


main()
