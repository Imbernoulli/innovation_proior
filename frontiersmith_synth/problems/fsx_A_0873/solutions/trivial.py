# TIER: trivial
"""Eager (call-by-value / innermost) reduction: fully normalize BOTH sides of every
application before ever checking whether the result forms a redex. This is the other
"obviously correct" first instinct -- reduce arguments before you use them -- and it
happens to sidestep the duplication trap (it always shrinks an S-redex's z argument
before the S ever fires, since z gets normalized as a child first). But it pays for that
by wastefully normalizing whatever sits on the *discarded* side of a K-redex, since it
never checks for K until after both sides are already fully reduced. This is exactly the
checker's own internal baseline construction.
"""
import sys

sys.setrecursionlimit(200000)


def is_app(n):
    return type(n) is tuple and n[0] == 'A'


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


def size(n):
    total = 0
    stack = [n]
    while stack:
        cur = stack.pop()
        total += 1
        if is_app(cur):
            stack.append(cur[1]); stack.append(cur[2])
    return total


def fire(m):
    kind = m[0]
    if kind in ('K', 'I'):
        return m[1], 1
    X, Y, Z = m[1], m[2], m[3]
    return ('A', ('A', X, Z), ('A', Y, Z)), 1 + 2 * size(Z)


class ParseError(Exception):
    pass


def parse_tokens(tokens):
    i, n = 0, len(tokens)
    work = []
    root, root_done = None, False
    while i < n:
        if root_done:
            raise ParseError("trailing tokens")
        tok = tokens[i]
        if tok == '@':
            work.append([]); i += 1; continue
        elif tok == 'L':
            i += 1
            node = ('L', int(tokens[i])); i += 1
        else:
            node = tok; i += 1
        while True:
            if not work:
                root, root_done = node, True
                break
            work[-1].append(node)
            if len(work[-1]) == 2:
                f, a = work[-1]
                node = ('A', f, a)
                work.pop()
                continue
            break
    if not root_done:
        raise ParseError("unclosed")
    return root


def cbv_emit(term):
    addrs = []

    def norm(n, prefix):
        if not is_app(n):
            return n
        f2 = norm(n[1], prefix + '0')
        a2 = norm(n[2], prefix + '1')
        cur = ('A', f2, a2)
        while True:
            m = match_redex(cur)
            if m is None:
                return cur
            newn, _c = fire(m)
            addrs.append(prefix if prefix else '.')
            if not is_app(newn):
                return newn
            cur = ('A', norm(newn[1], prefix + '0'), norm(newn[2], prefix + '1'))

    norm(term, '')
    return addrs


def main():
    tokens = sys.stdin.read().split()
    term = parse_tokens(tokens)
    addrs = cbv_emit(term)
    out = [str(len(addrs))]
    out.extend(addrs)
    sys.stdout.write(' '.join(out) + '\n')


if __name__ == "__main__":
    main()
