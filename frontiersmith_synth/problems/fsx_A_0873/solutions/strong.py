# TIER: strong
"""Hybrid strategy: fire K/I the instant they are recognized at the head (never touch
what a K is about to discard), but for an S-redex, first fully normalize the Z argument
(using this SAME strategy, recursively) *before* firing S -- because Z is about to be
physically duplicated, and duplicating it small is cheaper than duplicating it big and
paying to re-normalize both copies. This is the one-line deviation from plain
leftmost-outermost that the trap is built to punish: "reason about what is about to be
copied," not just "reduce the outermost thing you can see."
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


def strong_emit(term):
    addrs = []

    def norm(n, prefix):
        cur = n
        while True:
            m = match_redex(cur)
            if m is not None:
                if m[0] == 'S':
                    X, Y, Z = m[1], m[2], m[3]
                    Zr = norm(Z, prefix + '1')
                    newn, _c = fire(('S', X, Y, Zr))
                else:
                    newn, _c = fire(m)
                addrs.append(prefix if prefix else '.')
                cur = newn
                continue
            if is_app(cur):
                f2 = norm(cur[1], prefix + '0')
                if f2 != cur[1]:
                    cur = ('A', f2, cur[2])
                    continue
                a2 = norm(cur[2], prefix + '1')
                return ('A', f2, a2)
            return cur

    norm(term, '')
    return addrs


def main():
    tokens = sys.stdin.read().split()
    term = parse_tokens(tokens)
    addrs = strong_emit(term)
    out = [str(len(addrs))]
    out.extend(addrs)
    sys.stdout.write(' '.join(out) + '\n')


if __name__ == "__main__":
    main()
