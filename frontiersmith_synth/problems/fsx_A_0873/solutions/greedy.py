# TIER: greedy
"""Leftmost-outermost (normal-order) reduction: the textbook, PROVABLY-SAFE strategy --
it is the one strategy guaranteed to reach a normal form whenever one exists, so it is
the natural "obvious, correct, safe" thing to submit. It always fires the outermost
available redex, including an S-redex, the instant the S x y z shape is recognized --
without ever looking inside z first. On instances where z is still a large unreduced
blob, this duplicates that blob whole, and then repeats the duplication recursively on
every remaining head of the hydra beneath it.
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


def leftmost_outermost_emit(term):
    addrs = []

    def find_and_fire(n, prefix):
        m = match_redex(n)
        if m is not None:
            newn, _c = fire(m)
            addrs.append(prefix if prefix else '.')
            return newn, True
        if is_app(n):
            f2, ch = find_and_fire(n[1], prefix + '0')
            if ch:
                return ('A', f2, n[2]), True
            a2, ch2 = find_and_fire(n[2], prefix + '1')
            if ch2:
                return ('A', n[1], a2), True
            return n, False
        return n, False

    cur = term
    while True:
        cur, changed = find_and_fire(cur, '')
        if not changed:
            break
    return addrs


def main():
    tokens = sys.stdin.read().split()
    term = parse_tokens(tokens)
    addrs = leftmost_outermost_emit(term)
    out = [str(len(addrs))]
    out.extend(addrs)
    sys.stdout.write(' '.join(out) + '\n')


if __name__ == "__main__":
    main()
