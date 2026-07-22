#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic checker for Hydra-Combinator reduction.

Reads the SKI term from <in>. Reads the participant's reduction PLAN from <out>:
    R
    addr_1 addr_2 ... addr_R
(all whitespace separated; addr is '.' for the root or a nonempty string over {0,1}
navigating Function(0)/Argument(1) edges from the CURRENT root at that step).

Replays the plan against the input term using the FIXED S/K/I rewrite rules, validating
every step strictly. On any violation: `Ratio: 0.0`. Otherwise scores

    Ratio = min(1.0, KAPPA * B / max(1e-9, F))

where F is the total weighted operation count the plan actually paid (1 per K/I-step,
1 + 2*size(Z) per S-step -- the cost of materializing the duplicate of Z) and B is an
INTERNAL baseline the checker computes itself: the cost of the naive "always fully
reduce both sides before combining" (eager / call-by-value) strategy applied to the
SAME input. KAPPA = 0.14 is a fixed constant (not tuned per instance).
"""
import re
import sys

sys.setrecursionlimit(200000)

KAPPA = 0.14
STEP_CAP = 20000
ADDR_RE = re.compile(r'^(?:\.|[01]+)$')


def fail(msg):
    print("Ratio: 0.0  (%s)" % msg)
    sys.exit(0)


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


def has_any_redex(n):
    stack = [n]
    while stack:
        cur = stack.pop()
        if match_redex(cur) is not None:
            return True
        if is_app(cur):
            stack.append(cur[1]); stack.append(cur[2])
    return False


class ParseError(Exception):
    pass


def parse_tokens(tokens):
    i, n = 0, len(tokens)
    work = []
    root, root_done = None, False
    while i < n:
        if root_done:
            raise ParseError("trailing tokens after complete term")
        tok = tokens[i]
        if tok == '@':
            work.append([])
            i += 1
            continue
        elif tok == 'L':
            i += 1
            if i >= n:
                raise ParseError("truncated leaf")
            idtok = tokens[i]
            if not (idtok.isdigit() or (idtok[:1] == '-' and idtok[1:].isdigit())):
                raise ParseError("bad leaf id %r" % idtok)
            node = ('L', int(idtok))
            i += 1
        elif tok in ('S', 'K', 'I'):
            node = tok
            i += 1
        else:
            raise ParseError("bad token %r" % tok)
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
        raise ParseError("unclosed @ application(s)")
    return root


def get_at(node, addr):
    cur = node
    if addr == '.':
        return cur
    for ch in addr:
        if not is_app(cur):
            return None
        cur = cur[1] if ch == '0' else cur[2]
    return cur


def replace_at(node, addr, newsub):
    if addr == '.':
        return newsub

    def rec(cur, idx):
        if not is_app(cur):
            raise ParseError("address runs off tree")
        ch = addr[idx]
        if idx == len(addr) - 1:
            return ('A', newsub, cur[2]) if ch == '0' else ('A', cur[1], newsub)
        if ch == '0':
            return ('A', rec(cur[1], idx + 1), cur[2])
        return ('A', cur[1], rec(cur[2], idx + 1))

    return rec(node, 0)


def eager_baseline(term, step_cap=STEP_CAP):
    """Checker's own naive construction: fully normalize both children before ever
    checking whether the parent forms a redex (call-by-value to normal form)."""
    steps = [0]
    total = [0]

    def norm(n):
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
            total[0] += c
            steps[0] += 1
            if steps[0] > step_cap:
                raise RuntimeError("baseline step cap exceeded")
            if not is_app(newn):
                return newn
            cur = ('A', norm(newn[1]), norm(newn[2]))

    nf = norm(term)
    return nf, total[0]


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        in_tokens = f.read().split()
    try:
        term = parse_tokens(in_tokens)
    except ParseError as e:
        # malformed input file would be an authoring bug, not a participant fault --
        # but fail safe rather than crash.
        fail("bad input file: %s" % e)

    try:
        with open(outf) as f:
            out_tokens = f.read().split()
    except OSError:
        fail("cannot read output")

    if not out_tokens:
        fail("empty output")

    r_tok = out_tokens[0]
    if not (r_tok.isdigit() or (r_tok[:1] == '-' and r_tok[1:].isdigit())):
        fail("R is not an integer")
    try:
        R = int(r_tok)
    except ValueError:
        fail("R not parseable")
    if R < 0 or R > STEP_CAP:
        fail("R out of range [0, %d]" % STEP_CAP)
    if len(out_tokens) != 1 + R:
        fail("expected %d address tokens, got %d" % (R, len(out_tokens) - 1))

    addrs = out_tokens[1:]
    for a in addrs:
        if not ADDR_RE.match(a):
            fail("malformed address token %r" % a)

    cur = term
    F = 0
    for step_i, addr in enumerate(addrs):
        node = get_at(cur, addr)
        if node is None:
            fail("address %r runs off the term at step %d" % (addr, step_i))
        m = match_redex(node)
        if m is None:
            fail("no redex at address %r (step %d)" % (addr, step_i))
        newn, c = fire(m)
        try:
            cur = replace_at(cur, addr, newn)
        except ParseError as e:
            fail(str(e))
        F += c

    if has_any_redex(cur):
        fail("final term is not in normal form (reachable redex remains)")

    try:
        canonical_nf, B = eager_baseline(term)
    except RuntimeError as e:
        fail("internal baseline error: %s" % e)

    if cur != canonical_nf:
        fail("final term does not match the unique normal form")

    ratio = min(1.0, KAPPA * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
