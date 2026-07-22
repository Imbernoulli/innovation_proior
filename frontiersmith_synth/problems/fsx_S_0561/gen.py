#!/usr/bin/env python3
# Generator for fsx_S_0561 -- scrambled-circuit-core-recovery (format D)
#
# Emits an OBFUSCATED boolean circuit on n inputs to stdout. The circuit computes
# some target function f; the participant must output ANY circuit computing the
# SAME f with as few gates as possible. The obfuscation composes four layers:
#   (t1) syntactic identity padding      -- a naive peephole pass removes it
#   (t2) globally-constant dead logic     -- built from DISTANT structurally-different
#                                            nodes so no local x&~x rule fires; only
#                                            a functional (truth-table) view sees it is 0
#   (t3) redundant recomputation          -- a subfunction cloned via De Morgan and
#                                            XOR'd back; only functional-signature
#                                            equivalence across distant nodes reveals it
#   (t4) irreducible structural expansion -- De Morgan / XOR blow-ups that neither a
#                                            peephole nor a fold+CSE pass contracts
#                                            (they leave real head-room above `strong`)
#
# Circuit text schema (also produced by solutions, parsed by counter.py):
#   line 1:            n g
#   next g lines:      one gate each, gate k (0-indexed) has node id (n+k)
#                        "AND a b" | "OR a b" | "XOR a b"   (a,b are earlier node ids)
#                        "NOT a"                            (a earlier node id)
#                        "CONST0" | "CONST1"
#                      inputs are node ids 0..n-1 (x0..x_{n-1})
#   last line:         "OUTPUT r"   (r any node id 0..n+g-1)
#
# Deterministic: everything seeded from testId only.

import sys
import random


def input_tts(n):
    N = 1 << n
    M = (1 << N) - 1
    tts = []
    for i in range(n):
        half = 1 << i
        col = ((1 << half) - 1) << half     # bits [half,2half)=1 in first period
        w = 2 * half
        while w < N:
            col |= col << w
            w <<= 1
        tts.append(col & M)
    return tts, M


class Builder:
    def __init__(self, n, tts, M):
        self.n = n
        self.M = M
        self.tt = list(tts)          # tt[id] = truth-table integer of node id
        self.gates = []              # gate k -> (op, a, b)

    def add(self, op, a=-1, b=-1):
        M = self.M
        if op == 'CONST0':
            t = 0
        elif op == 'CONST1':
            t = M
        elif op == 'NOT':
            t = self.tt[a] ^ M
        elif op == 'AND':
            t = self.tt[a] & self.tt[b]
        elif op == 'OR':
            t = self.tt[a] | self.tt[b]
        elif op == 'XOR':
            t = self.tt[a] ^ self.tt[b]
        else:
            raise ValueError(op)
        idd = self.n + len(self.gates)
        self.gates.append((op, a, b))
        self.tt.append(t)
        return idd

    # ---- structural NOT of a node, built WITHOUT a literal NOT-of-that-node ----
    # pushes negation down via De Morgan so the result is a distant subgraph whose
    # truth table is ~tt(node) but which a syntactic peephole cannot recognise.
    def dm_not(self, node, budget):
        if node < self.n or budget[0] <= 0:
            budget[0] -= 1
            return self.add('NOT', node)
        budget[0] -= 1
        op, a, b = self.gates[node - self.n]
        if op == 'AND':
            return self.add('OR', self.dm_not(a, budget), self.dm_not(b, budget))
        if op == 'OR':
            return self.add('AND', self.dm_not(a, budget), self.dm_not(b, budget))
        if op == 'XOR':
            return self.add('XOR', a, self.dm_not(b, budget))
        if op == 'NOT':
            return a
        if op in ('CONST0', 'CONST1'):
            return self.add('CONST1' if op == 'CONST0' else 'CONST0')
        return self.add('NOT', node)

    # ---- structural CLONE of a node (same truth table, different structure) ----
    def clone_eq(self, node, budget):
        if node < self.n or budget[0] <= 0:
            return node
        budget[0] -= 1
        op, a, b = self.gates[node - self.n]
        if op == 'AND':                       # a&b = ~(~a | ~b)
            return self.add('NOT', self.add('OR', self.dm_not(a, budget), self.dm_not(b, budget)))
        if op == 'OR':                        # a|b = ~(~a & ~b)
            return self.add('NOT', self.add('AND', self.dm_not(a, budget), self.dm_not(b, budget)))
        if op == 'XOR':                       # a^b = ~a ^ ~b
            return self.add('XOR', self.dm_not(a, budget), self.dm_not(b, budget))
        if op == 'NOT':                       # ~a  = dm_not(a)
            return self.dm_not(a, budget)
        return node

    # ---- add a 2-input gate, optionally as an irreducible structural expansion --
    def add2(self, op, a, b, expand):
        if not expand:
            return self.add(op, a, b)
        if op == 'AND':                       # ~(~a | ~b)
            return self.add('NOT', self.add('OR', self.add('NOT', a), self.add('NOT', b)))
        if op == 'OR':                        # ~(~a & ~b)
            return self.add('NOT', self.add('AND', self.add('NOT', a), self.add('NOT', b)))
        if op == 'XOR':                       # (a & ~b) | (~a & b)
            t1 = self.add('AND', a, self.add('NOT', b))
            t2 = self.add('AND', self.add('NOT', a), b)
            return self.add('OR', t1, t2)
        return self.add(op, a, b)

    # ---- t1 identity padding on a wire (syntactically removable) ----
    def pad(self, w, rng):
        style = rng.randrange(5)
        if style == 0:                        # w ^ 0
            return self.add('XOR', w, self.add('CONST0'))
        if style == 1:                        # w & w
            return self.add('AND', w, w)
        if style == 2:                        # ~~w
            return self.add('NOT', self.add('NOT', w))
        if style == 3:                        # w & 1
            return self.add('AND', w, self.add('CONST1'))
        return self.add('OR', w, self.add('CONST0'))   # w | 0


def build(testId):
    n = [13, 13, 14, 14, 14, 15, 15, 15, 16, 16][(testId - 1) % 10]
    rng = random.Random(0x9E3779B1 * testId + 12345 * testId * testId + 7)
    tts, M = input_tts(n)
    B = Builder(n, tts, M)

    # cached input literals (xi and ~xi), so the core has a genuine, irreducible
    # functional signature (an XOR-tree of AND-terms does NOT collapse under a
    # fold+CSE pass -- this is what gives `strong` a stable, sizable gate count).
    notcache = {}

    def lit():
        i = rng.randrange(n)
        if rng.random() < 0.5:
            return i
        if i not in notcache:
            notcache[i] = B.add('NOT', i)
        return notcache[i]

    # ---------- core body: XOR-tree of (2-or-3 input) AND-terms ------------------
    k = 12 + testId                       # number of product terms
    core2 = []                            # ids of core AND/XOR gates (for t3 h)
    terms = []
    seen = set()
    guard = 0
    while len(terms) < k and guard < 40 * k:
        guard += 1
        la, lb = lit(), lit()
        if la == lb:
            continue
        key = tuple(sorted((la, lb)))
        if key in seen:
            continue
        seen.add(key)
        t = B.add2('AND', la, lb, rng.random() < 0.5)
        if rng.random() < 0.35:           # sometimes a 3-literal product
            t = B.add2('AND', t, lit(), rng.random() < 0.5)
        core2.append(t)
        terms.append(t)

    # balanced XOR-tree over the product terms
    while len(terms) > 1:
        nxt = []
        for i in range(0, len(terms) - 1, 2):
            g = B.add2('XOR', terms[i], terms[i + 1], rng.random() < 0.5)
            core2.append(g)
            nxt.append(g)
        if len(terms) % 2 == 1:
            nxt.append(terms[-1])
        terms = nxt
    out = terms[0]

    # a couple of mid-path syntactic pads (t1) that are genuinely used downstream
    for _ in range(3 + testId // 3):
        out = B.pad(out, rng)

    # ---------- t2: globally-constant dead logic (distant-node structure) --------
    n_dead = 5 + testId
    for _ in range(n_dead):
        sub = list(range(n))
        s = 8 + testId
        for _ in range(s):
            op = ('AND', 'OR', 'XOR')[rng.randrange(3)]
            a = sub[rng.randrange(len(sub))]
            b = sub[rng.randrange(len(sub))]
            sub.append(B.add(op, a, b))
        p = sub[-1]
        npc = B.dm_not(p, [s + 6])       # ~p as a distant subgraph
        dead = B.add('AND', p, npc)      # p & ~p == 0  (not syntactically obvious)
        out = B.add('OR', out, dead)     # OR 0  ->  no change

    # ---------- t3: redundant recomputation revealed only by functional signature -
    n_clone = 3 + testId // 2
    for _ in range(n_clone):
        if not core2:
            break
        h = core2[rng.randrange(len(core2))]
        q = B.clone_eq(h, [6])           # structurally different, tt(q)==tt(h)
        redund = B.add('XOR', h, q)      # h ^ q == 0
        out = B.add('OR', out, redund)

    # ---------- t1: output identity chain (syntactically removable) --------------
    chain = 12 + 2 * testId
    for _ in range(chain):
        out = B.pad(out, rng)

    return n, B.gates, out


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, gates, out = build(testId)
    lines = ["%d %d" % (n, len(gates))]
    for (op, a, b) in gates:
        if op in ('AND', 'OR', 'XOR'):
            lines.append("%s %d %d" % (op, a, b))
        elif op == 'NOT':
            lines.append("NOT %d" % a)
        else:
            lines.append(op)
    lines.append("OUTPUT %d" % out)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
