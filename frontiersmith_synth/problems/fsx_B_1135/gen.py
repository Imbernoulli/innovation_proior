#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE instance of the confluent term-rewriting-normalizer
problem to stdout. Deterministic: seeded only from testId.

Term signature (fixed across every test case, and restated in statement.md):
  u              leaf atom
  (drop T)       unary, rule: drop(x) -> x                (collapses; strictly shrinks)
  (dup  T)       unary, rule: dup(x)  -> (pair x x)        (duplicates x)
  (pair T1 T2)   binary constructor, NO rule (already normal form)

The system is non-overlapping / left-linear (orthogonal): drop and dup are the only
reducible root symbols, they never share a position, and pair has no rule at all --
so there are zero critical pairs and the system is confluent by construction. Every
instance is also terminating (each reduction step is a copy/removal of material
already present in the original finite term -- no rule ever manufactures a fresh
drop/dup symbol out of thin air), so every legal rewriting order reaches the SAME
unique normal form; only the number of steps taken differs.

Instance = a forest of independent "chains" dup^d( drop^k( u ) ) combined with pair
into one tree. d is capped at 3 (keeps the checker's outermost-strategy baseline from
exploding past the innovation-headroom cap); k and the chain COUNT grow with testId
to scale instance size ("large" per spec) without touching the d cap.
"""
import sys, random


def mk_chain(d, k):
    t = ('u',)
    for _ in range(k):
        t = ('drop', t)
    for _ in range(d):
        t = ('dup', t)
    return t


def combine(chains):
    if len(chains) == 1:
        return chains[0]
    return ('pair', chains[0], combine(chains[1:]))


def size(t):
    if t[0] == 'u':
        return 1
    if t[0] in ('drop', 'dup'):
        return 1 + size(t[1])
    return 1 + size(t[1]) + size(t[2])


def to_sexpr(t):
    if t[0] == 'u':
        return 'u'
    if t[0] == 'drop':
        return '(drop %s)' % to_sexpr(t[1])
    if t[0] == 'dup':
        return '(dup %s)' % to_sexpr(t[1])
    return '(pair %s %s)' % (to_sexpr(t[1]), to_sexpr(t[2]))


def build_instance(tid, rng):
    num_chains = 3 + tid
    d_base = 2 if tid <= 4 else 3
    k_base = 2 + min(3, (tid - 1) // 3)
    chains = []
    for _ in range(num_chains):
        d = max(1, min(3, d_base + rng.choice([-1, 0, 0, 1])))
        k = max(1, min(6, k_base + rng.choice([-1, 0, 0, 1])))
        chains.append(mk_chain(d, k))
    rng.shuffle(chains)
    return combine(chains)


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260726 + 97 * tid)
    term = build_instance(tid, rng)
    n = size(term)
    sys.stdout.write(str(n) + "\n")
    sys.stdout.write(to_sexpr(term) + "\n")


if __name__ == '__main__':
    main()
