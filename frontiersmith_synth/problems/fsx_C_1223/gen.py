#!/usr/bin/env python3
"""gen.py <testId> -- print ONE panic-mode-recovery instance to stdout.

A tiny toy language: statements are `A ;` (atom), `( chunk ) ;` (paren group,
chunk = comma-separated items), `{ stmt* }` (block), or `F ( chunk ; chunk ;
chunk ) stmt` (for-loop: header then one body statement). Some tokens are
corrupted and printed as `?` -- never a legal token anywhere.

Each testId prints K independent corrupted "programs" (token streams). The
generator composes short, independently-checkable FRAGMENTS: plain OK filler
(no error), a plain isolated `?` at statement level (easy: any reasonable
sync strategy recovers it), and three TRAP fragments engineered so that the
naive "always resync to the next `;`" strategy either (a) swallows a nearby
opening bracket whose own closer later shows up as a phantom mismatch, or
(b) swallows a second nearby `?` outright (a suppressed/missed error).
Fragments are individually self-closing (return to the top-level statement
context), so concatenating them in any order is always well-formed apart
from the deliberately injected `?` tokens. Difficulty (K, injected-error
density, trap density, nesting depth) increases with testId. Seeded by
testId (and a per-program sub-seed) only -> bit-for-bit reproducible.
"""
import sys
import random

# ---- hand-verified trap fragments (see AUTHOR NOTES) ----------------------
# Each fragment is a self-contained token list that starts and ends at the
# top-level statement context. `errs` = number of true `?` tokens it plants.

def frag_paren_swallow():
    # "(  ?  )  {  ;  A  }" : error inside a bare paren group, immediately
    # followed by a block whose OWN closing `}` sits just past an internal
    # `;`. Skip-to-`;`-only recovery swallows the paren's `)` AND the
    # block's `{`, lands on the internal `;`, then the block's real `}`
    # shows up with nothing open -> a phantom. A paren-context sync set of
    # {`,`,`)`} finds the paren's own `)` immediately -> no phantom.
    return ['(', '?', ')', '{', ';', 'A', '}'], 1


def frag_paren_suppress():
    # "(  ?  ,  ?  )  ;" : two errors close together inside one paren group.
    # Skip-to-`;`-only recovery jumps straight past BOTH `?`s (and the `)`)
    # to the trailing `;`, reporting only the first -> the second is
    # silently suppressed (cascading-error-suppression). A paren-context
    # sync set containing `,` stops at the comma and finds both.
    return ['(', '?', ',', '?', ')', ';'], 2


def frag_forhdr_swallow():
    # "F  (  ?  )  {  A  ;  }" : error right at the start of a for-header
    # (empty header). Skip-to-`;`-only swallows the header's `)` AND the
    # body block's `{`, lands on the block's internal `;`, then the block's
    # real `}` shows up with nothing open -> a phantom. A for-header sync
    # set containing `)` finds its own close immediately -> no phantom.
    return ['F', '(', '?', ')', '{', 'A', ';', '}'], 1


TRAP_FRAGS = [frag_paren_swallow, frag_paren_suppress, frag_forhdr_swallow]

OK_FRAGS = [
    lambda: ['A', ';'],
    lambda: ['(', 'A', ')', ';'],
    lambda: ['{', 'A', ';', '}'],
    lambda: ['F', '(', 'A', ';', 'A', ';', 'A', ')', 'A', ';'],
]


def frag_plain_error():
    return ['?', ';'], 1


def wrap_block(tokens, depth):
    for _ in range(depth):
        tokens = ['{'] + tokens + ['}']
    return tokens


def build_program(rng, err_seg_lo, err_seg_hi, trap_prob, wrap_lo, wrap_hi):
    n_err_segs = rng.randint(err_seg_lo, err_seg_hi)
    segs = []
    true_errs = 0
    for _ in range(n_err_segs):
        if rng.random() < trap_prob:
            frag_fn = rng.choice(TRAP_FRAGS)
            toks, e = frag_fn()
        else:
            toks, e = frag_plain_error()
        segs.append(list(toks))
        true_errs += e
        # occasional harmless filler right after a trap, for realism
        if rng.random() < 0.35:
            segs.append(list(rng.choice(OK_FRAGS)()))
    n_ok = rng.randint(1, 3)
    for _ in range(n_ok):
        segs.append(list(rng.choice(OK_FRAGS)()))
    rng.shuffle(segs)
    tokens = []
    for s in segs:
        tokens += s
    depth = rng.randint(wrap_lo, wrap_hi)
    tokens = wrap_block(tokens, depth)
    return tokens, true_errs


# (K, err_seg_lo, err_seg_hi, trap_prob, wrap_lo, wrap_hi)
SPECS = [
    (5, 1, 2, 0.00, 0, 0),   # 1  trivial: isolated statement-level errors only
    (5, 2, 3, 0.05, 0, 0),   # 2  easy
    (6, 2, 3, 0.25, 0, 1),   # 3
    (6, 3, 4, 0.35, 0, 1),   # 4
    (6, 3, 4, 0.45, 0, 1),   # 5
    (7, 3, 5, 0.55, 0, 1),   # 6
    (7, 4, 5, 0.60, 1, 1),   # 7
    (7, 4, 6, 0.65, 1, 2),   # 8
    (8, 4, 6, 0.70, 1, 2),   # 9  hard
    (8, 5, 7, 0.75, 1, 2),   # 10 hardest / adversarial
]


def main():
    tid = int(sys.argv[1])
    spec = SPECS[(tid - 1) % len(SPECS)]
    K, elo, ehi, tprob, wlo, whi = spec
    base_seed = 900001 + 97 * tid
    out = [str(K)]
    for p in range(K):
        rng = random.Random(base_seed * 1000003 + p * 7919 + 13)
        toks, _true_errs = build_program(rng, elo, ehi, tprob, wlo, whi)
        out.append(" ".join(toks))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
