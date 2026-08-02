#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE handshake-automaton instance to stdout.

Instance format:
    m
    L
    len s_1 ... s_len        (repeated L times: legal traces -- MUST be accepted)
    F
    len s_1 ... s_len        (repeated F times: forbidden traces -- MUST be rejected)

Construction (seeded ONLY by testId, fully deterministic):
  - A small set of "head" motifs H_1..H_p and "tail" motifs T_1..T_k are drawn from a
    usable alphabet {0..m-2}. Symbol (m-1) -- the ATTACK symbol -- never occurs in any
    head/tail motif; it is reserved for the rollback-exploit forbidden traces below.
  - Heads are split into 2 compatibility PATTERNS (i % 2). Each pattern names a subset
    of tails it may legally complete with. Legal traces L = { H_i + T_j : pattern of i
    allows tail j }. Two heads sharing a pattern are behaviourally IDENTICAL from that
    point on (same allowed continuations) -- the intended state-merge-equivalence
    opportunity: a correct minimizer can represent both heads' futures with ONE shared
    sub-automaton instead of duplicating it once per head.
  - Forbidden traces F combine four rollback/attack shapes:
      * downgrade  : H_i + T_j for a (i,j) pair pattern(i) does NOT allow (protocol
                      version / cipher downgrade to an incompatible tail)
      * replay     : a legal trace with its own tail segment repeated
      * reorder    : a legal trace with one adjacent pair of symbols transposed
                      (out-of-order delivery)
      * rollback   : H_i + ATTACK + (a full legal trace) -- injects the reserved ATTACK
                      symbol mid-handshake then resumes with an untouched legal
                      completion. Present ONLY on designated "trap" testIds. Because the
                      ATTACK symbol is used ON NO legal trace anywhere, ANY automaton
                      that (for undefined symbols) falls back to routing back to its own
                      start state -- instead of a dedicated dead/reject state -- will
                      wrongly re-run the appended legal trace from scratch and ACCEPT
                      this forbidden trace.
"""
import random
import sys


def params(test_id):
    p = 4 + (test_id - 1) // 4          # 4,4,4,4,5,5,5,5,6,6  (>=2 heads per pattern)
    k = 3 + (test_id - 1) // 5          # 3,3,3,3,3,4,4,4,4,4
    m = 4 + (1 if test_id >= 6 else 0) + (1 if test_id >= 9 else 0)   # 4..4,5..5,6,6
    hl = 2                              # keep head cost (unmergeable) small
    tl = 3 + (1 if test_id >= 6 else 0)  # tail length -- deeper shared subtrees
    trap = test_id in (3, 6, 8, 10)
    return p, k, m, hl, tl, trap


def gen_distinct_motifs(rng, count, length, alphabet):
    out = []
    seen = set()
    tries = 0
    while len(out) < count:
        tries += 1
        cand = tuple(rng.choice(alphabet) for _ in range(length))
        if cand not in seen or tries > 200 * count:
            seen.add(cand)
            out.append(cand)
    return out


def make_pattern_bits(rng, k):
    bits = []
    for _ in range(2):
        b = [rng.random() < 0.65 for _ in range(k)]
        if not any(b):
            b[0] = True
        bits.append(b)
    if bits[0] == bits[1]:
        flip = 0
        bits[1][flip] = not bits[1][flip]
        if not any(bits[1]):
            bits[1][(flip + 1) % k] = True
    return bits


def main():
    test_id = int(sys.argv[1])
    rng = random.Random(900001 * test_id + 137)
    p, k, m, hl, tl, trap = params(test_id)
    usable = list(range(m - 1))     # 0..m-2 ; symbol m-1 is ATTACK, reserved
    attack = m - 1

    heads = gen_distinct_motifs(rng, p, hl, usable)
    tails = gen_distinct_motifs(rng, k, tl, usable)
    pattern_bits = make_pattern_bits(rng, k)

    # compatibility: head i uses pattern (i % 2)
    def compatible(i, j):
        return pattern_bits[i % 2][j]

    L = []
    L_set = set()
    for i in range(p):
        for j in range(k):
            if compatible(i, j):
                tr = heads[i] + tails[j]
                if tr not in L_set:
                    L_set.add(tr)
                    L.append(tr)

    F = []
    F_set = set()

    def add_forbidden(tr):
        if tr and tr not in L_set and tr not in F_set:
            F_set.add(tr)
            F.append(tr)

    # 1) downgrade: incompatible (head, tail) splices
    incompat = [(i, j) for i in range(p) for j in range(k) if not compatible(i, j)]
    rng.shuffle(incompat)
    for (i, j) in incompat[:2]:
        add_forbidden(heads[i] + tails[j])

    # 2) replay: duplicate the tail segment of a legal trace
    legal_idx = list(range(len(L)))
    rng.shuffle(legal_idx)
    replay_src = []
    for i in range(p):
        for j in range(k):
            if compatible(i, j):
                replay_src.append((i, j))
    rng.shuffle(replay_src)
    for (i, j) in replay_src[:2]:
        add_forbidden(heads[i] + tails[j] + tails[j])

    # 3) reorder: transpose one adjacent differing pair inside a legal trace
    for idx in legal_idx[:3]:
        tr = list(L[idx])
        done = False
        for pos in range(len(tr) - 1):
            if tr[pos] != tr[pos + 1]:
                tr2 = tr[:]
                tr2[pos], tr2[pos + 1] = tr2[pos + 1], tr2[pos]
                add_forbidden(tuple(tr2))
                done = True
                break
        if done:
            continue

    # 4) rollback / reset-exploit -- trap testIds only
    if trap:
        anchor = L[0]
        heads_used = list(range(p))
        rng.shuffle(heads_used)
        cnt = 0
        for i in heads_used:
            add_forbidden(heads[i] + (attack,) + anchor)
            cnt += 1
            if cnt >= 2:
                break

    # safety floor: make sure F is never empty
    if not F:
        add_forbidden(heads[0] + (attack,) + L[0])

    out = [str(m), str(len(L))]
    for tr in L:
        out.append(str(len(tr)) + " " + " ".join(map(str, tr)))
    out.append(str(len(F)))
    for tr in F:
        out.append(str(len(tr)) + " " + " ".join(map(str, tr)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
