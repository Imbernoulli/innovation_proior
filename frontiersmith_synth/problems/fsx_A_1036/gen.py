#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the weighted-DFA-chorus passphrase problem.

Deterministic: everything is seeded from testId alone.

Construction (kept out of the statement, discovered only from the raw transition
tables): each DFA is a "prefix acceptor with a dead sink" -- states 0..L-1 walk a
fixed required symbol sequence req[0..L-1]; any wrong symbol falls into an absorbing
non-accepting sink state; matching the whole sequence lands in an absorbing accepting
state (so once accepted, a DFA stays accepted no matter what follows).

Every instance contains:
  - a nested "chain" of 3 DFAs (c1 subset c2 subset c3 as required prefixes, i.e. c1 is
    a prefix of c2 which is a prefix of c3) that are all mutually satisfiable by one
    sufficiently long string,
  - a "spoiler" DFA whose required prefix conflicts with the chain's root symbol, so
    NO string can satisfy the spoiler and any chain member at once (this keeps the full
    m-way intersection provably empty for every test case),
  - on TRAP cases, the spoiler is made the single heaviest DFA (weight > every chain
    member alone, but < the chain's combined weight) -- a "heaviest DFA first" heuristic
    locks onto the spoiler and can never recover the (better) chain,
  - on HARMONY cases, the spoiler is deliberately light (lighter than every chain
    member), so a heaviest-first heuristic naturally lands on the chain and the greedy
    recipe roughly matches the informed optimum,
  - a few small independent "distractor" DFAs on the larger cases, for texture.
"""
import random
import sys

A = 3  # shared alphabet size {0,1,2}

# (m, Lmax, is_trap)
CASE_CFG = {
    1: (4, 10, False),
    2: (4, 11, True),
    3: (5, 13, False),
    4: (5, 14, True),
    5: (6, 16, False),
    6: (6, 17, True),
    7: (7, 19, True),
    8: (7, 20, False),
    9: (8, 22, True),
    10: (8, 23, False),
}


def make_dfa(req, weight):
    """Prefix-acceptor DFA: states 0..L-1 in progress, state L = absorbing accept,
    state L+1 = absorbing dead sink."""
    L = len(req)
    accept_idx = L
    sink_idx = L + 1
    trans = []
    for t in range(L):
        row = [sink_idx] * A
        row[req[t]] = t + 1
        trans.append(row)
    trans.append([accept_idx] * A)
    trans.append([sink_idx] * A)
    return {"n": L + 2, "start": 0, "accept": [accept_idx], "w": weight, "trans": trans}


def rand_seq(rng, length, forbid_first=None):
    first = rng.randrange(A)
    if forbid_first is not None:
        choices = [s for s in range(A) if s != forbid_first]
        first = rng.choice(choices)
    return [first] + [rng.randrange(A) for _ in range(length - 1)]


def build_case(test_id):
    m, Lmax, trap = CASE_CFG[test_id]
    rng = random.Random(20260000 + 97 * test_id)

    # --- nested chain: c1 (len3) is a prefix of c2, c2 is a prefix of c3 ---
    inc = max(2, (Lmax - 3) // 3)
    c1_req = rand_seq(rng, 3)
    c2_req = c1_req + [rng.randrange(A) for _ in range(inc)]
    c3_req = c2_req + [rng.randrange(A) for _ in range(inc)]
    assert len(c3_req) <= Lmax - 1

    # --- spoiler: conflicts with the chain root at position 0 ---
    spoil_len = rng.choice([2, 3])
    spoil_req = rand_seq(rng, spoil_len, forbid_first=c1_req[0])

    if trap:
        h_w = 28 + rng.randint(0, 6)          # spoiler is the single heaviest DFA
        chain_w = [h_w - 1, h_w - 3, h_w - 5]  # each individually lighter than spoiler,
        spoil_w = h_w                          # but their SUM (~2.8x h_w) beats it
    else:
        chain_w = [rng.randint(18, 26) for _ in range(3)]
        spoil_w = rng.randint(3, max(3, min(chain_w) - 1))  # strictly lighter than every chain DFA

    dfas = [
        make_dfa(spoil_req, spoil_w),
        make_dfa(c1_req, chain_w[0]),
        make_dfa(c2_req, chain_w[1]),
        make_dfa(c3_req, chain_w[2]),
    ]

    n_extra = m - 4
    for _ in range(n_extra):
        dl = rng.randint(2, min(6, Lmax - 1))
        dw = rng.randint(2, max(2, spoil_w))
        d_req = rand_seq(rng, dl)
        dfas.append(make_dfa(d_req, dw))

    rng.shuffle(dfas)

    # sanity: confirm the FULL m-way intersection is genuinely empty (spoiler conflicts
    # with the chain root at position 0, so no string can satisfy the spoiler together
    # with any chain member -- hence not with all m DFAs at once).
    def conflicts(req_a, req_b):
        k = min(len(req_a), len(req_b))
        return req_a[:k] != req_b[:k]
    assert conflicts(spoil_req, c1_req)

    return m, Lmax, dfas


def print_case(test_id):
    m, Lmax, dfas = build_case(test_id)
    out = [f"{m} {A} {Lmax}"]
    for d in dfas:
        out.append(f"{d['n']} {d['start']} {d['w']}")
        out.append(f"{len(d['accept'])} " + " ".join(map(str, d["accept"])))
        for row in d["trans"]:
            out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1])
    print_case(tid)
