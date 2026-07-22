#!/usr/bin/env python3
"""gen.py <testId> -- prints one Gray-Adjacent State Encoding instance to stdout.

Hidden construction (never revealed on stdout):
  - Pick B (state bits) from a fixed difficulty ladder indexed by testId; K=2 always
    (M=4 input symbols).  N = 2**B states.
  - A HIDDEN bijection maps a "hidden index" j in [0,N) to a codeword = binary(j).
    For each of the B output (next-state) codeword bits we draw a random LOW-SUPPORT
    Boolean function phi_i of at most L in {2,3} of the (B+K) available variables
    (the B hidden-codeword bits of the current state plus the K symbol bits).  Under
    the hidden encoding this keeps every next-state-bit function cheap (few literals).
  - For 3 of the 10 testIds ("trap" cases) symbol 0's transitions are additionally
    forced state-INDEPENDENT (phi_i(*, symbol=0) = constant), so a heuristic that only
    looks at ONE symbol (e.g. symbol 0) to cluster states sees no signal at all there.
  - A uniformly random permutation pi (seeded by testId) relabels hidden index j to an
    actual state id perm[j].  The instance handed to the solver only exposes the actual
    (state, symbol) -> next_state table -- the hidden index / codewords / phi_i / pi are
    NOT printed anywhere.  Recovering low-literal structure requires noticing which
    actual states behave alike across ALL symbols, not the input order of state ids.
"""
import sys
import random

# difficulty ladder: (B) with K fixed at 2 -> N = 2**B, totalvars = B+K <= 7
B_LADDER = [2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
TRAP_TEST_IDS = {3, 6, 9}
K = 2


def bits_of(x, n):
    return tuple((x >> i) & 1 for i in range(n))


def build_case(test_id):
    idx = (test_id - 1) % len(B_LADDER)
    B = B_LADDER[idx]
    N = 1 << B
    M = 1 << K
    totalvars = B + K
    trap = test_id in TRAP_TEST_IDS

    rng = random.Random(1000003 * test_id + 7)

    supports = []
    tables = []
    trap_const = []
    for _i in range(B):
        L = min(rng.choice([2, 3]), totalvars)
        support = tuple(sorted(rng.sample(range(totalvars), L)))
        table = {}
        for combo in range(1 << L):
            key = tuple((combo >> k) & 1 for k in range(L))
            table[key] = rng.randint(0, 1)
        supports.append(support)
        tables.append(table)
        trap_const.append(rng.randint(0, 1))

    # hidden_next[j][s] = hidden index of next state for hidden index j, symbol s
    hidden_next = [[0] * M for _ in range(N)]
    for j in range(N):
        state_bits = bits_of(j, B)
        for s in range(M):
            sym_bits = bits_of(s, K)
            full = state_bits + sym_bits
            out_bits = []
            for i in range(B):
                if trap and s == 0:
                    out_bits.append(trap_const[i])
                else:
                    support = supports[i]
                    key = tuple(full[v] for v in support)
                    out_bits.append(tables[i][key])
            val = 0
            for k, b in enumerate(out_bits):
                val |= (b << k)
            hidden_next[j][s] = val

    perm = list(range(N))
    rng.shuffle(perm)

    trans = [[0] * M for _ in range(N)]
    for j in range(N):
        u = perm[j]
        for s in range(M):
            trans[u][s] = perm[hidden_next[j][s]]

    return N, K, M, trans


def main():
    test_id = int(sys.argv[1])
    N, Kc, M, trans = build_case(test_id)
    out = [f"{N} {Kc}"]
    for u in range(N):
        out.append(" ".join(str(x) for x in trans[u]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
