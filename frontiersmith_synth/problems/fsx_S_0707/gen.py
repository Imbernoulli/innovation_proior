#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE instance of the "leaked cipher layer" XOR-SLP
problem to stdout.  Deterministic: all randomness is seeded from testId only.

Hidden construction (NOT shown to the solver -- only the dense product matrix M
is printed):
    M = row-permute_P( S xor-composed-with (I + u v^T) )
  where, over GF(2):
    - S is an m x m matrix with EXACTLY 3 ones per row (a random sparse mixing
      layer: "S_i" = XOR of 3 random input columns);
    - u, v are random length-m 0/1 vectors of weight floor(m/2) (a planted
      RANK-1 correction "I + u v^T" bolted onto S);
    - P is a random row permutation (hides WHICH physical row came from which
      construction step, but not the row's content).

  For row i: if the parity of u over S_i's 3 support columns is EVEN, row i of
  (S composed with (I+uv^T)) is exactly S_i (sparse, weight 3, UNCHANGED).  If
  the parity is ODD, the row becomes S_i XOR v (dense, weight ~ m/2).  So the
  dense matrix M handed to the solver secretly contains a "sparse + shared
  rank-1 offset" structure: differencing any two of the dense rows cancels v
  and returns a weight <=6 vector -- an invariant a synthesizer can exploit,
  but only if it looks for it.
"""
import sys
import random

# small -> large / adversarial ladder (testId 1..10)
M_LADDER = [16, 20, 26, 32, 40, 48, 58, 70, 84, 100]


def m_for(test_id: int) -> int:
    if 1 <= test_id <= len(M_LADDER):
        return M_LADDER[test_id - 1]
    # extend deterministically beyond the ladder, just in case
    return M_LADDER[-1] + 12 * (test_id - len(M_LADDER))


def build_instance(test_id: int):
    m = m_for(test_id)
    rng = random.Random(1000003 * test_id + 7)

    for attempt in range(8):
        r = random.Random(1000003 * test_id + 7 + 97 * attempt)

        perm = list(range(m))
        r.shuffle(perm)

        s_rows = []            # s_rows[i] = bitmask (int), exactly 3 bits set
        for i in range(m):
            cols = r.sample(range(m), 3)
            bm = 0
            for c in cols:
                bm |= (1 << c)
            s_rows.append(bm)

        u_bits = set(r.sample(range(m), m // 2))
        v_idx = r.sample(range(m), m // 2)
        v_bm = 0
        for c in v_idx:
            v_bm |= (1 << c)

        sa_rows = []            # (S composed with (I+uv^T))_i
        for i in range(m):
            bm = s_rows[i]
            parity = 0
            b = bm
            while b:
                low = b & (-b)
                col = low.bit_length() - 1
                if col in u_bits:
                    parity ^= 1
                b ^= low
            row = bm ^ v_bm if parity else bm
            sa_rows.append(row)

        M_rows = [sa_rows[perm[k]] for k in range(m)]

        if all(row != 0 for row in M_rows):
            return m, M_rows

    raise RuntimeError("failed to build a non-degenerate instance")


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    m, rows = build_instance(test_id)

    out = [str(m)]
    for row in rows:
        out.append("".join("1" if (row >> j) & 1 else "0" for j in range(m)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
