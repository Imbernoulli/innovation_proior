import sys

# gen.py <testId> -- prints ONE VLIW bundle-scheduling instance to stdout.
#
# The dependence DAG is W_ind independent unary chains (each of length L, type 'A' /
# ALU slots, latency LAT_A) that all feed an incremental accumulator of (W_ind-1)
# binary "combine" ops (type 'M' / MEM-MUL slots, latency LAT_M), i.e. a reduction:
#   acc_1 = combine(chain_1_final, chain_2_final)
#   acc_j = combine(acc_{j-1}, chain_{j+1}_final)      j = 2..W_ind-1
#
# Every value has out-degree 1 (its unique consumer). Chain heads are ALL ready at
# cycle 1 with no register-pressure cost visible in the DAG shape itself: with
# LAT_M >> L*LAT_A (the serial accumulator chain, not chain compute, is the true
# critical path) a max-fill list scheduler happily launches every chain in parallel
# across the 5 free ALU slots because nothing in the DAG says "don't" -- there is no
# cycle-count reason not to. Doing so makes every chain finish almost immediately,
# so most/all of the W_ind chain-final values become simultaneously live while they
# wait their turn through the (necessarily serial, single-M-slot, LAT_M-spaced)
# combine phase -- forcing spills once W_ind exceeds the register file size R. A
# pressure-aware scheduler instead caps how many chains it keeps "in flight" at
# once (leaving ALU slots idle even though ready ops exist), paying no cycle
# penalty because a chain's own compute time (L*LAT_A) hides entirely inside the
# latency of the combine that will eventually retire it.
#
# Tests 1-3 are low-pressure warm-ups (R comfortably exceeds W_ind: no spill is ever
# forced, so max-fill list scheduling is close to optimal there). Tests 4-10 are the
# planted high-pressure kernels (W_ind > R): the trap.

W = 6
SLOT_TYPES = "AAAAAM"   # 5 ALU ('A') slots + 1 MEM/MUL ('M') slot per bundle, every cycle
LAT_A = 1
LAT_M = 5   # combine latency dwarfs a single chain's compute time (L*LAT_A): the serial
            # accumulator chain is the true critical path, so a chain's own computation
            # can always be hidden inside a prior combine's latency window "for free" --
            # a pressure-aware scheduler pays NO cycle penalty for staying narrow.

# testId -> (W_ind, L, R, SPILL_COST)
SPECS = {
    1:  (2, 3, 14, 6),
    2:  (3, 3, 14, 6),
    3:  (4, 4, 14, 8),
    4:  (5, 3, 2, 3),
    5:  (6, 3, 2, 3),
    6:  (6, 4, 3, 3),
    7:  (7, 3, 2, 3),
    8:  (8, 4, 3, 3),
    9:  (9, 4, 3, 3),
    10: (10, 4, 3, 3),
}


def main():
    tid = int(sys.argv[1])
    w_ind, L, R, spill_cost = SPECS[tid]

    # Round-robin (layer-major) numbering: index i holds ALL chains' step-t ops
    # before any chain's step-(t+1) op. Predecessors always land at a strictly
    # smaller index (still a valid DAG order), but a naive "process ops in
    # index order" construction now interleaves every chain instead of finishing
    # one chain before starting the next -- so the naive baseline doesn't
    # accidentally inherit the register-pressure-aware chain-at-a-time shape.
    ops = [None] * (w_ind * L)   # 0-indexed slot for 1-indexed op id - 1
    chain_final = [0] * w_ind
    for t in range(1, L + 1):
        for k in range(w_ind):
            idx = (t - 1) * w_ind + k + 1
            preds = [] if t == 1 else [(t - 2) * w_ind + k + 1]
            ops[idx - 1] = ('A', LAT_A, preds)
            if t == L:
                chain_final[k] = idx
    base = w_ind * L

    # combine chain: acc_1 = combine(chain_final[0], chain_final[1]); acc_j = combine(acc_{j-1}, chain_final[j+1])
    prev = chain_final[0]
    for j in range(1, w_ind):
        ops.append(('M', LAT_M, [prev, chain_final[j]]))
        prev = base + j
    N = len(ops)
    ops = [None] + ops  # make 1-indexed

    lines = ["%d %d %d %d" % (N, W, R, spill_cost), SLOT_TYPES]
    for i in range(1, N + 1):
        typ, lat, preds = ops[i]
        lines.append("%s %d %d %s" % (typ, lat, len(preds), " ".join(map(str, preds))))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
