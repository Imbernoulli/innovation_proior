import sys, random

# Deterministic instance generator for the "scriptorium wet-ink shelving" problem.
# Family: dirty-loop-writeback-cache.  Seeded ONLY by testId.
#
# Structure per round:
#   sweep 0  : write-sweep over the k+2 loop pages (ink goes wet)   -> W ops
#   sweep 1,2: read-sweeps over the same k+2 loop pages             -> R ops
#   idle     : a short idle burst re-reading warm pages             -> R ops
# The loop length is k+2 > k, so any recency rule (LRU) thrashes; the write
# sweep plants dirt that the farthest-in-future rule (Belady) keeps paying to
# write back.  The read sweeps + idle burst are the "safe window" a proactive
# cleaning plan exploits.

def main():
    tid = int(sys.argv[1])
    rng = random.Random(52200 + tid)

    # ---- cache size grows slowly with difficulty (small-scale spec) ----
    k = 4 + (tid - 1) // 3          # 4,4,4,5,5,5,6,6,6,7
    loop = k + 2                    # loop working set (> k  => forced churn)

    # ---- charges: dirty writeback dominates; proactive clean < writeback ----
    F  = 5
    Ce = 1
    De = 12 + 2 * (tid % 3)         # 12,14,16 cycling  (dirty evict)
    Pc = 3 + (tid % 2)              # 3 or 4            (proactive clean)
    # invariant kept by construction: Pc < De - Ce  (cleaning early always pays)

    # ---- how many rounds (drives instance size) ----
    rounds = 8 * tid * tid          # 8 .. 800

    # write density of the write-sweep, varied per test so the solver must read it
    wprob = 0.55 + 0.05 * (tid % 4)   # 0.55 .. 0.70
    idle_len = 2 + (tid % 3)          # 2..4 warm re-reads between rounds

    ops = []  # each entry: (isWrite:0/1, page)

    L = list(range(loop))           # loop pages 0..k+1
    for r in range(rounds):
        # rotate the sweep order a little so it is not perfectly periodic
        off = r % loop
        order = L[off:] + L[:off]
        # --- write sweep ---
        for p in order:
            isw = 1 if rng.random() < wprob else 0
            ops.append((isw, p))
        # --- two read sweeps ---
        for _ in range(2):
            for p in order:
                ops.append((0, p))
        # --- idle burst: re-read the most-recently-touched warm pages ---
        warm = order[-k:]
        for j in range(idle_len):
            ops.append((0, warm[j % len(warm)]))

    M = len(ops)
    out = []
    out.append("%d %d %d %d %d" % (k, F, Ce, De, Pc))
    out.append(str(M))
    for isw, p in ops:
        out.append(("W " if isw else "R ") + str(p))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
