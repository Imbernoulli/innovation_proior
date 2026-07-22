# TIER: greedy
# The obvious approach: each epoch bolt the top-p lamps by THIS epoch's raw access
# FREQUENCY, moving toward that target within the reconfiguration bandwidth.
#
# Two blind spots (by construction):
#   * frequency ranking bolts the DECOY lamp (highest count) even though L2 already
#     serves it cheaply -- a wasted slot;
#   * it is purely reactive, so at a regime epoch (several lamps turn hot at once,
#     more than Bmax) it cannot assemble the hot set in time and eats the misses.
import sys
from collections import Counter


def main():
    toks = sys.stdin.read().split()
    idx = 0
    def nxt():
        nonlocal idx
        v = toks[idx]; idx += 1; return v
    tid = int(nxt())
    N = int(nxt()); E = int(nxt()); L = int(nxt()); p = int(nxt()); q = int(nxt()); Bmax = int(nxt())
    cpin = int(nxt()); cl2 = int(nxt()); cmiss = int(nxt()); cswap = int(nxt())
    T = int(nxt())
    seq = [int(nxt()) for _ in range(T)]

    out = []
    prev = []
    for t in range(E):
        cnt = Counter(seq[t * L:(t + 1) * L])
        target = [k for k, _ in cnt.most_common(p)]
        tset = set(target)
        keep = [k for k in prev if k in tset]
        room = p - len(keep)
        add_pool = [k for k in target if k not in keep]
        # rank additions by frequency
        add_pool.sort(key=lambda k: -cnt[k])
        budget = p if t == 0 else Bmax        # epoch 0 = free install
        n_add = min(room, budget, len(add_pool))
        cur = keep + add_pool[:n_add]
        out.append(str(len(cur)) + ("" if not cur else " " + " ".join(str(k) for k in cur)))
        prev = cur
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
