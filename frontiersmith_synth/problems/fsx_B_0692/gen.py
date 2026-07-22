#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the "Plate-Stack Codegen" problem.

Instance format (stdout):
  P
  N M K
  v_1 v_2 ... v_N
  op_1 childL_1 childR_1
  ...
  op_M childL_M childR_M
  out_1 out_2 ... out_K

Construction: M nodes are split into K contiguous "blocks", one per output
(out_j = the last node of block j). Each block has three parts:
  1. a handful of "hub" nodes (each a fresh 2-leaf combination);
  2. a run of "consumer" nodes, each combining a leaf with either a hub
     (NEAR reuse -- stays single-output/"local"), a node from an EARLIER
     block (FAR reuse -- becomes genuinely multi-output/"cross"), or a
     fresh leaf;
  3. a left-to-right reduction chain folding all the consumers together
     into the block's output.
Consumers are *siblings* of each other inside the reduction tree, not
nested inside one another -- so a hub reused by consumer_1 and then again
by consumer_2 is genuinely close in the instruction stream (consumer_2
starts right after consumer_1 returns), unlike reusing something threaded
through a long dependency chain, where every "outer" use is unavoidably
nested around all the "inner" ones. That distinction is exactly the
mechanism a residency-aware scheduler exploits and a memory-only one
doesn't bother to notice.

All randomness is seeded purely from testId -> fully deterministic.
"""
import sys

P = 998244353  # fixed prime modulus, also documented in statement.md


class Rng:
    """Small deterministic xorshift32 PRNG (no dependency on Python's random
    module internals, for bit-for-bit reproducibility -- G4)."""

    def __init__(self, seed):
        self.state = (seed * 2654435769 + 1013904223) & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9

    def next32(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        self.state = x
        return x

    def randrange(self, n):
        return self.next32() % n

    def random(self):
        return self.next32() / 4294967296.0


# per-testId scale/shape parameters:
#   N leaves, K blocks(=outputs), hubs (per block), consumers (per block),
#   near_p (consumer reuses a hub), far_p (consumer reuses an earlier
#   block's node), cap (per-node expansion-size cap)
PARAMS = {
    1: dict(N=5, K=2, hubs=2, consumers=10, near_p=0.92, far_p=0.00, cap=900, hub_depth=6),
    2: dict(N=6, K=2, hubs=2, consumers=12, near_p=0.93, far_p=0.00, cap=1000, hub_depth=6),
    3: dict(N=6, K=3, hubs=2, consumers=12, near_p=0.93, far_p=0.02, cap=1100, hub_depth=6),
    4: dict(N=7, K=3, hubs=2, consumers=13, near_p=0.93, far_p=0.02, cap=1200, hub_depth=7),
    5: dict(N=7, K=4, hubs=2, consumers=13, near_p=0.93, far_p=0.03, cap=1300, hub_depth=7),
    6: dict(N=8, K=4, hubs=2, consumers=14, near_p=0.90, far_p=0.05, cap=1400, hub_depth=7),
    7: dict(N=8, K=5, hubs=2, consumers=14, near_p=0.93, far_p=0.02, cap=1500, hub_depth=7),
    8: dict(N=9, K=5, hubs=2, consumers=15, near_p=0.90, far_p=0.04, cap=1600, hub_depth=7),
    9: dict(N=10, K=6, hubs=2, consumers=16, near_p=0.91, far_p=0.03, cap=1700, hub_depth=7),
    10: dict(N=11, K=7, hubs=2, consumers=17, near_p=0.90, far_p=0.04, cap=1800, hub_depth=8),
}

OPS = ["ADD", "SUB", "MUL"]


def build(testId):
    prm = PARAMS[testId]
    rng = Rng(2000003 * testId + 919)
    N, K, H, C = prm["N"], prm["K"], prm["hubs"], prm["consumers"]

    leaf_vals = [rng.randrange(P) for _ in range(N)]
    sizes = {}
    for i in range(1, N + 1):
        sizes[("L", i)] = 1

    node_children = {}  # i -> (op, childL, childR)  (refs are ("L",i)/("N",i))
    outs = []
    i = 1

    def rand_leaf():
        return ("L", rng.randrange(N) + 1)

    def emit(op, cl, cr):
        nonlocal i
        node_children[i] = (op, cl, cr)
        sizes[("N", i)] = sizes[cl] + sizes[cr] + 1
        ref = ("N", i)
        i += 1
        return ref

    for j in range(1, K + 1):
        block_start = i

        # 1) hubs: a handful of fresh, freely-reusable values, each built
        # from a short chain of its own (not just one op) so that actually
        # recomputing one from scratch is expensive enough that caching it
        # -- however it gets cached -- is unambiguously worth it.
        hub_pool = []
        for _h in range(H):
            v = emit(OPS[rng.randrange(3)], rand_leaf(), rand_leaf())
            for _d in range(prm.get("hub_depth", 3)):
                v = emit(OPS[rng.randrange(3)], v, rand_leaf())
            hub_pool.append(v)

        # 2) consumers: each is a *sibling* combination of a leaf with a hub
        # (near reuse), an earlier block's node (far reuse), or a fresh leaf.
        # Consumers never reference each other, so reusing the same hub
        # across several of them is never a nested-dependency situation.
        consumers = []
        last_hub = None
        for _c in range(C):
            r = rng.random()
            other = None
            if r < prm["near_p"] and hub_pool:
                # Strongly prefer sticking with whichever hub the *previous*
                # consumer used: a run of consecutive consumers all reusing
                # the same hub is what actually keeps its spare within the
                # top-3 shuffle window the whole time -- switching hubs every
                # draw spreads reuse across targets that are each other's
                # near-neighbours only in creation order, not in the
                # instruction stream, so the spare is long gone by the time
                # a far-off consumer asks for it again.
                if last_hub is not None and rng.random() < 0.90:
                    other = last_hub
                else:
                    other = hub_pool[rng.randrange(len(hub_pool))]
                last_hub = other
            elif r < prm["near_p"] + prm["far_p"] and block_start > 1:
                other = ("N", rng.randrange(block_start - 1) + 1)
            if other is None:
                other = rand_leaf()
            leaf_ref = rand_leaf()
            if sizes[leaf_ref] + sizes[other] + 1 > prm["cap"]:
                other = rand_leaf()
            # `other` (the reusable part) always goes first: fetching an
            # already-resident value at depth 0/1 (DUP/OVER) *duplicates* it,
            # so if it lands there every time, a run of consumers can keep
            # passing the same spare forward. Depth 2 (ROT) only relocates,
            # not duplicates, so consistently landing shallow is what keeps
            # a hub alive across more than one extra reuse.
            consumers.append(emit(OPS[rng.randrange(3)], other, leaf_ref))

        # 3) fold the consumers together left-to-right into the block output
        # (accumulator always first, next consumer always second, so the gap
        # between one consumer finishing and the next one's own lookups
        # starting is as small as possible: just the reduction's own OP).
        acc = consumers[0]
        for k in range(1, len(consumers)):
            acc = emit(OPS[rng.randrange(3)], acc, consumers[k])
        outs.append(acc)

    M = i - 1
    return leaf_vals, node_children, outs, N, M, K


def fmt_ref(ref):
    return "%s%d" % (ref[0], ref[1])


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    testId = int(sys.argv[1])
    if testId not in PARAMS:
        testId = max(1, min(10, testId))
    leaf_vals, node_children, outs, N, M, K = build(testId)

    out_lines = []
    out_lines.append(str(P))
    out_lines.append("%d %d %d" % (N, M, K))
    out_lines.append(" ".join(str(v) for v in leaf_vals))
    for i in range(1, M + 1):
        op, cl, cr = node_children[i]
        out_lines.append("%s %s %s" % (op, fmt_ref(cl), fmt_ref(cr)))
    out_lines.append(" ".join(fmt_ref(r) for r in outs))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
