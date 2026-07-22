#!/usr/bin/env python3
"""
fsx_S_0714 -- Grid Librarian: Reordering Errands to Stop Re-Shelving.

Builds M hidden "reading-list" collections. Collection s owns a private pool
of P shelf slots. Collection s has L "consultation" errands, each reading a
random Lp-subset of that collection's pool and writing its own private
"result" slot (so different consultations of the same collection genuinely
overlap in which shelf slots they touch -- that overlap is the only trace of
collection membership). Each consultation errand ALSO requires a small,
random number (0..Dmax) of independent one-off "clearance" errands to be
done first (each clearance touches one dedicated, never-shared slot) --
these gate WHEN a consultation becomes eligible without leaking anything
about which collection it belongs to.

Every slot address and every errand id is then relabelled with a random
bijection (seeded by testId): the numeric codes give no hint of collection
membership. Only the read-address CO-OCCURRENCE pattern (which slots get
touched together by the same errand) still betrays it -- recoverable, not
visible, and NOT reconstructable from the dependence edges alone (clearance
errands are privately wired one-to-one to the consultation they unlock, so
the dependency graph alone never reveals which consultations share a
collection).
"""
import sys
import random

# (M collections, L consultations/collection, P pool size, Lp reads/consultation,
#  Dmax max clearance errands per consultation, K cart size) per testId.
TESTS = [
    (4, 8, 6, 3, 1, 8),     # 1  small warm-up
    (5, 8, 6, 3, 1, 8),     # 2  small warm-up
    (6, 10, 8, 4, 1, 10),   # 3  trap begins
    (6, 10, 8, 4, 1, 10),   # 4  trap
    (8, 12, 8, 4, 1, 10),   # 5  trap
    (8, 12, 8, 4, 1, 10),   # 6  trap
    (8, 14, 8, 4, 1, 10),   # 7  trap
    (10, 12, 8, 4, 1, 12),  # 8  trap
    (10, 14, 8, 4, 1, 12),  # 9  trap
    (10, 14, 10, 4, 2, 12), # 10 largest / hardest
]


def main():
    tid = int(sys.argv[1])
    M, L, P, Lp, Dmax, K = TESTS[(tid - 1) % len(TESTS)]
    rng = random.Random(2000003 * tid + 11)

    next_addr = 0
    pool_addrs = {}
    for s in range(M):
        pool_addrs[s] = list(range(next_addr, next_addr + P))
        next_addr += P

    # true ops: (true_id, reads_true[list], write_true)
    true_ops = []
    edges_true = []  # (clearance_true_id, consultation_true_id)
    oc = 0
    for s in range(M):
        for _i in range(L):
            reads = rng.sample(pool_addrs[s], Lp)
            write = next_addr
            next_addr += 1
            mid = oc
            oc += 1
            true_ops.append((mid, reads, write))
            d = rng.randrange(Dmax + 1)
            for _ in range(d):
                daddr = next_addr
                next_addr += 1
                did = oc
                oc += 1
                true_ops.append((did, [daddr], daddr))
                edges_true.append((did, mid))

    total_ops = oc
    total_addr = next_addr

    addr_perm = list(range(total_addr))
    rng.shuffle(addr_perm)
    op_perm = list(range(total_ops))
    rng.shuffle(op_perm)

    op_lines = []
    for tid_, reads, write in true_ops:
        r_labels = [addr_perm[a] for a in reads]
        rng.shuffle(r_labels)
        w_label = addr_perm[write]
        lbl = op_perm[tid_]
        op_lines.append((lbl, r_labels, w_label))

    edges = [(op_perm[u], op_perm[v]) for u, v in edges_true]

    op_lines.sort(key=lambda x: x[0])
    edges.sort()

    N = total_ops
    M_edges = len(edges)
    out = [f"{N} {M_edges} {K}"]
    for lbl, r_labels, w_label in op_lines:
        out.append(f"{lbl} {len(r_labels)} " + " ".join(map(str, r_labels)) + f" {w_label}")
    for u, v in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
