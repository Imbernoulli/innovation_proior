# TIER: strong
"""
Per-row retention BINNING + load-aware phase assignment (the intended insight).

Per bank b, let P = min(rho) (the bank's weakest row, exactly the interval `greedy` forces on
every row). Rows are assigned a private residue class r in [0,P) -- since P is always >= Rb
(the row count), P offers at least as many residues as there are rows, so every row can own a
DISTINCT residue with zero capacity risk, no matter how each row's firing pattern is chosen.

  - retention binning: row i only fires every k_i = floor(rho_i / P) periods (k_i=1 for the
    weakest rows, k_i up to dozens for strong rows -- exactly "refresh weak rows often, strong
    rows rarely"), instead of every single period like `greedy`. This alone frees most of a
    bank's refresh slots.
  - bank-parallelism: each bank is optimised completely independently (no cross-bank
    contention exists in the model), so per-bank residue assignment is exact, not heuristic.
  - refresh-vs-access conflict: residues are handed out cheapest-load-first to the rows that
    fire most often (the weak rows pay the full column cost every period, so they get the
    coldest columns first); this actively steers refreshes away from where the access trace is
    hammering, which `greedy`'s fixed round-robin offsets never attempt.

This is a good heuristic, not a proven optimum (choices made for one row can block a better
placement for another -- the joint problem is a hard combinatorial assignment), so it still
leaves headroom above what it achieves.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    T = int(nx())
    Bnum = int(nx())
    banks = []
    for _ in range(Bnum):
        Rb = int(nx())
        rho = [int(nx()) for _ in range(Rb)]
        banks.append((Rb, rho))
    M = int(nx())
    reqs_by_bank = [[] for _ in range(Bnum)]
    for _ in range(M):
        s = int(nx()); b = int(nx()); w = int(nx())
        reqs_by_bank[b].append((s, w))

    events = []
    for b, (Rb, rho) in enumerate(banks):
        P = min(rho)
        col_score = [0] * P
        for (s, w) in reqs_by_bank[b]:
            col_score[s % P] += w

        # cheapest residues first
        residues_sorted = sorted(range(P), key=lambda r: col_score[r])
        avail = list(residues_sorted)  # pop(0) = cheapest remaining

        # weak (small-rho, fire-every-period) rows claim the coldest residues first
        order = sorted(range(Rb), key=lambda i: rho[i])

        row_residue = [None] * Rb
        for i in order:
            r = avail.pop(0)
            row_residue[i] = r

        for i in range(Rb):
            r = row_residue[i]
            k_i = max(1, rho[i] // P)
            t = r
            while t < T:
                events.append((t, b, i))
                t += k_i * P

    out = [str(len(events))]
    for (t, b, row) in events:
        out.append(f"{t} {b} {row}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
