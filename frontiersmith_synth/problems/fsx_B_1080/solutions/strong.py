# TIER: strong
"""The insight: since no rebalancing is ever possible, the binding must be
shaped for the FINAL frequency schedule, not the frequencies observed as
symbols arrive -- and since that final schedule is fully computable from the
input before anything is bound, the arrival order is a red herring for the
BINDING decision itself (it only matters for the naive/adaptive algorithm's
mental model, not for a solver with full foresight).

Exchange argument: for any two symbols i, j with f_i > f_j bound to slots of
length d_i > d_j, swapping their slots strictly decreases
f_i*d_i + f_j*d_j  ->  f_i*d_j + f_j*d_i
since (f_i-f_j)*(d_i-d_j) > 0. So at the optimum no such inversion can exist:
sort symbols by final frequency (descending) and slots by length (ascending)
and pair them up -- the pairing is globally optimal for the fixed slot
multiset (rearrangement inequality), independent of arrival order."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    f = [int(data[idx + i]) for i in range(n)]; idx += n
    L = [int(data[idx + i]) for i in range(n)]; idx += n

    order = sorted(range(n), key=lambda i: (-f[i], i))  # biggest final freq first
    slots_sorted = sorted(L)                             # shortest slot first

    d = [0] * n
    for rank, sym in enumerate(order):
        d[sym] = slots_sorted[rank]

    print(" ".join(str(x) for x in d))


if __name__ == "__main__":
    main()
