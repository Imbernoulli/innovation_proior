# TIER: greedy
"""The obvious first attempt: at every period, dump ALL currently free capital
into whichever single instrument has the highest rate THIS period. Never looks
ahead at the rest of the lock window, never anticipates future rate windows,
never leaves capital idle to wait for a better entry point. This is exactly
the trap the problem plants: a decoy instrument's one-period spike looks like
the obvious choice, but honoring its long lock through the ensuing slump (and
missing a much better window that opens on a different instrument two periods
later) is a bad trade -- greedy cannot see that.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    T = int(data[p]); p += 1
    K = int(data[p]); p += 1
    C0 = int(data[p]); p += 1
    L = [int(data[p + i]) for i in range(K)]; p += K
    rate = []
    for _t in range(T):
        row = [int(data[p + i]) for i in range(K)]
        p += K
        rate.append(row)

    free_cash = [0.0] * (T + 2)
    free_cash[1] = float(C0)
    out_rows = []

    for t in range(1, T + 1):
        avail = free_cash[t]
        row = [0.0] * K
        if avail > 0:
            best_k = max(range(K), key=lambda k: rate[t - 1][k])
            row[best_k] = avail
            Lk = L[best_k]
            window_end = min(t + Lk - 1, T)
            mult = 1.0
            for s in range(t, window_end + 1):
                mult *= (1.0 + rate[s - 1][best_k] / 10000.0)
            val = avail * mult
            bucket = min(t + Lk, T + 1)
            free_cash[bucket] += val
        else:
            free_cash[t + 1] += 0.0
        out_rows.append(" ".join(("%.6f" % x) for x in row))

    sys.stdout.write("\n".join(out_rows) + "\n")


if __name__ == "__main__":
    main()
