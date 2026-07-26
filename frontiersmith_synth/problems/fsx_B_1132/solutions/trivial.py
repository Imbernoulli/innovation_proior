# TIER: trivial
"""Feed each day's freshly-available (unspoiled) stock in its own
arrival ratio, scaled down only as far as needed to respect the fixed
daily capacity. No chemistry optimization, no adaptation-awareness, no
salvage planning -- exactly the checker's own reference construction."""
import sys
from collections import deque


def read_instance():
    data = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    T = int(nxt()); K = int(nxt())
    alpha_milli = int(nxt())
    switch_cost_i100 = int(nxt())
    cap = int(nxt())
    c = [int(nxt()) for _ in range(K)]
    s = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            v = int(nxt())
            s[i][j] = v; s[j][i] = v
    thr = [int(nxt()) for _ in range(K)]
    pen = [int(nxt()) for _ in range(K)]
    shelf = [int(nxt()) for _ in range(K)]
    M0 = [int(nxt()) for _ in range(K)]
    arr = [[int(nxt()) for _ in range(K)] for _ in range(T)]
    n_spikes = int(nxt())
    spike = [[None] * K for _ in range(T)]
    for _ in range(n_spikes):
        d = int(nxt()); typ = int(nxt()); amt = int(nxt()); sh = int(nxt())
        if 0 <= d < T and 0 <= typ < K:
            spike[d][typ] = (amt, sh)
    return T, K, cap, shelf, arr, spike


def main():
    T, K, cap, shelf, arr, spike = read_instance()
    inv = [deque() for _ in range(K)]
    out_lines = []
    for t in range(T):
        dq_all = inv
        for k in range(K):
            dq = dq_all[k]
            while dq and (t - dq[0][0]) >= dq[0][2]:
                dq.popleft()
            if arr[t][k] > 0:
                dq.append([t, float(arr[t][k]), shelf[k]])
            if spike[t][k]:
                amt, sh = spike[t][k]
                if amt > 0:
                    dq.append([t, float(amt), sh])
        avail = [sum(b[1] for b in dq_all[k]) for k in range(K)]
        tot = sum(avail)
        if tot <= 1e-9:
            row = [0.0] * K
        else:
            f = min(1.0, cap / tot)
            row = [v * f for v in avail]
        out_lines.append(" ".join("%.6f" % v for v in row))
        for k in range(K):
            need = row[k]
            dq = dq_all[k]
            while need > 1e-6 and dq:
                b = dq[0]
                take = need if need < b[1] else b[1]
                b[1] -= take
                need -= take
                if b[1] <= 1e-6:
                    dq.popleft()
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
