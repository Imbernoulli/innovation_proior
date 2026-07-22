# TIER: strong
# INSIGHT: fault tolerance is a COVERING problem over deletions, not a duplication
# problem.  Run a compact sorting network P once, then append a SHARED "mop-up"
# stage R made of a few odd-even transposition rounds.  When any single comparator
# in P is deleted, P's output is only mildly disordered, and the SAME handful of
# mop-up comparators repairs every one of those failure cases at once; when a
# comparator in R is deleted, P already sorted so R sees an ordered input where
# each comparator is a no-op (deleting a no-op is harmless).  So one small R
# amortizes redundancy across all faults -> total size S + |R|, far below 2S.
#
# We find the MINIMAL number of mop-up rounds m that makes the whole network pass
# the single-comparator deletion sweep (verified here with the 0/1 principle), then
# emit P + R.
import sys


def batcher(n):
    net = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        a = i + j; b = i + j + k
                        if a < n and b < n:
                            net.append((a, b))
            k //= 2
        p *= 2
    return net


def oets(n, m):
    net = []
    for r in range(m):
        i = 0 if r % 2 == 0 else 1
        while i + 1 < n:
            net.append((i, i + 1)); i += 2
    return net


def init_wires(n):
    size = 1 << n
    W = [0] * n
    for i in range(n):
        block = 1 << i
        val = 0; pos = 0
        while pos < size:
            pos += block
            if pos < size:
                val |= (((1 << block) - 1) << pos); pos += block
        W[i] = val
    return W


def targets(n):
    size = 1 << n
    pc = [bin(t).count("1") for t in range(size)]
    T = [0] * n
    for i in range(n):
        need = n - i; tgt = 0
        for t in range(size):
            if pc[t] >= need:
                tgt |= (1 << t)
        T[i] = tgt
    return T


def sorts(net, n, W0, T):
    W = list(W0)
    for (a, b) in net:
        x = W[a]; y = W[b]
        W[a] = x & y; W[b] = x | y
    for i in range(n):
        if W[i] != T[i]:
            return False
    return True


def fault_tolerant(net, n, W0, T):
    if not sorts(net, n, W0, T):
        return False
    for k in range(len(net)):
        if not sorts(net[:k] + net[k + 1:], n, W0, T):
            return False
    return True


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = batcher(n)
    W0 = init_wires(n)
    T = targets(n)
    best = None
    for m in range(0, n + 6):
        net = P + oets(n, m)
        if fault_tolerant(net, n, W0, T):
            best = net
            break
    if best is None:                          # ultra-safe fallback: duplication
        best = []
        for (a, b) in P:
            best.append((a, b)); best.append((a, b))
    lines = [str(len(best))] + ["%d %d" % (a, b) for (a, b) in best]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
