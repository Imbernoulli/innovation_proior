import sys
from collections import deque

# Deterministic scorer for threshold-firebreak-blocking.
#   objective: MINIMIZE the final count of activated NON-SOURCE nodes after a
#   linear-threshold cascade runs to a fixed point on the graph with the
#   participant's chosen firebreak (removed) nodes deleted.
#
# Input:
#   N M K S
#   s_1 .. s_S
#   theta_i amp_i   for i = 1..N
#   u v w           for each of M directed edges  (repeated M times)
#
# Participant output:
#   R
#   id_1 .. id_R      (0 <= R <= K, distinct, each in [1,N], none a source)


def fail(msg):
    print("Ratio: 0.000000 (%s)" % msg)
    sys.exit(0)


def simulate(N, sources, theta, amp, adj, removed):
    active = [False] * (N + 1)
    inw = [0.0] * (N + 1)
    q = deque()
    for s in sources:
        if s in removed:
            continue
        active[s] = True
        q.append(s)
    while q:
        u = q.popleft()
        au = amp[u]
        for (v, w) in adj[u]:
            if v in removed or active[v]:
                continue
            inw[v] += au * w
            if inw[v] >= theta[v] - 1e-9:
                active[v] = True
                q.append(v)
    src_set = set(sources)
    F = 0
    for i in range(1, N + 1):
        if active[i] and i not in src_set:
            F += 1
    return F


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inp_path, out_path = sys.argv[1], sys.argv[2]

    try:
        toks = open(inp_path).read().split()
        it = iter(toks)
        N = int(next(it)); M = int(next(it)); K = int(next(it)); S = int(next(it))
        if N <= 0 or M < 0 or K < 0 or S <= 0:
            fail("bad header")
        sources = [int(next(it)) for _ in range(S)]
        theta = [0.0] * (N + 1)
        amp = [1.0] * (N + 1)
        for i in range(1, N + 1):
            theta[i] = float(next(it))
            amp[i] = float(next(it))
        adj = [[] for _ in range(N + 1)]
        for _ in range(M):
            u = int(next(it)); v = int(next(it)); w = float(next(it))
            adj[u].append((v, w))
    except Exception:
        fail("bad input (should not happen)")

    src_set = set(sources)

    # ---- parse participant output ----
    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("no output")
    oit = iter(out_toks)
    try:
        R = int(next(oit))
    except Exception:
        fail("missing R")
    if not (0 <= R <= K):
        fail("R out of [0,K]")

    removed_list = []
    try:
        for _ in range(R):
            removed_list.append(int(next(oit)))
    except Exception:
        fail("could not read R node ids")

    extra = list(oit)
    if extra:
        fail("trailing tokens after node list")

    seen = set()
    for x in removed_list:
        if not (1 <= x <= N):
            fail("removed id out of range")
        if x in src_set:
            fail("cannot remove a source node")
        if x in seen:
            fail("duplicate removed id")
        seen.add(x)

    removed = set(removed_list)

    F = simulate(N, sources, theta, amp, adj, removed)
    B = simulate(N, sources, theta, amp, adj, set())
    B = max(1, B)

    sc = min(1000.0, 100.0 * B / max(1, F))
    print("F=%d B=%d R=%d Ratio: %.6f" % (F, B, R, sc / 1000.0))


if __name__ == "__main__":
    main()
