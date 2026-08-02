#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for the power-gating-partition problem.

1. Parses the instance: N blocks, D domain limit, K traces, T timesteps, leakage power L,
   wakeup energy W, and K*N activity strings (each length T, chars '0'/'1').
2. Parses the participant's artifact; validates STRICTLY:
     - well-formed integer tokens (garbage/empty/huge/nan/inf -> Ratio: 0.0)
     - 1 <= Du <= D
     - each block's domain id in [1, Du]; all N assignments present
     - each domain's threshold theta_d is an integer in [0, 1_000_000]
   Any violation -> "Ratio: 0.0" and exit 0.
3. Replays every trace against the submitted (partition, thresholds). A domain's leakage RATE
   is proportional to its own SIZE (member-block count): while domain d (|members|=s) is ON --
   either servicing real work (any member active) or idling-but-not-yet-gated -- it draws
   L*s per timestep (a domain is a shared power rail: the more blocks bundled onto it, the
   more leakage it draws whenever the rail is live, regardless of how many of its members are
   the ones actually working). Between active periods, a maximal idle run of length Lr is
   powered OFF (paying a flat, size-INDEPENDENT wakeup energy W once, unless the run runs off
   the end of the trace, in which case no wakeup is ever needed) iff Lr >= theta_d; otherwise
   the domain is kept needlessly ON through it, still paying L*s per idle step. F = total
   energy summed over all domains and all K traces.
4. Baseline B = L*N*T*K: the energy of the checker's own trivial construction (one domain
   holding all N blocks, size N, never gated, held ON for the full horizon on every trace) --
   a fixed constant, independent of the actual trace content. Minimization ratio:
   sc = min(1000, 100*B/max(1e-9,F)); print("Ratio: %.6f" % (sc/1000)).
"""
import sys

THETA_MAX = 1_000_000


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


def run_lengths(demand):
    """Split a 0/1 string into maximal runs: list of (state:int, length:int), in order."""
    runs = []
    n = len(demand)
    i = 0
    while i < n:
        c = demand[i]
        j = i + 1
        while j < n and demand[j] == c:
            j += 1
        runs.append((int(c), j - i))
        i = j
    return runs


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = iter(itoks)

    def inext():
        return next(ip)

    try:
        N = int(inext())
        D = int(inext())
        K = int(inext())
        T = int(inext())
        if N <= 0 or D <= 0 or K <= 0 or T <= 0:
            fail("bad instance header")
        L = int(inext())
        W = int(inext())
        if L <= 0 or W < 0:
            fail("bad instance energies")
        traces = []  # K lists of N strings, length T
        for _k in range(K):
            rows = []
            for _i in range(N):
                s = inext()
                if len(s) != T or any(ch not in "01" for ch in s):
                    fail("malformed instance trace (should not happen)")
                rows.append(s)
            traces.append(rows)
    except (StopIteration, ValueError):
        fail("malformed instance (should not happen)")

    # ---- parse participant output (untrusted) ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except OSError:
        fail("cannot read output")

    if not otoks:
        fail("empty output")

    op = iter(otoks)
    try:
        Du = int(next(op))
    except (StopIteration, ValueError):
        fail("bad domain count")
    if not (1 <= Du <= D):
        fail("domain count out of range")

    dom = []
    try:
        for _ in range(N):
            v = int(next(op))
            if not (1 <= v <= Du):
                fail("block domain id out of range")
            dom.append(v)
    except (StopIteration, ValueError):
        fail("malformed domain assignment (non-finite/garbage/truncated)")

    theta = []
    try:
        for _ in range(Du):
            v = int(next(op))
            if not (0 <= v <= THETA_MAX):
                fail("threshold out of range")
            theta.append(v)
    except (StopIteration, ValueError):
        fail("malformed threshold list (non-finite/garbage/truncated)")

    # ---- group members per domain ----
    members = [[] for _ in range(Du)]  # 0-indexed domain -> list of block indices
    for i, d in enumerate(dom):
        members[d - 1].append(i)

    # ---- score ----
    F = 0
    for rows in traces:
        for d in range(Du):
            blk_idx = members[d]
            size = len(blk_idx)
            if size == 0:
                continue  # empty domain: no rail, no leakage
            th = theta[d]
            rate = L * size
            if size == 1:
                demand = rows[blk_idx[0]]
            else:
                # OR-combine member rows into the domain's shared-rail demand signal
                chunks = [rows[b] for b in blk_idx]
                demand_chars = []
                for t in range(T):
                    on = "1" if any(c[t] == "1" for c in chunks) else "0"
                    demand_chars.append(on)
                demand = "".join(demand_chars)

            runs = run_lengths(demand)
            n_runs = len(runs)
            for ridx, (state, length) in enumerate(runs):
                if state == 1:
                    F += rate * length  # servicing real work: still draws the rail's leakage
                else:
                    is_last = (ridx == n_runs - 1)
                    if length >= th:
                        if not is_last:
                            F += W
                        # else: gated + trailing -> never wakes again -> 0 cost
                    else:
                        F += rate * length

    # Baseline B: single domain holding all N blocks, never gated -> L*N per timestep, always.
    B = L * N * T * K
    if B <= 0:
        fail("degenerate instance (non-positive baseline)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
