#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for the DRAM refresh-schedule problem.

1. Parses the instance (T, banks' per-row retention bounds, the access trace).
2. Parses the participant's refresh schedule; validates STRICTLY:
     - well-formed integer tokens (garbage/empty/huge/nan/inf -> Ratio: 0.0)
     - bank/row/slot in range
     - at most one refresh command per (bank, slot)  [bank-parallelism: banks independent]
     - every row's refresh gaps (including the -1..first and last..T boundaries) <= its
       retention bound
   Any violation -> "Ratio: 0.0" and exit 0.
3. Computes F = total stall weight (sum of access-request weights whose (slot,bank) is
   occupied by a refresh command).
4. Baseline B = total weight of ALL access requests (the weight the checker's own trivial
   "refresh every slot in every bank" construction would incur, since that construction
   occupies every (bank,slot) and therefore stalls everything). Minimization ratio:
       sc = min(1000, 100*B/max(1e-9,F));  print("Ratio: %.6f" % (sc/1000))
"""
import sys


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


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
        T = int(inext())
        Bnum = int(inext())
        if T <= 0 or Bnum <= 0:
            fail("bad instance header")
        banks_rho = []
        for _ in range(Bnum):
            Rb = int(inext())
            if Rb <= 0:
                fail("bad Rb")
            rho = [int(inext()) for _ in range(Rb)]
            banks_rho.append(rho)
        M = int(inext())
        reqs = []
        for _ in range(M):
            s = int(inext())
            b = int(inext())
            w = int(inext())
            reqs.append((s, b, w))
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
        K = int(next(op))
    except (StopIteration, ValueError):
        fail("bad event count")
    if K < 0 or K > 4_000_000:
        fail("event count out of range")

    events = []
    try:
        for _ in range(K):
            s_tok = next(op)
            b_tok = next(op)
            r_tok = next(op)
            s = int(s_tok)
            bk = int(b_tok)
            rw = int(r_tok)
            events.append((s, bk, rw))
    except (StopIteration, ValueError):
        fail("malformed refresh event (non-finite/garbage/truncated)")

    # ---- validate ----
    occupied = {}          # (bank, slot) -> row
    per_row_times = {}     # (bank, row) -> list[slot]
    for (s, bk, rw) in events:
        if not (0 <= bk < Bnum):
            fail("bank out of range")
        Rb = len(banks_rho[bk])
        if not (0 <= rw < Rb):
            fail("row out of range")
        if not (0 <= s < T):
            fail("slot out of range")
        key = (bk, s)
        if key in occupied:
            fail("capacity violation: two refreshes in same bank+slot")
        occupied[key] = rw
        per_row_times.setdefault((bk, rw), []).append(s)

    for bk in range(Bnum):
        rho_list = banks_rho[bk]
        for rw in range(len(rho_list)):
            rho = rho_list[rw]
            times = sorted(per_row_times.get((bk, rw), []))
            prev = -1
            for t in times:
                if t - prev > rho:
                    fail(f"retention violated bank={bk} row={rw}")
                prev = t
            if T - prev > rho:
                fail(f"final retention violated bank={bk} row={rw}")

    # ---- score ----
    F = 0
    B = 0
    for (s, b, w) in reqs:
        B += w
        if (b, s) in occupied:
            F += w

    if B <= 0:
        fail("degenerate instance (no positive-weight requests)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
