#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic op-count scorer for
Mirror Pond Rake Budget (format D: FLOPs / exact op accounting).

Reads the instance (N, T, MAXOPS, h0[]) from <in>.
Reads the participant's straight-line plan from <out>:
    line 1:      M            (number of op lines that follow)
    next M lines: "P i"       point-polish cell i, cost 1
                  "B a w"     rake-average block [a, a+w), cost w

Executes the plan EXACTLY as given (order matters), verifies max_i|h_i| <= T,
then scores  ratio = min(1, 0.1 * ops_baseline / ops_yours)  (fewer ops better),
where ops_baseline is a simple, always-valid point-only construction the
checker builds itself.  Any feasibility violation -> Ratio: 0.0.
"""
import sys


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def is_pow2(x):
    return x >= 2 and (x & (x - 1)) == 0


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_tokens = f.read().split()
    try:
        N = int(in_tokens[0]); T = int(in_tokens[1]); MAXOPS = int(in_tokens[2])
        h0 = [int(x) for x in in_tokens[3:3 + N]]
        if len(h0) != N:
            raise ValueError("bad instance")
    except Exception as e:
        fail("could not parse instance: %s" % e)

    try:
        with open(out_path, "r") as f:
            out_text = f.read()
    except Exception as e:
        fail("could not read output: %s" % e)

    out_lines = out_text.split("\n")
    # strip a single trailing empty line artifact but keep everything else
    toks_first = out_lines[0].split() if out_lines else []
    if not toks_first:
        fail("empty output / missing op count")
    try:
        M = int(toks_first[0])
    except Exception:
        fail("op count M is not an integer")

    if M < 0:
        fail("negative op count")
    if M > MAXOPS:
        fail("M=%d exceeds MAXOPS=%d" % (M, MAXOPS))

    # collect remaining non-empty lines as op lines (bounded scan, M <= MAXOPS)
    op_lines = []
    for ln in out_lines[1:]:
        s = ln.strip()
        if s == "":
            continue
        op_lines.append(s)
        if len(op_lines) > M:
            break  # never read further than needed; extra trailing junk is fine

    if len(op_lines) < M:
        fail("declared M=%d ops but only %d op lines present" % (M, len(op_lines)))

    h = h0[:]
    total_cost = 0
    for line_no in range(M):
        parts = op_lines[line_no].split()
        if not parts:
            fail("empty op line at %d" % (line_no + 1))
        code = parts[0]
        if code == "P":
            if len(parts) != 2:
                fail("malformed P op at line %d" % (line_no + 1))
            try:
                i = int(parts[1])
            except Exception:
                fail("non-integer cell index at line %d" % (line_no + 1))
            if not (0 <= i < N):
                fail("P index %d out of range at line %d" % (i, line_no + 1))
            v = h[i]
            if v > 0:
                h[i] = v - 1
            elif v < 0:
                h[i] = v + 1
            total_cost += 1
        elif code == "B":
            if len(parts) != 3:
                fail("malformed B op at line %d" % (line_no + 1))
            try:
                a = int(parts[1]); w = int(parts[2])
            except Exception:
                fail("non-integer block params at line %d" % (line_no + 1))
            if not is_pow2(w):
                fail("block width %d is not a power of two >=2 at line %d" % (w, line_no + 1))
            if a < 0 or w > N or a % w != 0 or a + w > N:
                fail("block [%d,%d) misaligned/out of range at line %d" % (a, a + w, line_no + 1))
            s = sum(h[a:a + w])
            avg = s // w  # exact floor division, deterministic
            if avg != 0:
                for i in range(a, a + w):
                    h[i] -= avg
            total_cost += w
        else:
            fail("unknown op code %r at line %d" % (code, line_no + 1))

    # reject non-finite / absurd state defensively (should be impossible given
    # the integer-only op set, but keep the invariant checked explicitly)
    for x in h:
        if x != x or abs(x) > 10 ** 15:
            fail("state left non-finite/absurd")

    resid = max(abs(x) for x in h)
    if resid > T:
        fail("final max|h_i|=%d exceeds threshold T=%d" % (resid, T))

    # internal baseline: point-only construction (shave every cell to T)
    baseline = sum(max(0, abs(x) - T) for x in h0)
    if baseline <= 0:
        baseline = 1  # defensive; all shipped instances have baseline > 0

    sc = min(1000.0, 100.0 * baseline / max(1e-9, float(total_cost)))
    print("cost=%d baseline=%d resid=%d" % (total_cost, baseline, resid))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
