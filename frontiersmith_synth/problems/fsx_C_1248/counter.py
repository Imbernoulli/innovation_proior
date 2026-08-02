import sys, math

# Format D checker -- pipeline stage-partition + forwarding-budget total execution time.
#   Objective (minimize) F = total_cycles * cycle_time, given the artifact's chosen
#   stage partition (cut points) and forwarding-path subset.
#   Baseline B = execution time of the no-pipelining construction (S=1, no forwarding
#   needed since all blocks land in one stage -> zero hazard/branch penalty).
#   Ratio = min(1.0, 0.1 * B / max(1e-9, F))


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_int_token(tok, name):
    if tok is None:
        fail("missing token for %s" % name)
    try:
        fv = float(tok)
    except Exception:
        fail("non-numeric token for %s: %r" % (name, tok))
    if not math.isfinite(fv):
        fail("non-finite token for %s: %r" % (name, tok))
    try:
        iv = int(tok)
    except Exception:
        fail("non-integer token for %s: %r" % (name, tok))
    if float(iv) != fv:
        fail("non-integer token for %s: %r" % (name, tok))
    return iv


def main():
    if len(sys.argv) < 3:
        fail("usage: counter.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = open(in_path).read().split()
    ii = [0]

    def inxt():
        v = itoks[ii[0]]
        ii[0] += 1
        return v

    N = int(inxt()); K = int(inxt()); L = int(inxt())
    Br = int(inxt()); Mb = int(inxt()); resolve_block = int(inxt())
    Budget = int(inxt()); I = int(inxt())
    c = [int(inxt()) for _ in range(N)]
    haz = []
    for _ in range(K):
        need_b = int(inxt()); res_b = int(inxt()); dist = int(inxt()); freq = int(inxt())
        haz.append((need_b, res_b, dist, freq))

    try:
        content = open(out_path).read()
    except Exception as e:
        fail("cannot read output: %s" % e)

    otoks = content.split()
    oi = [0]

    def onxt():
        if oi[0] >= len(otoks):
            return None
        v = otoks[oi[0]]
        oi[0] += 1
        return v

    S = parse_int_token(onxt(), "S")
    if not (1 <= S <= N):
        fail("S out of range: %d (need 1<=S<=%d)" % (S, N))

    cuts = []
    for _ in range(S - 1):
        cuts.append(parse_int_token(onxt(), "cut point"))
    for i in range(len(cuts)):
        if not (1 <= cuts[i] <= N - 1):
            fail("cut point out of range: %d" % cuts[i])
    for i in range(len(cuts) - 1):
        if not (cuts[i] < cuts[i + 1]):
            fail("cut points not strictly increasing")

    P = parse_int_token(onxt(), "P")
    if not (0 <= P <= K):
        fail("P out of range: %d" % P)
    fwd = [parse_int_token(onxt(), "forwarding index") for _ in range(P)]
    if len(set(fwd)) != len(fwd):
        fail("duplicate forwarding indices")
    for v in fwd:
        if not (1 <= v <= K):
            fail("forwarding index out of range: %d" % v)

    if oi[0] != len(otoks):
        fail("trailing garbage after expected tokens")

    # ---- stage_of ----
    boundaries = [0] + cuts + [N]
    stage_of = [0] * (N + 1)
    for s_idx in range(1, S + 1):
        lo, hi = boundaries[s_idx - 1] + 1, boundaries[s_idx]
        for b in range(lo, hi + 1):
            stage_of[b] = s_idx

    stage_delay = [0] * (S + 1)
    for b in range(1, N + 1):
        stage_delay[stage_of[b]] += c[b - 1]
    T = max(stage_delay[1:S + 1]) + L

    fwd_set = set(fwd)
    cost_total = 0
    for k in fwd_set:
        need_b, res_b, dist, freq = haz[k - 1]
        gap = stage_of[res_b] - stage_of[need_b]
        cost_total += gap
    if cost_total > Budget:
        fail("forwarding budget exceeded: %d > %d" % (cost_total, Budget))

    total_stall = 0
    for k in range(1, K + 1):
        need_b, res_b, dist, freq = haz[k - 1]
        gap = stage_of[res_b] - stage_of[need_b]
        stall = 0 if k in fwd_set else max(0, gap - dist)
        total_stall += freq * stall

    resolve_stage = stage_of[resolve_block]
    branch_penalty = Mb * (resolve_stage - 1)

    total_cycles = I + total_stall + branch_penalty
    F = total_cycles * T
    if F <= 0:
        fail("non-positive execution time")

    B_stage_delay = sum(c) + L
    B_cycles = I
    B = B_cycles * B_stage_delay
    if B <= 0:
        fail("degenerate baseline")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d T=%d cycles=%d stall=%d branch_pen=%d S=%d" %
          (F, B, T, total_cycles, total_stall, branch_penalty, S))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
