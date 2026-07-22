import sys

MAX_TOKEN_LEN = 12
MAX_LINES = 400_000


def fail(reason):
    print("Ratio: 0.0  # %s" % reason)
    sys.exit(0)


def parse_nonneg_int(tok):
    if len(tok) == 0 or len(tok) > MAX_TOKEN_LEN:
        return None
    t = tok
    if t[0] == "+":
        t = t[1:]
    if len(t) == 0 or not t.isdigit():
        return None
    try:
        return int(t)
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        fail("bad_args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        with open(in_path, "r") as f:
            in_lines = f.read().split("\n")
        K, P, A = (int(x) for x in in_lines[0].split())
        d_raw = in_lines[1].split()
        if len(d_raw) != K:
            fail("bad_input_distances")
        D = [int(x) for x in d_raw]
        if K < 2 or P < 1 or A < 1 or any(d < 1 for d in D):
            fail("bad_input_ranges")
    except Exception:
        fail("unparseable_input")

    try:
        with open(out_path, "r") as f:
            out_text = f.read()
    except Exception:
        fail("no_output")

    lines = [ln for ln in out_text.split("\n") if ln.strip() != ""]
    if len(lines) == 0:
        fail("no_ticks")
    if len(lines) > MAX_LINES:
        fail("too_many_ticks")

    cumulative = [0] * K
    total_energy = 0

    for ln in lines:
        toks = ln.split()
        if len(toks) != K:
            fail("wrong_column_count")
        speeds = []
        for tok in toks:
            v = parse_nonneg_int(tok)
            if v is None:
                fail("unparseable_or_negative_speed")
            speeds.append(v)
        tick_power = 0
        for i in range(K):
            v = speeds[i]
            cumulative[i] += v
            if cumulative[i] > D[i]:
                fail("arm_overshoots_target_distance")
            tick_power += v * v
        if tick_power > P:
            fail("power_cap_exceeded")
        total_energy += tick_power

    for i in range(K):
        if cumulative[i] != D[i]:
            fail("chain_not_completed")

    T = len(lines)
    F = A * T + total_energy
    if F <= 0:
        fail("nonpositive_cost")

    # Internal baseline B: the fully serial, speed-1-only construction
    # (move one arm at a time, one distance unit per tick -- always
    # feasible since 1*1 = 1 <= P by construction). This is the "do the
    # obviously-correct dumbest thing" reference.
    S1 = sum(D)
    B = (A + 1) * S1

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("T=%d energy=%d cost=%d baseline=%d Ratio: %.6f" % (T, total_energy, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
