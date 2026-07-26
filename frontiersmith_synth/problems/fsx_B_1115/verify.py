import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad input file")
    try:
        out_txt = open(sys.argv[2]).read()
    except Exception:
        fail("no output")
    out = out_txt.split()

    # ---- parse instance ----
    try:
        it = iter(inp)
        T = int(next(it))
        times = [int(next(it)) for _ in range(T)]
        demands = [int(next(it)) for _ in range(T)]
        S1 = int(next(it)); S2 = int(next(it)); S3 = int(next(it))
        d1 = float(next(it)); d2 = float(next(it)); d3 = float(next(it))
    except Exception:
        fail("bad instance")
    S = {1: S1, 2: S2, 3: S3}
    tT = times[-1]

    # ---- internal baseline B: one-lot-per-pulse, zero decay, max scrap ----
    B = sum(demands) + T * (S1 + S2 + S3)
    B = max(1e-9, B)

    # ---- parse participant output ----
    try:
        it2 = iter(out)
        N = int(next(it2))
    except Exception:
        fail("bad header")
    if not (1 <= N <= 5000):
        fail("N out of range: %d" % N)

    lots = {1: [], 2: [], 3: []}   # stage -> list of (time, R)
    raw_total = 0.0
    try:
        for _ in range(N):
            stage = int(next(it2))
            time = next(it2)
            r = next(it2)
            if stage not in (1, 2, 3):
                fail("bad stage %s" % stage)
            # time must be an exact integer token
            tval = float(time)
            if not math.isfinite(tval) or tval != int(tval):
                fail("non-integer time %s" % time)
            tval = int(tval)
            if not (0 <= tval <= tT):
                fail("time out of range %s" % tval)
            rval = float(r)
            if not math.isfinite(rval):
                fail("non-finite R")
            if rval < -1e-9 or rval > 1e9:
                fail("R out of range %s" % r)
            rval = max(0.0, rval)
            if rval < S[stage] - 1e-6:
                fail("lot at stage %d doesn't cover its own scrap: R=%.6f S=%d" % (stage, rval, S[stage]))
            lots[stage].append((tval, rval))
            if stage == 1:
                raw_total += rval
    except StopIteration:
        fail("truncated output")
    except (ValueError, TypeError):
        fail("unparsable token")
    # reject trailing garbage tokens beyond declared N triples (seekEof-style discipline)
    remaining = list(it2)
    if remaining:
        fail("trailing tokens after declared N lots")

    if not math.isfinite(raw_total) or raw_total <= 0:
        fail("degenerate raw material total")

    EPS = 1e-6

    def simulate_buffer(buf_id, add_events, withdraw_events, decay):
        """add_events/withdraw_events: list of (time, amount). Returns True if feasible."""
        events = []
        for (t, a) in add_events:
            events.append((t, 0, a))     # adds (type 0) before withdrawals (type 1) at same time
        for (t, a) in withdraw_events:
            events.append((t, 1, a))
        events.sort(key=lambda e: (e[0], e[1]))
        level = 0.0
        last_t = None
        for (t, typ, amt) in events:
            if last_t is not None:
                elapsed = t - last_t
                if elapsed > 0:
                    level *= (1.0 - decay) ** elapsed
            if typ == 0:
                level += amt
            else:
                level -= amt
                if level < -EPS:
                    return False
                if level < 0:
                    level = 0.0
            last_t = t
        return True

    # Buffer 1: adds = stage-1 lots (O1 = R1 - S1); withdrawals = stage-2 lots' draws (R2)
    b1_adds = [(t, r - S1) for (t, r) in lots[1]]
    b1_withdraws = [(t, r) for (t, r) in lots[2]]
    if not simulate_buffer(1, b1_adds, b1_withdraws, d1):
        fail("Buffer 1 underflow")

    # Buffer 2: adds = stage-2 lots (O2 = R2 - S2); withdrawals = stage-3 lots' draws (R3)
    b2_adds = [(t, r - S2) for (t, r) in lots[2]]
    b2_withdraws = [(t, r) for (t, r) in lots[3]]
    if not simulate_buffer(2, b2_adds, b2_withdraws, d2):
        fail("Buffer 2 underflow")

    # Buffer 3: adds = stage-3 lots (O3 = R3 - S3); withdrawals = mandatory demand pulses
    b3_adds = [(t, r - S3) for (t, r) in lots[3]]
    b3_withdraws = list(zip(times, demands))
    if not simulate_buffer(3, b3_adds, b3_withdraws, d3):
        fail("Buffer 3 underflow (demand not met)")

    F = raw_total
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
