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
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    # ---- parse instance ----
    try:
        it = iter(inp)
        V = int(next(it)); T = int(next(it))
        variants = []
        for _ in range(V):
            fleet = int(next(it))
            p = float(next(it))
            reward = float(next(it))
            penalty = float(next(it))
            D = int(next(it))
            W = float(next(it))
            variants.append(dict(fleet=fleet, p=p, reward=reward, penalty=penalty, D=D, W=W))
    except Exception:
        fail("bad instance")

    def finite(x):
        return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))

    for v in variants:
        if not (finite(v["p"]) and finite(v["reward"]) and finite(v["penalty"]) and finite(v["W"])):
            fail("non-finite instance field")

    # ---- parse participant output strictly: V blocks of (k, then k (t,c) pairs) ----
    try:
        oit = iter(out)
        schedules = []
        for v in range(V):
            k = int(next(oit))
            if k < 0 or k > 4 * T + 10:
                fail("k_v out of range for variant %d" % v)
            stages = []
            for _ in range(k):
                t_tok = next(oit)
                c_tok = next(oit)
                t = int(t_tok)
                c = int(c_tok)
                if not finite(t) or not finite(c):
                    fail("non-finite stage token")
                stages.append((t, c))
            schedules.append(stages)
        # nothing must remain
        try:
            next(oit)
            fail("trailing data in output")
        except StopIteration:
            pass
    except SystemExit:
        raise
    except Exception:
        fail("parse error / malformed schedule")

    # ---- feasibility ----
    for v_idx, (v, stages) in enumerate(zip(variants, schedules)):
        seen_t = set()
        total_c = 0
        for (t, c) in stages:
            if t < 0 or t >= T:
                fail("issue time out of range (variant %d)" % v_idx)
            if t in seen_t:
                fail("duplicate issue time (variant %d)" % v_idx)
            seen_t.add(t)
            if c < 1:
                fail("nonpositive cohort size (variant %d)" % v_idx)
            total_c += c
        if total_c > v["fleet"]:
            fail("scheduled devices exceed fleet (variant %d)" % v_idx)

    # ---- objective: sliding rollback-window simulation ----
    EPS = 1e-6

    def ev(v):
        return (1.0 - v["p"]) * v["reward"] - v["p"] * v["penalty"]

    def simulate(v, stages):
        total = 0.0
        accepted = []  # list of (t, c)
        e = ev(v)
        for (t, c) in sorted(stages, key=lambda x: x[0]):
            inflight = sum(cc * v["p"] for (tt, cc) in accepted if tt > t - v["D"])
            if inflight + c * v["p"] <= v["W"] + EPS:
                accepted.append((t, c))
                total += c * e
        return total

    F = 0.0
    for v, stages in zip(variants, schedules):
        F += simulate(v, stages)

    # ---- internal baseline B: one cautious single-shot cohort per variant at t=0 ----
    def baseline(v):
        if v["p"] <= 0:
            return 0.0
        c = math.floor(v["W"] / v["p"])
        c = min(c, v["fleet"])
        if c <= 0:
            return 0.0
        return c * ev(v)

    B = sum(baseline(v) for v in variants)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * max(0.0, F) / B)
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
