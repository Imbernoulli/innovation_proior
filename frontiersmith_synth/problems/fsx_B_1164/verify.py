#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the cleanroom-airlock-caste problem.

Feasibility (any violation -> Ratio: 0.0):
  - every job id 1..J is serviced by exactly one robot, exactly once
  - within a robot's event list (as given, in order), events are chronologically
    consistent: a job's start time >= the robot's current free time and >= the
    job's release; a decon cycle can only be joined at/after the robot is free
  - a job may only start if its grade <= the robot's current contamination level
    (contamination is a one-way ratchet: it only ever becomes numerically LOWER,
    i.e. dirtier, as jobs are worked, until a decon resets it to L)
  - all NC decon cycles are mutually non-overlapping in time (single shared
    airlock == a mutex) and each has at most C participants

Objective (minimize): F = sum(weight_j * max(0, finish_j - deadline_j)) + KCOST * NC
The checker also builds its own naive reference schedule (round-robin job
assignment, ignoring grade, with an individual un-batched decon cycle fired
every time one is needed) to get a baseline cost B, then prints
    Ratio = min(1.0, 0.1 * B / max(1e-9, F))
"""
import sys


def fail(reason):
    sys.stdout.write("INFEASIBLE: %s\n" % reason)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def read_tokens(path, cap_tokens=6_000_000):
    try:
        with open(path, "r") as f:
            data = f.read()
    except Exception as e:
        fail(f"cannot read file: {e}")
    toks = data.split()
    if len(toks) > cap_tokens:
        fail("output far too large")
    return toks


def parse_int(tok):
    """Strict integer parse (rejects nan/inf/floats/garbage)."""
    if tok is None:
        return None
    s = tok
    neg = s.startswith("-")
    body = s[1:] if neg else s
    if body == "" or not body.isdigit():
        return None
    return int(s)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_tokens(in_path)
    it = iter(itoks)

    def nxt_in():
        try:
            return next(it)
        except StopIteration:
            fail("truncated input (bug in generator)")

    L = int(nxt_in()); R = int(nxt_in()); J = int(nxt_in())
    T = int(nxt_in()); C = int(nxt_in()); KCOST = int(nxt_in())

    jobs = []  # 0-indexed here; job id j (1-based) -> jobs[j-1]
    for _ in range(J):
        g = int(nxt_in()); rel = int(nxt_in()); dur = int(nxt_in())
        dl = int(nxt_in()); w = int(nxt_in())
        jobs.append((g, rel, dur, dl, w))

    # ---------------- generic bounded-token output parser ----------------
    def make_parser(otoks):
        n = len(otoks)
        posbox = [0]

        def get():
            if posbox[0] >= n:
                return None
            v = otoks[posbox[0]]
            posbox[0] += 1
            return v

        def get_int(lo=None, hi=None):
            v = get()
            iv = parse_int(v)
            if iv is None:
                fail("non-integer / nan / inf / missing token in output")
            if lo is not None and iv < lo:
                fail("value out of range (too small)")
            if hi is not None and iv > hi:
                fail("value out of range (too large)")
            return iv

        def get_lit(word):
            v = get()
            if v != word:
                fail(f"expected literal '{word}', got {v!r}")

        return get, get_int, get_lit

    # ---------------- parse & simulate a submitted itinerary ----------------
    def parse_and_simulate(otoks, label):
        get, get_int, get_lit = make_parser(otoks)

        NC = get_int(lo=0, hi=max(4 * (R + J) + 10, 100000))
        cyc_start = []
        for _ in range(NC):
            s = get_int(lo=0, hi=10**9)
            cyc_start.append(s)

        # mutex: cycles must be pairwise non-overlapping in time
        order = sorted(range(NC), key=lambda i: cyc_start[i])
        for k in range(1, NC):
            prev, cur = order[k - 1], order[k]
            if cyc_start[cur] < cyc_start[prev] + T:
                fail("two decon cycles overlap on the shared airlock")

        cyc_count = [0] * NC
        seen_job = [False] * (J + 1)
        total_cost = 0

        for r in range(1, R + 1):
            get_lit("ROBOT")
            rid = get_int(lo=1, hi=R)
            if rid != r:
                fail(f"ROBOT block out of order (expected {r}, got {rid})")
            m = get_int(lo=0, hi=max(4 * J + 10, 100000))
            cur_time = 0
            cur_contam = L
            for _ in range(m):
                tag = get()
                if tag == "J":
                    jid = get_int(lo=1, hi=J)
                    s = get_int(lo=0, hi=10**9)
                    if seen_job[jid]:
                        fail(f"job {jid} serviced more than once")
                    seen_job[jid] = True
                    g, rel, dur, dl, w = jobs[jid - 1]
                    if s < cur_time:
                        fail(f"robot {r} job {jid} starts before robot is free")
                    if s < rel:
                        fail(f"robot {r} job {jid} starts before release")
                    if g > cur_contam:
                        fail(f"robot {r} job {jid} needs decon first (grade {g} > contam {cur_contam})")
                    finish = s + dur
                    cur_time = finish
                    cur_contam = g
                    late = finish - dl
                    if late > 0:
                        total_cost += w * late
                elif tag == "D":
                    cid = get_int(lo=1, hi=max(NC, 1))
                    if cid > NC:
                        fail("decon cycle id out of range")
                    st = cyc_start[cid - 1]
                    if cur_time > st:
                        fail(f"robot {r} cannot join cycle {cid}: not free until {cur_time} > cycle start {st}")
                    cyc_count[cid - 1] += 1
                    cur_time = st + T
                    cur_contam = L
                else:
                    fail(f"unknown event tag {tag!r}")

        # trailing garbage is ignored (bounded by cap_tokens already)
        for jid in range(1, J + 1):
            if not seen_job[jid]:
                fail(f"job {jid} never serviced")
        for cid in range(NC):
            if cyc_count[cid] > C:
                fail(f"decon cycle {cid + 1} over capacity ({cyc_count[cid]} > {C})")

        total_cost += KCOST * NC
        return total_cost

    otoks = read_tokens(out_path)
    F = parse_and_simulate(otoks, "participant")

    # ---------------- internal naive baseline B ----------------
    # Round-robin job assignment (by (release, job_id) order) ignoring grade
    # entirely; every time a robot's next job needs a higher grade than its
    # current contamination, it fires its OWN individual decon cycle right
    # away (no batching with anyone else). The shared airlock mutex is still
    # respected (cycles queue if the airlock is busy).
    order_jobs = sorted(range(1, J + 1), key=lambda j: (jobs[j - 1][1], j))
    base_cur_time = [0] * (R + 1)
    base_cur_contam = [L] * (R + 1)
    airlock_free = 0
    base_cost = 0
    base_nc = 0
    for idx, jid in enumerate(order_jobs):
        r = (idx % R) + 1
        g, rel, dur, dl, w = jobs[jid - 1]
        t = base_cur_time[r]
        if g > base_cur_contam[r]:
            start = max(airlock_free, t)
            end = start + T
            airlock_free = end
            base_nc += 1
            base_cur_time[r] = end
            base_cur_contam[r] = L
            t = end
        s = max(t, rel)
        finish = s + dur
        base_cur_time[r] = finish
        base_cur_contam[r] = g
        late = finish - dl
        if late > 0:
            base_cost += w * late
    base_cost += KCOST * base_nc
    B = max(base_cost, 1)  # keep numerator positive/sane

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d NC_used_by_participant_accounted_in_F=yes" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
