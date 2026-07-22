import sys, json, random, isorun

# ==========================================================================
# fsx_B_0919 -- thermal-spread-placement (Format B, isolated candidate)
# Theme: datacenter rack workload placement under thermal limits.
#
# A 3x3 rack grid of N=9 nodes serves a fixed stream of M jobs over a T-step
# horizon. The candidate sees the WHOLE job stream up front (arrival step,
# duration/demand, heat generated while active) and must, in ONE shot,
# schedule every job onto (node, start_time >= arrival), respecting node
# exclusivity (one job occupies a node at a time). Each occupied node
# generates heat while it runs a job; heat DIFFUSES to grid neighbors every
# step and decays; crossing an upper threshold HI throttles the node to a
# crawl (throttle_rate << 1), and -- HYSTERESIS -- the node stays throttled
# until it cools all the way down to a LOWER threshold LO < HI, not merely
# back below HI. Throttled nodes still (weakly) generate heat and diffuse
# it into neighbors, so a hotspot can drag a whole neighborhood into
# throttle and hold it there for a long time (far longer than it took to
# get hot). Objective: total useful compute delivered (throttle-adjusted),
# summed over jobs, normalized against the absolute ceiling (finishing
# every job at full untouched rate).
# ==========================================================================

T = 60
ROWS, COLS = 3, 3
N = ROWS * COLS
HI = 7.0
THROTTLE_RATE = 0.12


def neighbors(i):
    r, c = divmod(i, COLS)
    out = []
    if r > 0: out.append(i - COLS)
    if r < ROWS - 1: out.append(i + COLS)
    if c > 0: out.append(i - 1)
    if c < COLS - 1: out.append(i + 1)
    return out


NEIGH = [neighbors(i) for i in range(N)]

# (seed_off, M, burst, decay, alpha, LO)  -- burst=True bunches most arrivals
# early + strong diffusion + harsh hysteresis gap = the planted trap: a
# "pack the momentarily-coolest node" policy builds a hotspot that throttles
# a whole neighborhood for most of the remaining horizon.
SPECS = [
    (101, 30, False, 0.14, 0.02, 5.5),   # control: spread arrivals, weak coupling, mild hysteresis
    (102, 28, False, 0.13, 0.03, 5.0),   # control
    (103, 32, False, 0.10, 0.05, 3.5),   # mild
    (104, 36, True, 0.09, 0.06, 3.0),    # mild-bursty
    (105, 42, True, 0.07, 0.06, 2.0),    # TRAP
    (106, 44, True, 0.06, 0.05, 1.5),    # TRAP
    (107, 46, True, 0.06, 0.08, 1.2),    # TRAP
    (108, 42, True, 0.07, 0.06, 2.0),    # trap-ish
    (109, 26, False, 0.14, 0.02, 5.5),   # control
    (110, 48, True, 0.06, 0.10, 1.0),    # TRAP (harshest)
]


def gen_jobs(rng, M, burst):
    jobs = []
    for jid in range(M):
        if burst and rng.random() < 0.8:
            arrival = rng.randint(0, max(1, T // 4) - 1)
        else:
            arrival = rng.randint(0, T - 1)
        demand = rng.randint(3, 9)
        heat_rate = round(rng.uniform(0.3, 1.0), 4)
        jobs.append({"id": jid, "arrival": arrival, "demand": demand, "heat_rate": heat_rate})
    jobs.sort(key=lambda j: (j["arrival"], j["id"]))
    return jobs


def make_instances():
    out = []
    for si, (seed_off, M, burst, decay, alpha, LO) in enumerate(SPECS):
        rng = random.Random(4200 + seed_off)
        jobs = gen_jobs(rng, M, burst)
        D = sum(j["demand"] for j in jobs)
        pub = {
            "T": T, "N": N, "grid_rows": ROWS, "grid_cols": COLS,
            "decay": decay, "alpha": alpha, "HI": HI, "LO": LO,
            "throttle_rate": THROTTLE_RATE,
            "jobs": jobs,
        }
        out.append({"public": pub, "hidden": {"D": float(D)}})
    return out


def baseline(inst):
    """Reference the evaluator computes itself (no candidate involved): the
    absolute ceiling if every job ran to completion at the untouched full
    rate (never throttled, never overlapping). Real schedules score below
    this; it is only used to normalize."""
    return inst["hidden"]["D"]


def simulate(pub, schedule):
    """Ground-truth physics simulation. `schedule`: dict job_id -> (node, start)
    for scheduled jobs only (skipped jobs absent). Returns (ok, per_job_contrib
    dict) or (False, None) on any infeasibility (overlap / out-of-range)."""
    jobs_by_id = {j["id"]: j for j in pub["jobs"]}
    Tn, Nn = pub["T"], pub["N"]
    decay, alpha, HIv, LOv, trate = pub["decay"], pub["alpha"], pub["HI"], pub["LO"], pub["throttle_rate"]

    occ = [[None] * Tn for _ in range(Nn)]  # occ[node][t] = job_id or None
    for jid, (node, start) in schedule.items():
        j = jobs_by_id.get(jid)
        if j is None:
            return False, None
        if not isinstance(node, int) or not isinstance(start, int):
            return False, None
        if node < 0 or node >= Nn:
            return False, None
        if start < j["arrival"] or start + j["demand"] > Tn:
            return False, None
        for t in range(start, start + j["demand"]):
            if occ[node][t] is not None:
                return False, None  # exclusivity violation
            occ[node][t] = jid

    H = [0.0] * Nn
    throttled = [False] * Nn
    contrib = {jid: 0.0 for jid in schedule}

    for t in range(Tn):
        m = [trate if throttled[i] else 1.0 for i in range(Nn)]
        gen = [0.0] * Nn
        for i in range(Nn):
            jid = occ[i][t]
            if jid is not None:
                hr = jobs_by_id[jid]["heat_rate"]
                gen[i] = hr * m[i]
                contrib[jid] += m[i]
        newH = [0.0] * Nn
        for i in range(Nn):
            diff = sum(H[k] - H[i] for k in NEIGH[i])
            newH[i] = H[i] + gen[i] - decay * H[i] + alpha * diff
        H = newH
        for i in range(Nn):
            if throttled[i]:
                if H[i] <= LOv:
                    throttled[i] = False
            else:
                if H[i] > HIv:
                    throttled[i] = True

    return True, contrib


def score(inst, answer):
    pub = inst["public"]
    if not isinstance(answer, dict) or "schedule" not in answer:
        return False, 0.0
    sched_list = answer["schedule"]
    if not isinstance(sched_list, list):
        return False, 0.0
    valid_ids = {j["id"] for j in pub["jobs"]}
    schedule = {}
    seen_ids = set()
    for e in sched_list:
        if not isinstance(e, dict) or "id" not in e or "node" not in e:
            return False, 0.0
        jid = e["id"]
        if isinstance(jid, bool) or not isinstance(jid, int) or jid not in valid_ids:
            return False, 0.0
        if jid in seen_ids:
            return False, 0.0  # duplicate entry for same job (skip or real)
        seen_ids.add(jid)
        node = e["node"]
        if isinstance(node, bool) or not isinstance(node, int):
            return False, 0.0
        if node == -1:
            continue  # explicit skip
        start = e.get("start")
        if isinstance(start, bool) or not isinstance(start, int):
            return False, 0.0
        schedule[jid] = (node, start)

    ok, contrib = simulate(pub, schedule)
    if not ok:
        return False, 0.0
    total = sum(contrib.values())
    if total != total or total < -1e-9:
        return False, 0.0
    return True, max(0.0, total)


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        D = baseline(inst)
        r = obj / D if D > 0 else 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r if (r == r) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
