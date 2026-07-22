# TIER: strong
# INSIGHT (the genuine leverage, not "EDF plus more iterations"):
#  (1) BATCH-AWARE SEQUENCING: same-family jobs are setup-free back to back, so an
#      ordering that visits a family and drains as much of its pending queue as still
#      meets deadlines amortizes ONE setup over many jobs -- unlike EDF, which hops
#      families every step whenever deadlines interleave across families.
#  (2) FAMILY-CLUSTERING DIAGNOSIS: which family to visit next is decided by comparing
#      each family's *most urgent pending deadline* (a family-level EDF), not a fixed
#      family order -- this reads the instance's setup matrix and deadline structure
#      instead of assuming any canonical family sequence.
#  (3) REINSERTION LOCAL MOVE: clustering an entire family as ONE contiguous block is
#      itself sometimes a trap -- a family can hold one high-weight, tight-deadline job
#      stranded among many low-value, loose-deadline packmates. Visiting the whole
#      family early to save that job wastes time on the packmates; visiting it at its
#      natural slot loses the stranded job. The fix is to remove single jobs from the
#      current sequence and try reinserting them at every position, keeping any move
#      that raises total admitted weight -- this can pull the stranded job out to an
#      early slot (paying one extra setup) while leaving the rest of its family later.
#
# We build THREE candidate base sequences (plain EDF as a safety net, the best full
# family-block permutation, and the batch-interleave heuristic above), refine EACH with
# reinsertion local search to a local optimum, and keep whichever refined sequence
# admits the most weight. Because EDF is always one of the candidates, this can never
# do worse than the naive recipe -- it only ever finds more.
import sys, json, itertools


def simulate(inst, order):
    n = inst["n"]; fam = inst["fam"]; p = inst["p"]; d = inst["d"]; w = inst["w"]; setup = inst["setup"]
    seen = set()
    t = 0.0
    prev = None
    gained = 0.0
    for idx in order:
        if not (0 <= idx < n) or idx in seen:
            return None
        seen.add(idx)
        f = fam[idx]
        su = 0.0 if prev is None or prev == f else setup[prev][f]
        tentative = t + su + p[idx]
        if tentative <= d[idx] + 1e-9:
            t = tentative
            prev = f
            gained += w[idx]
    return gained


def edf_order(inst):
    n = inst["n"]; d = inst["d"]
    return sorted(range(n), key=lambda i: (d[i], i))


def family_jobs(inst):
    n = inst["n"]; fam = inst["fam"]; d = inst["d"]
    fj = {}
    for i in range(n):
        fj.setdefault(fam[i], []).append(i)
    for f in fj:
        fj[f].sort(key=lambda i: (d[i], i))
    return fj


def family_block_order(inst):
    fj = family_jobs(inst)
    fams = list(fj.keys())
    best_order, best_val = None, -1.0
    # F is small (<=6) in every instance, so an exhaustive family-order search is cheap
    for perm in itertools.permutations(fams):
        order = []
        for f in perm:
            order.extend(fj[f])
        val = simulate(inst, order)
        if val is not None and val > best_val:
            best_val = val
            best_order = order
    return best_order


def batch_interleave_order(inst):
    """Family-granularity EDF with batching: repeatedly visit the family whose next
    pending job has the smallest deadline, consume that family's queue while jobs still
    fit, and stop batching (re-pick a family) the moment the next job in the current
    family would miss."""
    fam = inst["fam"]; p = inst["p"]; d = inst["d"]; setup = inst["setup"]
    fj = family_jobs(inst)
    ptrs = {f: 0 for f in fj}
    order = []
    t = 0.0
    prev = None
    remaining = sum(len(v) for v in fj.values())
    while remaining > 0:
        best_f, best_d = None, None
        for f, jobs in fj.items():
            if ptrs[f] < len(jobs):
                dd = d[jobs[ptrs[f]]]
                if best_d is None or dd < best_d:
                    best_d = dd
                    best_f = f
        if best_f is None:
            break
        jobs = fj[best_f]
        su = 0.0 if prev is None or prev == best_f else setup[prev][best_f]
        first = True
        while ptrs[best_f] < len(jobs):
            idx = jobs[ptrs[best_f]]
            s = su if first else 0.0
            tentative = t + s + p[idx]
            if tentative <= d[idx] + 1e-9:
                t = tentative
                order.append(idx)
                prev = best_f
                ptrs[best_f] += 1
                remaining -= 1
                first = False
            else:
                break
        if first:
            # this family's next job doesn't fit right now even alone -- defer it
            # permanently (it's revisited by the reinsertion pass below if it can help)
            ptrs[best_f] += 1
            remaining -= 1
    return order


def reinsertion(inst, order, passes=3):
    n = inst["n"]
    order = list(order)
    cur = simulate(inst, order)
    if cur is None:
        cur = -1.0
    for _ in range(passes):
        improved = False
        for j in range(n):
            if j in order:
                pos = order.index(j)
                base = order[:pos] + order[pos + 1:]
            else:
                base = list(order)
            best_val, best_pos = cur, None
            for pos in range(len(base) + 1):
                cand = base[:pos] + [j] + base[pos:]
                v = simulate(inst, cand)
                if v is not None and v > best_val:
                    best_val, best_pos = v, pos
            if best_pos is not None:
                order = base[:best_pos] + [j] + base[best_pos:]
                cur = best_val
                improved = True
        if not improved:
            break
    return order, cur


def solve(inst):
    candidates = [edf_order(inst)]
    fb = family_block_order(inst)
    if fb is not None:
        candidates.append(fb)
    candidates.append(batch_interleave_order(inst))

    best_order, best_val = None, -1.0
    for c in candidates:
        o2, v2 = reinsertion(inst, c, passes=3)
        if v2 > best_val:
            best_order, best_val = o2, v2
    return best_order


inst = json.load(sys.stdin)
order = solve(inst)
print(json.dumps({"order": order}))
