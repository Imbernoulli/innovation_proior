#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the priority-inversion
scheduler problem.

<in>  : L J Tmax
        c_1 .. c_L                       (static ceiling priority per lock)
        J lines: pri arr dl w k  (len_1 lock_1) .. (len_k lock_k)   -- one job

<out> : exactly L whitespace-separated INTEGER tokens, each in {0,1,2}
        (one per lock, in lock order 1..L): 0=none, 1=inherit, 2=ceiling --
        the candidate's chosen protocol for every shared lock.

Feasibility: the artifact must be exactly L tokens, each an integer literal
in {0,1,2}; anything else (wrong count, non-integer, out of range, nan/inf)
-> Ratio: 0.0. We then run the SAME fixed-priority preemptive simulation as
the reference solutions with the candidate's per-lock protocol choice, and
separately with protocol "none" on every lock (the checker's own baseline
construction B). Lower simulated deadline-miss cost is better (minimization):

    sc = min(1000, 100 * B / max(1e-9, F))
    Ratio = sc / 1000
"""
import sys

PROTO_NAMES = ("none", "inherit", "ceiling")
CODE_TO_NAME = {0: "none", 1: "inherit", 2: "ceiling"}


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    L = int(nxt()); J = int(nxt()); Tmax = int(nxt())
    ceilings = {}
    for l in range(1, L + 1):
        ceilings[l] = int(nxt())
    jobs = []
    for _ in range(J):
        pri = int(nxt()); arr = int(nxt()); dl = int(nxt()); w = int(nxt())
        k = int(nxt())
        segs = []
        for _ in range(k):
            ln = int(nxt()); lk = int(nxt())
            segs.append((ln, lk))
        jobs.append({"pri": pri, "arr": arr, "dl": dl, "w": w, "segs": segs})
    return L, J, Tmax, ceilings, jobs


def simulate(L, jobs, ceilings, protocols, Tmax):
    """protocols: dict lock_id(1..L) -> one of PROTO_NAMES."""
    n = len(jobs)
    seg_idx = [0] * n
    rem = [jobs[i]["segs"][0][0] if jobs[i]["segs"] else 0 for i in range(n)]
    finish = [None] * n
    lock_holder = [None] * (L + 1)  # 1-indexed; None = free

    for t in range(Tmax):
        if all(f is not None for f in finish):
            break
        blocked_want = {}
        runnable = []
        for i in range(n):
            if finish[i] is not None:
                continue
            if jobs[i]["arr"] > t:
                continue
            seg_len, seg_lock = jobs[i]["segs"][seg_idx[i]]
            if seg_lock != 0 and lock_holder[seg_lock] not in (None, i):
                blocked_want.setdefault(seg_lock, []).append(i)
                continue
            runnable.append(i)
        if not runnable:
            continue

        def eff(i):
            base = jobs[i]["pri"]
            held = [l for l in range(1, L + 1) if lock_holder[l] == i]
            if not held:
                return base
            best = base
            for l in held:
                proto = protocols[l]
                if proto == "ceiling":
                    best = min(best, ceilings[l])
                elif proto == "inherit":
                    waiters = blocked_want.get(l, [])
                    if waiters:
                        best = min(best, min(jobs[w]["pri"] for w in waiters))
                # "none": no boost
            return best

        runnable.sort(key=lambda i: (eff(i), i))
        chosen = runnable[0]
        seg_len, seg_lock = jobs[chosen]["segs"][seg_idx[chosen]]
        if seg_lock != 0 and lock_holder[seg_lock] != chosen:
            lock_holder[seg_lock] = chosen
        rem[chosen] -= 1
        if rem[chosen] == 0:
            if seg_lock != 0:
                lock_holder[seg_lock] = None
            seg_idx[chosen] += 1
            if seg_idx[chosen] >= len(jobs[chosen]["segs"]):
                finish[chosen] = t + 1
            else:
                nlen, _ = jobs[chosen]["segs"][seg_idx[chosen]]
                rem[chosen] = nlen

    cost = 0
    for i in range(n):
        f = finish[i] if finish[i] is not None else Tmax
        late = max(0, f - jobs[i]["dl"])
        cost += jobs[i]["w"] * late
    return cost


def parse_output(path, L):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) != L:
        return None
    protocols = {}
    for idx, tok in enumerate(toks):
        # strict integer literal only: reject "nan"/"inf"/floats/anything else
        if not (tok.lstrip("+-").isdigit()):
            return None
        v = int(tok)
        if v not in CODE_TO_NAME:
            return None
        protocols[idx + 1] = CODE_TO_NAME[v]
    return protocols


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    L, J, Tmax, ceilings, jobs = read_input(in_path)

    protocols = parse_output(out_path, L)
    if protocols is None:
        print("infeasible artifact (need exactly %d integer tokens in {0,1,2})" % L)
        print("Ratio: 0.0")
        return 0

    F = simulate(L, jobs, ceilings, protocols, Tmax)
    baseline_protocols = {l: "none" for l in range(1, L + 1)}
    B = simulate(L, jobs, ceilings, baseline_protocols, Tmax)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
