# TIER: strong
"""The insight: the choice of protocol per lock is NOT a single global rule
(inheritance always, or ceiling always) -- it depends on that lock's own
contention pattern (does a real high-priority waiter actually show up, and
when, relative to who else might steal cycles from the holder first?).
Since the number of locks is small, the per-lock choice can be decided
exactly by simulating every combination of {none, inherit, ceiling}^L with
the SAME deterministic scheduler the checker uses, and keeping the
assignment with the lowest total deadline-miss cost. This is a genuine
reformulation (decompose the policy space and search it exactly) rather
than "apply one recipe everywhere"."""
import sys
from itertools import product

PROTOS = ("none", "inherit", "ceiling")
NAME_TO_CODE = {"none": 0, "inherit": 1, "ceiling": 2}


def read_input(data):
    it = iter(data)

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
    n = len(jobs)
    seg_idx = [0] * n
    rem = [jobs[i]["segs"][0][0] if jobs[i]["segs"] else 0 for i in range(n)]
    finish = [None] * n
    lock_holder = [None] * (L + 1)

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


def main():
    data = sys.stdin.read().split()
    L, J, Tmax, ceilings, jobs = read_input(data)

    best_cost = None
    best_assign = None
    for combo in product(PROTOS, repeat=L):
        protocols = {l: combo[l - 1] for l in range(1, L + 1)}
        c = simulate(L, jobs, ceilings, protocols, Tmax)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_assign = combo
    print(" ".join(str(NAME_TO_CODE[p]) for p in best_assign))


if __name__ == "__main__":
    main()
