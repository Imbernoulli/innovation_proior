# TIER: strong
# INSIGHT: the small "soft" jobs are disguised re-dress actions.  You have to run
# them anyway, and each one shaves the wheel.  So the real problem is to schedule
# production and this HIDDEN maintenance jointly on one timeline -- inserting a
# soft job right before wear gets dangerous is a re-dress you were going to be
# paid to perform, at a fraction of T_r.  Explicit re-dresses are only for the
# residual damage that soft capacity cannot absorb.
#
# Construction: run big hard jobs while the wheel is fresh; keep wear inside a low
# band by spending soft jobs (cheapest-first) as maintenance; fall back to an
# explicit re-dress only when the soft pool is empty and wear is still high.  Then
# a short local search (reset toggles + swaps) polishes the schedule.  The greedy
# biggest-first-with-threshold-resets schedule is included as a fallback seed, so
# the result is never worse than that recipe.
import sys


def simulate(seq, T_r, W_max, jobs):
    w = 0
    cost = 0
    for tok in seq:
        if tok == 0:
            cost += T_r
            w = 0
        else:
            s, d = jobs[tok - 1]
            cost += s * (1 + w) * (1 + w)
            w = w + d
            if w < 0:
                w = 0
            elif w > W_max:
                w = W_max
    return cost


def build_band(cap, hard, soft_pool0, W_max):
    soft_pool = list(soft_pool0)
    w = 0
    seq = []
    for (s, d, idx) in hard:
        while w > cap and soft_pool:
            sj = soft_pool.pop(0)
            seq.append(sj[2])
            w = max(0, w + sj[1])
        if w > cap and not soft_pool:
            seq.append(0)
            w = 0
        seq.append(idx)
        w = w + d
        if w < 0:
            w = 0
        elif w > W_max:
            w = W_max
    # leftover soft jobs -- process at the end where wear is low (cheap)
    for sj in soft_pool:
        seq.append(sj[2])
        w = max(0, w + sj[1])
    return seq


def greedy_seed(hard, soft):
    order = sorted(hard + soft, key=lambda x: -x[0])
    theta = 3
    out = []
    w = 0
    for (s, d, ix) in order:
        if w > theta:
            out.append(0)
            w = 0
        out.append(ix)
        w = w + d
        if w < 0:
            w = 0
    return out


def local_search(seq, T_r, W_max, jobs, budget):
    best = list(seq)
    bc = simulate(best, T_r, W_max, jobs)
    it = 0
    improved = True
    while improved and it < budget:
        improved = False
        # 1) remove each redundant re-dress
        i = 0
        while i < len(best):
            if best[i] == 0:
                cand = best[:i] + best[i + 1:]
                c = simulate(cand, T_r, W_max, jobs)
                if c < bc:
                    best, bc = cand, c
                    improved = True
                    continue
            i += 1
            it += 1
            if it >= budget:
                break
        # 2) try inserting a re-dress before each job position
        i = 0
        while i < len(best) and it < budget:
            if best[i] != 0 and (i == 0 or best[i - 1] != 0):
                cand = best[:i] + [0] + best[i:]
                c = simulate(cand, T_r, W_max, jobs)
                if c < bc:
                    best, bc = cand, c
                    improved = True
            i += 1
            it += 1
        # 3) adjacent swaps of jobs
        i = 0
        while i + 1 < len(best) and it < budget:
            if best[i] != 0 and best[i + 1] != 0:
                cand = list(best)
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = simulate(cand, T_r, W_max, jobs)
                if c < bc:
                    best, bc = cand, c
                    improved = True
            i += 1
            it += 1
    return best, bc


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    T_r = int(next(it))
    W_max = int(next(it))
    jobs = []
    for i in range(N):
        s = int(next(it))
        d = int(next(it))
        jobs.append((s, d))

    hard = [(jobs[i][0], jobs[i][1], i + 1) for i in range(N) if jobs[i][1] > 0]
    soft = [(jobs[i][0], jobs[i][1], i + 1) for i in range(N) if jobs[i][1] < 0]
    hard.sort(key=lambda x: -x[0])                    # big hard jobs first
    soft_pool0 = sorted(soft, key=lambda x: (x[0], x[1]))  # cheap, most-restoring first

    candidates = []
    for cap in range(0, W_max + 1):
        candidates.append(build_band(cap, hard, soft_pool0, W_max))
    candidates.append(greedy_seed(hard, soft))

    best = None
    bc = None
    for seq in candidates:
        c = simulate(seq, T_r, W_max, jobs)
        if bc is None or c < bc:
            best, bc = seq, c

    best, bc = local_search(best, T_r, W_max, jobs, budget=4000)

    sys.stdout.write(" ".join(str(x) for x in best))


if __name__ == "__main__":
    main()
