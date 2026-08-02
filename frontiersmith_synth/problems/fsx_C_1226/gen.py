#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE garbage-collector-schedule instance to stdout.
Deterministic: all randomness seeded ONLY from testId.
"""
import sys, random

# ---------------------------------------------------------------------------
# Shared cohort simulator (kept byte-identical in verify.py). Objects are
# tracked in per-step "cohorts" (all objects allocated at the same step share
# a lifetime), so state is O(#live cohorts) <= O(T).
# ---------------------------------------------------------------------------
def simulate(T, Y, K, Bpause, c0_minor, c0_major, k_young, k_old, f_rate, penalty,
             alloc, lifetime, actions):
    if len(actions) != T:
        return False, "wrong action count", 0.0, 0
    young = []   # [remaining, death_time, survive_count]
    old = []     # [remaining, death_time]
    total = 0.0
    violations = 0
    for t in range(1, T + 1):
        a = actions[t - 1]
        if a not in (0, 1, 2):
            return False, f"invalid action token at step {t}", 0.0, 0
        if a != 0:
            pre_y = sum(c[0] for c in young)
            nxt = []
            for rem, death, surv in young:
                if t >= death:
                    continue
                surv += 1
                if surv >= K:
                    old.append([rem, death])
                else:
                    nxt.append([rem, death, surv])
            young = nxt
            cost = c0_minor + k_young * pre_y
            if a == 2:
                pre_o = sum(c[0] for c in old)
                nxt_o = []
                for rem, death in old:
                    if t >= death:
                        continue
                    nxt_o.append([rem, death])
                old = nxt_o
                cost = c0_major + k_young * pre_y + k_old * pre_o
            if cost > Bpause + 1e-9:
                violations += 1
                total += penalty * (cost - Bpause)
            total += cost
        if alloc[t - 1] > 0:
            young.append([alloc[t - 1], t + lifetime[t - 1], 0])
        yres = sum(c[0] for c in young)
        if yres > Y:
            return False, f"young heap overflow at step {t}: {yres} > {Y}", 0.0, 0
        ores = sum(c[0] for c in old)
        total += f_rate * (yres + ores)
    return True, "", total, violations


def trivial_max_resident(T, K, alloc, lifetime):
    """Max young-resident under the 'collect every step' policy, uncapped."""
    young = []
    best = 0
    for t in range(1, T + 1):
        pre_y = sum(c[0] for c in young)
        nxt = []
        for rem, death, surv in young:
            if t >= death:
                continue
            surv += 1
            if surv < K:
                nxt.append([rem, death, surv])
        young = nxt
        if alloc[t - 1] > 0:
            young.append([alloc[t - 1], t + lifetime[t - 1], 0])
        yres = sum(c[0] for c in young)
        best = max(best, yres)
    return best


def greedy_actions(T, Y, K, alloc, lifetime):
    """Reference 'collect only when the heap would overflow' policy."""
    young = []
    actions = [0] * T
    for t in range(1, T + 1):
        pre_y = sum(c[0] for c in young)
        if pre_y + alloc[t - 1] > Y:
            actions[t - 1] = 1
            nxt = []
            for rem, death, surv in young:
                if t >= death:
                    continue
                surv += 1
                if surv < K:
                    nxt.append([rem, death, surv])
            young = nxt
        if alloc[t - 1] > 0:
            young.append([alloc[t - 1], t + lifetime[t - 1], 0])
    return actions


def build_profile(rng, tid):
    """Return (T, baseline_lo, baseline_hi, bursts) where bursts is a list of
    (start, length, rate_lo, rate_hi). >=3 of the 10 testIds get real bursts."""
    burst_ids = {4, 6, 7, 8, 9, 10}
    if tid == 1:
        T = 24; lo, hi = 1, 3; bursts = []
    elif tid == 2:
        T = 34; lo, hi = 1, 5; bursts = []
    elif tid == 3:
        T = 42; lo, hi = 2, 6; bursts = []
    elif tid == 5:
        T = 60; lo, hi = 1, 4; bursts = []
    elif tid == 4:
        T = 50; lo, hi = 1, 3
        bursts = [(20, 9, 22, 30)]
    elif tid == 6:
        T = 68; lo, hi = 1, 3
        bursts = [(15, 8, 20, 26), (45, 8, 20, 26)]
    elif tid == 7:
        T = 80; lo, hi = 1, 4
        bursts = [(30, 12, 24, 32)]
    elif tid == 8:
        T = 95; lo, hi = 1, 4
        bursts = [(20, 10, 26, 34), (60, 10, 26, 34)]
    elif tid == 9:
        T = 112; lo, hi = 1, 3
        bursts = [(10, 9, 28, 36), (48, 9, 28, 36), (86, 9, 28, 36)]
    else:  # 10 (largest / hardest)
        T = 135; lo, hi = 1, 3
        bursts = [(12, 11, 30, 40), (55, 11, 30, 40), (95, 11, 30, 40)]
    return T, lo, hi, bursts


def make_alloc(rng, T, lo, hi, bursts):
    alloc = [rng.randint(lo, hi) for _ in range(T)]
    for start, length, rlo, rhi in bursts:
        for i in range(length):
            t = start + i
            if 1 <= t <= T:
                alloc[t - 1] = rng.randint(rlo, rhi)
    return alloc


def make_lifetime(rng, T, K):
    """Most objects are short-lived; a minority are long-lived enough to be
    promoted after K survivals (generational-survival mechanism)."""
    life = []
    for t in range(1, T + 1):
        roll = rng.random()
        if roll < 0.65:
            life.append(rng.randint(1, 3))
        elif roll < 0.90:
            life.append(rng.randint(4, 10))
        else:
            life.append(rng.randint(12, max(13, T // 2)))
    return life


def window_max_sum(alloc, w):
    w = max(1, min(w, len(alloc)))
    cur = sum(alloc[:w])
    best = cur
    for i in range(w, len(alloc)):
        cur += alloc[i] - alloc[i - w]
        best = max(best, cur)
    return best


def main():
    tid = int(sys.argv[1])
    rng = random.Random(1_000_003 * tid + 7)

    T, lo, hi, bursts = build_profile(rng, tid)
    K = 3
    alloc = make_alloc(rng, T, lo, hi, bursts)
    lifetime = make_lifetime(rng, T, K)

    # --- size Y: physical heap capacity is provisioned for the worst known
    # burst (survive it fully UNCOLLECTED), not for steady-state traffic.
    # Smooth instances therefore get a heap only modestly bigger than what
    # normal pacing needs; bursty instances get a heap sized to the burst,
    # which is the mismatch "collect only when full" cannot see coming.
    if bursts:
        long_w = min(T, max(length for _, length, _, _ in bursts) + 2)
    else:
        long_w = min(T, 7)
    long_peak = window_max_sum(alloc, long_w)
    Y = max(10, int(long_peak * 1.15) + 5)
    for _ in range(20):
        ga = greedy_actions(T, Y, K, alloc, lifetime)
        feas, _, _, _ = simulate(T, Y, K, 10 ** 9, 1, 1, 1, 1, 0.0, 0, alloc, lifetime, ga)
        if feas:
            break
        Y = int(Y * 1.3) + 5
    else:
        Y = int(Y * 2) + 20  # last-resort safety net

    c0_minor = 10
    c0_major = 30
    k_young = 1
    k_old = 2
    f_rate = 0.05
    penalty = 5  # cost PER UNIT of overage above Bpause -- severity-scaled,
                 # so one huge overdue pause is punished far more than many
                 # small ones (not just "a violation happened").

    # Pause budget paces against the peak batch a MAXIMALLY frequent
    # collector would still see (the structurally-unavoidable floor set by
    # the promotion threshold and short-lived garbage), with headroom --
    # so a genuinely rate-paced schedule can clear the budget every time.
    # Y, in contrast, is sized to survive a burst fully UNCOLLECTED, which
    # is far above that floor on bursty instances -- the mismatch "collect
    # only when full" cannot see coming.
    floor = trivial_max_resident(T, K, alloc, lifetime)
    target_batch = max(3, round(floor * 1.8))
    Bpause = c0_minor + k_young * target_batch

    out = []
    out.append(f"{T} {Y} {K} {Bpause} {c0_minor} {c0_major} {k_young} {k_old} {f_rate:.4f} {penalty}")
    out.append(" ".join(map(str, alloc)))
    out.append(" ".join(map(str, lifetime)))
    print("\n".join(out))


if __name__ == "__main__":
    main()

