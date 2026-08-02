#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE adhesive-interface-design instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small -> large/adversarial.

Story: a bond line between two substrates with mismatched thermal-expansion
coefficients is split into N discrete segments. A library of M candidate
adhesive layer types is given (sorted by increasing shear stiffness k_j; each
type also has a shear-strength capacity s_j, which the generator makes
increasing in k_j too -- a stiffer adhesive is also individually a STRONGER
one, exactly the "stiffer wins" intuition the trap exploits). The solver
assigns one adhesive type to every segment: the artifact IS this per-segment
choice, i.e. the compliant-layer grading profile along the bond line.

Mechanism composition:
  - thermal-expansion-mismatch: a fixed CTE mismatch dAlpha and a held-out
    thermal cycling profile (dT_1..dT_C) drive a differential-slip forcing
    term along the bond line.
  - interfacial-stress-concentration: the checker's two-sweep shear-lag
    recursion concentrates the resulting shear stress near the free EDGES of
    the bond line, and MORE stiffness makes that concentration worse (shorter
    decay length -> more mismatch strain dumped onto fewer, more stressed
    segments) -- while in the sub-critical size regime a stiffer *uniform*
    bond line still outperforms a soft one (matches "stiffer wins" at small
    scale), so naive stiffness-maximizing looks individually rational there.
  - compliant-layer-grading: the artifact is exactly a per-segment stiffness
    profile; a graded (soft-edge / stiff-core) profile is the only way to cap
    the edge concentration without giving up the core's stiffness advantage.

Trap (>=3 of 10, ids 8/9/10 here): N is calibrated (via an internal, seeded
search over candidate N -- no wall-time/GPU/randomness beyond testId) to sit
well past the size at which uniform-max-stiffness's edge stress exceeds its
own strength cap: cycling life collapses to ~0 there, while a uniform-softest
design (the checker's own baseline) and a graded design both remain healthy.
On the other 7 ids, N is calibrated so uniform-max-stiffness is genuinely the
best UNIFORM choice (life comfortably above uniform-softest) -- greedy is a
plausible, individually-rational recipe there, just not the best available
move; a graded design still beats it by exploiting the edge/core split.
"""
import sys
import random
import math

P_EXP = 3  # fixed Basquin-type fatigue exponent (stated in statement.md)

TRAP_IDS = {8, 9, 10}


# ---------------------------------------------------------------- physics --
def sim_H(N, Csub, k_list):
    """Homogeneous unit-load sweep (delta[0]=1, no thermal forcing). Returns
    1-indexed (H, shear) arrays of length N+1. Pure math tool used to solve
    the free-free thermal boundary condition below -- NOT a mechanical test."""
    H = [0.0]*(N+1); slip = [0.0]*(N+1); shear = [0.0]*(N+1)
    slip[0] = 1.0
    for i in range(1, N+1):
        shear[i] = k_list[i-1]*slip[i-1]
        H[i] = H[i-1] + shear[i]
        slip[i] = slip[i-1] + Csub*H[i]
    return H, shear


def sim_T(N, Csub, dAlpha, k_list):
    """Particular thermal sweep (delta[0]=0, unit-dT forcing dAlpha each step)."""
    T = [0.0]*(N+1); tslip = [0.0]*(N+1); tshear = [0.0]*(N+1)
    for i in range(1, N+1):
        tshear[i] = k_list[i-1]*tslip[i-1]
        T[i] = T[i-1] + tshear[i]
        tslip[i] = tslip[i-1] + Csub*T[i] + dAlpha
    return T, tshear


def cycling_life(N, Csub, dAlpha, k_list, s_list, dTs):
    """Deterministic stress model -> cycles-to-failure for a given per-segment
    design. Returns 0.0 on any non-finite/degenerate outcome."""
    H, shear = sim_H(N, Csub, k_list)
    if H[N] <= 0 or not math.isfinite(H[N]):
        return 0.0
    T, tshear = sim_T(N, Csub, dAlpha, k_list)
    if not math.isfinite(T[N]):
        return 0.0
    d0 = -T[N]/H[N]
    R = 0.0
    for i in range(1, N+1):
        th = tshear[i] + d0*shear[i]
        if not math.isfinite(th):
            return 0.0
        r = abs(th)/s_list[i-1]
        if r > R:
            R = r
    if R <= 0:
        return 0.0
    Q = sum(abs(dt)**P_EXP for dt in dTs)
    try:
        denom = Q*(R**P_EXP)
    except OverflowError:
        return 0.0
    if not math.isfinite(denom) or denom <= 0:
        return 0.0
    val = 1.0/denom
    return val if math.isfinite(val) else 0.0


def make_lib(k0, r, M, s0, slope):
    k_lib = [round(k0*(r**j), 4) for j in range(M)]
    s_lib = [round(s0 + slope*(k**0.5), 4) for k in k_lib]
    return k_lib, s_lib


def adv_at_N(N, Csub, dAlpha, k_lib, s_lib, dTs):
    Lsoft = cycling_life(N, Csub, dAlpha, [k_lib[0]]*N, [s_lib[0]]*N, dTs)
    Lmax = cycling_life(N, Csub, dAlpha, [k_lib[-1]]*N, [s_lib[-1]]*N, dTs)
    if Lsoft <= 0:
        return 0.0
    return Lmax/Lsoft


def calibrate_sweet_N(Csub, dAlpha, k_lib, s_lib, dTs, lo, hi, target):
    """Smallest-log-distance-to-target N (robust to the near-discontinuous
    collapse -- a naive 'first N above target' overshoots past narrow windows)."""
    best_gap, best_N, best_adv = None, lo, -1.0
    logt = math.log(target)
    for N in range(lo, hi+1):
        adv = adv_at_N(N, Csub, dAlpha, k_lib, s_lib, dTs)
        gap = abs(math.log(max(adv, 1e-12)) - logt)
        if best_gap is None or gap < best_gap:
            best_gap, best_N, best_adv = gap, N, adv
    return best_N, best_adv


def find_trap_N(sweet_N, Csub, dAlpha, k_lib, s_lib, dTs, hi):
    for N in range(sweet_N+1, hi):
        if adv_at_N(N, Csub, dAlpha, k_lib, s_lib, dTs) < 1e-4:
            return N
    return hi


def design_ramp(N, M, w, mid):
    w = min(w, N//2)
    d = [mid]*N
    for i in range(w):
        idx = int(round(i/(w-1)*mid)) if w > 1 else mid
        d[i] = idx; d[N-1-i] = idx
    return d


def strong_search_life(N, M, Csub, dAlpha, k_lib, s_lib, dTs):
    """Used only INSIDE the generator's own calibration to keep headroom sane
    (reject a draw whose best graded design would blow past the score cap)."""
    best = -1.0
    cand_w = sorted(set([1,2,3,4,5,6,8,10,12,15,18,22,26,30,35,40,45,50,55,60, N//2]))
    for w in cand_w:
        if w > N//2:
            continue
        for mid in range(M):
            d = design_ramp(N, M, w, mid)
            L = cycling_life(N, Csub, dAlpha, [k_lib[a] for a in d], [s_lib[a] for a in d], dTs)
            if L > best:
                best = L
    for j in range(M):
        L = cycling_life(N, Csub, dAlpha, [k_lib[j]]*N, [s_lib[j]]*N, dTs)
        if L > best:
            best = L
    return best


# --------------------------------------------------------------- build() --
def build(tid):
    is_trap = tid in TRAP_IDS
    base_seed = 500000 + 977*tid
    attempt = 0
    N = M = Csub = dAlpha = dTs = k_lib = s_lib = None

    while True:
        rng = random.Random(base_seed + attempt*104729)
        # round Csub/dAlpha to the SAME precision they will be printed+re-parsed
        # with (8dp) -- the calibration below must operate on the exact numbers
        # the checker will see, not full float precision: near the collapse
        # cliff a print/parse round-trip can otherwise flip the outcome.
        Csub = round(rng.uniform(3.2e-4, 5.8e-4), 8)
        r = rng.uniform(2.2, 3.0)
        M = rng.choice([5, 6, 7])
        slope = rng.uniform(3.3, 5.8)
        k0 = rng.uniform(25, 48)
        s0 = rng.uniform(3.5, 5.5)
        dAlpha = round(rng.uniform(2.0e-5, 4.2e-5), 8)
        dTmag = rng.uniform(38, 72)
        C = rng.randint(6, 10)
        dTs = [round(dTmag*rng.uniform(0.78, 1.18)*(1 if rng.random() < 0.5 else -1), 3)
               for _ in range(C)]
        k_lib, s_lib = make_lib(k0, r, M, s0, slope)  # already rounded to 4dp

        lo, hi_search = 10, 90
        sweet_N, sweet_adv = calibrate_sweet_N(Csub, dAlpha, k_lib, s_lib, dTs, lo, hi_search, target=2.3)

        def stable_band(Ncenter):
            """Require Ncenter-2..Ncenter+2 to ALL sit comfortably on the same
            side of the collapse cliff (no >4x jump across the window) -- a
            knife-edge N is fragile to any downstream rounding."""
            vals = [adv_at_N(Ncenter + d, Csub, dAlpha, k_lib, s_lib, dTs) for d in (-2, -1, 0, 1, 2)]
            return min(vals) > 0 and max(vals)/max(min(vals), 1e-12) < 4.0

        found = False
        if not is_trap:
            if 1.9 <= sweet_adv <= 6.0 and stable_band(sweet_N):
                N_try = sweet_N
                Lsoft = cycling_life(N_try, Csub, dAlpha, [k_lib[0]]*N_try, [s_lib[0]]*N_try, dTs)
                Lstrong = strong_search_life(N_try, M, Csub, dAlpha, k_lib, s_lib, dTs)
                if Lsoft > 0 and Lstrong/Lsoft <= 8.5:
                    N = N_try
                    found = True
        else:
            trap_N = find_trap_N(sweet_N, Csub, dAlpha, k_lib, s_lib, dTs, hi=200)
            N_try = int(trap_N * 1.35) + 4
            deep_ok = (trap_N < 195 and N_try <= 130
                       and adv_at_N(N_try, Csub, dAlpha, k_lib, s_lib, dTs) < 1e-6
                       and adv_at_N(N_try + 3, Csub, dAlpha, k_lib, s_lib, dTs) < 1e-6)
            if deep_ok:
                Lsoft = cycling_life(N_try, Csub, dAlpha, [k_lib[0]]*N_try, [s_lib[0]]*N_try, dTs)
                Lstrong = strong_search_life(N_try, M, Csub, dAlpha, k_lib, s_lib, dTs)
                if Lsoft > 0 and Lstrong/Lsoft <= 8.5:
                    N = N_try
                    found = True

        if found:
            break
        attempt += 1
        if attempt > 40:
            N = sweet_N if not is_trap else min(int(sweet_N*2.5) + 4, 130)
            break

    lines = [f"{N} {M}",
             f"{Csub:.8f} {dAlpha:.8f}",
             f"{len(dTs)}",
             " ".join(f"{dt:.3f}" for dt in dTs)]
    for (k, s) in zip(k_lib, s_lib):
        lines.append(f"{k:.4f} {s:.4f}")
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
