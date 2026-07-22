import sys, json, math, random, statistics, isorun

# ==========================================================================
# fsx_B_0904 -- thermal-hysteresis-idling (Format B, isolated candidate)
# Theme: "forge furnace idling between sporadic jobs".
#
# A forge furnace serves a stream of jobs separated by idle gaps of unknown
# length. After each job, the operator must choose ONE of K+1 graded SETBACK
# LEVELS (0 = hold at full operating temperature ... K = full shutdown) to
# hold for the WHOLE gap. Holding costs energy per unit time (cheapest at
# full shutdown); when the next job arrives, the furnace must be brought
# back to operating temperature, and that RESTART cost is STATE-DEPENDENT:
# it grows with how deep the chosen setback level cooled the furnace and
# with how long the gap lasted, via a CONCAVE saturating curve. Gap lengths
# are drawn from a hidden two-regime (short/long) process whose regime
# PERSISTS from gap to gap (autocorrelated) -- reading recently realized
# gaps lets a policy infer which regime is currently active and predict.
#
# The 40-gap test episode is graded in 10 causal CHUNKS of 4 gaps: before
# each chunk the candidate is invoked FRESH (isolated OS-sandboxed
# subprocess, per Format-B contract) with the TRUE gaps of all STRICTLY
# EARLIER chunks (plus a pre-episode calibration history) and must commit
# ONE setback level for the whole upcoming chunk. This guarantees the
# candidate can never see a gap it has not already causally observed.
# ==========================================================================

K = 4
TAU = [1.0, 0.7, 0.4, 0.15, 0.0]
HOLD_POWER = [12.0, 7.0, 3.5, 1.5, 0.0]
COOL_RATE = [0.010, 0.022, 0.040, 0.065, 0.10]
REHEAT_COEFF = 260.0
REHEAT_EXP = 0.4

CALIB = 150
M = 40
CHUNK = 4
N_CHUNKS = M // CHUNK
MAXG = 5000.0  # generation safety cap only; P(Exp(70) > 5000) ~ 1e-31, never actually hit

SPECS = [
    # (mu_short, mu_long, p_stay) -- p_stay is the per-gap regime persistence
    (4, 45, 0.90),
    (3, 60, 0.92),
    (5, 38, 0.50),   # control: near-i.i.d., low persistence advantage
    (6, 50, 0.85),
    (4, 70, 0.93),
    (5, 14, 0.85),
    (7, 16, 0.50),   # control: near-i.i.d.
    (4, 12, 0.90),
    (8, 15, 0.75),
    (6, 13, 0.88),
]


def cost1(level, g):
    """Energy cost of holding `level` for one gap of length g: linear holding
    power plus a CONCAVE (exponent < 1) saturating reheat cost that depends
    on the deficit reached (state-dependent restart cost)."""
    tv = TAU[level]
    d = (1.0 - tv) * (1.0 - math.exp(-COOL_RATE[level] * g))
    return HOLD_POWER[level] * g + REHEAT_COEFF * (d ** REHEAT_EXP)


def gen_episode(mu_s, mu_l, p_stay, rng):
    regime_short = rng.random() < 0.5
    seq = []
    for t in range(CALIB + M):
        if t > 0 and rng.random() >= p_stay:
            regime_short = not regime_short
        mu = mu_s if regime_short else mu_l
        seq.append(min(rng.expovariate(1.0 / mu), MAXG))
    return seq


def make_instances():
    out = []
    for si, (mu_s, mu_l, p_stay) in enumerate(SPECS):
        rng = random.Random(9000 + si)
        seq = gen_episode(mu_s, mu_l, p_stay, rng)
        calib_gaps = seq[:CALIB]
        episode_gaps = seq[CALIB:]

        w_hold = sum(cost1(0, g) for g in episode_gaps)
        w_off = sum(cost1(K, g) for g in episode_gaps)
        Wc = max(w_hold, w_off)
        Oc = sum(min(cost1(l, g) for l in range(K + 1)) for g in episode_gaps)

        pub_common = {
            "K": K, "tau": list(TAU), "hold_power": list(HOLD_POWER),
            "cool_rate": list(COOL_RATE), "reheat_coeff": REHEAT_COEFF,
            "reheat_exp": REHEAT_EXP, "mu_short": float(mu_s), "mu_long": float(mu_l),
            "p_stay": float(p_stay), "n_chunks": N_CHUNKS, "chunk_size": CHUNK,
            "calib_gaps": [round(x, 6) for x in calib_gaps],
        }
        out.append({
            "public": pub_common,
            "hidden": {"episode_gaps": episode_gaps, "W": Wc, "O": Oc},
        })
    return out


def baseline(inst):
    """Trivial-construction reference the evaluator computes itself: the
    worse of the two naive constant extremes (always-hold, always-shutdown)."""
    return inst["hidden"]["W"]


def score(inst, cand_path):
    """Drive the candidate causally, one isolated subprocess call per
    4-gap chunk, and replay its committed levels against the TRUE hidden
    gaps. Any invalid/failed chunk answer fails the WHOLE instance."""
    pub = inst["public"]
    gaps = inst["hidden"]["episode_gaps"]
    Wc, Oc = inst["hidden"]["W"], inst["hidden"]["O"]
    history = list(pub["calib_gaps"])
    total = 0.0
    for c in range(N_CHUNKS):
        view = dict(pub)
        view["chunk_index"] = c
        view["history"] = [round(x, 6) for x in history]
        del view["calib_gaps"]
        ans, st = isorun.run_candidate(cand_path, view, timeout=10)
        if st != "OK" or not isinstance(ans, dict) or "level" not in ans:
            return False, 0.0
        lv = ans["level"]
        if isinstance(lv, bool) or not isinstance(lv, (int, float)):
            return False, 0.0
        if lv != lv or lv in (float("inf"), float("-inf")):
            return False, 0.0
        lvi = int(round(lv))
        if abs(lv - lvi) > 1e-6 or lvi < 0 or lvi > K:
            return False, 0.0
        chunk_true = gaps[c * CHUNK:(c + 1) * CHUNK]
        for g in chunk_true:
            total += cost1(lvi, g)
        history.extend(chunk_true)
    if total != total or total < 0:
        return False, 0.0
    if Wc <= Oc + 1e-9:
        return True, Wc  # degenerate instance: any feasible answer -> ratio computed as 0 below
    return True, total


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        try:
            ok, obj = score(inst, cand)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        Wc, Oc = inst["hidden"]["W"], inst["hidden"]["O"]
        if Wc <= Oc + 1e-9:
            vec.append(0.0)
            continue
        r = (Wc - obj) / (Wc - Oc)
        r = max(0.0, min(1.0, r))
        vec.append(r if (r == r) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
