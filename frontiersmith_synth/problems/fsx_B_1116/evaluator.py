import sys, json, math, random, isorun
from collections import Counter

# ==========================================================================
# fsx_B_1116 -- counterpoint-rule-garden (Format B, isolated candidate)
# Theme: "first-species counterpoint writing policy".
#
# Mechanisms composed into ONE objective:
#   (1) species-rule-table:        every hard legality rule (which vertical
#       scale-degree distances are consonant/"perfect", the melodic leap cap,
#       the required minimum fraction of contrary motion) is DATA read from
#       the public instance, not baked into the statement's prose.
#   (2) contrary-motion-coupling:  a hard, GLOBAL, session-wide requirement
#       that a minimum fraction of the counterpoint line's melodic moves be
#       in contrary motion against the cantus firmus -- this couples the two
#       lines' direction choices across the whole piece, not note-by-note.
#   (3) corpus-level-style-diversity: after ALL 10 instances have been
#       answered, a SHARED bonus term rewards the *entropy* (across the whole
#       10-piece corpus) of WHERE each piece's unique melodic climax sits.
#       A candidate that always resolves every instance the same structural
#       way collapses this bonus for every single piece in the corpus, even
#       though each instance is scored by a fresh, isolated subprocess call
#       that never sees any other instance's answer. The only way to earn
#       this bonus is to treat the corpus itself as the unit of design: read
#       the instance's own id/corpus_size and deliberately diversify.
#
# Innovation hook: the best policy must structurally vary its OWN output
# distribution as a seeded function of instance_id -- it manages its own
# style across the corpus, not just one instance in isolation.
# ==========================================================================

RUNG_MOD = 7
CONSONANT_CLASSES = [0, 2, 4, 5]   # unison/8ve, 3rd, 5th, 6th  ((cp-cf) mod 7)
PERFECT_CLASSES = [0, 4]           # unison/8ve, 5th
BOUNDARY_CLASSES = [0, 4]          # first & last vertical must be perfect
CLIMAX_BUCKETS = 5


def _arch_walk(rng, L, start, turn_frac, noise_prob, big_step_prob):
    """Seeded cantus-firmus melody shaped as a single ascend-then-descend arch
    with its peak near turn_frac of the way through the piece -- the standard
    first-species convention (climax roughly two-thirds through). noise_prob
    occasionally wiggles AGAINST the arch trend so the line is not perfectly
    monotone; 0 noise (used by the dedicated contrary-motion trap instances)
    yields a long, hard, near-monotonic run."""
    turn_idx = max(1, min(L - 2, round(turn_frac * (L - 1))))
    cf = [start]
    for i in range(1, L):
        direction = 1 if i <= turn_idx else -1
        if rng.random() < noise_prob:
            direction = -direction
        step = direction * (2 if rng.random() < big_step_prob else 1)
        cf.append(cf[-1] + step)
    return cf


def _build_instance(spec):
    rng = random.Random(spec["seed"])
    L = spec["L"]
    cf = _arch_walk(rng, L, spec["start"], spec["turn_frac"],
                     spec.get("noise_prob", 0.3), spec.get("big_step_prob", 0.45))
    for pos in spec.get("force_plateau", []):
        pos = pos if pos >= 0 else L + pos
        cf[pos] = max(cf)
    lo_pad, hi_pad = spec.get("range_pad", (6, 8))
    lo = min(cf) - lo_pad
    hi = max(cf) + hi_pad
    rules = {
        "consonant_classes": list(CONSONANT_CLASSES),
        "perfect_classes": list(PERFECT_CLASSES),
        "boundary_classes": list(BOUNDARY_CLASSES),
        "max_leap": spec["max_leap"],
        "min_contrary_frac": spec["min_contrary_frac"],
        "climax_buckets": CLIMAX_BUCKETS,
    }
    return {
        "instance_id": spec["idx"],
        "corpus_size": spec["corpus_size"],
        "name": spec["name"],
        "cantus": cf,
        "cp_range": [lo, hi],
        "rules": rules,
    }


SPECS = [
    # "mild": full arch (climax ~70% through, standard convention), generous
    # register and leap room, modest contrary quota -- any careful heuristic
    # can satisfy the hard rules here.
    dict(name="cf0", seed=20101, L=8,  start=0, turn_frac=0.70, noise_prob=0.35, big_step_prob=0.4,
         max_leap=6, min_contrary_frac=0.22, range_pad=(6, 8)),
    dict(name="cf1", seed=20103, L=10, start=5, turn_frac=0.70, noise_prob=0.35, big_step_prob=0.4,
         max_leap=6, min_contrary_frac=0.22, range_pad=(6, 8)),
    dict(name="cf2", seed=20105, L=8,  start=3, turn_frac=0.65, noise_prob=0.35, big_step_prob=0.4,
         max_leap=6, min_contrary_frac=0.22, range_pad=(6, 8)),
    dict(name="cf3", seed=20107, L=9,  start=4, turn_frac=0.70, noise_prob=0.35, big_step_prob=0.4,
         max_leap=6, min_contrary_frac=0.22, range_pad=(6, 8)),
    dict(name="cf4", seed=20109, L=10, start=2, turn_frac=0.65, noise_prob=0.35, big_step_prob=0.4,
         max_leap=6, min_contrary_frac=0.22, range_pad=(6, 8)),
    # "trap": still the same 0.65-0.75 arch (so the climax still naturally
    # sits in the same relative zone every time -- the corpus-diversity trap
    # for ALL 10), but near-monotonic (noise_prob=0) with a TIGHT register
    # and short leash. A one-step-lookback heuristic that always grabs the
    # locally-best contrary tone burns its register room early and cannot
    # keep affording contrary moves once the long climb/descent runs on --
    # the global contrary-motion quota needs real planning, not a local rule.
    dict(name="cf5", seed=20102, L=10, start=2, turn_frac=0.70, noise_prob=0.0, big_step_prob=0.4,
         max_leap=3, min_contrary_frac=0.45, range_pad=(2, 3)),
    dict(name="cf6", seed=20104, L=11, start=9, turn_frac=0.72, noise_prob=0.0, big_step_prob=0.4,
         max_leap=3, min_contrary_frac=0.45, range_pad=(2, 3)),
    dict(name="cf7", seed=20106, L=12, start=1, turn_frac=0.67, noise_prob=0.0, big_step_prob=0.35,
         max_leap=3, min_contrary_frac=0.45, range_pad=(2, 3), force_plateau=[7, -2]),
    dict(name="cf8", seed=20108, L=13, start=14, turn_frac=0.70, noise_prob=0.0, big_step_prob=0.4,
         max_leap=3, min_contrary_frac=0.45, range_pad=(2, 3)),
    dict(name="cf9", seed=20110, L=14, start=1, turn_frac=0.68, noise_prob=0.05, big_step_prob=0.35,
         max_leap=3, min_contrary_frac=0.42, range_pad=(2, 3), force_plateau=[4, -3]),
]


def make_instances():
    n = len(SPECS)
    insts = []
    for i, s in enumerate(SPECS):
        s = dict(s); s["idx"] = i; s["corpus_size"] = n
        insts.append(_build_instance(s))
    return insts


def _validate_answer(inst, ans):
    """Returns (ok, contrary_excess, diversity_norm, climax_bucket)."""
    cantus = inst["cantus"]
    L = len(cantus)
    lo, hi = inst["cp_range"]
    rules = inst["rules"]
    cc = set(rules["consonant_classes"])
    pc = set(rules["perfect_classes"])
    bc = set(rules["boundary_classes"])
    max_leap = rules["max_leap"]
    min_cf = rules["min_contrary_frac"]
    buckets = rules["climax_buckets"]

    if not isinstance(ans, dict):
        return False, 0.0, 0.0, None
    cp = ans.get("cp")
    if not isinstance(cp, list) or len(cp) != L:
        return False, 0.0, 0.0, None
    for x in cp:
        if not isinstance(x, int) or isinstance(x, bool):
            return False, 0.0, 0.0, None
        if x < lo or x > hi:
            return False, 0.0, 0.0, None

    vclasses = [(cp[i] - cantus[i]) % RUNG_MOD for i in range(L)]
    for vc in vclasses:
        if vc not in cc:
            return False, 0.0, 0.0, None
    if vclasses[0] not in bc or vclasses[-1] not in bc:
        return False, 0.0, 0.0, None
    for i in range(L - 1):
        if vclasses[i] in pc and vclasses[i + 1] in pc and vclasses[i] == vclasses[i + 1]:
            return False, 0.0, 0.0, None
    for i in range(L - 1):
        if abs(cp[i + 1] - cp[i]) > max_leap:
            return False, 0.0, 0.0, None

    total_moves = L - 1
    contrary = 0
    for i in range(total_moves):
        dcp = cp[i + 1] - cp[i]
        dcf = cantus[i + 1] - cantus[i]
        if dcp != 0 and dcf != 0 and (dcp > 0) != (dcf > 0):
            contrary += 1
    contrary_frac = contrary / total_moves if total_moves > 0 else 1.0
    if contrary_frac < min_cf - 1e-9:
        return False, 0.0, 0.0, None

    mx = max(cp)
    if cp.count(mx) != 1:
        return False, 0.0, 0.0, None

    cats = []
    for i in range(total_moves):
        d = abs(cp[i + 1] - cp[i])
        if d <= 1:
            cats.append("step")
        elif d <= 3:
            cats.append("skip")
        else:
            cats.append("leap")
    cnt = Counter(cats)
    n = sum(cnt.values())
    ent = 0.0
    for v in cnt.values():
        p = v / n
        ent -= p * math.log(p)
    ent_norm = min(1.0, ent / math.log(3)) if n > 0 else 0.0

    climax_idx = cp.index(mx)
    bucket = min(buckets - 1, int(climax_idx / (L - 1) * buckets)) if L > 1 else 0

    contrary_excess = 0.0
    if min_cf < 1.0:
        contrary_excess = (contrary_frac - min_cf) / (1.0 - min_cf)
    contrary_excess = max(0.0, min(1.0, contrary_excess))

    return True, contrary_excess, ent_norm, bucket


def main():
    cand = sys.argv[1]
    insts = make_instances()
    results = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst, timeout=20)
        if st != "OK":
            results.append((False, 0.0, 0.0, None))
            continue
        try:
            r = _validate_answer(inst, ans)
        except Exception:
            r = (False, 0.0, 0.0, None)
        results.append(r)

    buckets = insts[0]["rules"]["climax_buckets"]
    valid_buckets = [r[3] for r in results if r[0]]
    corpus_bonus = 0.0
    if len(valid_buckets) >= 2:
        cnt = Counter(valid_buckets)
        n = len(valid_buckets)
        ent = 0.0
        for v in cnt.values():
            p = v / n
            ent -= p * math.log(p)
        corpus_bonus = ent / math.log(buckets) if buckets > 1 else 0.0
        corpus_bonus = max(0.0, min(1.0, corpus_bonus))

    vec = []
    for ok, cexc, divn, _bucket in results:
        if not ok:
            vec.append(0.0)
            continue
        r = 0.10 + 0.30 * cexc + 0.20 * divn + 0.25 * corpus_bonus
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
