import sys, json, random, isorun

# ==========================================================================
# fsx_B_0174 -- offline-decision-policy (Format B, isolated candidate)
# Theme: "geothermal wells"  --  a downhole logging campaign.
#
# Each borehole is logged by a modular tool string built from a subset of the
# K sensor-module types. Before a module can read, it is calibrated/warmed at
# the wellhead IN SEQUENCE, and the rig CACHES the calibrated prefix: if the
# next tool string begins with the SAME ordered sequence of modules that some
# earlier string began with, that leading calibration is reused (cache HIT)
# and rig time is saved. You choose ONE global canonical ordering of the module
# types; every borehole assembles its required modules in that global relative
# order and the boreholes are logged in a FIXED field schedule.
#
# This is exactly a prefix (KV-cache) reuse problem: maximise the total number
# of reused calibration steps == minimise the number of distinct prefixes in
# the trie built from all tool-string sequences.
#
# Objective (MAX): the anchor's normalised hit-rate improvement over a naive
# reference ordering,  score = 0.1 + 0.8 * (hit - base_hit)/(1 - base_hit),
# clipped to [0,1].  The naive reference (natural module-id order) maps to 0.1.
# The evaluator computes the reuse itself via a pure-python trie; the candidate
# only ever sees the PUBLIC instance and returns an ordering.
# ==========================================================================

K_MODULES = 30
IMP_GAIN = 0.8          # slope of the normalised-improvement mapping


def _reuse(order, wells):
    """Total reused (cached) calibration steps and total steps, for a given
    global module ordering, under a growing prefix cache (trie)."""
    rank = {m: i for i, m in enumerate(order)}
    trie = {}
    total = 0
    reused = 0
    for w in wells:
        seq = sorted(w, key=lambda m: rank[m])
        total += len(seq)
        node = trie
        i = 0
        while i < len(seq) and seq[i] in node:
            node = node[seq[i]]
            i += 1
        reused += i
        for s in seq[i:]:
            node[s] = {}
            node = node[s]
    return reused, total


def _mapped(cand_reuse, ref_reuse, T):
    """Normalised hit-rate improvement over the reference, mapped to [0,1]."""
    if T <= 0:
        return 0.0
    hit = cand_reuse / T
    base = ref_reuse / T
    if base >= 1.0:
        imp = 0.0
    else:
        imp = (hit - base) / (1.0 - base)
    v = 0.1 + IMP_GAIN * imp
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def make_instances():
    """Deterministic, seeded geothermal logging campaigns.
    Structure combines (a) high-frequency common modules that a good order puts
    first, (b) tightly co-occurring module GROUPS that reward co-occurrence-aware
    grouping (frequency-sort alone scatters them), (c) medium-frequency
    independent noise modules, and (d) rare unique tags -- so that a naive order
    is poor, frequency-sort is decent, and a grouping/local-search order is best.
    No easy optimum; several viable strategies."""
    out = []
    for s in range(10):
        rng = random.Random(3000 + s)
        K = K_MODULES
        ids = list(range(K))
        rng.shuffle(ids)                     # popularity NOT aligned with module id
        commons = ids[:3]
        ptr = 3
        groups = []
        for gs in (3, 3, 4, 2):
            groups.append(ids[ptr:ptr + gs]); ptr += gs
        independents = ids[ptr:ptr + 6]; ptr += 6
        tags = ids[ptr:]
        M = rng.randint(180, 220)
        wells = []
        for _ in range(M):
            st = set()
            for g in commons:
                if rng.random() < 0.95:
                    st.add(g)
            grp = groups[rng.randrange(len(groups))]
            for m in grp:
                if rng.random() < 0.97:
                    st.add(m)
            for m in independents:
                if rng.random() < 0.35:
                    st.add(m)
            if rng.random() < 0.5:
                st.add(rng.choice(tags))
            if not st:
                st.add(rng.choice(ids))
            wells.append(sorted(st))
        rng.shuffle(wells)
        pub = {
            "K": K,
            "wells": wells,
            "baseline_order": list(range(K)),
        }
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    """Score of the naive reference construction (natural module-id order).
    By construction its normalised improvement is 0 -> maps to exactly 0.1."""
    pub = inst["public"]
    ref = pub["baseline_order"]
    r, _ = _reuse(ref, pub["wells"])
    _, T = _reuse(ref, pub["wells"])
    return _mapped(r, r, T)      # == 0.1


def score(inst, ans):
    """Strictly validate the candidate's ordering; return (ok, normalised score)."""
    pub = inst["public"]
    K = pub["K"]
    wells = pub["wells"]
    if not isinstance(ans, dict) or "order" not in ans:
        return False, None
    order = ans["order"]
    if not isinstance(order, list) or len(order) != K:
        return False, None
    seen = [False] * K
    clean = []
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return False, None
        if v < 0 or v >= K:
            return False, None
        if seen[v]:
            return False, None
        seen[v] = True
        clean.append(v)
    if not all(seen):
        return False, None
    ref = pub["baseline_order"]
    ref_reuse, T = _reuse(ref, wells)
    cand_reuse, _ = _reuse(clean, wells)
    v = _mapped(cand_reuse, ref_reuse, T)
    if v != v:
        return False, None
    return True, float(v)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
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
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
