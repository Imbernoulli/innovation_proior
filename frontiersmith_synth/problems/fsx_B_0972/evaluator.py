import sys, json, random, isorun

# ==========================================================================
# fsx_B_0972 -- online-macro-acquisition (Format B, isolated candidate)
# Theme: "invent shorthand live while the dictation continues"  --  a live
# stenographer coining permanent radio-brevity shorthand while the dictation
# keeps streaming in.  Mechanisms composed: irrevocable-online-commitment
# (a coined shorthand can never be redefined or dropped, and the codebook has
# a hard lifetime cap) + streaming-structure-estimation (the stenographer
# only ever sees the transcript UP TO NOW; the true dominant phrase of the
# whole session is only weakly visible early on -- it must be inferred from
# growing evidence, not chased by raw local frequency).
#
# PROTOCOL. The dictation session streams in ROUNDS. The same candidate
# program is invoked ONCE PER ROUND (a fresh, isolated subprocess call each
# time -- no shared memory between calls). Each call's stdin is the PUBLIC
# view for that round only: the codebook committed so far, the raw dictated
# lines of every EARLIER round ("history", already scored, provided for
# reference), and this round's NEW lines that must be transcribed right now.
# Future rounds are never sent -- the online nature is a real information
# restriction enforced by the harness, not just flavor text.
#
# SCORING. Per round: total tokens WRITTEN in the transcribed lines (a raw
# term or a shorthand reference each cost 1) plus a one-time entry fee for
# every newly coined shorthand this round (fee = overhead + phrase length).
# Total session cost = sum over all rounds, MINIMIZED. An identity
# transcription (never coin anything) is always a valid, safe fallback and
# costs exactly the raw token count -- the calibrated 0.1 baseline.
# ==========================================================================

LIB_MAXLEN = 24  # generous upper bound on a coined shorthand's phrase length


# ----------------------------- instance generator ---------------------------
def _build_instance(spec):
    seed = spec["seed"]
    L = spec["L"]
    n_d = spec["n_d"]
    n_distract = spec["n_distract"]
    c_schedule = spec["c_schedule"]
    fpr = spec["filler_per_round"]
    fla = spec["filler_len_avg"]
    overhead = spec["overhead"]
    cap = spec["cap"]
    rng = random.Random(seed)
    R = len(c_schedule)
    assert R >= n_distract

    # Opaque per-instance token tag derived from (but not textually equal to) the
    # seed, so the raw seed number never appears verbatim inside any token the
    # candidate can read -- avoids the cheapest form of instance-fingerprinting.
    _tagrng = random.Random(f"tag:{seed}")
    alnum = "abcdefghijklmnopqrstuvwxyz0123456789"
    tag = "".join(_tagrng.choice(alnum) for _ in range(10))

    templates = [[f"s{tag}_{k}_{p}" for p in range(L)] for k in range(n_distract + 1)]
    dom = templates[-1]
    distract_templates = templates[:n_distract]

    def pad(pool, lo=0, hi=2):
        return [rng.choice(pool) for _ in range(rng.randint(lo, hi))]

    def templ_line(tmpl, pool):
        return pad(pool) + list(tmpl) + pad(pool)

    def filler_line(pool):
        n = max(1, fla + rng.randint(-1, 1))
        return [rng.choice(pool) for _ in range(n)]

    rounds = []
    for r in range(R):
        # per-round-unique filler vocabulary: guarantees no ACCIDENTAL cross-round
        # n-gram recurrence -- only the planted templates ever genuinely recur.
        pool = [f"f{tag}_{r}_{i}" for i in range(40)]
        lines = []
        if r < n_distract:
            for _ in range(n_d):
                lines.append(templ_line(distract_templates[r], pool))
        for _ in range(c_schedule[r]):
            lines.append(templ_line(dom, pool))
        for _ in range(fpr):
            lines.append(filler_line(pool))
        rng.shuffle(lines)
        rounds.append(lines)
    return {"rounds": rounds, "cap": cap, "overhead": overhead, "name": spec["name"]}


def make_instances():
    specs = [
        dict(name="s0", seed=1001, L=8, overhead=4, n_d=55, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 150, 130], filler_per_round=3, filler_len_avg=4, cap=6),
        dict(name="s1", seed=1002, L=6, overhead=3, n_d=45, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 120, 100], filler_per_round=2, filler_len_avg=3, cap=6),
        dict(name="s2", seed=1003, L=7, overhead=4, n_d=50, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 160, 140], filler_per_round=3, filler_len_avg=5, cap=6),
        dict(name="s3", seed=1004, L=5, overhead=2, n_d=40, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 90, 80], filler_per_round=2, filler_len_avg=3, cap=6),
        dict(name="s4", seed=1005, L=8, overhead=4, n_d=60, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 200, 180], filler_per_round=4, filler_len_avg=4, cap=6),
        dict(name="s5", seed=1006, L=6, overhead=3, n_d=35, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 120, 105], filler_per_round=2, filler_len_avg=4, cap=6),
        dict(name="s6", seed=1007, L=7, overhead=3, n_d=45, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 130, 115], filler_per_round=3, filler_len_avg=4, cap=6),
        dict(name="s7_mild", seed=1008, L=6, overhead=3, n_d=15, n_distract=4,
             c_schedule=[10, 10, 10, 10, 10, 10], filler_per_round=2, filler_len_avg=3, cap=6),
        dict(name="s8_mild", seed=1009, L=5, overhead=3, n_d=10, n_distract=3,
             c_schedule=[3, 3, 3, 3, 3], filler_per_round=2, filler_len_avg=3, cap=6),
        dict(name="s9", seed=1010, L=8, overhead=4, n_d=52, n_distract=6,
             c_schedule=[2, 2, 2, 2, 3, 3, 140, 125], filler_per_round=3, filler_len_avg=4, cap=6),
    ]
    return [{"public": None, "hidden": _build_instance(s)} for s in specs]


def baseline(inst):
    h = inst["hidden"]
    return sum(len(line) for lines in h["rounds"] for line in lines)


# ----------------------------- per-round validation --------------------------
def _validate_round(ans, lines, library, cap, overhead):
    """Returns (ok, round_cost, updated_library) for ONE round's answer."""
    if not isinstance(ans, dict):
        return False, 0.0, library
    new_macros = ans.get("new_macros")
    rewrites = ans.get("rewrites")
    if not isinstance(new_macros, list) or not isinstance(rewrites, list):
        return False, 0.0, library
    if len(library) + len(new_macros) > cap:
        return False, 0.0, library

    new_patterns = []
    for nm in new_macros:
        if not isinstance(nm, dict):
            return False, 0.0, library
        pat = nm.get("pattern")
        if not isinstance(pat, list) or not (2 <= len(pat) <= LIB_MAXLEN):
            return False, 0.0, library
        if not all(isinstance(t, str) for t in pat):
            return False, 0.0, library
        new_patterns.append(pat)

    def_cost = sum(overhead + len(p) for p in new_patterns)
    updated_library = list(library) + [
        {"id": len(library) + i, "pattern": p} for i, p in enumerate(new_patterns)
    ]

    if len(rewrites) != len(lines):
        return False, 0.0, library

    round_token_cost = 0
    for line, rew in zip(lines, rewrites):
        if not isinstance(rew, list) or not all(isinstance(t, str) for t in rew):
            return False, 0.0, library
        if len(rew) == 0 and len(line) != 0:
            return False, 0.0, library
        expanded = []
        for tok in rew:
            if tok.startswith("$") and tok[1:].isdigit():
                mid = int(tok[1:])
                if mid < 0 or mid >= len(updated_library):
                    return False, 0.0, library
                expanded.extend(updated_library[mid]["pattern"])
            else:
                expanded.append(tok)
        if expanded != line:
            return False, 0.0, library
        round_token_cost += len(rew)

    round_cost = round_token_cost + def_cost
    if round_cost != round_cost:  # nan guard
        return False, 0.0, library
    return True, float(round_cost), updated_library


def score(inst, cand):
    """Runs the FULL multi-round online session for one instance against the
    candidate program (a fresh isolated subprocess call per round). Returns
    (ok, total_cost)."""
    h = inst["hidden"]
    cap, overhead = h["cap"], h["overhead"]
    library = []
    total_cost = 0.0
    for r, lines in enumerate(h["rounds"]):
        public = {
            "round": r,
            "total_rounds": len(h["rounds"]),
            "cap": cap,
            "overhead": overhead,
            "library": [dict(m) for m in library],
            "history": [ [list(ln) for ln in h["rounds"][rr]] for rr in range(r) ],
            "lines": [list(ln) for ln in lines],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            return False, 0.0
        try:
            ok, rcost, library = _validate_round(ans, lines, library, cap, overhead)
        except Exception:
            ok = False
        if not ok:
            return False, 0.0
        total_cost += rcost
    if total_cost <= 0.0:
        return False, 0.0
    return True, total_cost


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
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
