#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0768 -- "Multi-Scale Motif Coder"
(family: motif-segment-coder; format B, quality-metric, objective=minimize).

THEME.  A symbol sequence (an "alphabet trace") contains a handful of hidden motifs --
short exact repeats -- planted at IRREGULAR, MULTI-SCALE spacing: some motifs are short
and recur close together, others are long and recur far apart, and both scales are
interleaved in the same trace, on top of a random background.

The candidate must compress the sequence into a DICTIONARY of reusable motifs plus a
SEGMENTATION of the sequence into literal runs and dictionary references, minimizing a
transparent bit-cost formula the evaluator computes deterministically:

    total_bits = sum( dict_header_bits + bits_per_symbol*len(e)  for e in dictionary
                       entries referenced by >=1 "ref" segment )
               + sum( bits_per_symbol * len            for each "lit" segment )
               + sum( ptr_bits                         for each "ref" segment )

A "ref" segment costs a FLAT ptr_bits regardless of the referenced motif's length, so
capturing a long-range repeat as ONE reference is far cheaper than encoding it (or
fragments of it) literally -- but every declared dictionary entry is paid for ONCE,
whether it is used once or many times, so padding the dictionary with rarely-reused
motifs is a net loss. This composes THREE mechanisms into one objective:
  - motif-length-diagnosis: figure out which length SCALES actually recur (vs. which
    lengths are just noise) before spending any dictionary budget on them;
  - greedy-dictionary-build: from the diagnosed scales, pick the motifs whose expected
    reuse pays for their one-time header cost;
  - dp-segment-boundary: given a fixed dictionary, the CHEAPEST way to cut the sequence
    into literal/reference segments is a shortest-path DP over positions, not a greedy
    left-to-right scan (greedy longest-match parsing is well known to be sub-optimal for
    this exact reason).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "seq": [N ints in [0, alphabet_size)],
             "alphabet_size": int, "bits_per_symbol": int, "ptr_bits": int,
             "dict_header_bits": int, "max_dict_entries": int, "max_motif_len": int}
  stdout: ONE JSON object:
            {"dictionary": [[...ints...], ...],     # each entry: 2..max_motif_len ints
             "segments": [{"type": "lit", "len": L}, {"type": "ref", "dict_idx": i}, ...]}
          Segments must partition [0, N) IN ORDER with no gaps or overlaps. A "ref"
          segment consumes len(dictionary[dict_idx]) symbols, and those symbols in `seq`
          at the segment's position must EXACTLY equal that dictionary entry's content.

  Any of the following makes the instance score 0.0: segments that don't exactly cover
  [0, N); a "ref" whose dict_idx is out of range or whose content mismatches `seq`; a
  dictionary entry outside [2, max_motif_len] symbols or with an out-of-range/non-int
  symbol; more than max_dict_entries declared entries; a crash, timeout, or non-JSON
  output.

SCORING (deterministic; no wall-time).  Per instance:
    y_base = bits_per_symbol * N              (the whole sequence encoded as one literal
                                                run -- the weak, dictionary-free baseline)
    y_cand = total_bits of the validated answer (see formula above)
  normalized (minimization, F/B analog):
    r = clamp( 0.1 * y_base / max(y_cand, 1e-12), 0, 1 )
  Matching the pure-literal baseline scores ~0.1; doing worse scores below 0.1; genuine
  compression scores higher. The instance mix includes cases where the two motif scales
  are tightly clustered (favorable to any reasonable matcher) and cases where they are
  scattered across a wide, irregular span (only a boundary-DP that isn't limited by a
  fixed lookback window can capture them), including harder held-out traces.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. All references and
validation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- cost model constants -------------------------
ALPHABET_SIZE = 40
BITS_PER_SYMBOL = 6
PTR_BITS = 6
DICT_HEADER_BITS = 6
MAX_DICT_ENTRIES = 64
MAX_MOTIF_LEN = 40
MIN_MOTIF_LEN = 2


def _dict_entry_cost(L):
    return DICT_HEADER_BITS + BITS_PER_SYMBOL * L


def _literal_cost(L):
    return BITS_PER_SYMBOL * L


# ----------------------------- instance family -----------------------------
def _build_seq(seed, n, short_len, long_len, n_short_motifs, n_long_motifs,
               short_occs, long_occs, min_gap, max_gap, cluster):
    rng = _rng(seed)
    seq = [rng(0, ALPHABET_SIZE - 1) for _ in range(n)]
    short_motifs = [tuple(rng(0, ALPHABET_SIZE - 1) for _ in range(short_len))
                     for _ in range(n_short_motifs)]
    long_motifs = [tuple(rng(0, ALPHABET_SIZE - 1) for _ in range(long_len))
                    for _ in range(n_long_motifs)]

    if cluster:
        # FRIENDLY layout: all occurrences of ONE motif are placed consecutively (small
        # local gaps between repeats); only the GROUP order is shuffled.
        groups = []
        for m in short_motifs:
            groups.append([('s', m)] * (short_occs // max(1, n_short_motifs)))
        for m in long_motifs:
            groups.append([('l', m)] * (long_occs // max(1, n_long_motifs)))
        for i in range(len(groups) - 1, 0, -1):
            j = rng(0, i)
            groups[i], groups[j] = groups[j], groups[i]
        events = [e for g in groups for e in g]
    else:
        # TRAP layout: individual occurrences of BOTH scales are fully interleaved and
        # shuffled, so recurrences of the same motif land arbitrarily far apart -- a
        # fixed lookback window misses them no matter the local gap size.
        events = []
        for k in range(short_occs):
            events.append(('s', short_motifs[k % n_short_motifs]))
        for k in range(long_occs):
            events.append(('l', long_motifs[k % n_long_motifs]))
        for i in range(len(events) - 1, 0, -1):
            j = rng(0, i)
            events[i], events[j] = events[j], events[i]

    lead = rng(min_gap, max_gap)
    gaps = [lead] + [rng(min_gap, max_gap) for _ in range(len(events) - 1)]
    total_content = sum(len(c) for _, c in events)
    room = n - total_content - 1
    total_gap = sum(gaps)
    if total_gap > max(room, 0) and total_gap > 0:
        scale = max(room, 0) / total_gap
        gaps = [max(0, int(g * scale)) for g in gaps]

    pos = 0
    for (kind, content), gap in zip(events, gaps):
        pos += gap
        L = len(content)
        if pos + L > n:
            pos = max(0, n - L)
        seq[pos:pos + L] = list(content)
        pos += L
    return seq


def _build_instances():
    """Deterministic instance family: (seed, n, short_len, long_len, n_short_motifs,
    n_long_motifs, short_occs, long_occs, min_gap, max_gap, cluster)."""
    specs = [
        (9101, 220, 5, 14, 2, 2, 22, 6, 2, 5, True),     # friendly: clustered repeats
        (9102, 260, 6, 16, 2, 2, 22, 6, 2, 6, True),     # friendly
        (9103, 320, 5, 18, 2, 2, 26, 10, 20, 110, False),# trap: irregular multi-scale
        (9104, 340, 7, 20, 2, 2, 22, 8, 20, 130, False), # trap
        (9105, 260, 6, 15, 2, 2, 22, 8, 2, 6, True),     # friendly
        (9106, 380, 5, 22, 2, 2, 30, 8, 30, 170, False), # trap: wide long-motif spacing
        (9107, 200, 4, 12, 2, 2, 18, 6, 2, 6, True),     # dense friendly
        (9108, 400, 8, 24, 2, 2, 20, 8, 4, 9, True),     # held-out: larger, friendly
        (9109, 340, 6, 17, 3, 2, 22, 10, 14, 100, False),# trap: 3 short-motif family
        (9110, 380, 7, 19, 2, 2, 24, 10, 20, 120, False),# held-out trap
    ]
    out = []
    for (seed, n, sl, ll, nsm, nlm, socc, locc, mingap, maxgap, cluster) in specs:
        seq = _build_seq(seed, n, sl, ll, nsm, nlm, socc, locc, mingap, maxgap, cluster)
        out.append({"name": f"trace{seed}", "n": n, "seq": seq})
    return out


# ----------------------------- baseline / validation / scoring -------------
def _baseline_bits(inst):
    return _literal_cost(inst["n"])


def _validate_and_score(inst, answer):
    """Return (ok: bool, total_bits: float|None)."""
    seq = inst["seq"]
    n = inst["n"]
    if not isinstance(answer, dict):
        return False, None
    dictionary = answer.get("dictionary")
    segments = answer.get("segments")
    if not isinstance(dictionary, list) or not isinstance(segments, list):
        return False, None
    if len(dictionary) > MAX_DICT_ENTRIES:
        return False, None

    parsed_dict = []
    for e in dictionary:
        if not isinstance(e, list) or not (MIN_MOTIF_LEN <= len(e) <= MAX_MOTIF_LEN):
            return False, None
        content = []
        for v in e:
            if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v < ALPHABET_SIZE):
                return False, None
            content.append(v)
        parsed_dict.append(tuple(content))

    if len(segments) > 4 * n + 16:
        return False, None

    pos = 0
    used = set()
    seg_cost = 0
    for seg in segments:
        if not isinstance(seg, dict):
            return False, None
        typ = seg.get("type")
        if typ == "lit":
            L = seg.get("len")
            if isinstance(L, bool) or not isinstance(L, int) or L <= 0:
                return False, None
            if pos + L > n:
                return False, None
            seg_cost += _literal_cost(L)
            pos += L
        elif typ == "ref":
            idx = seg.get("dict_idx")
            if isinstance(idx, bool) or not isinstance(idx, int):
                return False, None
            if not (0 <= idx < len(parsed_dict)):
                return False, None
            content = parsed_dict[idx]
            L = len(content)
            if pos + L > n:
                return False, None
            if tuple(seq[pos:pos + L]) != content:
                return False, None
            used.add(idx)
            seg_cost += PTR_BITS
            pos += L
        else:
            return False, None
    if pos != n:
        return False, None

    dict_cost = sum(_dict_entry_cost(len(parsed_dict[i])) for i in used)
    total = dict_cost + seg_cost
    return True, total


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {
            "name": inst["name"], "n": inst["n"], "seq": list(inst["seq"]),
            "alphabet_size": ALPHABET_SIZE, "bits_per_symbol": BITS_PER_SYMBOL,
            "ptr_bits": PTR_BITS, "dict_header_bits": DICT_HEADER_BITS,
            "max_dict_entries": MAX_DICT_ENTRIES, "max_motif_len": MAX_MOTIF_LEN,
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = _validate_and_score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = _baseline_bits(inst)
        r = 0.1 * b / max(obj, 1e-12)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
