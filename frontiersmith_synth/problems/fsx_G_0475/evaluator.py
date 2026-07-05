#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0475 -- "Length-Generalization Probe: Formal-Language
Transducers" (family: ml-synth-generalization; format B, quality-metric).

THEME (tiny-transformer length-generalization probe).  A recurring finding in the
"tiny transformer probe" literature is that a model can nail a synthetic formal-
language task on the string lengths it was trained on, yet fall apart on LONGER
held-out lengths -- because it memorized a length-specific shortcut instead of the
true, length-independent rule.  This problem distills that phenomenon into a
deterministic, CPU-only probe: each instance is one synthetic formal-language
TRANSDUCTION task (a function f mapping an input string to an output string).  You
are shown a handful of (input, output) TRAINING pairs at SHORT lengths, plus a set
of longer, held-out TEST inputs.  You must predict the TEST outputs.  Memorizing
the training strings is worthless -- the test lengths are strictly longer and never
appear in training; you must recover the underlying rule and extrapolate it.

The tasks (per instance, one hidden rule): identity, reversal, bitwise complement,
a caesar shift, parity, modular symbol-counting, sorting, character doubling,
consecutive de-duplication, left-rotation, fixed-suffix extraction, adjacent-pair
swapping, and a composite (reverse-then-complement).  Several are classic length-
generalization TRAPS: e.g. the count-modulo task looks exactly like plain counting
until the count exceeds the modulus (which only happens at the longer test
lengths), and the composite rule is not something a small rule-library recognizes.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "alphabet": [c0, c1, ...],          # the input alphabet (ordered)
             "train": [[in_str, out_str], ...],  # SHORT-length training pairs
             "tests": [in_str, ...]}             # LONGER held-out inputs (predict these)
  stdout: ONE JSON object:
            {"pred": [out_str, ...]}             # one predicted output per test input,
                                                 # SAME order as "tests"
  A valid answer is a dict whose "pred" is a list of exactly len(tests) STRINGS.
  Anything else (wrong length, non-string element, non-dict, crash, timeout, non-
  JSON, nan/inf) -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the score is the fraction of
TEST inputs whose predicted output EXACTLY equals the true output f(input):
    q = (# exact matches) / (# tests)          in [0, 1]
The reported Ratio is the mean of q over all instances; the Vector holds the per-
instance q.  A pure echo/memorizer scores ~0.1 (only the identity instances); a
rule-library extrapolator scores high but NOT 1.0 -- the count-modulo traps and the
composite rule leave real headroom for a smarter length-generalizer.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS SANDBOX via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (train pairs + test
INPUTS).  The true test OUTPUTS and the hidden rule name live only in THIS parent
process, so a frame-walking / filesystem-snooping candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean q over all instances, in [0,1]>
  Vector: [q_1, q_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):                       # inclusive integer in [lo, hi]
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- the hidden rule library ---------------------
# Each task is a length-INDEPENDENT function f: input string -> output string.
# COUNT_MOD_M is the modulus for the count-modulo trap task.
COUNT_MOD_M = 7
SHIFT_ORDER = "abcd"


def apply_task(task, s):
    if task == "copy":
        return s
    if task == "reverse":
        return s[::-1]
    if task == "complement":                                   # bits
        return "".join("1" if c == "0" else "0" for c in s)
    if task == "shift":                                        # caesar +1 over SHIFT_ORDER
        n = len(SHIFT_ORDER)
        return "".join(SHIFT_ORDER[(SHIFT_ORDER.index(c) + 1) % n] for c in s)
    if task == "parity":                                       # bits -> single bit
        return "1" if (s.count("1") % 2) else "0"
    if task == "count_mod":                                    # count of 'a' modulo M
        return str(s.count("a") % COUNT_MOD_M)
    if task == "sort":
        return "".join(sorted(s))
    if task == "double":
        return "".join(c * 2 for c in s)
    if task == "dedup":                                        # collapse consecutive repeats
        out = []
        for c in s:
            if not out or out[-1] != c:
                out.append(c)
        return "".join(out)
    if task == "rotate":                                       # left-rotate by 1
        return s[1:] + s[:1] if s else s
    if task == "last2":                                        # fixed 2-char suffix
        return s[-2:]
    if task == "swap_pairs":                                   # swap adjacent pairs
        l = list(s)
        for i in range(0, len(l) - 1, 2):
            l[i], l[i + 1] = l[i + 1], l[i]
        return "".join(l)
    if task == "rev_complement":                               # composite: reverse then complement (bits)
        return "".join("1" if c == "0" else "0" for c in s)[::-1]
    raise ValueError("unknown task " + str(task))


# ----------------------------- instance family -----------------------------
def _gen_strings(nxt, alphabet, count, lo_len, hi_len):
    """Deterministically generate `count` strings with lengths in [lo_len, hi_len]."""
    A = len(alphabet)
    out = []
    for _ in range(count):
        L = nxt(lo_len, hi_len)
        out.append("".join(alphabet[nxt(0, A - 1)] for _ in range(L)))
    return out


def _build_instances():
    """Deterministic instance family.

    Each spec: (seed, task, alphabet, train_len_lo, train_len_hi,
                test_len_lo, test_len_hi).  Training lengths are SHORT; test lengths
    are strictly LONGER (length generalization).  count_mod uses SHORT training so
    count('a') < M there -- the modular wrap only shows up at test length (the trap).
    """
    specs = [
        (1101, "copy",          "abc",  3, 7, 13, 22),
        (1102, "copy",          "abc",  3, 7, 13, 22),
        (1103, "reverse",       "abc",  3, 7, 13, 22),
        (1104, "reverse",       "abc",  3, 7, 13, 22),
        (1105, "complement",    "01",   3, 7, 13, 22),
        (1106, "shift",         "abcd", 3, 7, 13, 22),
        (1107, "parity",        "01",   3, 7, 13, 22),
        (1108, "count_mod",     "ab",   3, 6, 14, 22),      # trap (short train, no wrap)
        (1109, "count_mod",     "ab",   3, 6, 14, 22),      # trap
        (1110, "sort",          "abc",  3, 7, 13, 22),
        (1111, "double",        "abc",  3, 7, 13, 22),
        (1112, "dedup",         "abc",  3, 7, 13, 22),
        (1113, "rotate",        "abc",  3, 7, 13, 22),
        (1114, "last2",         "abc",  3, 7, 13, 22),
        (1115, "swap_pairs",    "abc",  3, 7, 13, 22),
        (1116, "rev_complement", "01",  3, 7, 13, 22),      # composite / held-out
    ]
    N_TRAIN, N_TEST = 12, 10
    out = []
    for seed, task, alphabet, tl_lo, tl_hi, sl_lo, sl_hi in specs:
        nxt = _rng(seed)
        train_in = _gen_strings(nxt, alphabet, N_TRAIN, tl_lo, tl_hi)
        test_in = _gen_strings(nxt, alphabet, N_TEST, sl_lo, sl_hi)
        train = [[x, apply_task(task, x)] for x in train_in]
        test_out = [apply_task(task, x) for x in test_in]
        out.append({
            "name": f"probe{seed}_{task}",
            "alphabet": list(alphabet),
            "train": train,
            "tests": test_in,
            "test_out": test_out,          # HIDDEN
        })
    return out


# ----------------------------- validation / scoring ------------------------
def score(inst, answer):
    """Return (ok, q).  ok=False -> instance scores 0.  q = exact-match fraction."""
    if not isinstance(answer, dict):
        return False, 0.0
    pred = answer.get("pred")
    tests = inst["tests"]
    if not isinstance(pred, list) or len(pred) != len(tests):
        return False, 0.0
    for p in pred:
        if not isinstance(p, str):
            return False, 0.0
    truth = inst["test_out"]
    hits = sum(1 for p, t in zip(pred, truth) if p == t)
    return True, hits / len(truth)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {"name": inst["name"], "alphabet": list(inst["alphabet"]),
                  "train": [list(p) for p in inst["train"]], "tests": list(inst["tests"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, q = score(inst, ans)
        except Exception:
            ok, q = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        if not (q == q) or q in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        vec.append(q)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
