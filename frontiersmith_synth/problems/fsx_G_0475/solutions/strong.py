# TIER: strong
# Broad rule-library consistency search.  Enumerate a rich library of length-
# independent transduction rules; keep the FIRST rule consistent with every
# training pair, then extrapolate it to the longer test inputs.  This length-
# generalizes correctly on every rule it knows.
#
# It still leaves headroom, on purpose:
#   * the count-modulo TRAP: training (short) never wraps, so plain "count" is
#     consistent with training but WRONG on the longer tests where the count
#     exceeds the modulus;
#   * the composite reverse-then-complement rule is NOT in the library, so no rule
#     fits -> it falls back to echo and misses.
# A smarter length-generalizer (inferring the modulus, or composing primitives)
# would beat this.
import sys, json

inst = json.load(sys.stdin)
train = inst["train"]
tests = inst["tests"]

# infer a caesar-shift order from the declared alphabet
ALPHA = "".join(inst.get("alphabet", []))
ORDER = "".join(sorted(set(ALPHA))) if ALPHA else ""


def r_copy(s):
    return s


def r_reverse(s):
    return s[::-1]


def r_complement(s):
    return "".join("1" if c == "0" else "0" for c in s)


def r_parity(s):
    return "1" if (s.count("1") % 2) else "0"


def r_shift(s):
    n = len(ORDER)
    if n == 0:
        raise ValueError
    return "".join(ORDER[(ORDER.index(c) + 1) % n] for c in s)


def r_sort(s):
    return "".join(sorted(s))


def r_double(s):
    return "".join(c * 2 for c in s)


def r_dedup(s):
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def r_rotate(s):
    return s[1:] + s[:1] if s else s


def r_last2(s):
    return s[-2:]


def r_swap_pairs(s):
    l = list(s)
    for i in range(0, len(l) - 1, 2):
        l[i], l[i + 1] = l[i + 1], l[i]
    return "".join(l)


def r_count_plain(s):
    # plain (unbounded) count of 'a' -- matches the count-modulo trap on the short
    # training pairs, but does NOT apply the modulus on the longer tests.
    return str(s.count("a"))


# order: specific/common first
LIBRARY = [r_copy, r_reverse, r_complement, r_parity, r_shift, r_sort,
           r_double, r_dedup, r_rotate, r_last2, r_swap_pairs, r_count_plain]


def consistent(rule):
    try:
        for x, y in train:
            if rule(x) != y:
                return False
        return True
    except Exception:
        return False


chosen = None
for rule in LIBRARY:
    if consistent(rule):
        chosen = rule
        break

if chosen is None:
    pred = list(tests)                      # no rule fits -> echo fallback
else:
    try:
        pred = [chosen(x) for x in tests]
    except Exception:
        pred = list(tests)

print(json.dumps({"pred": pred}))
