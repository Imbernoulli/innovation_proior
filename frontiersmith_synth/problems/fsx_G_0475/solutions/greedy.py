# TIER: greedy
# Small fixed rule library.  Try a handful of the most common length-independent
# rules (identity, reversal, bitwise complement, parity); keep the first one that
# reproduces EVERY training pair, then apply it to the (longer) test inputs.  If
# none fit, fall back to echo.  This nails the easy tasks but is blind to sorting,
# doubling, shifts, counting, rotations, etc. -> a middle-of-the-ladder score.
import sys, json

inst = json.load(sys.stdin)
train = inst["train"]
tests = inst["tests"]


def r_copy(s):
    return s


def r_reverse(s):
    return s[::-1]


def r_complement(s):
    return "".join("1" if c == "0" else "0" for c in s)


def r_parity(s):
    return "1" if (s.count("1") % 2) else "0"


LIBRARY = [r_copy, r_reverse, r_complement, r_parity]


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
    pred = list(tests)                      # fall back to echo
else:
    pred = [chosen(x) for x in tests]

print(json.dumps({"pred": pred}))
