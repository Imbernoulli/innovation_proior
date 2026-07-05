# TIER: strong
# Recover the matching algorithm from the labelled examples and generalise it to the
# OOD queries. A cable is called well-spliced iff:
#   (1) it is a proper stack-matched Dyck word over the K connector types, AND
#   (2) its maximum nesting depth <= D_hat, where D_hat is inferred conservatively as
#       the deepest nesting ever observed among the WELL-SPLICED training cables.
# (1) generalises perfectly to arbitrary length (solves the length-OOD split). (2) is
# a defensible inductive bias, but the training set never probes the true depth budget,
# so D_hat under-estimates it -> the deep-but-valid OOD cables are missed -> headroom.
import sys, json

inst = json.load(sys.stdin)
opens = inst["open_symbols"]
closes = inst["close_symbols"]
open_of = {c: i for i, c in enumerate(opens)}
close_of = {c: i for i, c in enumerate(closes)}


def match_depth(s):
    """Return (well_matched, max_depth). max_depth meaningful only if well_matched."""
    stack = []
    maxd = 0
    for ch in s:
        if ch in open_of:
            stack.append(open_of[ch])
            if len(stack) > maxd:
                maxd = len(stack)
        elif ch in close_of:
            if not stack or stack[-1] != close_of[ch]:
                return False, 0
            stack.pop()
        else:
            return False, 0
    return (len(stack) == 0), maxd


# infer the depth budget from the positively-labelled training cables
d_hat = 0
for s, lab in inst["train"]:
    if lab == 1:
        ok, d = match_depth(s)
        if ok and d > d_hat:
            d_hat = d
if d_hat == 0:
    d_hat = 1

labels = []
for s in inst["queries"]:
    ok, d = match_depth(s)
    labels.append(1 if (ok and d <= d_hat) else 0)

print(json.dumps({"labels": labels}))
