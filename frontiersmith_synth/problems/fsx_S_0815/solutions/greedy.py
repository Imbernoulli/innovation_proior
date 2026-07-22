# TIER: greedy
# Target-schedule chasing: hold every bus back toward the nominal headway using
# ONLY its own observed gap (gain_fwd=0), with one fixed, "safe-looking" gain
# used unconditionally on every instance -- it never looks at the instance's
# actual dwell-feedback strength (beta), never accounts for what its own
# holding does to the bus immediately behind it, and never dials back once the
# ring is close to settled. "Hold firmly back toward schedule whenever you're
# running early" is the very first idea anyone reaches for, and a fairly
# assertive gain (>1, i.e. more than a 1-for-1 correction) looks like the safe,
# conservative choice -- surely holding MORE can only help against bunching?
# It does help on some instances, but on others the very same fixed gain
# overcorrects: it repeatedly punches each bus's headway past the target,
# which the ring's own dwell-feedback then amplifies right back, so this
# "obviously safe" policy can end up worse than doing nothing at all, and it
# keeps spending hold budget long after the gaps have evened out.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"gain_back": 1.5, "gain_fwd": 0.0, "target_frac": 1.0, "cap_frac": 1.0}))
