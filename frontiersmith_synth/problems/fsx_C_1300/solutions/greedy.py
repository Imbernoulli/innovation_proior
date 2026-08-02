# TIER: greedy
# Delayed-threshold policy: never drops below level 1; escalates to level
# 2/3 by thresholding a rolling average of the DELAYED `reported_cases`
# series, extrapolating it forward with the naive "same day-over-day ratio"
# rule (no correction for the reporting/incubation dark window). Never
# looks at the leading indicator. This is the obvious "watch the case count
# dashboard and react" policy a typical coder writes first -- realistic and
# reactive, but on fast-growth instances the case count it trusts is still
# reporting news from `d_rep` days ago, so it escalates only after the true
# outbreak has already grown well past what the stale count shows.
import sys, json

inst = json.load(sys.stdin)
FUT = inst["future_days"]
reported = list(inst["reported_cases"])


def recent_avg(hist):
    w = hist[-5:] if len(hist) >= 5 else hist
    return sum(w) / max(1, len(w))


hist = list(reported)
seq = []
for _ in range(FUT):
    avg = recent_avg(hist)
    if avg > 3000:
        lvl = 3
    elif avg > 1200:
        lvl = 2
    else:
        lvl = 1
    seq.append(lvl)
    # naive forward projection: assume the series keeps the last observed
    # day-over-day ratio (no attempt to correct for reporting delay)
    if len(hist) >= 2 and hist[-2] > 0:
        ratio = hist[-1] / hist[-2]
    else:
        ratio = 1.1
    nxt = max(0.0, hist[-1] * ratio) if hist else 0.0
    hist.append(nxt)

print(json.dumps({"levels": seq}))
