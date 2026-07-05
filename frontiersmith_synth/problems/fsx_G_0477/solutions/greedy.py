# TIER: greedy
# Univariate max |z-score|: standardize each sensor independently and flag the row by
# its most extreme single-sensor deviation. Catches GLOBAL/point faults (a pegged
# sensor) but is blind to correlation-break / contextual / density faults whose
# marginals look perfectly normal -> ~chance there.
import sys, json, math

inst = json.load(sys.stdin)
X = inst["X"]; n = inst["n"]; d = inst["d"]

mean = [0.0] * d
for row in X:
    for j in range(d):
        mean[j] += row[j]
for j in range(d):
    mean[j] /= n

var = [0.0] * d
for row in X:
    for j in range(d):
        dv = row[j] - mean[j]
        var[j] += dv * dv
std = [math.sqrt(var[j] / max(1, n - 1)) or 1e-9 for j in range(d)]

scores = []
for row in X:
    z = 0.0
    for j in range(d):
        a = abs((row[j] - mean[j]) / std[j])
        if a > z:
            z = a
    scores.append(z)

print(json.dumps({"scores": scores}))
