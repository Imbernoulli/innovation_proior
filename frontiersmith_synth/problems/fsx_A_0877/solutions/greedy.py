# TIER: greedy
# Per-symbol static inversion: treat the channel as if it were memoryless
# (use only the leading tap h_0), inverting the saturating nonlinearity
# independently for each target symbol. This is the obvious first attempt
# at "channel inversion" -- and it is fine when the memory taps h_1..h_L
# are small. But it ignores the fact that its OWN recent symbols are still
# leaking into the current output through the channel's memory: once the
# memory taps carry real weight, the uncancelled inter-symbol interference
# compounds across the sequence and the reconstructed output can land
# further from the target than transmitting nothing at all.
import sys, json, math


def atanh_safe(u):
    u = max(-0.999999, min(0.999999, u))
    return math.atanh(u)


inst = json.load(sys.stdin)
h0 = inst["h"][0]
A = inst["A"]
xmax = inst["xmax"]
target = inst["target"]

x = []
for t in target:
    v = A * atanh_safe(t / A)     # desired pre-nonlinearity value, symbol-by-symbol
    xt = v / h0                   # invert using ONLY the leading tap
    xt = max(-xmax, min(xmax, xt))
    x.append(xt)

print(json.dumps({"x": x}))
