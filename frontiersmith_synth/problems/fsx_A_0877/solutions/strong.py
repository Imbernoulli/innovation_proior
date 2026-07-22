# TIER: strong
# Causal decision-feedback precoding (Tomlinson/Volterra-style): the channel
# is causal -- y_t depends only on x_t and EARLIER symbols x_{t-1..t-L}, never
# on future ones. So process t = 0, 1, ..., N-1 IN ORDER: by the time we pick
# x_t, the symbols x_{t-1..t-L} that also leak into y_t are already fixed and
# KNOWN, so their contribution can be subtracted out before inverting the
# nonlinearity for x_t. This jointly precompensates the whole sequence for
# the channel's memory instead of inverting one symbol at a time -- exactly
# cancelling the inter-symbol interference the greedy tier ignores (up to
# the amplitude budget: when the exact causal solution would need
# |x_t| > xmax, we clip and let the shortfall show up as residual error,
# rather than overshoot the energy budget).
import sys, json, math


def atanh_safe(u):
    u = max(-0.999999, min(0.999999, u))
    return math.atanh(u)


inst = json.load(sys.stdin)
h = inst["h"]
A = inst["A"]
xmax = inst["xmax"]
target = inst["target"]
n = inst["n"]

x = [0.0] * n
for t in range(n):
    v_des = A * atanh_safe(target[t] / A)   # desired pre-nonlinearity value at t
    isi = 0.0
    for k in range(1, len(h)):
        if t - k >= 0:
            isi += h[k] * x[t - k]          # already-known contribution from earlier symbols
    xt = (v_des - isi) / h[0]               # cancel it, then invert with the leading tap
    xt = max(-xmax, min(xmax, xt))
    x[t] = xt

print(json.dumps({"x": x}))
