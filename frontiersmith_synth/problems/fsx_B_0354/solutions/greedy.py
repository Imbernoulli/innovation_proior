# TIER: greedy
# Classic single-echelon newsvendor: pre-position each site to its own critical
# fractile  q_i = mean_i + z*std_i,  z = Phi^{-1}( p_i / (p_i + h_i) ),  and hold no
# vessel reserve.  Because shortage cost p >> holding cost h, the fractile sits high
# (~80% service), sharply cutting shortage vs the mean policy.  But it ignores the
# per-site service target tau, ignores risk-pooling on the vessel, and never spends
# down toward the budget, so it leaves clear gains on the table.
import sys, json, math


def inv_norm(p):
    if p <= 0:
        return -1e9
    if p >= 1:
        return 1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


inst = json.load(sys.stdin)
q = []
for s in inst["sites"]:
    z = inv_norm(s["p"] / (s["p"] + s["h"]))
    q.append(max(0.0, s["mean"] + z * s["std"]))
print(json.dumps({"q": q, "q0": 0.0}))
