# TIER: strong
# INSIGHT (spectral-regime diagnosis before choosing the acceleration):
#   The observed step vectors obey dx_{k+1} = M dx_k, so the dominant behaviour
#   satisfies a 2-term scalar recurrence dx_{k+2} ~ alpha*dx_{k+1} + beta*dx_k.
#   Fit (alpha,beta) by least squares over ALL components/steps; the dominant
#   eigenvalues are the roots of z^2 - alpha z - beta = 0.
#     * REAL root (disc >= 0): a single geometric mode. Annihilate it with
#       over-relaxation omega* = 1/(1-lambda), and Aitken's Delta^2 (which is
#       exact for one real mode) mops up the rest -> aitken ON.
#     * COMPLEX pair (disc < 0): lambda = a +/- i b, |lambda| near 1, OSCILLATORY.
#       Over-relaxation blows it up; the |amplification|-minimising factor is
#       omega* = (1-a)/((1-a)^2 + b^2) < 1 (UNDER-relaxation), and Aitken on an
#       oscillation is unstable -> aitken OFF.
import sys, json, math

inst = json.load(sys.stdin)
its = inst["iterates"]
lo, hi = inst["omega_min"], inst["omega_max"]
n = inst["n"]

dx = [[its[k + 1][i] - its[k][i] for i in range(n)] for k in range(len(its) - 1)]
m = len(dx)

# Least-squares fit of dx[k+2] ~ alpha*dx[k+1] + beta*dx[k]. Use only the LATER
# triples, where the sub-dominant modes have decayed and the recurrence is
# governed cleanly by the dominant eigen-pair -- this is what lets the complex
# (oscillatory) regime be distinguished from the real one.
s11 = s12 = s22 = t1 = t2 = 0.0
k0 = max(0, (m - 2) - 4)      # last ~4 triples
for k in range(k0, m - 2):
    for i in range(n):
        u = dx[k + 1][i]     # coefficient of alpha
        v = dx[k][i]         # coefficient of beta
        w = dx[k + 2][i]     # target
        s11 += u * u
        s12 += u * v
        s22 += v * v
        t1 += u * w
        t2 += v * w

det = s11 * s22 - s12 * s12
if abs(det) < 1e-18:
    alpha, beta = 1.0, 0.0
else:
    alpha = (t1 * s22 - t2 * s12) / det
    beta = (s11 * t2 - s12 * t1) / det

disc = alpha * alpha + 4.0 * beta

if disc >= 0.0:
    # real dominant mode
    r1 = (alpha + math.sqrt(disc)) / 2.0
    r2 = (alpha - math.sqrt(disc)) / 2.0
    lam = r1 if abs(r1) >= abs(r2) else r2
    lam = min(max(lam, -0.999), 0.999)
    omega = 1.0 / (1.0 - lam) if lam < 1.0 else 1.9
    omega = min(max(omega, max(lo, 0.5)), min(hi, 1.9))
    aitken = 3        # Delta^2 is (near-)exact for one real geometric mode
else:
    # complex conjugate pair a +/- i b
    a = alpha / 2.0
    b = math.sqrt(-disc) / 2.0
    denom = (1.0 - a) ** 2 + b * b
    omega = (1.0 - a) / denom if denom > 1e-15 else 1.0
    omega = min(max(omega, max(lo, 0.05)), min(hi, 1.9))
    aitken = 0        # never Aitken an oscillatory mode

print(json.dumps({"omega": omega, "aitken_period": aitken}))
