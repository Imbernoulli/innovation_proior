# TIER: strong
# Two-part insight over the greedy recipe:
#
# 1) NARROW the hidden sensitivity from the WHOLE accumulation curve, not the
#    last point. The true probe concentration at step t (for a patient with
#    sensitivity S) is basis[t] * S, where basis[t] = sum_{j<t} rho^{t-1-j} *
#    probe_dose[j] is a fully public, known quantity (rho and the probe doses
#    are given). So this is a 1-parameter linear regression through the
#    origin: S_hat = sum(basis*reading) / sum(basis^2), a projection that
#    downweights noisy points with small "signal" (basis) and is far more
#    robust than reading a single ratio off the last, noisiest, most
#    accumulation-confounded point.
#
# 2) DOSE TO THE SIMULATED STEADY STATE, not the current reading. Under the
#    known recursion C_{t+1} = rho*C_t + S*dose_t, holding a constant dose d
#    converges to C* = S*d/(1-rho) -- so the dose that targets C* = target is
#    d_star = target*(1-rho)/S_hat, which explicitly divides OUT the
#    accumulation multiplier 1/(1-rho) that greedy ignores entirely. The
#    first dose additionally does a one-step "deadbeat" correction from the
#    model-predicted end-of-probe concentration so the very first treatment
#    step already lands near target instead of drifting up to it.
import sys, json

inst = json.load(sys.stdin)
T = inst["treatment_steps"]
target = inst["target"]
rho = inst["rho"]
probe_doses = inst["probe_doses"]
dose_max = inst["dose_max"]

# public basis[t-1] = concentration at probe step t for a unit-sensitivity patient
basis = []
c = 0.0
for d in probe_doses:
    c = rho * c + d
    basis.append(c)

doses = []
for pat in inst["patients"]:
    readings = pat["probe_readings"]
    num = sum(b * r for b, r in zip(basis, readings))
    den = sum(b * b for b in basis)
    S_hat = num / den if den > 1e-9 else 1.0
    S_hat = max(1e-6, S_hat)

    c_end_hat = S_hat * basis[-1]                       # model-predicted end-of-probe conc.
    d0 = (target - rho * c_end_hat) / S_hat              # one-step deadbeat correction
    d_star = target * (1.0 - rho) / S_hat                # true steady-state-targeting dose

    d0 = max(0.0, min(dose_max, d0))
    d_star = max(0.0, min(dose_max, d_star))
    seq = [d0] + [d_star] * (T - 1)
    doses.append(seq)

print(json.dumps({"doses": doses}))
