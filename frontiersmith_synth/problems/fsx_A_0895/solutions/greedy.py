# TIER: greedy
# Titrate to the CURRENT reading: estimate sensitivity from the single most
# recent probe measurement and the single most recent probe dose (S_hat =
# last_reading / last_dose), then hold a constant dose = target / S_hat for
# every remaining step. This is the "obvious" recipe -- it reacts to what the
# patient just showed, and it implicitly assumes the drug has NO memory (the
# next reading is a fresh, one-step response to the next dose, exactly as the
# most recent reading looked like a fresh response to the most recent dose).
# It never accounts for `rho`, so on patients whose drug clears slowly
# (rho close to 1) the constant dose keeps compounding on top of un-cleared
# residual and drifts the true steady state far past the target -- overshoot
# that shows up a couple of steps into treatment, exactly when it's too late
# to react (this candidate gets no further feedback).
import sys, json

inst = json.load(sys.stdin)
T = inst["treatment_steps"]
target = inst["target"]
probe_doses = inst["probe_doses"]
dose_max = inst["dose_max"]

d_last = probe_doses[-1]
doses = []
for pat in inst["patients"]:
    r_last = pat["probe_readings"][-1]
    S_hat = r_last / d_last if d_last > 1e-9 else 1.0
    S_hat = max(1e-6, S_hat)
    d = target / S_hat
    d = max(0.0, min(dose_max, d))
    doses.append([d] * T)

print(json.dumps({"doses": doses}))
