# TIER: strong
# Two-part insight:
#  (1) fit an exponential growth rate off the much-less-delayed
#      `leading_indicator` series (log-linear regression on its most recent
#      points), then project forward only the SHORT `d_lead`-day dark window
#      to estimate the TRUE current severity -- correcting exactly the bias
#      the delayed-threshold approach ignores.
#  (2) using that estimate, search a small deterministic set of PULSED
#      schedules (bursts of restriction separated by rest days, at several
#      levels/durations) plus the four constant-level schedules, simulating
#      each internally with the estimated rate + the real fatigue/cost
#      model, and keep the cheapest. Pulsing keeps compliance from
#      collapsing to the fatigue floor, unlike holding one level forever.
# Fully deterministic (closed-form regression + exhaustive small search, no
# randomness).
import sys, json, math

inst = json.load(sys.stdin)
FUT = inst["future_days"]
leading = list(inst["leading_indicator"])
levels_menu = [(lv["m"], lv["cost"]) for lv in inst["levels"]]
K = len(levels_menu)
fat = inst["fatigue"]
DECAY, RECOVER, FLOOR = fat["decay"], fat["recover"], fat["floor"]
H_CAP = inst["hospital_capacity"]
OVERFLOW_PEN = inst["overflow_penalty"]
BASE_WEIGHT = inst["health_weight"]
D_LEAD = inst["d_lead"]


def fit_growth_rate(series, npts=12):
    pts = [(i, v) for i, v in enumerate(series) if v > 0]
    if len(pts) < 3:
        return 1.15, (pts[-1][1] if pts else 10.0)
    pts = pts[-npts:]
    n = len(pts)
    sx = sum(i for i, v in pts)
    sy = sum(math.log(v) for i, v in pts)
    sxx = sum(i * i for i, v in pts)
    sxy = sum(i * math.log(v) for i, v in pts)
    denom = n * sxx - sx * sx
    slope = 0.02 if abs(denom) < 1e-9 else (n * sxy - sx * sy) / denom
    r0_est = max(0.9, min(1.6, math.exp(slope)))
    last_v = pts[-1][1]
    return r0_est, last_v


r0_est, last_v = fit_growth_rate(leading)
x_now_est = last_v * (r0_est ** D_LEAD)   # correct for the leading indicator's own (short) dark window


def sim_est(seq):
    xt = x_now_est
    streak = 0
    compliance = 1.0
    h = 0.0
    e = 0.0
    for lv in seq:
        m, c = levels_menu[lv]
        if lv > 0:
            streak += 1
            compliance = max(FLOOR, 1.0 - DECAY * streak)
        else:
            streak = 0
            compliance = min(1.0, compliance + RECOVER)
        eff_m = 1.0 - compliance * (1.0 - m)
        xt = xt * r0_est * eff_m
        h += BASE_WEIGHT * xt + OVERFLOW_PEN * max(0.0, xt - H_CAP)
        e += c
    return h + e


def make_pulse(on_len, off_len, on_level):
    seq = []
    while len(seq) < FUT:
        seq += [on_level] * on_len + [0] * off_len
    return seq[:FUT]


candidates = []
for on_level in (1, 2, 3):
    for on_len in (4, 5, 6, 8):
        for off_len in (2, 3, 4):
            candidates.append(make_pulse(on_len, off_len, on_level))
for lv in range(K):
    candidates.append([lv] * FUT)

best, best_cost = None, None
for cand in candidates:
    cost = sim_est(cand)
    if best_cost is None or cost < best_cost:
        best_cost = cost
        best = cand

print(json.dumps({"levels": best}))
