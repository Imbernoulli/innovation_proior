# TIER: greedy
# Classic ski-rental reasoning: compute ONE aggregate average gap length from
# everything observed so far, and choose the cheaper of the two EXTREME
# setback levels (hold vs full shutdown) for that average. No graded ladder,
# no regime/autocorrelation modeling -- the textbook "one hold-or-kill
# threshold" answer.
import sys, json, math

def cost(level, g, tau, hold_power, cool_rate, A, p):
    tv = tau[level]
    d = (1.0 - tv) * (1.0 - math.exp(-cool_rate[level] * g))
    return hold_power[level] * g + A * (d ** p)

def main():
    inst = json.load(sys.stdin)
    K = inst["K"]
    tau = inst["tau"]; hold_power = inst["hold_power"]; cool_rate = inst["cool_rate"]
    A = inst["reheat_coeff"]; p = inst["reheat_exp"]
    hist = inst["history"]
    m = sum(hist) / len(hist) if hist else 10.0
    c0 = cost(0, m, tau, hold_power, cool_rate, A, p)
    cK = cost(K, m, tau, hold_power, cool_rate, A, p)
    level = 0 if c0 <= cK else K
    print(json.dumps({"level": level}))

main()
