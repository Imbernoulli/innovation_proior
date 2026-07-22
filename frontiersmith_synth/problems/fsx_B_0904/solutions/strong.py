# TIER: strong
# Exploits BOTH planted mechanisms together:
#   1. gap-autocorrelation-reading: a Bayesian forward filter (2-state HMM,
#      using the given regime persistence p_stay) over every gap observed so
#      far turns the raw history into a posterior belief about which regime
#      (short/long) is currently active -- far more responsive after a
#      regime switch than a running average, because each new observation's
#      LIKELIHOOD under the two regimes is combined with the transition
#      prior instead of just averaged in.
#   2. setback-ladder-policies + state-dependent (concave) restart cost: the
#      committed level is chosen to minimize the FULL EXPECTED cost across
#      the belief-weighted mixture of regimes (numeric quadrature over each
#      regime's exponential gap distribution), not the cost at a single
#      point-estimate. Because the reheat term is concave, this expectation
#      argument is what makes an INTERMEDIATE ladder level the right hedge
#      near regime uncertainty, instead of committing to one of the two
#      textbook extremes.
import sys, json, math

def cost(level, g, tau, hold_power, cool_rate, A, p):
    tv = tau[level]
    d = (1.0 - tv) * (1.0 - math.exp(-cool_rate[level] * g))
    return hold_power[level] * g + A * (d ** p)

def expected_cost(level, mu, tau, hold_power, cool_rate, A, p, nsteps=250):
    # E_{G~Exp(mu)}[cost(level, G)] via midpoint quadrature over [0, 14*mu]
    hi = 14.0 * mu
    h = hi / nsteps
    acc = 0.0
    for i in range(nsteps):
        g = (i + 0.5) * h
        pdf = (1.0 / mu) * math.exp(-g / mu)
        acc += cost(level, g, tau, hold_power, cool_rate, A, p) * pdf * h
    return acc

def main():
    inst = json.load(sys.stdin)
    K = inst["K"]
    tau = inst["tau"]; hold_power = inst["hold_power"]; cool_rate = inst["cool_rate"]
    A = inst["reheat_coeff"]; p = inst["reheat_exp"]
    mu_s = inst["mu_short"]; mu_l = inst["mu_long"]; p_stay = inst["p_stay"]
    hist = inst["history"]

    def lik(g, mu):
        return (1.0 / mu) * math.exp(-g / mu)

    bS, bL = 0.5, 0.5  # stationary prior for a symmetric 2-state chain
    for g in hist:
        predS = p_stay * bS + (1 - p_stay) * bL
        predL = p_stay * bL + (1 - p_stay) * bS
        lS, lL = lik(g, mu_s) * predS, lik(g, mu_l) * predL
        z = lS + lL
        bS, bL = (lS / z, lL / z) if z > 0 else (0.5, 0.5)

    predS = p_stay * bS + (1 - p_stay) * bL
    predL = p_stay * bL + (1 - p_stay) * bS

    ec_s = [expected_cost(l, mu_s, tau, hold_power, cool_rate, A, p) for l in range(K + 1)]
    ec_l = [expected_cost(l, mu_l, tau, hold_power, cool_rate, A, p) for l in range(K + 1)]
    ecosts = [predS * ec_s[l] + predL * ec_l[l] for l in range(K + 1)]
    level = min(range(K + 1), key=lambda l: ecosts[l])
    print(json.dumps({"level": level}))

main()
