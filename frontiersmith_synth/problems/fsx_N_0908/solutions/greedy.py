# TIER: greedy
"""The obvious first instinct: plan capacity proportional to expected fraud volume times
harm value, every round, self-simulating the population forward from p0 under the
DETERMINISTIC law given in the statement (this is a single-shot submission with no live
feedback, so the candidate must plan the whole trajectory in advance). This is a perfectly
sensible, well-informed reactive plan -- it is NOT a naive lagged or all-or-nothing rule.

Its blind spot: it plans PRECISELY against the noise-free forecast and never hedges. Every
round some unpredictable fraction of attacker mass takes one more opportunistic hop along
the same migration graph -- a shift this plan could not have foreseen from the deterministic
law alone -- and a precisely-targeted plan is fully exposed wherever that hop lands."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    K, T = inst["K"], inst["T"]
    V, R, value, M, beta = inst["V"], inst["R"], inst["value"], inst["M"], inst["beta"]

    p = list(inst["p0"])
    alloc = []
    for t in range(T):
        Vt, Rt = V[t], R[t]
        vol = [Vt * p[j] for j in range(K)]
        weight = [max(vol[j], 1e-9) * value[j] for j in range(K)]
        s = sum(weight)
        row = [Rt * x / s for x in weight]
        alloc.append(row)

        cov = [0.0 if vol[j] <= 1e-12 else min(1.0, row[j] / vol[j]) for j in range(K)]
        newp = [0.0] * K
        for i in range(K):
            stay = p[i] * (1 - beta * cov[i])
            newp[i] += stay
            leave = p[i] * beta * cov[i]
            if leave > 0:
                Mi = M[i]
                for j in range(K):
                    if j != i and Mi[j] > 0:
                        newp[j] += leave * Mi[j]
        p = newp

    print(json.dumps({"alloc": alloc}))


main()
