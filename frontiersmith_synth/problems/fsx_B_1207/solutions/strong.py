# TIER: strong
# The insight: resistant strains are rare pre-treatment ONLY because they pay
# a fitness cost c. At the pre-treatment mutation/transfer-selection balance,
#     (mu+tau)*(1-p0) = p0*(1-p0)*c   =>   c = (mu+tau) / p0
# so the flat pre-treatment level p0 (estimated as the mean of the visible
# observations) reveals the hidden cost c -- even though c itself never
# appears in the input. Once c is known, the SAME replicator dynamics
#     dp/dt = (mu+tau)*(1-p) + p*(1-p)*(alpha*D(t) - c)
# can be solved in closed form for t after treatment starts (D(t)=D, a
# constant given in the input): with A = mu+tau, Bc = alpha*D - c, R = A+Bc,
#     K0 = (A + Bc*p0) / (1 - p0)
#     p(t) = (K0*exp(R*(t-T0)) - A) / (K0*exp(R*(t-T0)) + Bc)
# This single expression, with A/Bc/R/K0/T0 baked in as numeric constants
# computed from the training data, reproduces the sweep (or its absence)
# instead of extrapolating flatness.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    _t = int(data[idx]); idx += 1
    mu, tau, alpha, D, T0, _T1 = (float(x) for x in data[idx:idx + 6]); idx += 6
    n_train = int(data[idx]); idx += 1
    obs = []
    for _ in range(n_train):
        _tt = float(data[idx]); idx += 1
        yy = float(data[idx]); idx += 1
        obs.append(yy)
    p0 = sum(obs) / len(obs) if obs else 0.1
    p0 = min(0.999, max(1e-6, p0))

    A = mu + tau
    c_hat = A / p0
    Bc = alpha * D - c_hat
    R = A + Bc
    if abs(R) < 1e-6:
        R = 1e-6 if R >= 0 else -1e-6
    denom0 = (1.0 - p0)
    if denom0 < 1e-9:
        denom0 = 1e-9
    K0 = (A + Bc * p0) / denom0
    if K0 <= 0:
        K0 = 1e-9

    expr = "((%.10g*exp(%.10g*(t-%.10g))-%.10g)/(%.10g*exp(%.10g*(t-%.10g))+%.10g))" % (
        K0, R, T0, A, K0, R, T0, Bc)
    print(expr)


if __name__ == "__main__":
    main()
