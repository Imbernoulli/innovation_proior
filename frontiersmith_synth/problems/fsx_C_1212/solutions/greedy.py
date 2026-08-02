# TIER: greedy
"""Textbook fundamental-diagram fit with a defensive extrapolation cap: fit
a proportional flow model q ~= k*rho through the origin by ordinary least
squares on the training rows -- the "I recognize this, it's a fundamental
diagram" reflex -- and, because nobody trusts a raw linear fit to be
extrapolated forever, CAP the prediction at the best flow ever observed in
training (a common defensive habit: "the road can't do better than the best
it has already shown me"). Written as pure arithmetic via the identity
min(x, c) = (x + c - sqrt((x - c)^2)) / 2 (sqrt done as **0.5, no function
calls needed).

This single-valued, capped curve is far more stable than an uncapped linear
or quadratic extrapolation (it never explodes), and it correctly captures
the sub-critical trend AND the fact that flow saturates somewhere near
capacity. But it is still exactly ONE curve per density, monotonically
non-decreasing, with a hard floor at the observed maximum: it has no notion
of a metastable branch that quietly DECLINES past the critical density, no
notion of a lower discharge branch after a perturbation triggers breakdown,
and it completely ignores the perturbation covariate P. Wherever the
capacity-drop and metastable decline pull the true held-out flow well BELOW
the plateau this cap assumes, the miss is large and grows with how deep the
held-out density reaches into the broken-down region."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        rho = float(data[idx]); P = float(data[idx + 1]); q = float(data[idx + 2])
        idx += 3
        rows.append((rho, P, q))

    num = sum(rho * q for rho, P, q in rows)
    den = sum(rho * rho for rho, P, q in rows)
    k = num / den if den > 1e-12 else 0.0
    qmax_obs = max(q for rho, P, q in rows)

    # min(k*rho, qmax_obs) written as pure arithmetic (no function calls)
    print(
        "(%.6f * rho + %.6f - ((%.6f * rho - %.6f) ** 2) ** 0.5) / 2"
        % (k, qmax_obs, k, qmax_obs)
    )


if __name__ == "__main__":
    main()
