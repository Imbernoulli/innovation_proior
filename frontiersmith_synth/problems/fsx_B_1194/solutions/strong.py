# TIER: strong
# The insight: the visible data is 100% subcritical, so a fit of the amplitude
# column itself carries zero signal about the post-onset branch (the trivial
# trap). But the DECAY RATE'S approach to zero, extrapolated past its
# zero-crossing, gives the growth-rate law g(R)=a(R-Rc). That growth rate is
# NOT the amplitude -- the given Landau constant L fixes how the cubic
# nonlinearity saturates it into an equilibrium oscillation:
#     A(R) = sqrt( relu(g(R)) / L )
# Same honest OLS fit of decay_rate ~ alpha + beta*R as the greedy recipe --
# the only difference is applying the stated amplitude-growth-rate relation
# instead of reporting the growth rate as if it already were the amplitude.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0"); return
    M = int(data[0])
    L = float(data[2])
    vals = data[3:]
    Rs = []
    decays = []
    for i in range(M):
        Rs.append(float(vals[3 * i]))
        decays.append(float(vals[3 * i + 1]))

    n = len(Rs)
    mx = sum(Rs) / n
    my = sum(decays) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(Rs, decays))
    sxx = sum((x - mx) ** 2 for x in Rs)
    beta = sxy / sxx if sxx > 1e-12 else 0.0
    alpha = my - beta * mx

    # g(R) = -(alpha + beta*R);  A(R) = sqrt( relu(g(R)) / L )
    print("sqrt ( relu ( ( -1.0 * %.10g ) + ( -1.0 * %.10g ) * R ) / %.10g )"
          % (alpha, beta, L))


if __name__ == "__main__":
    main()
