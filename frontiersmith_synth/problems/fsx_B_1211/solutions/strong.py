# TIER: strong
# The insight: the calving trigger is the RATIO D(t)/H(t) crossing phi, not the
# raw thinning/growth rate -- buttressing loss is a genuine regime SWITCH, not
# one more linear covariate to fold into a rate fit. Given phi we can compute
# the exact phase-1 -> phase-2 switch time algebraically:
#     t1 = (phi*H0 - D0) / (c0 + phi*gamma)
#     D1 = D0 + c0*t1            H1 = H0 - gamma*t1
# and after the switch the rift closes the remaining gap to kappa at the fixed,
# KNOWN accelerated rate c0*(1+BETA). That lets us invert the FULL two-phase
# relation for kappa, per training row, WITHOUT the phase-1-only bias:
#     T - t1 = (kappa*H1 - D1) / (c0*(1+BETA) + kappa*gamma)
#   =>  kappa = (D1 + (T-t1)*c0*(1+BETA)) / (H1 - (T-t1)*gamma)
# Averaging this unbiased per-row estimate over the training table and emitting
# the exact piecewise closed form (with BETA and the fitted kappa baked in)
# generalizes correctly to held-out segments dominated by phase 2.
import sys

BETA = 3.0


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    kappas = []
    for i in range(n):
        H0 = float(vals[6 * i]); D0 = float(vals[6 * i + 1])
        c0 = float(vals[6 * i + 2]); gamma = float(vals[6 * i + 3])
        phi = float(vals[6 * i + 4]); T = float(vals[6 * i + 5])
        t1 = (phi * H0 - D0) / (c0 + phi * gamma)
        D1 = D0 + c0 * t1
        H1 = H0 - gamma * t1
        tprime = T - t1
        denom = H1 - tprime * gamma
        if abs(denom) < 1e-9:
            continue
        kappa = (D1 + tprime * c0 * (1.0 + BETA)) / denom
        kappas.append(kappa)
    kappa_hat = sum(kappas) / len(kappas) if kappas else 0.85

    T1 = "( ( phi * H0 - D0 ) / ( c0 + phi * gamma ) )"
    D1e = "( D0 + c0 * %s )" % T1
    H1e = "( H0 - gamma * %s )" % T1
    expr = "%s + ( ( %.10g ) * %s - %s ) / ( c0 * ( %.10g ) + ( %.10g ) * gamma )" % (
        T1, kappa_hat, H1e, D1e, 1.0 + BETA, kappa_hat)
    print(expr)


if __name__ == "__main__":
    main()
