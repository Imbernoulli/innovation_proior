# TIER: strong
# The insight: the declared UNITS are hard algebraic constraints, not decoration.
# For F [M L T^-2] built from rho [M L^-3], V [L T^-1], D [L], mu [M L^-1 T^-1],
# dimensional homogeneity FORCES the force scale to be the monomial rho V^2 D^2
# and leaves exactly ONE independent dimensionless group, the Reynolds number
# Re = rho V D / mu.  So the whole 4-variable law must collapse to
#       F = rho V^2 D^2 * g(Re),     g dimensionless.
# We do not FIT the four exponents (the notebook can't even identify rho, mu) --
# the units hand them to us.  That reduces the problem to fitting the single 1-D
# curve g(Re): form Cd = F / (rho V^2 D^2), and regress it on {1, Re^-1/2} in the
# training range.  Because the prefactor is exact, this extrapolates to the
# held-out fluid/scale where the greedy free fit diverges.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    # collapse to (Re, Cd) using the dimensionally forced prefactor
    xs = []  # basis rows [1, Re^-0.5]
    cds = []
    for i in range(n):
        rho = float(vals[5 * i]); V = float(vals[5 * i + 1])
        D = float(vals[5 * i + 2]); mu = float(vals[5 * i + 3])
        F = float(vals[5 * i + 4])
        Re = rho * V * D / mu
        Cd = F / (rho * V * V * D * D)
        xs.append((1.0, Re ** (-0.5)))
        cds.append(Cd)
    # 2x2 normal equations for Cd ~ a0 + a1 * Re^-0.5
    s11 = s12 = s22 = t1 = t2 = 0.0
    for (u, w), c in zip(xs, cds):
        s11 += u * u; s12 += u * w; s22 += w * w
        t1 += u * c; t2 += w * c
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-18:
        a0 = sum(cds) / len(cds); a1 = 0.0
    else:
        a0 = (t1 * s22 - t2 * s12) / det
        a1 = (s11 * t2 - s12 * t1) / det
    # emit the dimensionally forced closed form
    print("(%.10g + %.10g * (rho*V*D/mu)**(-0.5)) * rho * V**2 * D**2" % (a0, a1))


if __name__ == "__main__":
    main()
