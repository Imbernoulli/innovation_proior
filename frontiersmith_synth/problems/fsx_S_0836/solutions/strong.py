# TIER: strong
# The insight: don't fit the PROFILE map, fit the CONSERVED FLUX -- and use
# the CONSERVATION LAW IN INTEGRAL (not differential) form, which is far more
# noise-robust than differencing a noisy point grid twice.
#
# Integrating rho_t + (f(rho))_x = 0 in x from 0 to X gives
#     d/dt [ F(X,t) ]  =  -( f(rho(X,t)) - f(rho(0,t)) ),   F(X,t) := int_0^X rho dx
# i.e. the flux DIFFERENCE between any point X and the reference x=0 equals
# minus the time-derivative of a SPATIAL INTEGRAL of the sensor readings.
# Spatial integration averages sensor noise down; only ONE noisy
# differentiation remains (in time, across the K snapshots), applied to the
# already-smooth integral, not to the raw density.
#
# Positing f(rho) = b0*rho + b1*rho^2 (a concave "fundamental diagram" shape
# -- the simplest family that lets characteristic speed f'(rho)=b0+b1*rho
# actually DEPEND on density, which a rigid-translation model fundamentally
# cannot), the unknowns enter LINEARLY:
#     -dF/dt(x_m,t_k) = b0*(rho_m-rho_0) + b1*(rho_m^2-rho_0^2)
# (rho_0 = rho(0,t_k), the moving reference at that snapshot). Solve this by
# least squares over every (sensor, time) sample -- no spatial finite
# differences of rho at all, and only 2 unknowns keeps the estimate stable
# under sensor noise. Because this reconstructs the actual density-dependent
# flux, it extrapolates the correct Rankine-Hugoniot shock speed to the
# unseen high-density regime where the held-out shockwave forms.
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0"); return
    Mn, Kn, t = int(data[0]), int(data[1]), int(data[2])
    vals = list(map(float, data[3:]))
    rows = np.array(vals, dtype=np.float64).reshape(Mn * Kn, 3)

    ts = sorted(set(round(v, 9) for v in rows[:, 0]))
    xs = sorted(set(round(v, 9) for v in rows[:, 1]))
    tidx = {tv: i for i, tv in enumerate(ts)}
    xidx = {xv: i for i, xv in enumerate(xs)}
    grid = np.zeros((Kn, Mn))
    for tk, xm, rv in rows:
        grid[tidx[round(tk, 9)], xidx[round(xm, 9)]] = rv
    ts = np.array(ts); xs = np.array(xs)

    # --- light denoise in x: truncated Fourier fit per snapshot (periodic) ---
    H = max(1, min(4, Mn // 3))
    cols = [np.ones(Mn)]
    for h in range(1, H + 1):
        cols.append(np.cos(2 * np.pi * h * xs))
        cols.append(np.sin(2 * np.pi * h * xs))
    Xf = np.stack(cols, axis=1)
    rho_s = np.zeros((Kn, Mn))
    for k in range(Kn):
        coef, *_ = np.linalg.lstsq(Xf, grid[k], rcond=None)
        rho_s[k] = Xf @ coef

    # --- cumulative spatial integral F_k(x_m) = int_0^{x_m} rho_s dx (trapezoid) ---
    dx = 1.0 / Mn
    F = np.zeros((Kn, Mn))
    for k in range(Kn):
        seg = 0.5 * (rho_s[k, :-1] + rho_s[k, 1:]) * dx
        F[k, 1:] = np.cumsum(seg)

    # --- denoise in t: quadratic fit per position, differentiated analytically ---
    deg = 2
    dFdt = np.zeros((Kn, Mn))
    for m in range(Mn):
        c = np.polyfit(ts, F[:, m], deg)
        dc = np.polyder(c)
        dFdt[:, m] = np.polyval(dc, ts)

    rho0 = rho_s[:, 0:1]                       # (Kn,1) moving reference density
    d1 = rho_s - rho0
    d2 = rho_s ** 2 - rho0 ** 2

    A = np.stack([d1.reshape(-1), d2.reshape(-1)], axis=1)
    btgt = -dFdt.reshape(-1)

    scale = np.maximum(np.sqrt(np.mean(A * A, axis=0)), 1e-9)
    An = A / scale
    lam = 0.01 * An.shape[0]
    G = An.T @ An + lam * np.eye(2)
    rhs = An.T @ btgt
    coef_n = np.linalg.solve(G, rhs)
    b0, b1 = coef_n / scale
    print("%.6f * rho + %.6f * rho ** 2" % (b0, b1))


if __name__ == "__main__":
    main()
