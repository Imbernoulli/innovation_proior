#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Highway sensor loops before the jam (family: traffic-shockwave-law).

A ring road of length 1 (x in [0,1), periodic) carries density rho(x,t) in
[0,1] obeying the scalar conservation law rho_t + f(rho)_x = 0, where f is a
HIDDEN flux function of density alone (a "fundamental diagram"). Each test id
fixes a different hidden f (drawn from a concave cubic family f(rho) =
V*rho*(1-rho)*(1+S*(rho-0.5))) and a different smooth initial density profile.

The solver only ever SEES noisy readings from a sparse ring of inductive
sensor loops, recorded on a CALM day: the profile is smooth over the whole
recorded window (the generator computes an analytic safety margin from the
hidden law + initial slope so no shock forms before the last recorded time).
The held-out grading initial condition is a STEEP, high-density profile that
DOES break into a shockwave; it is regenerated only inside the grader and is
never printed here, nor are the hidden flux parameters V,S.

STDOUT prints ONLY:
    M K t
    t_0 x_0 rho_0
    ...
    t_{K-1} x_{M-1} rho_{MK-1}
(M sensor positions x_m = m/M, K observation times t_k with t_0 = 0.)
"""
import sys, random
import numpy as np

K = 8                      # number of observation snapshots (fixed, public)
N_FINE = 800                # internal PDE-solve resolution (not printed)
TARGET_T_TRAIN = 0.08
SAFETY = 0.45
CFL = 0.45


# ---------------- hidden law (lives in gen.py AND verify.py, never printed) ----------------
def flux_params(t):
    rng = random.Random(9173231 + t * 104729)
    V = rng.uniform(0.85, 1.35)
    S = rng.uniform(-0.55, 0.55)
    W = rng.uniform(0.05, 0.12)
    freq = rng.choice([2, 3])
    return V, S, W, freq


def f_true_vec(rho, V, S, W, freq):
    base = V * rho * (1.0 - rho) * (1.0 + S * (rho - 0.5))
    wiggle = W * V * rho * (1.0 - rho) * np.sin(2.0 * np.pi * freq * rho)
    return base + wiggle


def fprime_true_vec(rho, V, S, W, freq):
    # numeric derivative (robust to the composite base+wiggle shape; only
    # used internally for CFL / breaking-time estimates, never printed)
    eps = 1e-4
    rp = np.clip(rho + eps, 0.0, 1.0)
    rm = np.clip(rho - eps, 0.0, 1.0)
    return (f_true_vec(rp, V, S, W, freq) - f_true_vec(rm, V, S, W, freq)) / (rp - rm)


def train_ic_params(t):
    rng = random.Random(55411 + t * 7919)
    base = rng.uniform(0.05, 0.15)
    bumps = []
    for _ in range(3):
        c = rng.uniform(0.0, 1.0)
        w = rng.uniform(0.16, 0.28)
        h = rng.uniform(0.10, 0.35)
        bumps.append((c, w, h))
    return base, bumps


def periodic_dist(x, c):
    d = x - c
    d = d - np.round(d)
    return d


def bump_vec(x, c, w, h):
    d = periodic_dist(x, c)
    out = np.zeros_like(x)
    mask = np.abs(d) < w
    out[mask] = h * 0.5 * (1.0 + np.cos(np.pi * d[mask] / w))
    return out


def train_ic(x, t):
    base, bumps = train_ic_params(t)
    rho = np.full_like(x, base)
    for c, w, h in bumps:
        rho = rho + bump_vec(x, c, w, h)
    return np.clip(rho, 0.0, 0.95)


# ---------------- fixed entropy-respecting solver (Lax-Friedrichs, periodic, clipped) --------
def lxf_evolve(rho0, flux_fn, dx, dt, n_steps):
    rho = rho0.copy()
    for _ in range(n_steps):
        fr = flux_fn(rho)
        rho_l = np.roll(rho, 1); rho_r = np.roll(rho, -1)
        f_l = np.roll(fr, 1); f_r = np.roll(fr, -1)
        rho = 0.5 * (rho_l + rho_r) - (dt / (2.0 * dx)) * (f_r - f_l)
        rho = np.clip(rho, 0.0, 1.0)
    return rho


def main():
    if len(sys.argv) < 2:
        print("1 1 1"); print("0.0 0.0 0.0"); return
    t = int(sys.argv[1])

    V, S, W, freq = flux_params(t)

    x_fine = np.linspace(0.0, 1.0, N_FINE, endpoint=False)
    dx_fine = 1.0 / N_FINE
    rho0_fine = train_ic(x_fine, t)

    # analytic-ish safety margin so the recorded window stays shock-free
    rr = np.linspace(0.001, 0.999, 401)
    fpp = np.gradient(fprime_true_vec(rr, V, S, W, freq), rr)
    Fpp = float(np.max(np.abs(fpp)))
    drho = (np.roll(rho0_fine, -1) - np.roll(rho0_fine, 1)) / (2.0 * dx_fine)
    Rp = float(np.max(np.abs(drho)))
    t_break_est = 1.0 / (Fpp * Rp + 1e-9)
    T_train = min(TARGET_T_TRAIN, SAFETY * t_break_est)
    T_train = max(T_train, 0.006)

    max_speed = float(np.max(np.abs(fprime_true_vec(rr, V, S, W, freq))))
    dt = CFL * dx_fine / max(max_speed, 0.3)
    dt_obs = T_train / (K - 1)
    steps_per_obs = max(1, int(round(dt_obs / dt)))
    dt_actual = dt_obs / steps_per_obs

    def flux_fn(r):
        return f_true_vec(r, V, S, W, freq)

    snaps = [rho0_fine.copy()]
    cur = rho0_fine.copy()
    for _ in range(K - 1):
        cur = lxf_evolve(cur, flux_fn, dx_fine, dt_actual, steps_per_obs)
        snaps.append(cur.copy())

    # difficulty ladder: fewer sensors, more noise as t grows
    frac = (t - 1) / 9.0
    M = int(round(32 - 14 * frac))
    sigma = 0.006 + (0.022 - 0.006) * frac

    idx = (np.round(np.arange(M) * (1.0 / M) * N_FINE).astype(int)) % N_FINE
    x_m = np.arange(M) / M

    rng2 = random.Random(662607 + t * 15485863)

    out = []
    out.append("%d %d %d" % (M, K, t))
    for k in range(K):
        tk = k * dt_obs
        snap = snaps[k]
        for m in range(M):
            val = float(snap[idx[m]]) + rng2.gauss(0.0, sigma)
            val = min(1.0, max(0.0, val))
            out.append("%.6f %.6f %.6f" % (tk, x_m[m], val))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
