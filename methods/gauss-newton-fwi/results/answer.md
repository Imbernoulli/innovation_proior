# Full-Waveform Inversion via the adjoint-state method

## Problem

Reconstruct a spatial wave-speed model of the subsurface, c(x) (equivalently the squared slowness
m = 1/c²), from recorded seismograms. Sources are fired at the surface and an array of receivers
records the waveforms. Fit the data in the least-squares sense: minimize the L2 waveform misfit
between predicted and observed traces over the model,

    J(m) = ½ Σ_s ∫₀ᵀ ‖ S u_s(·,t) − d_s(t) ‖² dt,

where, for each shot s, u_s solves the acoustic wave equation in model m and S samples the wavefield
at the receivers. The model has millions of grid-point parameters and one scalar misfit; the central
difficulty is computing ∂J/∂m affordably, and reaching the correct minimum despite a highly
non-convex misfit.

## Key idea

Treat the wave equation as a constraint F(u,m)=0 and apply the adjoint-state method. The gradient of
the waveform misfit with respect to *all* model parameters is obtained from **one forward wavefield
plus one adjoint wavefield**, independent of the number of parameters, instead of one simulation per
parameter. The adjoint of the wave operator is the wave equation run **backward in time, sourced by
the data residual injected at the receiver positions**; with the change of variable q(t)=λ(T−t) it is
just another forward solve. Because the model parameter m=1/c² multiplies ∂²u/∂t² in the wave
operator, the gradient is the **zero-lag time correlation of the forward wavefield's second time
derivative with the back-propagated residual wavefield**:

    ∂J/∂m(x) = − Σ_s ∫₀ᵀ λ_s(x,t) ∂²u_s(x,t)/∂t² dt   (m = 1/c²),

with the adjoint field λ_s solving the *same* wave operator,

    m ∂²λ_s/∂t² − Δλ_s = Σ_r S^T_{s,r}(S_{s,r} u_s − d_{s,r}),   λ_s(T)=0, ∂_t λ_s(T)=0.

This is the imaging/migration principle read as an exact derivative. For a velocity
parameterization the chain rule m=1/c² (∂m/∂c=−2/c³) gives

    ∂J/∂c(x) = (2/c³) Σ_s ∫₀ᵀ λ_s(x,t) ∂²u_s(x,t)/∂t² dt.

Steepest descent on this gradient is mis-scaled by the geometrical spreading of the waves
(over-updating shallow, under-updating deep). The **Gauss–Newton** model J=½‖r‖² with Hessian
H_GN = Re(JᴴJ) (dropping the multiple-scattering second-derivative term) corrects this: its diagonal
is the zero-lag autocorrelation of the partial-derivative wavefields — a source-illumination
preconditioner — and the full step δm = −H_GN⁻¹∇J is solved matrix-free by conjugate gradients
(each H_GN-vector product = one Born forward-sensitivity solve + one adjoint solve).

The L2 misfit over oscillatory seismograms is severely non-convex: **cycle skipping** occurs when a
predicted arrival is more than half a period off its observed twin, trapping a local optimizer in a
spurious minimum that matches the wrong cycle. Since lower frequencies have longer periods (a wider
half-period tolerance), the remedy is **multiscale frequency continuation**: invert the lowest
frequencies first to recover the smooth background velocity, then progressively add higher-frequency
bands, so each scale starts within half a period of the data.

## Algorithm

Per shot s, per outer iteration:
1. **Forward solve.** March m ∂²u_s/∂t² − Δu_s = f_s forward in time; record the trace S u_s at
   receivers and store the wavefield u_s.
2. **Residual.** r_s = S u_s − d_s^obs; accumulate J += ½‖r_s‖².
3. **Adjoint solve.** March the same wave operator sourced by the **time-reversed residual** injected
   at the receivers (q_s(t)=λ_s(T−t), forward-in-time), getting the back-propagated field.
4. **Gradient (zero-lag correlation).** Accumulate g(x) += −∫ λ_s(x,t) ∂²u_s(x,t)/∂t² dt over time and
   shots. (Multiply by 2/c³ if updating velocity.)
5. **Precondition.** Divide g by the source-illumination diagonal pseudo-Hessian (or take a matrix-free
   Gauss–Newton step via CG).
6. **Update.** Line-search a step α and set m ← m − α·g_pc.

Wrap steps 1–6 in a multiscale loop: low-pass the wavelet/data to the lowest band, iterate, raise the
corner frequency, repeat.

## Code

A self-contained 2D constant-density acoustic FD modeler, residual at receivers, adjoint
(time-reversed residual) simulation, gradient by correlating the forward field's ∂²/∂t² with the
back-propagated field, and a preconditioned gradient-descent / Gauss–Newton update.

```python
import numpy as np

# ---------------------------------------------------------------- forward modeling
def laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
                       - 4.0 * u[1:-1, 1:-1]) / dx**2
    return lap

def damping_mask(shape, nb, dx, amp=0.0015):
    """Multiplicative absorbing sponge in a border of nb cells."""
    nz, nx = shape
    d = np.ones((nz, nx))
    for i in range(nb):
        w = (1.0 - amp * (nb - i)**2)
        d[i, :] *= w; d[nz-1-i, :] *= w; d[:, i] *= w; d[:, nx-1-i] *= w
    return d

def forward(m, wavelet, src, rec, nt, dt, dx, damp, store=False):
    """March  m u_tt - Lap(u) = f .  m = 1/c^2 per cell.  Return shot record (and wavefield)."""
    nz, nx = m.shape
    u_prev = np.zeros((nz, nx)); u_cur = np.zeros((nz, nx))
    d = np.zeros((nt, len(rec)))
    U = np.zeros((nt, nz, nx)) if store else None
    sz, sx = src
    for it in range(nt):
        lap = laplacian(u_cur, dx)
        f = np.zeros((nz, nx)); f[sz, sx] += wavelet[it]            # source injection
        u_next = (2.0 * u_cur - u_prev + (dt**2) * (lap + f) / m)   # leapfrog: m u_tt = Lap u + f
        u_next *= damp                                             # absorbing boundary
        for k, (rz, rx) in enumerate(rec): d[it, k] = u_cur[rz, rx]  # sample receivers
        if store: U[it] = u_cur
        u_prev, u_cur = u_cur * damp, u_next
    return (d, U) if store else d

def residual_misfit(d_pred, d_obs):
    r = d_pred - d_obs
    return r, 0.5 * np.sum(r * r)

# ---------------------------------------------------------------- adjoint + gradient
def gradient_one_shot(m, wavelet, src, rec, d_obs, nt, dt, dx, damp):
    """dJ/dm for one shot:  forward field, then SAME solver sourced by the time-reversed
       residual at receivers; gradient = - sum_t (d^2u/dt^2)(t) * lambda(t),  lambda(t)=q(T-t)."""
    d_pred, U = forward(m, wavelet, src, rec, nt, dt, dx, damp, store=True)
    r, J = residual_misfit(d_pred, d_obs)                          # data residual at receivers
    nz, nx = m.shape

    Utt = np.zeros_like(U)                                         # forward field 2nd time deriv
    Utt[1:-1] = (U[2:] - 2.0 * U[1:-1] + U[:-2]) / dt**2

    q_prev = np.zeros((nz, nx)); q_cur = np.zeros((nz, nx))
    g = np.zeros((nz, nx))
    for it in range(nt):                                          # adjoint marched forward in q
        lap = laplacian(q_cur, dx)
        fa = np.zeros((nz, nx))
        for k, (rz, rx) in enumerate(rec):
            fa[rz, rx] += r[nt - 1 - it, k]                       # time-reversed residual source
        q_next = (2.0 * q_cur - q_prev + (dt**2) * (lap + fa) / m)
        q_next *= damp
        g += - Utt[nt - 1 - it] * q_cur                          # zero-lag correlation, lambda=q(T-t)
        q_prev, q_cur = q_cur * damp, q_next
    return g, J          # g = dJ/dm ;  for velocity update multiply by 2/c^3 = 2 * m**1.5

# ---------------------------------------------------------------- inversion driver
def line_search(m, p, g, eval_J, J0, frac=0.05, c1=1e-4, shrink=0.5, n=20):
    """Backtracking Armijo for the step m <- m - alpha*p (descent direction -p).
       Directional derivative of J along -p is -g.p, so sufficient decrease is
       J(m - alpha p) <= J0 - c1 * alpha * (g.p).  The trial step is scaled so the
       first model perturbation is a small fraction of the model magnitude."""
    gdotp = float(np.sum(g * p))
    if gdotp <= 0.0:
        return 0.0                                      # not a descent direction
    alpha = frac * (np.max(np.abs(m)) / (np.max(np.abs(p)) + 1e-30))
    for _ in range(n):
        if eval_J(m - alpha * p) <= J0 - c1 * alpha * gdotp:
            return alpha
        alpha *= shrink
    return 0.0

def invert(m0, wavelet, src_list, rec, d_obs_list, nt, dt, dx, damp, n_iter):
    m = m0.copy()
    def total_misfit(mm):
        J = 0.0
        for src, d_obs in zip(src_list, d_obs_list):
            d_pred = forward(mm, wavelet, src, rec, nt, dt, dx, damp)
            J += residual_misfit(d_pred, d_obs)[1]
        return J
    for k in range(n_iter):
        g = np.zeros_like(m); illum = np.zeros_like(m); J = 0.0
        for src, d_obs in zip(src_list, d_obs_list):
            _, U = forward(m, wavelet, src, rec, nt, dt, dx, damp, store=True)
            gs, Js = gradient_one_shot(m, wavelet, src, rec, d_obs, nt, dt, dx, damp)
            g += gs; J += Js
            illum += np.sum(U * U, axis=0)                        # diag(H_GN) ~ source illumination
        g_pc = g / (illum + 1e-6 * illum.max())                  # Gauss-Newton-like rescaling
        alpha = line_search(m, g_pc, g, total_misfit, J)
        m = m - alpha * g_pc                                     # model update (in slowness m=1/c^2)
        print(f"iter {k:3d}  J = {J:.4e}  step = {alpha:.2e}")
    return m

# ---------------------------------------------------------------- multiscale wrapper
def ricker(nt, dt, f0):
    t = (np.arange(nt) * dt) - 1.0 / f0
    a = (np.pi * f0 * t)**2
    return (1.0 - 2.0 * a) * np.exp(-a)

def lowpass(trace_or_wavelet, dt, fmax):
    """Crude spectral low-pass to band-limit data/wavelet for a multiscale stage."""
    X = np.fft.rfft(trace_or_wavelet, axis=0)
    freqs = np.fft.rfftfreq(trace_or_wavelet.shape[0], dt)
    X[freqs > fmax] = 0.0
    return np.fft.irfft(X, n=trace_or_wavelet.shape[0], axis=0)

def multiscale_invert(m0, wavelet, src_list, rec, d_obs_list, nt, dt, dx, damp,
                      freq_bands, iters_per_band):
    """Invert lowest frequencies first, then raise the band -> stay within half a period."""
    m = m0.copy()
    for fmax in freq_bands:                                       # e.g. [3, 6, 10] Hz
        w_b   = lowpass(wavelet, dt, fmax)
        d_b   = [lowpass(d, dt, fmax) for d in d_obs_list]
        m = invert(m, w_b, src_list, rec, d_b, nt, dt, dx, damp, iters_per_band)
    return m
```
