OK, let me think this through from scratch. I have a seismic survey: sources fired at the surface,
receivers recording the wiggles that come back. Buried in those traces is the wave speed everywhere
underground, c(x). I want the computer to *recover* c(x) by fitting the data — find the model whose
simulated seismograms match the recorded ones. I have an acoustic forward solver: give it a velocity
model and a source, it marches the wave equation in time and hands me the predicted trace at each
receiver. So the object I want to minimize is the waveform misfit, the L2 distance between predicted
and observed traces, summed over receivers and time and shots,

    J(m) = ½ Σ_s ∫₀ᵀ ‖ S u_s(·,t) − d_s(t) ‖² dt,

where u_s solves the wave equation in model m for shot s, and S samples the wavefield at the
receivers. I'll parameterize the model by the squared slowness m = 1/c² to start, because that's what
sits linearly in the wave operator; I can convert to velocity at the end. To push m downhill I need
the gradient ∂J/∂m at every grid point.

The brute-force way is staring at me. Perturb one grid point's slowness by a little δ, re-run the
whole forward simulation, see how much J changed, divide:

    ∂J/∂m_i ≈ (J(m + δ e_i) − J(m)) / δ.

This is exact in the limit and conceptually trivial. But count the forward solves: one baseline plus
one per model parameter. And the model parameter is *every grid point of the earth* — a 2D model is
already hundreds of thousands of cells, a 3D one is millions. One full wave simulation per cell, times
millions of cells, every gradient, every iteration. That's not a wall, it's a mountain. And there's
the usual step-size disease on top: δ too big contaminates the slope with curvature, δ too small and
the difference drowns in the solver's discretization jitter. So finite differences are out, not by a
little but by six orders of magnitude.

Let me try to be cleverer but stay "forward." The reason FD costs M solves is I'm re-running the
nonlinear forward map M times. What if I only linearize? The forward wavefield satisfies the wave
equation, which I'll write abstractly as a constraint F(u, m) = 0 — the residual of "m u_tt − Δu − f"
is zero at the true solution. Differentiate that with respect to one model parameter:

    (∂F/∂u)(∂u/∂m_i) + ∂F/∂m_i = 0   ⟹   (∂F/∂u)(∂u/∂m_i) = −∂F/∂m_i.

That's a *linear* wave equation for the wavefield sensitivity ∂u/∂m_i — solve it once (no re-running
the nonlinear forward), then dJ/dm_i = (∂J/∂u)·(∂u/∂m_i). Cleaner than FD, exact, no step-size
dilemma. But… the right-hand side −∂F/∂m_i is different for every grid point i, so I have to solve
that linear wave equation once for *each* model parameter. Still M solves. I've traded M nonlinear
solves for M linear solves — the M never left the expensive part. The structure of the trade hasn't
changed. Hmm.

Stare at what I'm actually computing. Write the misfit change under a model change δm:

    δJ = (∂J/∂u)·δu + (∂J/∂m)·δm.

The second term is the *direct* dependence of the misfit on the model — but J only sees m through the
wavefield, so for the waveform misfit that direct term is zero; everything is in δu. And δu is the
expensive object: it's the wavefield's response to the model change, the thing that needs a linearized
wave solve per parameter. The constraint, linearized, ties δu to δm:

    (∂F/∂u) δu + (∂F/∂m) δm = 0,   so   δu = −(∂F/∂u)⁻¹ (∂F/∂m) δm.

Substitute into δJ:

    δJ = − (∂J/∂u)·(∂F/∂u)⁻¹ (∂F/∂m) δm.

There it is, fully assembled. And now look at the order of operations. That's a row vector — the data
residual mapped to the wavefield, ∂J/∂u — times an inverse of the wave operator, times the matrix
∂F/∂m. I've been computing it left-to-right: (∂F/∂u)⁻¹(∂F/∂m) first, which is M right-hand sides, M
wave solves, the forward sensitivities. But matrix products associate however I like. Group it
right-to-left:

    δJ = − [ (∂J/∂u)·(∂F/∂u)⁻¹ ] (∂F/∂m) δm.

The thing in brackets is one row vector solved against the inverse — a *single* linear wave solve,
completely independent of how many model parameters there are. Then I dot the result into the columns
of ∂F/∂m, once per parameter, but each of those is a cheap local product, not a wave solve. The M
moved out of the expensive part and into the cheap part. That's the whole trick, and it's just
associativity — the same move that turns "one solve per input" into "one solve, period."

Let me make that single solve concrete and respectable. Define the field λ by

    (∂J/∂u)·(∂F/∂u)⁻¹ = λᵀ   ⟺   (∂F/∂u)ᵀ λ = ∂J/∂u.

So λ solves a linear system whose operator is the **transpose (adjoint)** of the wave operator and
whose right-hand side is ∂J/∂u — the data residual mapped back onto the wavefield grid. One solve, no
m in it anywhere. Then

    δJ = − λᵀ (∂F/∂m) δm,   i.e.   ∂J/∂m = − (∂F/∂m)ᵀ λ,

and the gradient with respect to *all* model parameters falls out of cheap local products of the
single field λ against ∂F/∂m. Cost of the whole gradient: one forward solve (to get u, hence ∂J/∂u),
plus one adjoint solve for λ, plus cheap assembly — independent of M. The earth can have a million
cells and the gradient costs the same two solves.

I can reach the same λ-system the Lagrangian way, which shows where it comes from and is reassuring.
The wave equation F(u,m) = 0 means I'm free to add any multiple of it to J without changing anything.
Form the Lagrangian L = J − ⟨λ, F(u,m)⟩ with λ a field of multipliers, one per scalar wave-equation
constraint (i.e. one per grid point per time step). Then

    δL = (∂J/∂u)·δu − ⟨λ, (∂F/∂u)δu + (∂F/∂m)δm⟩
       = [ (∂J/∂u) − (∂F/∂u)ᵀλ ]·δu − ⟨(∂F/∂m)ᵀλ, δm⟩.

λ is mine to choose. The δu bracket is the only place the expensive wavefield sensitivity appears, so
I choose λ to annihilate it: (∂F/∂u)ᵀ λ = ∂J/∂u. Same adjoint equation. Then ∂J/∂m = −(∂F/∂m)ᵀλ. The
multiplier field λ *is* the costate — the Lagrange-multiplier field from Lions' optimal-control-of-PDEs
machinery — and the "solve once, get the gradient against all controls" structure is exactly Bryson &
Ho's trajectory-optimization trick: there the costate is integrated backward in time in one sweep and
delivers the gradient against every control at once; here the wave analog is one transposed/backward
solve delivering the gradient against every grid point at once. Chavent already carried the
adjoint-state idea into inverse problems abstractly. The thing to do is make it concrete for the
acoustic wave equation — get the actual adjoint source, the actual final conditions, the actual
weighting in the gradient — and turn it into an iterative model-fitting scheme.

So let me stop being abstract and write the adjoint of the *actual* wave operator. The forward problem
in the time domain, with m = 1/c² the squared slowness, the wave operator L = m ∂²/∂t² − Δ, source f_s
at the shot, and zero initial conditions:

    m ∂²u_s/∂t² − Δu_s = f_s,    u_s(0) = 0,  ∂_t u_s(0) = 0.

The misfit, written out:

    J = ½ Σ_{s,r} ∫₀ᵀ (S_{s,r} u_s − d_{s,r})² dt.

Build the augmented functional, multiplying the constraint by a costate field λ_s and integrating over
the domain and over time, and I have to be careful to also carry multipliers for the two initial
conditions because they're constraints too:

    L = J − Σ_s ∫₀ᵀ ⟨ λ_s, m ∂²u_s/∂t² − Δu_s − f_s ⟩ dt − (initial-condition terms).

Set ∂L/∂u_s = 0. The misfit contributes its derivative ∂J/∂u_s = Σ_r S^T_{s,r}(S_{s,r}u_s − d_{s,r}) —
the data residual, mapped from the receivers back onto the wavefield grid by the transpose of the
sampling operator. That residual is an injection at the receiver locations and nowhere else. The
constraint contributes the adjoint of the wave operator acting on λ_s. The Laplacian is self-adjoint;
the only subtle piece is the ∂²/∂t² term, which I have to handle by integrating by parts in time
twice. Do it carefully:

    ∫₀ᵀ ⟨ λ_s, m ∂²(δu_s)/∂t² ⟩ dt = ∫₀ᵀ ⟨ m ∂²λ_s/∂t², δu_s ⟩ dt
        + [ ⟨λ_s, m ∂_t δu_s⟩ − ⟨m ∂_t λ_s, δu_s⟩ ]₀ᵀ.

Two integrations by part, each flipping a time derivative onto λ_s; ∂²/∂t² is self-adjoint in the
interior, which is the nice part. The boundary terms at t = 0 and t = T are the interesting part. At
t = 0 the forward field's variation δu_s and its time derivative vanish (the initial conditions are
fixed: u_s(0) = 0, ∂_t u_s(0) = 0), so those boundary terms die on their own. At t = T nothing pins
δu_s, so to kill the t = T boundary terms I'm forced to impose conditions on λ_s instead:

    λ_s(T) = 0,    ∂_t λ_s(T) = 0.

**Final** conditions, not initial. So the adjoint field equation is

    m ∂²λ_s/∂t² − Δλ_s = Σ_r S^T_{s,r}(S_{s,r}u_s − d_{s,r}),    λ_s(T) = 0,  ∂_t λ_s(T) = 0.

Read it. The left side is the *same wave operator* as the forward problem — same m, same Laplacian,
self-adjoint. The right side is the data residual injected at the receiver positions. And it carries
*final* conditions, so it must be solved **backward in time**, from T down to 0. The adjoint of the
wave equation is the wave equation run in reverse, sourced by the residual at the receivers. The
residual — the part of the recorded wiggle the current model fails to predict — is shot back down into
the earth as if the receivers were sources.

Solving something backward in time is awkward to code. Let me change variables to flip it into a
forward march. Set q_s(t) = λ_s(T − t). Then ∂²q_s/∂t² = ∂²λ_s/∂t² evaluated at T − t (the chain rule
brings two minus signs that cancel), and the final conditions λ_s(T) = 0, ∂_tλ_s(T) = 0 become initial
conditions q_s(0) = 0, ∂_t q_s(0) = 0. The adjoint becomes an ordinary *forward* wave solve:

    m ∂²q_s/∂t² − Δq_s = Σ_r S^T_{s,r}(S_{s,r}u_s(T − t) − d_{s,r}(T − t)),   q_s(0) = 0, ∂_t q_s(0) = 0.

Same solver as the forward modeler, just with a different source: the **time-reversed residual**
injected at the receivers. So I get λ for free from my existing forward code — I never wrote a
separate backward integrator.

Now the gradient. With λ chosen, ∂J/∂m = −(∂F/∂m)ᵀλ. The only place m appears in the wave operator is
in front of ∂²u/∂t², so ∂F/∂m at grid point x is just ∂²u_s/∂t² evaluated there. Therefore

    ∂J/∂m(x) = − Σ_s ∫₀ᵀ λ_s(x,t) ∂²u_s(x,t)/∂t² dt
             = − Σ_s ∫₀ᵀ q_s(x, T − t) ∂²u_s(x,t)/∂t² dt.

So the gradient at every grid point is the **zero-lag time correlation of the forward field's second
time derivative with the back-propagated residual field**, summed over shots. The forward wavefield u
sweeps down from the source; the residual field λ sweeps in from the receivers, back in time; wherever
they coincide *at the same time and place* the integrand lights up, and that's where the model is
wrong. This is exactly the imaging principle read as a derivative — Lailly and Claerbout's picture
that the source wavefield correlated with a receiver-side wavefield at zero lag images the reflectors
— except now I see it isn't a heuristic, it's literally ∂J/∂m of the L2 waveform misfit. The migration
operator *is* the gradient.

Let me sanity-check the ∂²/∂t² weight, because it's the easy place to drop a factor or a sign, and
the frequency domain makes it transparent. In frequency, ∂²/∂t² → −ω², the wave operator is
A = −ω²m − Δ, the forward solve is A u_s = f_s, and the adjoint is Aᴴ λ_s = residual. The single place
m enters A is the −ω² m term, so ∂A/∂m = −ω², and the gradient is ∂J/∂m(x) = Re Σ_ω Σ_s (−ω²) ·
(− λ*_s(x,ω) u_s(x,ω)) = Re Σ_ω Σ_s ω² λ*_s u_s. The ω² is precisely the −∂²/∂t² turned around, and the
overall minus from −(∂F/∂m)ᵀλ has cancelled the minus in ∂A/∂m = −ω², leaving a clean +ω². Back in
time domain that's the −∂²/∂t² weighting with the leading sign I wrote: ∂J/∂m = −Σ_s ∫ λ_s ∂²u_s/∂t²
dt. Signs and factor agree both ways. Good.

Now, I parameterized in squared slowness because it's linear in the operator, but I actually want to
update *velocity* c, and physical intuition reads in velocity. Chain rule. m = 1/c², so ∂m/∂c = −2/c³.
Then

    ∂J/∂c = (∂J/∂m)(∂m/∂c) = (−2/c³)·(∂J/∂m)
          = (−2/c³)·(− Σ_s ∫₀ᵀ λ_s ∂²u_s/∂t² dt) = (2/c³) Σ_s ∫₀ᵀ λ_s ∂²u_s/∂t² dt.

So the velocity gradient is the same forward·adjoint correlation, scaled pointwise by 2/c³. The factor
matters: it reweights the update by velocity cubed, and the two minus signs (from the slowness
gradient and from ∂m/∂c) cancel, so the velocity gradient ends up with the *opposite* overall sign to
the slowness gradient — which is right, because raising c lowers m, and the misfit responds to m. I'll
keep the inversion in m to avoid carrying the 2/c³ around, and convert c = 1/√m only when I report the
model; but it's good to know the velocity-domain gradient explicitly, because if I'd inverted in c
directly and forgotten the 2/c³ the deep, slow part of the model would be mis-scaled.

So the gradient is two wave solves per shot, regardless of M. But will steepest descent on it actually
work? Two problems are sitting there. First, conditioning. The waves spread geometrically, so u and λ
decay with depth — the integrand λ ∂²u/∂t² is huge near the sources and tiny deep down. A raw −α∇J
step therefore slams the shallow model and barely moves the deep model. That's the Hessian talking.
Write the misfit as J = ½‖r(m)‖² with r the residual; the gradient is ∇J = Re(Jᴴ r) where J = ∂r/∂m is
the Fréchet (Born) derivative, and the exact Hessian is Re(Jᴴ J) + (a term in ∂²r/∂m² weighted by r).
Drop the second term — that's the Gauss–Newton approximation, exact in the limit of a small residual
and cheaper because it needs only first derivatives — and the model is H_GN = Re(Jᴴ J), with the
descent step δm = −H_GN⁻¹ ∇J. The diagonal of H_GN is the zero-lag *auto*correlation of the
partial-derivative wavefields, which is essentially the squared illumination — strong where the waves
are strong, weak where they're weak. So H_GN⁻¹ is exactly the deconvolution that undoes the
geometrical spreading and rescales the deep model up. The catch: H_GN is M×M and dense — I can't form
it, let alone invert it. So I approximate. The diagonal of H_GN is a cheap "pseudo-Hessian" /
source-illumination preconditioner I can accumulate from the forward wavefield energy (Σ_s ∫ u_s² dt
at each point, with a small stabilizer), and dividing the gradient by it gives a depth-balanced update
without ever forming the full Hessian. When I want the genuine Gauss–Newton step I solve H_GN δm = −∇J
matrix-free with conjugate gradients, where each H_GN-times-vector is one Born (forward-sensitivity)
solve plus one adjoint solve — no explicit matrix. Either way the second-order scaling is what turns a
crawling steepest descent into something that converges in a handful of iterations.

The second problem is the one that can send the whole thing to the wrong answer: the misfit is
violently non-convex. Why? Seismic traces are oscillatory — wave trains with a dominant period. Take
one predicted wiggle and the matching observed wiggle. If my model is off so that the predicted wiggle
arrives shifted by, say, a third of a period, the L2 misfit pulls it toward lining up with the correct
cycle — fine, the gradient is informative. But if it's shifted by *more than half a period*, the
nearest observed cycle is now the *neighbouring* one, and least-squares, which only knows local
distance, prefers to match the predicted wiggle to that wrong neighbouring cycle. The gradient then
points toward a model that fits a phase-shifted copy of the data — a spurious local minimum. This is
cycle skipping, and the half-period threshold is sharp: matching the data to within half a period is
the necessary condition for the local optimizer to head toward the truth and not into a wrong basin.

I can't convexify the wave physics, but I can choose *which data* I show the optimizer, and the
half-period criterion tells me exactly how. The threshold is half a *period* — and the lower the
frequency, the longer the period, so the wider the window of model error that still stays within half
a cycle. Low-frequency data are far more forgiving: with only the long-period content present, even a
crude starting model keeps every predicted arrival within half a period of its observed twin, so the
misfit for the smooth, long-wavelength part of the model is gentle and nearly convex. The fine,
high-frequency detail is what folds the misfit surface into a minefield. So invert frequency by
frequency, low to high. Start with the lowest frequencies in the data, fit them to recover the smooth
background velocity — the part low frequencies can constrain and the part that controls phase. With
that background in hand, the predicted high-frequency arrivals are now already within half a period of
the observed ones, so I can safely add the next frequency band and let it sharpen the model, and so
on up. Each scale hands the next a starting model good enough to stay out of the wrong basin. That's
multiscale frequency continuation: not a trick bolted on, but the direct operational reading of the
half-period rule.

Let me land this on real code, on a 2D acoustic grid where I can check every number. State is the
pressure wavefield u on a grid; the model is m = 1/c² per cell; the source is a Ricker wavelet; I
march u_tt = (Δu + f)/m with a leapfrog stencil and a damping sponge at the edges. The forward solve
records the trace at receivers *and* stores the wavefield so I can correlate it later. The adjoint
solve is the same time loop sourced by the time-reversed residual at the receivers; the gradient
accumulates −u ⊙ (∂²λ/∂t²) — equivalently −(∂²u/∂t²) ⊙ λ, the same correlation — over time and shots.

```python
import numpy as np

def laplacian(u, dx):
    # 2nd-order 5-point Laplacian on the 2D grid
    lap = np.zeros_like(u)
    lap[1:-1,1:-1] = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2]
                      - 4.0*u[1:-1,1:-1]) / dx**2
    return lap

def forward(m, wavelet, src, rec, nt, dt, dx, damp, store=False):
    """March  m u_tt - Lap(u) = f.  m = 1/c^2 per cell.  Return shot record (and wavefield)."""
    nz, nx = m.shape
    u_prev = np.zeros((nz, nx)); u_cur = np.zeros((nz, nx))
    d = np.zeros((nt, len(rec)))
    U = np.zeros((nt, nz, nx)) if store else None
    for it in range(nt):
        lap = laplacian(u_cur, dx)
        f = np.zeros((nz, nx)); f[src] += wavelet[it]          # inject source
        # leapfrog:  u_next = 2u - u_prev + dt^2 (Lap u + f)/m ,  with a damping sponge
        u_next = (2.0*u_cur - u_prev + (dt**2)*(lap + f)/m) * 1.0
        u_next *= damp; u_cur_d = u_cur * damp                 # absorbing boundary
        for k,(rz,rx) in enumerate(rec): d[it,k] = u_cur[rz,rx]  # sample receivers
        if store: U[it] = u_cur
        u_prev, u_cur = u_cur_d, u_next
    return (d, U) if store else d

def residual_misfit(d_pred, d_obs):
    r = d_pred - d_obs
    return r, 0.5*np.sum(r*r)

def gradient_one_shot(m, wavelet, src, rec, d_obs, nt, dt, dx, damp):
    # 1) forward: predicted trace + stored forward wavefield U
    d_pred, U = forward(m, wavelet, src, rec, nt, dt, dx, damp, store=True)
    r, J = residual_misfit(d_pred, d_obs)                       # data residual at receivers

    # 2) adjoint = SAME wave solver sourced by the TIME-REVERSED residual injected at receivers,
    #    marched forward in q(t)=lambda(T-t); accumulate the gradient on the fly.
    nz, nx = m.shape
    q_prev = np.zeros((nz, nx)); q_cur = np.zeros((nz, nx))
    g = np.zeros((nz, nx))
    # precompute the forward field's 2nd time derivative for the correlation
    Utt = np.zeros_like(U)
    Utt[1:-1] = (U[2:] - 2.0*U[1:-1] + U[:-2]) / dt**2
    for it in range(nt):
        lap = laplacian(q_cur, dx)
        fa = np.zeros((nz, nx))
        for k,(rz,rx) in enumerate(rec): fa[rz,rx] += r[nt-1-it, k]   # time-reversed residual source
        q_next = (2.0*q_cur - q_prev + (dt**2)*(lap + fa)/m)
        q_next *= damp; q_cur_d = q_cur * damp
        # zero-lag correlation:  g += - (d^2u/dt^2)(t) * lambda(t),  lambda(t)=q(T-t)
        g += - Utt[nt-1-it] * q_cur
        q_prev, q_cur = q_cur_d, q_next
    return g, J     # g = dJ/dm ;  multiply by 2/c^3 = 2 m^{1.5} to get dJ/dc if updating velocity

def invert(m0, wavelet, src_list, rec, d_obs_list, nt, dt, dx, damp, n_iter):
    m = m0.copy()
    for k in range(n_iter):
        g = np.zeros_like(m); J = 0.0
        illum = np.zeros_like(m)                                # diagonal pseudo-Hessian
        for s,(src, d_obs) in enumerate(zip(src_list, d_obs_list)):
            d_pred, U = forward(m, wavelet, src, rec, nt, dt, dx, damp, store=True)
            r, Js = residual_misfit(d_pred, d_obs); J += Js
            gs, _ = gradient_one_shot(m, wavelet, src, rec, d_obs, nt, dt, dx, damp)
            g += gs
            illum += np.sum(U*U, axis=0)                        # source illumination ~ diag(H_GN)
        g_pc = g / (illum + 1e-6*illum.max())                   # Gauss-Newton-like rescaling
        alpha = line_search(m, g_pc, ...)                       # backtracking on J
        m = m - alpha * g_pc                                    # model update (in slowness)
    return m
```

The Gauss–Newton step, when I want the real thing instead of the diagonal, is the same pieces wired
matrix-free: H_GN δm = −∇J solved with conjugate gradients, where each application of H_GN to a vector
is one Born forward-sensitivity solve followed by one adjoint solve — no matrix ever formed. And the
outer multiscale loop wraps all of this: low-pass the wavelet and the data to the lowest frequency
band, run a few iterations to fix the smooth background, raise the corner frequency, repeat — each
band starts from a model already within half a period of the data so the local optimizer never cycle-skips.

So the causal chain, end to end: I want ∂J/∂m to drive a velocity model downhill; finite differences
and forward sensitivities both cost one wave simulation per model parameter — millions — because the
count of parameters sits in the expensive part. Treating the wave equation as a constraint F(u,m)=0
and writing δJ = −(∂J/∂u)(∂F/∂u)⁻¹(∂F/∂m)δm, then regrouping that product right-to-left, moves the
parameter count out of the expensive part: define the costate λ by the *transposed* wave equation
(∂F/∂u)ᵀλ = ∂J/∂u, and the gradient is −(∂F/∂m)ᵀλ — one forward solve plus one adjoint solve, gradient
against all grid points at once. The adjoint of the wave operator is the wave equation run backward in
time, sourced by the data residual injected at the receivers; with the change of variable q(t)=λ(T−t)
it's just another forward solve. Since m sits in front of ∂²u/∂t², the gradient is the zero-lag
correlation ∂J/∂m = −Σ_s ∫ λ_s ∂²u_s/∂t² dt — the imaging principle, but exact — and in velocity it
carries the factor 2/c³ from m=1/c². Steepest descent on this is mis-scaled by geometrical spreading,
so I deconvolve by the Gauss–Newton Hessian — its diagonal as a cheap illumination preconditioner, or
the full H_GN solved matrix-free by conjugate gradients. And because the oscillatory misfit cycle-skips
whenever a predicted arrival is more than half a period off, I invert low frequencies first and add
higher bands as the background sharpens, so every scale stays in the right basin. Two solves per shot,
no matter how many model parameters.
