The first-order run told me exactly where the budget is being wasted, and it told me in FID. DDIM came
in at 34.23 on SD v1.5, 28.41 on SD v2.0, and 51.52 on SDXL — usable images on every variant (it never
diverged, which is the one thing first order reliably buys), but those FIDs are high, and the *shape*
of the gap is the tell. SD v2.0, the easiest variant, sits lowest; SDXL, with the largest latent and
the most structure to resolve in twenty coarse steps, is worst by a wide margin. Put numbers on "wide":
SDXL at 51.52 is $1.81\times$ SD v2.0's 28.41, a gap of 23.11 FID points, whereas SD v1.5 sits only 5.82
above SD v2.0. So the variants are not uniformly bad — they fan out, and they fan out in order of how
much structure there is to resolve. That is precisely the signature I expected from a first-order solver:
the error scales with the curvature of the denoiser trajectory between adjacent noise levels, and where
there is more fine structure to resolve, twenty straight-line steps leave the most on the table. The
failure is not instability (nothing diverged) and it is not a guidance problem (CFG++ is fixed and shared
across variants) — it is *step-efficiency*: the first-order truncation error the floor quantified, paid
twenty times over and worst where the trajectory bends sharpest. The fix is not to ask for more calls — I
do not have them — but to extract the *bend* of the trajectory from the calls I already spend, i.e. to go
to second order.

Before I reach for "second order," though, I have to be careful, because the obvious move — grab an
off-the-shelf high-order ODE solver and point it at the guided model — is a known trap, and understanding
why tells me *which* second-order construction is safe. The object my solver integrates is not the plain
$\epsilon_\theta$ but the CFG++-guided prediction, which carries a scale $s$ multiplying a difference of
model outputs. A high-order solver works by Taylor-expanding that integrand and matching its first (or
second) derivative along the trajectory; the expansion is only trustworthy inside a convergence radius set
by how large those derivatives are. But $s$ multiplies the integrand *and its derivatives together*, so
at a large guidance scale the derivatives are amplified by roughly $s$ and the convergence radius shrinks
by the same factor. At a fixed step size the guided integrand has then wandered outside the region where a
second- or third-order expansion is accurate, and the high-order terms — which are supposed to *reduce*
error — inject garbage instead; the higher the order, the more amplified the term, so a third-order solver
can be *worse* than a second, which can be worse than first-order DDIM. That is exactly backwards from the
unconditional intuition, and it is why DDIM survived the large-$s$ regime at all: with nothing to
extrapolate, there is nothing for $s$ to blow up. The substrate fixes `cfg_guidance = 7.5`, so the guided
prediction's $\lambda$-derivatives are amplified by roughly that factor — enough that the
convergence-radius worry is real, not theoretical. So the honest posture is not "bolt on as much order as
fits" but to take *one* order of curvature correction and stop; it is not obvious going higher would even
help at $s=7.5$, and I have a mechanism saying it might hurt.

Now set up what "second order" means for this update, because there is a budget trap waiting inside it.
The diffusion ODE, written on the clean-image (data) prediction, is semi-linear: the exact step from $s$
to $t$ carries a linear factor exactly and only ever approximates an integral of $z_\theta(\lambda)$ in
the half-log-SNR variable $\lambda$. A first-order solver freezes $z_\theta$ at the left endpoint — that
is DDIM, the $k=1$ member, exactly the line I just ran. So the second-order solver is not a disconnected
new object; it is the curvature correction bolted onto the exact robust first-order step I already trust,
which matters given the guidance argument — I am extending DDIM, the one member that survived large $s$,
not replacing it. To get $k=2$ I need an estimate of the *first derivative* of $z_\theta$ along the
trajectory, and a derivative needs a second sample. There are two ways to get it. The multistep way
reuses the $z_\theta$ I computed at the
previous step for free — no extra call — but it depends on history and its local error constant is set
by the spacing of past steps. The singlestep way takes a fresh evaluation at an intermediate point
strictly inside the current interval and uses the finite difference of the two clean-image predictions
as the derivative; it costs an extra call per step but stands alone, so each step's error constant is
governed entirely by within-step geometry. I build the singlestep version here: I want the cleanest, most
history-independent second-order step, and the substrate already hands me exactly the two ingredients a
singlestep Heun step needs — a clean-image estimate at the current level and the machinery to take a
provisional step and re-evaluate.

Here is where I have to be honest about the budget, because this is the constraint DDIM did not feel.
DDIM spent one `predict_noise` call per step and used all twenty. A singlestep second-order step makes
*two* model calls — one at the current timestep, one at the provisional endpoint — so if I kept twenty
steps I would blow the NFE budget to forty. I cannot. The only way a two-call-per-step method fits in
twenty calls is to **halve the number of steps**: walk every *other* timestep on the harness's grid,
ten boundaries, two calls each, twenty NFE. So the trade is explicit and it is not free — I go from
twenty first-order chords to ten second-order ones. Each step now spans a *double* interval (the stride
is `2 * self.skip`), so the per-step $h$ is twice as large, and a first-order solver on this coarse grid
would be much worse than the twenty-step DDIM I just measured. The bet is that second-order accuracy
over the double interval beats first-order accuracy over the single interval at the same total call
count — that the curvature correction more than pays back the coarser grid. That is exactly the
question the FID will answer, and it is a real question; it is not obvious a priori that ten
second-order steps beat twenty first-order ones, because halving the grid is a genuine cost.

Now derive the actual update, grounded in what this loop gives me rather than the textbook log-SNR
form, because the substrate's Tweedie helper and `self.alpha(t)` make a simpler arithmetic-mean Heun
the natural fill. At the current timestep $t$ I have $\bar\alpha_t$ from `self.alpha(t)` and
$\bar\alpha_{t-2\,\text{skip}}$ for the next (doubled) level. I form the guided prediction
$\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$ and the clean-image estimate by Tweedie,
$z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$. That is the slope at the
*start* of the interval. The DDIM (Euler) move would commit to it: a provisional step to the next
level, $z^{\text{euler}}=\sqrt{\bar\alpha_{t-2s}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-2s}}\,\epsilon_{uc}$,
renoised in the CFG++ style with the unconditional noise. But instead of committing, I treat that as a
*predictor* and evaluate the model there to get the slope at the *endpoint*. So a second model call at
the predicted point $z^{\text{euler}}$, at timestep $t-2s$, gives a second guided prediction
$\tilde\epsilon_2$ and a second clean-image estimate
$z_{0|t}^{(2)}=(z^{\text{euler}}-\sqrt{1-\bar\alpha_{t-2s}}\,\tilde\epsilon_2)/\sqrt{\bar\alpha_{t-2s}}$.
Now I have a clean-image estimate at both ends of the interval, and the corrected step averages them —
classic Heun's method, the trapezoidal rule for the slope:
$\bar z_0=\tfrac12(z_{0|t}+z_{0|t}^{(2)})$. Written as $\bar z_0=z_{0|t}+\tfrac12\,(z_{0|t}^{(2)}-z_{0|t})$
it is the DDIM start estimate plus *half* the finite difference between the two endpoint estimates, and
that finite difference is, to leading order, the interval times the first $\lambda$-derivative of
$z_\theta$ — so the correction is precisely a half-step of the slope, the trapezoidal first-derivative
term DDIM ignored. Then the *actual* step uses this averaged clean estimate,
$z_{t-2s}=\sqrt{\bar\alpha_{t-2s}}\,\bar z_0+\sqrt{1-\bar\alpha_{t-2s}}\,\epsilon_{uc}$, again renoised
with the unconditional noise to stay in the CFG++ convention the substrate fixes.

What the step needs, to beat the chord, is a good estimate of the clean image *averaged across the
interval*, not just its left-endpoint value: the exact data-prediction step weights $z_\theta$ over the
whole $\lambda$-interval, and DDIM approximates that weighted average by the single start value. The
trapezoidal average is the standard $O(h^2)$ quadrature of that integral against DDIM's $O(h)$ endpoint
rule — so the averaging converts an $O(h)$ error in the interval-averaged clean image into $O(h^2)$,
picking up exactly the linear bend of the trajectory the chord ignores, the curvature DDIM was paying for.

This is deliberately simpler than the fully general singlestep solver, which places the intermediate point
at a tunable fraction $r_1$ of the interval in $\lambda$ and weights the two predictions with $\lambda$-space
exponential-integrator coefficients. I take the intermediate point to be the *endpoint itself* ($r_1=1$,
trapezoidal Heun rather than the midpoint variant) and weight the two clean estimates equally with the
plain $\sqrt{\bar\alpha}$ coefficients — the natural fill given what the loop exposes: `self.alpha(t)`
hands me $\bar\alpha$ directly, Tweedie hands me $z_0$ directly, and the arithmetic mean of two endpoint
$z_0$ estimates is exactly the trapezoidal second-order correction in data space. Trapezoidal and midpoint
are both $O(h^2)$; what I lose by not tuning $r_1$ or the $\mathrm{expm1}$ constants is a slightly larger
error *constant*, not the order — an acceptable trade for a ten-step solver, and given the
guidance-amplification argument I would rather have a robust, minimally-tuned step than a finely-tuned one
whose extra sensitivity I would have to defend at large $s$.

One subtlety the doubled grid forces: the last step. With ten boundaries, the final one lands at the
end of the chain and there is no further point to evaluate a slope at — and more practically, taking a
second model call at the very last step would either overrun the budget or have nowhere useful to
correct toward. So the final step degrades gracefully to first order: just the Euler (DDIM) move,
$z=z^{\text{euler}}$, no averaging. Every interior step is second-order Heun; the last is first-order
DDIM. This is the standard "lower-order final" stabilizer, and it costs nothing — at the very end the
trajectory is nearly straight (low noise, the clean estimate barely moves), so a first-order step there
is almost lossless.

The budget arithmetic is exact and it is the whole reason the method has this shape: nine interior
second-order steps at two calls each ($18$) plus the first-order last step ($1$) is $19$ NFE, comfortably
inside twenty, where the unhalved $20\times2=40$ would be disqualified on the spot. And the halving buys a
correctness point for free: the intermediate re-evaluation is at timestep $t-2s$, an *actual* level on the
original thousand-step grid, so `predict_noise` is queried at a valid, trained noise level, not an
interpolated one.

Reading the DDIM numbers, the second-order correction should pull FID down on all three variants, since
the curvature I am now capturing is exactly the error DDIM ignored, with the biggest absolute improvement
on **SDXL**, where DDIM was worst (51.52) and there is the most room to close. The real risk is the halved
grid: if the doubled step is large enough that $O(h^2)$ over ten coarse steps does not beat $O(h)$ over
twenty fine ones, FID could fail to improve where it bites most. If SDXL in particular does not move, the
next move is already written: get the high-order correction *without* paying for it in steps — buy the
derivative from calls I am already spending, so the grid need not halve.
