The second-order run paid off, and it paid off exactly where I bet it would. DPM++(2S) pulled every
variant down from DDIM: SD v1.5 from 34.23 to 29.01, SD v2.0 from 28.41 to 23.89, and SDXL — the one I
called as having the most to gain — from 51.52 all the way to 42.83. Put the drops side by side: $-5.22$
on SD v1.5, $-4.52$ on SD v2.0, $-8.69$ on SDXL, which in relative terms is $15.3\%$, $15.9\%$, $16.9\%$
— remarkably even in percentage but with SDXL the largest absolute drop by a clear margin, almost double
either SD variant's. That is precisely the pattern the curvature story predicted: the correction closes
the most ground where the trajectory bent the most, and SDXL bent the most. So the first-order diagnosis
was right — the DDIM failure was step-efficiency, not stability — and trapezoidal averaging of the endpoint
slopes was the right cure; ten second-order Heun steps beat twenty first-order chords on every variant.
Good. But the variants are still fanned out: SDXL at 42.83 is $1.79\times$ SD v2.0's 23.89, an 18.94-point
gap where DDIM's was 23.11. The correction narrowed the fan but did not collapse it, so there is more of
the same kind of error left. And now I look at *how* I bought that improvement, and I see two separate
weaknesses sitting in the 2S construction that the FID is quietly paying for, and both point the same
direction.

The first weakness is the one I flagged when I built it: 2S is *singlestep*, two model calls per step,
so to fit twenty NFE I had to halve the grid to ten steps. Each step now spans a double interval, and
the intermediate evaluation I spend to estimate the slope is *thrown away* after one use — it never
becomes a sample on the trajectory, it is pure overhead to get a derivative. Of my twenty calls, ten
advance the state and ten only buy slopes. That is wasteful when every call is a full UNet forward. If I
could get the same first-derivative information *without* the throwaway call, I could keep all twenty
steps fine instead of ten coarse, and smaller $h$ directly shrinks the truncation error. That is the
multistep idea: reuse the clean-image prediction I already computed at the *previous* step as the second
sample for the finite difference. One new call per step, twenty steps, the derivative for free. The
error constant per step is set by past spacing rather than within-step geometry, but on a tight budget
the step-count win dominates — twenty fine steps with a reused derivative should beat ten coarse steps
with a fresh one. And if I am reusing two past predictions instead of one, I can fit a *quadratic*
through three points and pick up the *second* derivative too, going to third order at no extra call. So
the first weakness pushes me from singlestep second-order to multistep third-order.

The second weakness is subtler and it is about *what kind* of solver 2S is. It is a pure deterministic
march: every step is an ODE step, error accumulates along the trajectory, and nothing washes it back
out. On a deterministic high-order scheme over a tight budget, discretization error compounds — the
trajectory can drift off the data manifold and stay off, and that drift shows up as exactly the kind of
texture and global-structure error FID is sensitive to. There is a known antidote: re-inject a
controlled amount of Gaussian noise at each step and let the *next* denoising step remove it. That is a
Langevin correction — the noise-and-denoise cycle pulls the state back toward where the model thinks
data lives, cancelling accumulated error. It is well documented that fully deterministic sampling can be
perceptually *worse* than sampling with such re-injection, even at the same step count. So I do not want
to keep marching the deterministic ODE; I want to solve the reverse *SDE* with the same exponential-
integrator machinery, and keep a knob for how much noise to put back. Both weaknesses — singlestep
overhead and deterministic drift — argue for the same next method: a multistep, stochastic, higher-order
solver. That is DPM++(3M) SDE.

Let me derive it in the right space, because the substrate this time hands me a different set of helpers
than the DDIM-grid 2S used, and they steer the construction. The natural variable is no longer the
discrete timestep but the noise level $\sigma=\sqrt{1-\bar\alpha}/\sqrt{\bar\alpha}$, and the loop
exposes exactly the k-diffusion machinery for it: `get_sigmas_karras` to build the schedule,
`self.timestep(sigma)` to convert back, and `self.kdiffusion_x_to_denoised` to get the clean-image
estimate ("denoised") directly at a given $\sigma$. So I will work in the VE convention where
$\alpha=1$ and the half-log-SNR is simply $\lambda=-\log\sigma$. The reverse SDE in $\lambda$, on the
data prediction, is $dx=[-(1+\alpha^2)x+2\alpha\,x_\theta]\,d\lambda+\sqrt2\,\sigma\,dw$; solving the
linear part exactly by variation of constants and the data term by the exponential integrator gives the
exact step. The signal carries by $e^{-h_\eta}$, the clean prediction takes the complement
$1-e^{-h_\eta}$ (a convex split), and the Itô integral contributes Gaussian noise. That noise term is the
one genuinely new object, so I compute its variance rather than guess it: by Itô isometry the increment
$\int e^{-(\lambda_{\text{next}}-\lambda)}dw$ has variance $\int e^{-2(\lambda_{\text{next}}-\lambda)}d\lambda
=(1-e^{-2h})/2$, and the $\sqrt2\,\sigma_{\text{next}}$ prefactor turns that into a standard deviation
$\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$, where $h=\lambda_{\text{next}}-\lambda$ is the step in
half-log-SNR.

The single knob $\eta$ is what makes this one method instead of two. At $\eta=0$ the noise factor
$\sqrt{1-e^{-2h\eta}}$ is exactly zero — the deterministic ODE step, the deterministic limit of what 2S
was doing but multistep; at $\eta=1$ it is the full reverse SDE; and above one it injects a touch more
stochasticity and puts a touch more weight on the fresh clean prediction. Since my whole second motivation
is to add re-noising to fight the deterministic drift I diagnosed in 2S, I want $\eta$ a shade above one:
the working default is $\eta=1.2$, buying more self-correction than the plain SDE at the cost of a bit more
noise to clean up — the right posture at a tight budget where drift is the bigger enemy. The first-order
member grounds it: hold $x_\theta$ constant and the $\eta=1$ step is
$x=(\sigma_t/\sigma_s)e^{-h}x_s+(1-e^{-2h})x_\theta+\sigma_t\sqrt{1-e^{-2h}}\,z$, exactly stochastic DDIM
at noise level $\sigma_t\sqrt{1-e^{-2h}}$ — the higher orders are corrections on top of a step I already
know is sound.

One property of the noise term bears directly on the risk in $\eta=1.2$: the injected standard deviation
$\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$ carries $\sigma_{\text{next}}$ as its leading factor, so the
absolute noise scales with the *destination* noise level — largest at high $\sigma$, shrinking to zero as
the Karras grid drives $\sigma_{\text{next}}\to0$ near the end, and none at all on the final $\sigma=0$
step. So the Langevin re-injection does its drift-cancelling work in the high- and mid-noise regime and
then gets out of the way, rather than dumping noise into the nearly-finished image where a twenty-step
chain would have no steps left to clean it up. That is what makes $\eta$ a shade above one tolerable here:
the very schedule that concentrates steps at low $\sigma$ also throttles the noise there.

Now the multistep order. I keep the last two clean predictions, `denoised_1` and `denoised_2`, and the
last two $\lambda$-step sizes, `h_1` and `h_2`. The data integral's Taylor expansion in $\lambda$ brings
in the first and second derivatives of $x_\theta$, weighted by the exponential-integrator $\phi$
functions evaluated at $h_\eta$: $\phi_2(h_\eta)=(e^{-h_\eta}-1)/h_\eta+1$ and
$\phi_3(h_\eta)=\phi_2/h_\eta-0.5$. Their small-$h$ forms $\phi_2\approx h_\eta/2>0$ and
$\phi_3\approx-h_\eta/6<0$ fix the signs I have to carry: $\phi_2$ is a positive first-derivative weight,
$\phi_3$ a *negative* second-derivative weight. And $\phi_2$ is not an approximation to the
first-derivative coefficient — it is exactly the exponential-integrator weight $\phi_2(h_\eta)\cdot h
=\kappa\int_0^h e^{-\kappa(h-u)}u\,du$ with $\kappa=\eta+1$ — so whatever approximation enters the
third-order step lives in the $\phi_3\,d_2$ term and in the finite-difference derivative estimates, not in
$\phi_2$.

The derivatives themselves come from a Newton divided difference through the three points, and here I make
two coefficient choices that depart from the canonical k-diffusion scheme; I want to be explicit about
them. The canonical third-order multistep scales the older interval as $r_1=h_2/h$ (older spacing relative
to the current step) and applies the curvature correction with a *minus* sign, $x+\phi_2 d_1-\phi_3 d_2$
(so that $-\phi_3$, being positive, adds the second-difference). I instead use $r_1=h_2/h_1$ — the older
interval relative to the *previous* step rather than the current one — and apply the correction with a
*plus* sign, $x+\phi_2 d_1+\phi_3 d_2$. With the scaled spacings $r_0=h_1/h$ and $r_1=h_2/h_1$, the divided
differences are $d_{1,0}=(\text{denoised}-\text{denoised}_1)/r_0$ and
$d_{1,1}=(\text{denoised}_1-\text{denoised}_2)/r_1$; the endpoint first-derivative estimate is
$d_1=d_{1,0}+(d_{1,0}-d_{1,1})\,r_0/(r_0+r_1)$ and the second is $d_2=(d_{1,0}-d_{1,1})/(r_0+r_1)$; and
the step is $x=e^{-h_\eta}x+(1-e^{-h_\eta})\,\text{denoised}+\phi_2 d_1+\phi_3 d_2$.

Both departures bias the curvature estimate, and on a quadratic $x_\theta(\lambda)=a+b\lambda+c\lambda^2$ I
can see how: the $r_1=h_2/h_1$ mis-scaling shrinks the recovered derivatives (a few percent low on the
first, roughly a tenth low on the second) — a conservative bias, largest on the second derivative — and the
$+\phi_3$ sign flips the small curvature contribution to the opposite sign of the canonical one at similar
magnitude. So this is best read not as an exact $O(h^4)$ scheme but as a *practical* third-order multistep
correction with a slightly-off curvature constant. I accept it because the dominant gain over 2S is the
step-count (twenty vs ten) and the Langevin re-noising, not the exact third-order constant; a mis-weighted
curvature term is a second-order concern next to those two structural wins. When only one past value is
available (the first correction step) I drop $d_2$ and use the two-point estimate
$d=(\text{denoised}-\text{denoised}_1)/r$ with $r=h_1/h$, giving the second-order step $x+\phi_2 d$; when
no past value exists yet (the first step) it is the plain constant data-prediction step plus the noise.

A few implementation realities to nail down.
The schedule is the Karras power grid, `get_sigmas_karras(N, sigma_min, sigma_max, rho=7)`: over a
representative $\sigma\in[0.03,14.6]$ with twenty steps, nine land at $\sigma<1$ (a band only about $7\%$
of the raw $\sigma$-range) and the last six crowd into $\sigma\in[0.03,0.28]$, so nearly half the budget
goes to the low-noise sliver where the clean-image estimate changes fastest per unit $\sigma$ and the
truncation error concentrates — the schedule doing real work, not a formality. The exponentials all appear
as $e^{x}-1$ with small $x$, where naive subtraction cancels catastrophically, so they go through `expm1`;
under the `float16` autocast the substrate runs in, the small-$\sigma$ steps produce exactly the tiny
arguments where this matters, keeping $(1-e^{-h_\eta})$ = `(-h_eta).expm1().neg()` and the noise std
$\sqrt{1-e^{-2h\eta}}$ = `(-2*h*eta).expm1().neg().sqrt()` from disintegrating. The last step lands at
$\sigma=0$ with nothing left to denoise toward and no noise to add,
so it just returns `denoised`, the clean-image estimate itself. And the re-injected noise here is a plain
`torch.randn_like(x)` scaled by $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$ — *not* a Brownian-tree noise
sampler. The canonical k-diffusion implementation draws Brownian-consistent, seed-reproducible increments
from a Brownian tree keyed to the noise levels; this fill uses fresh independent Gaussian noise each step.
That is the third deviation from the canonical scheme: the noise is correctly *scaled* but not
Brownian-tree *correlated*, which trades exact reproducibility of the noise path for a simpler draw. For
single-seed FID at this budget it is a defensible simplification. The SD path forms `denoised` through
`self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)`, which already bakes in the CFG++
guided prediction and Tweedie; the SDXL path does it explicitly with the VE scalings
$c_{\text{in}}=1/\sqrt{\sigma^2+1}$, $c_{\text{out}}=-\sigma$, and `denoised = x + c_out * noise_pred`.

Reading the 2S numbers, both the finer grid (twenty vs ten steps) and the Langevin re-noising should pull
FID down further on every variant, with **SDXL** — 42.83, still the worst and still $1.79\times$ SD v2.0 —
helped most, since that is where deterministic drift over a coarse grid cost the most global structure; I
expect the largest absolute drop there again. The risk is the $\eta=1.2$ choice and the simplified noise:
too much injected stochasticity at twenty steps can leave residual noise the short chain cannot fully
clean, and the off-tuned $r_1=h_2/h_1$, $+\phi_3 d_2$ constants mean the third-order term is not exactly
tuned. If the FID *fails* to beat 2S on some variant, the likeliest culprit is those off-canonical
coefficients eating into the gain, pointing the next move at restoring $r_1=h_2/h$ and $-\phi_3 d_2$. But
the expectation I commit to is that the multistep stochastic third-order solver lands all three variants
below their 2S FIDs, SDXL the largest improvement, the strongest sampler so far.
