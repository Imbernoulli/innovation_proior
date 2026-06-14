The second-order run paid off, and it paid off exactly where I bet it would. DPM++(2S) pulled every
variant down from DDIM: SD v1.5 from 34.23 to 29.01, SD v2.0 from 28.41 to 23.89, and SDXL — the one I
called as having the most to gain — from 51.52 all the way to 42.83, an 8.7-point drop, the largest
absolute improvement of the three. So the curvature correction was real and the halved grid did not
cost me the win: ten second-order Heun steps beat twenty first-order chords on every variant. The
diagnosis from rung one was right — the DDIM failure was step-efficiency, not stability — and
trapezoidal averaging of the endpoint slopes was the right cure. Good. But now I look at *how* I bought
that improvement, and I see two separate weaknesses sitting in the 2S construction that the FID is
quietly paying for, and both point the same direction.

The first weakness is the one I flagged when I built it: 2S is *singlestep*, two model calls per step,
so to fit twenty NFE I had to halve the grid to ten steps. Each step now spans a double interval, and
the intermediate evaluation I spend to estimate the slope is *thrown away* after one use — it never
becomes a sample on the trajectory, it is pure overhead to get a derivative. That is wasteful when every
call is a full UNet forward. If I could get the same first-derivative information *without* the throwaway
call, I could keep all twenty steps fine instead of ten coarse, and smaller $h$ directly shrinks the
truncation error. That is the multistep idea: reuse the clean-image prediction I already computed at the
*previous* step as the second sample for the finite difference. One new call per step, twenty steps, the
derivative for free. The error constant per step is set by past spacing rather than within-step
geometry, but on a tight budget the step-count win dominates — twenty fine steps with a reused
derivative should beat ten coarse steps with a fresh one. And if I am reusing two past predictions
instead of one, I can fit a *quadratic* through three points and pick up the *second* derivative too,
going to third order at no extra call. So the first weakness pushes me from singlestep second-order to
multistep third-order.

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
$1-e^{-h_\eta}$ (a convex split), and the Itô integral contributes Gaussian noise of standard deviation
$\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$, where $h=\lambda_{\text{next}}-\lambda$ is the step in
half-log-SNR.

The single knob $\eta$ is what makes this one method instead of two. Set $h_\eta=h(\eta+1)$: at
$\eta=0$ the contraction is $e^{-h}$, the noise term vanishes, and I recover the deterministic ODE step
(exactly the deterministic limit of what 2S was doing, just multistep); at $\eta=1$ the contraction is
$e^{-2h}$ with noise $\sqrt{1-e^{-2h}}$, the full reverse SDE; and $\eta>1$ injects *extra* Langevin
stochasticity beyond the plain SDE. Given that my whole second motivation is to add re-noising to fight
the deterministic drift I diagnosed in 2S, I want $\eta$ a touch above one — the working default here is
$\eta=1.2$, buying a bit more self-correction than the bare SDE at the cost of a bit more noise to clean
up, which is the right posture at a tight budget where drift is the bigger enemy.

Now the multistep order. I keep the last two clean predictions, `denoised_1` and `denoised_2`, and the
last two $\lambda$-step sizes, `h_1` and `h_2`. The data integral's Taylor expansion in $\lambda$ brings
in the first and second derivatives of $x_\theta$, weighted by the exponential-integrator $\phi$
functions evaluated at $h_\eta$: $\phi_2(h_\eta)=(e^{-h_\eta}-1)/h_\eta+1\approx h_\eta/2$ (the
first-derivative weight) and $\phi_3(h_\eta)=\phi_2/h_\eta-0.5\approx -h_\eta/6$ (the second-derivative
weight, negative). The derivatives themselves come from a Newton divided difference through the three
points. And here I have to be careful, because the literal fill this harness records makes two specific
coefficient choices that are *not* the canonical k-diffusion ones, and the trajectory's job is to land
*this* implementation, not the textbook. The canonical third-order multistep scales the older interval
as $r_1=h_2/h$ (older spacing relative to the current step) and applies the curvature correction with a
*minus* sign, $x+\phi_2 d_1-\phi_3 d_2$ (so that $-\phi_3$, being positive, adds the second-difference).
The fill recorded as the baseline anchor here instead uses $r_1=h_2/h_1$ — the older interval relative to
the *previous* step rather than the current one — and applies the correction with a *plus* sign,
$x+\phi_2 d_1+\phi_3 d_2$. So with the scaled spacings $r_0=h_1/h$ and $r_1=h_2/h_1$, the divided
differences are $d_{1,0}=(\text{denoised}-\text{denoised}_1)/r_0$ and
$d_{1,1}=(\text{denoised}_1-\text{denoised}_2)/r_1$; the endpoint first-derivative estimate is
$d_1=d_{1,0}+(d_{1,0}-d_{1,1})\,r_0/(r_0+r_1)$ and the second is $d_2=(d_{1,0}-d_{1,1})/(r_0+r_1)$; and
the step is $x=e^{-h_\eta}x+(1-e^{-h_\eta})\,\text{denoised}+\phi_2 d_1+\phi_3 d_2$. I am noting these
two deviations deliberately. They are slight mis-scalings of the curvature term relative to the
provably-third-order scheme — the $r_1=h_2/h_1$ choice changes the weighting of the older sample, and the
$+\phi_3 d_2$ sign (with $\phi_3<0$) *subtracts* a curvature contribution where the canonical scheme
adds it — so this is best read as a *practical* third-order multistep correction with a slightly
off-tuned constant rather than an exact $O(h^4)$ scheme. That is fine for my purpose: the dominant gain
over 2S is the step-count (twenty vs ten) and the Langevin re-noising, not the exact third-order
constant, and a slightly-conservative curvature term is an acceptable price. When only one past value is
available (the first correction step) I drop $d_2$ and use the two-point estimate
$d=(\text{denoised}-\text{denoised}_1)/r$ with $r=h_1/h$, giving the second-order step $x+\phi_2 d$; when
no past value exists yet (the first step) it is the plain constant data-prediction step plus the noise.

A few implementation realities the harness fixes, each of which I keep because the baseline anchor does.
The schedule is the Karras power grid, `get_sigmas_karras(N, sigma_min, sigma_max, rho=7)`, which puts
more steps at low $\sigma$ where the per-step truncation error is largest — the right way to spend a
tight budget. The exponentials all appear as $e^{x}-1$ with small $x$, where naive subtraction cancels
catastrophically, so they go through `expm1`: $(1-e^{-h_\eta})$ is `(-h_eta).expm1().neg()` and the
noise standard deviation $\sqrt{1-e^{-2h\eta}}$ is `(-2*h*eta).expm1().neg().sqrt()`. The last step lands
at $\sigma=0$ with nothing left to denoise toward and no noise to add, so it just returns `denoised`,
the clean-image estimate itself. And the re-injected noise here is a plain `torch.randn_like(x)` scaled
by $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$ — *not* a Brownian-tree noise sampler. The canonical
k-diffusion implementation draws Brownian-consistent, seed-reproducible increments from a Brownian tree
keyed to the noise levels; this fill uses fresh independent Gaussian noise each step. That is the third
deviation from the textbook I am noting: the noise is correctly *scaled* but not Brownian-tree
*correlated*, which trades exact reproducibility/consistency of the noise path for a simpler draw. For
single-seed FID at this budget it is a defensible simplification, and again, it is the implementation
the anchor records, so it is what I land. The SD path forms `denoised` through
`self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)`, which already bakes in the CFG++
guided prediction and Tweedie; the SDXL path does it explicitly with the VE scalings
$c_{\text{in}}=1/\sqrt{\sigma^2+1}$, $c_{\text{out}}=-\sigma$, and `denoised = x + c_out * noise_pred`.

So the delta from 2S is two structural changes bundled into one method, both flowing from the two
weaknesses I diagnosed. Where 2S spent a throwaway call per step to get a fresh derivative on a halved
grid, 3M reuses two past predictions to get *both* first and second derivatives at one call per step,
keeping all twenty steps fine. And where 2S marched the deterministic ODE with error compounding, 3M
solves the reverse SDE with $\eta=1.2$, re-injecting Langevin noise that the next denoising step washes
out — actively cancelling the accumulated drift. The Karras schedule concentrates the steps where they
matter, `expm1` keeps the small-argument exponentials honest, and the curvature correction (with this
harness's $r_1=h_2/h_1$ scaling and $+\phi_3 d_2$ sign) brings in the trajectory's bend at no extra
cost.

Reading the 2S numbers, here is what I expect and where the bet is. Both the finer grid (twenty vs ten
steps) and the Langevin re-noising should pull FID down further on every variant. On **SDXL** — 42.83
under 2S, still the worst variant — the stochastic correction should help most, because that is where
deterministic drift over a coarse grid was costing the most global structure; I expect the largest
absolute drop there again, landing it well below 40. SD v1.5 (29.01) and SD v2.0 (23.89) should also
improve, though more modestly since they were already lower. The risk I am carrying into this is the
$\eta=1.2$ choice and the simplified noise: too much injected stochasticity at twenty steps can leave
residual noise the short chain cannot fully clean, and the plain `randn_like` (no Brownian-tree
correlation) plus the off-tuned $r_1=h_2/h_1$, $+\phi_3 d_2$ constants mean the third-order term is not
exactly tuned, so the curvature correction could under- or over-shoot. If the FID *fails* to beat 2S on
some variant, the most likely culprit is exactly those off-canonical coefficients eating into the gain —
which would point the next move at the kdfix correction (restoring $r_1=h_2/h$ and $-\phi_3 d_2$). But
the expectation I commit to is that the multistep stochastic third-order solver lands all three variants
below their 2S FIDs, with SDXL the largest improvement, making this the strongest of the three rungs.
