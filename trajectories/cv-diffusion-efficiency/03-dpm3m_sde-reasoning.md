The second-order run paid off, and it paid off exactly where I bet it would. DPM++(2S) pulled every
variant down from DDIM: SD v1.5 from 34.23 to 29.01, SD v2.0 from 28.41 to 23.89, and SDXL — the one I
called as having the most to gain — from 51.52 all the way to 42.83. Put the drops side by side: $-5.22$
on SD v1.5, $-4.52$ on SD v2.0, $-8.69$ on SDXL, which in relative terms is $15.3\%$, $15.9\%$, $16.9\%$
— remarkably even in percentage but with SDXL the largest absolute drop by a clear margin, almost double
either SD variant's. That is precisely the pattern the curvature story predicted: the correction closes
the most ground where the trajectory bent the most, and SDXL bent the most. So the diagnosis from rung one
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

The single knob $\eta$ is what makes this one method instead of two, and I want to check its endpoints
numerically so I know exactly what each setting does. Set $h_\eta=h(\eta+1)$. Take a representative step
$h=0.5$. At $\eta=0$: $h_\eta=0.5$, contraction $e^{-h_\eta}=0.6065$, data weight $1-e^{-h_\eta}=0.3935$,
and the noise factor $\sqrt{1-e^{-2h\eta}}=0$ exactly — the deterministic ODE step, no noise, exactly the
deterministic limit of what 2S was doing but multistep. At $\eta=1$: $h_\eta=1.0$, contraction $0.3679$,
data weight $0.6321$, noise factor $0.7951$ — the full reverse SDE. At $\eta=1.2$: $h_\eta=1.1$,
contraction $0.3329$, data weight $0.6671$, noise factor $0.8359$ — a touch *more* injected stochasticity
than the bare SDE, and a touch more weight on the fresh clean prediction. Given that my whole second
motivation is to add re-noising to fight the deterministic drift I diagnosed in 2S, I want $\eta$ a shade
above one: the working default here is $\eta=1.2$, buying a bit more self-correction than the plain SDE at
the cost of a bit more noise to clean up, which is the right posture at a tight budget where drift is the
bigger enemy. The endpoints confirm the knob does what the derivation claims — zero noise at $\eta=0$,
climbing monotonically, and the data weight rising with it.

Before I build order on top of this, one sanity hook that the first-order member sits on ground I already
trust: hold $x_\theta$ constant and the $\eta=1$ step is $x=(\sigma_t/\sigma_s)e^{-h}x_s+(1-e^{-2h})x_\theta
+\sigma_t\sqrt{1-e^{-2h}}\,z$. Stochastic DDIM with noise level $\eta_{\text{DDIM}}=\sigma_t\sqrt{1-e^{-2h}}$
is $x=x_\theta+\sqrt{\sigma_t^2-\eta_{\text{DDIM}}^2}\,(x_s-x_\theta)/\sigma_s+\eta_{\text{DDIM}}z$. These
should be the same sampler if my SDE derivation is right, so I check the deterministic parts on numbers
($\sigma_s=1.3$, $\sigma_t=0.8$, $x_s=0.7$, $x_\theta=0.2$): mine gives $0.38934911$, stochastic DDIM gives
$0.38934911$, difference zero — and the noise levels coincide by construction. So the first-order stochastic
member *is* stochastic DDIM, and an independent re-derivation landing back on it is the evidence I want that
the exponential-integrator SDE machinery has not wandered off; the higher orders are corrections on top of a
step I already know is sound.

There is one property of the noise term that I want to note because it directly bears on the risk I am
taking with $\eta=1.2$: the injected standard deviation is $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$, and
the leading factor is $\sigma_{\text{next}}$ itself. So the absolute noise I add scales with the noise level
of the *destination* — largest at high $\sigma$, and shrinking to zero as the Karras grid drives
$\sigma_{\text{next}}\to0$ near the end. Concretely, at $\eta=1.2$ the $\sqrt{1-e^{-2h\eta}}$ factor is
around $0.84$, but at the second-to-last step ($\sigma_{\text{next}}\approx0.28$ on the grid I tabulated)
the injected std is only $\approx0.23$, and at the last stochastic step ($\sigma_{\text{next}}\approx0.05$)
it is $\approx0.04$, and the final step at $\sigma=0$ adds none at all. So the Langevin re-injection does its
drift-cancelling work in the high- and mid-noise regime and then gets out of the way, rather than dumping
noise into the nearly-finished image where a twenty-step chain would have no steps left to clean it up. That
is what makes $\eta$ a shade above one tolerable at this budget: the very schedule that concentrates steps at
low $\sigma$ also throttles the noise there.

Now the multistep order. I keep the last two clean predictions, `denoised_1` and `denoised_2`, and the
last two $\lambda$-step sizes, `h_1` and `h_2`. The data integral's Taylor expansion in $\lambda$ brings
in the first and second derivatives of $x_\theta$, weighted by the exponential-integrator $\phi$
functions evaluated at $h_\eta$: $\phi_2(h_\eta)=(e^{-h_\eta}-1)/h_\eta+1$ (the first-derivative weight)
and $\phi_3(h_\eta)=\phi_2/h_\eta-0.5$ (the second-derivative weight). I want the signs and small-$h$
behaviour of these pinned down, not eyeballed, because the sign of the curvature term depends on them.
Evaluating: at $h_\eta=1.0$, $\phi_2=+0.3679$ and $\phi_3=-0.1321$; at $h_\eta=0.5$, $\phi_2=+0.2131$,
$\phi_3=-0.0739$; at $h_\eta=0.2$, $\phi_2=+0.0937$ (against the small-$h$ Taylor $h_\eta/2=0.1000$),
$\phi_3=-0.0317$ (against $-h_\eta/6=-0.0333$); at $h_\eta=0.05$, $\phi_2=0.02459$ vs $0.02500$ and
$\phi_3=-0.00823$ vs $-0.00833$. So $\phi_2\approx h_\eta/2>0$ and $\phi_3\approx-h_\eta/6<0$ across the
range, the Taylor forms tightening as $h_\eta$ shrinks — $\phi_2$ is a positive first-derivative weight
and $\phi_3$ is a *negative* second-derivative weight, which I have to carry through the sign of its term.
And $\phi_2$ is not a convenient approximation to the first-derivative coefficient — it *is* it. The exact
weight on the first derivative is $\kappa\int_0^h e^{-\kappa(h-u)}u\,du$ with $\kappa=\eta+1$ and
$h_\eta=\kappa h$; integrating by parts gives $h-(1-e^{-h_\eta})/\kappa$, and $\phi_2(h_\eta)\cdot h$ equals
exactly that. Numerically at $h=0.5$, $\eta=1.2$: $\phi_2\cdot h=0.19675958$ and the exact integral
$=0.19675958$, difference zero. So the first-derivative term is carried *exactly*; whatever approximation
enters the third-order step lives entirely in the second-derivative ($\phi_3\,d_2$) term and in the
finite-difference estimates of the derivatives, not in $\phi_2$.

The derivatives themselves come from a Newton divided difference through the three points. And here I have
to be careful, because the literal fill this harness records makes two specific coefficient choices that
are *not* the canonical k-diffusion ones, and the trajectory's job is to land *this* implementation, not
the textbook. The canonical third-order multistep scales the older interval as $r_1=h_2/h$ (older spacing
relative to the current step) and applies the curvature correction with a *minus* sign, $x+\phi_2 d_1-\phi_3 d_2$
(so that $-\phi_3$, being positive, adds the second-difference). The fill recorded as the baseline anchor
here instead uses $r_1=h_2/h_1$ — the older interval relative to the *previous* step rather than the
current one — and applies the correction with a *plus* sign, $x+\phi_2 d_1+\phi_3 d_2$. With the scaled
spacings $r_0=h_1/h$ and $r_1=h_2/h_1$, the divided differences are $d_{1,0}=(\text{denoised}-\text{denoised}_1)/r_0$
and $d_{1,1}=(\text{denoised}_1-\text{denoised}_2)/r_1$; the endpoint first-derivative estimate is
$d_1=d_{1,0}+(d_{1,0}-d_{1,1})\,r_0/(r_0+r_1)$ and the second is $d_2=(d_{1,0}-d_{1,1})/(r_0+r_1)$; and
the step is $x=e^{-h_\eta}x+(1-e^{-h_\eta})\,\text{denoised}+\phi_2 d_1+\phi_3 d_2$.

I do not want to hand-wave "slightly off-tuned"; I want to know exactly how far off these two deviations
put the curvature estimate, so I run both schemes on a quadratic where I know the true derivatives. Take
$x_\theta(\lambda)=a+b\lambda+c\lambda^2$ measured from the current endpoint, with $h=0.5$, $h_1=0.4$,
$h_2=0.6$, and $b=0.7$, $c=0.5$; the true scaled derivatives the estimates should recover are $h\,D_1=hb=0.35000$
and $(h^2/2)D_2=h^2 c=0.12500$. The *canonical* scaling ($r_1=h_2/h$) recovers them exactly, and it is
worth seeing why on paper: with the past points at $L=-h_1$ and $L=-(h_1+h_2)$, the near difference is
$d_{1,0}=h(b-c\,h_1)$ and the far one $d_{1,1}=h(b-c(2h_1+h_2))$, so $d_{1,0}-d_{1,1}=h\,c(h_1+h_2)$; with
$r_0+r_1=(h_1+h_2)/h$ this gives $d_2=(d_{1,0}-d_{1,1})/(r_0+r_1)=c\,h^2$ exactly, and the slope
extrapolation $d_1=d_{1,0}+(d_{1,0}-d_{1,1})r_0/(r_0+r_1)=h(b-c h_1)+h c h_1=hb$, the backward-difference
bias $-c h_1$ cancelling against the extrapolation term. Numerically $d_1=0.35000$, $d_2=0.12500$. That
clean cancellation is exactly what the $r_1=h_2/h_1$ mis-scaling perturbs. The *off-canonical* fill ($r_1=h_2/h_1$) gives $d_1=0.33696$ (about $4\%$ low) and
$d_2=0.10870$ (about $13\%$ low). So the $r_1$ mis-scaling does not destroy the estimate — it shrinks it, a
conservative bias, largest on the second derivative. The sign flip is the sharper deviation: with $\eta=1.2$
so $h_\eta=1.1$, $\phi_3=-0.14226$, the canonical curvature contribution is $-\phi_3 d_2^{\text{can}}=+0.017782$
while the fill's is $+\phi_3 d_2^{\text{off}}=-0.015463$ — opposite sign. Where the canonical scheme *adds*
a curvature contribution, the fill *subtracts* one of similar magnitude. So this is best read not as an
exact $O(h^4)$ scheme but as a *practical* third-order multistep correction with a slightly-off, and
slightly-conservative-to-inverted, curvature constant. That is a real caveat, now quantified rather than
asserted, and I am choosing to keep it because it is the anchor implementation and because the dominant
gain over 2S is the step-count (twenty vs ten) and the Langevin re-noising, not the exact third-order
constant; a mis-weighted curvature term is a second-order concern next to those two structural wins. When
only one past value is available (the first correction step) I drop $d_2$ and use the two-point estimate
$d=(\text{denoised}-\text{denoised}_1)/r$ with $r=h_1/h$, giving the second-order step $x+\phi_2 d$; when
no past value exists yet (the first step) it is the plain constant data-prediction step plus the noise.

A few implementation realities the harness fixes, each of which I keep because the baseline anchor does,
and one of which I want to justify with a number because it looks like a nicety and is not. The schedule is
the Karras power grid, `get_sigmas_karras(N, sigma_min, sigma_max, rho=7)`, which puts more steps at low
$\sigma$ where the per-step truncation error is largest — the right way to spend a tight budget, and I can
put a number on "more." Building the $\rho=7$ grid over a representative $\sigma\in[0.03,14.6]$ with twenty
steps, nine of the twenty land at $\sigma<1$ — a band that is only about $7\%$ of the raw $\sigma$-range —
and the last six crowd into $\sigma\in[0.03,0.28]$. So nearly half the budget is spent in the low-noise
sliver where the clean-image estimate is changing fastest per unit $\sigma$ and the truncation error
concentrates, rather than uniformly across a range dominated by the high-$\sigma$ region where the
trajectory is nearly straight. That is the schedule doing real work, not a formality. The
exponentials all appear as $e^{x}-1$ with small $x$, where naive subtraction cancels catastrophically, so
they go through `expm1`. This is not cosmetic: in float32, computing $e^x-1$ as `exp(x)-1` for $x=10^{-3}$
already carries relative error $2.3\times10^{-5}$, at $x=10^{-5}$ it is $1.3\times10^{-3}$, and at
$x=10^{-7}$ it is $0.19$ — a fifth of the value gone — whereas `expm1` holds relative error near $3\times10^{-8}$
throughout. Under the `float16` autocast the substrate runs in, the small-$\sigma$ steps of the Karras grid
produce exactly these tiny arguments, so `expm1` is what keeps $(1-e^{-h_\eta})$ = `(-h_eta).expm1().neg()`
and the noise standard deviation $\sqrt{1-e^{-2h\eta}}$ = `(-2*h*eta).expm1().neg().sqrt()` from
disintegrating. The last step lands at $\sigma=0$ with nothing left to denoise toward and no noise to add,
so it just returns `denoised`, the clean-image estimate itself. And the re-injected noise here is a plain
`torch.randn_like(x)` scaled by $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$ — *not* a Brownian-tree noise
sampler. The canonical k-diffusion implementation draws Brownian-consistent, seed-reproducible increments
from a Brownian tree keyed to the noise levels; this fill uses fresh independent Gaussian noise each step.
That is the third deviation from the textbook I am noting: the noise is correctly *scaled* but not
Brownian-tree *correlated*, which trades exact reproducibility/consistency of the noise path for a simpler
draw. For single-seed FID at this budget it is a defensible simplification, and again, it is the
implementation the anchor records, so it is what I land. The SD path forms `denoised` through
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
cost, if with the conservative constant I measured.

Reading the 2S numbers, here is what I expect and where the bet is. Both the finer grid (twenty vs ten
steps) and the Langevin re-noising should pull FID down further on every variant. On **SDXL** — 42.83
under 2S, still the worst variant and still $1.79\times$ SD v2.0 — the stochastic correction should help
most, because that is where deterministic drift over a coarse grid was costing the most global structure;
I expect the largest absolute drop there again, and if both structural wins land I would expect it under
40 for the first time. SD v1.5 (29.01) and SD v2.0 (23.89) should also improve, though more modestly since
they were already lower. The risk I am carrying into this is the $\eta=1.2$ choice and the simplified
noise: too much injected stochasticity at twenty steps can leave residual noise the short chain cannot
fully clean, and the plain `randn_like` (no Brownian-tree correlation) plus the off-tuned $r_1=h_2/h_1$,
$+\phi_3 d_2$ constants — the $13\%$-low, sign-flipped curvature term I measured — mean the third-order
term is not exactly tuned, so the curvature correction could under- or over-shoot. If the FID *fails* to
beat 2S on some variant, the most likely culprit is exactly those off-canonical coefficients eating into
the gain, which would point the next move at restoring $r_1=h_2/h$ and $-\phi_3 d_2$. But the expectation
I commit to is that the multistep stochastic third-order solver lands all three variants below their 2S
FIDs, with SDXL the largest improvement, making this the strongest of the three rungs.
