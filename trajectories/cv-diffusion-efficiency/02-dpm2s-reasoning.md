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
across variants) — it is *step-efficiency*. Each DDIM step holds the clean-image estimate $z_{0|t}$
constant across the interval and moves on a straight chord; the true trajectory bends, and I am paying the
full first-order truncation error twenty times over, worst where the bend is sharpest. The fix is not to
ask for more calls — I do not have them — but to extract the *bend* of the trajectory from the calls I
already spend, i.e. to go to second order.

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
extrapolate, there is nothing for $s$ to blow up. So the honest posture here is not "bolt on as much order
as fits" — it is to take *one* order of curvature correction, with the smallest error constant I can
manage, and stop. And the guidance scale here is not incidental: the substrate fixes `cfg_guidance = 7.5`,
so the guided prediction sits about seven-and-a-half times as far from the unconditional as the raw
conditional difference, and its $\lambda$-derivatives are amplified in the same ballpark. That is a large
enough amplification that the convergence-radius worry is real, not theoretical — a finely-tuned high-order
step whose accuracy depends on those derivatives staying small is exactly what such a scale breaks. Second
order, carefully, is the right increment; it is not obvious that going higher would even help at $s=7.5$,
and I have a mechanism saying it might hurt.

Now set up what "second order" means for this update before I commit, because there is a budget trap
waiting inside it. The diffusion ODE, written on the clean-image (data) prediction, is semi-linear: the
exact step from $s$ to $t$ carries a linear factor exactly and only ever approximates an integral of
$z_\theta(\lambda)$ in the half-log-SNR variable $\lambda$. A first-order solver freezes $z_\theta$ at
the left endpoint — that is DDIM, the $k=1$ member, exactly the line I just ran, and I can confirm the
identification rather than assert it. Hold $z_{0|t}$ constant across the step and the data-prediction step
is $z_{t-1}=(\sigma_{t-1}/\sigma_t)\,z_t-\sqrt{\bar\alpha_{t-1}}(e^{-h}-1)\,z_{0|t}$ with
$\sigma=\sqrt{1-\bar\alpha}$; substitute $z_{0|t}=(z_t-\sigma_t\epsilon)/\sqrt{\bar\alpha_t}$ and the
coefficient on $z_t$ collapses to $\sigma_{t-1}/\sigma_t$ and the coefficient on $\epsilon$ to
$\sqrt{\bar\alpha_{t-1}}(1-e^{-h})=\sqrt{\bar\alpha_{t-1}}\cdot\sigma_{t-1}/\sigma_t\cdot(\dots)$ — which
is the DDIM step $z_{t-1}=\sqrt{\bar\alpha_{t-1}}z_{0|t}+\sqrt{1-\bar\alpha_{t-1}}\epsilon$ written the
other way round. So the second-order solver is not a disconnected new object; it is the curvature
correction bolted onto the exact robust first-order step I already trust, which matters given the
guidance argument — I am extending DDIM, the one member that survived large $s$, not replacing it. To get $k=2$ I need an
estimate of the *first derivative* of $z_\theta$ along the trajectory, and a derivative needs a second
sample. There are two ways to get it. The multistep way reuses the $z_\theta$ I computed at the
previous step for free — no extra call — but it depends on history and its local error constant is set
by the spacing of past steps. The singlestep way takes a fresh evaluation at an intermediate point
strictly inside the current interval and uses the finite difference of the two clean-image predictions
as the derivative; it costs an extra call per step but stands alone, so each step's error constant is
governed entirely by the within-step geometry. The harness leaves me the whole `sample` body, so I can
build either. I will build the singlestep version, because at this rung I want the cleanest, most
history-independent second-order step I can write — the error constant depends only on this interval's
geometry, not on how the last few steps happened to be spaced — and because the substrate already hands
me, at each timestep, exactly the two ingredients a singlestep Heun step needs: a clean-image estimate at
the current level and the machinery to take a provisional step and re-evaluate.

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
$\bar z_0=\tfrac12(z_{0|t}+z_{0|t}^{(2)})$. It is worth reading that average as what it is, because it
makes the second-order structure explicit: $\bar z_0=z_{0|t}+\tfrac12\,(z_{0|t}^{(2)}-z_{0|t})$, i.e. the
start estimate plus *half* the finite difference between the two endpoint estimates. That finite
difference $z_{0|t}^{(2)}-z_{0|t}$ is, to leading order, the interval times the first $\lambda$-derivative
of $z_\theta$ along the trajectory — so the correction I am adding to the DDIM start value is precisely a
half-step of the slope, which is the trapezoidal first-derivative term. DDIM used $z_{0|t}$ alone; Heun
uses $z_{0|t}$ plus this slope correction. Then the *actual* step uses this averaged clean estimate,
$z_{t-2s}=\sqrt{\bar\alpha_{t-2s}}\,\bar z_0+\sqrt{1-\bar\alpha_{t-2s}}\,\epsilon_{uc}$, again renoised
with the unconditional noise to stay in the CFG++ convention the substrate fixes.

I should not just assert that averaging the two endpoint estimates "captures the curvature" — that is the
whole claim of the rung, so I want to see it as a number. What the step needs, to beat the chord, is a good
estimate of the clean image *averaged across the interval*, not just its left-endpoint value; the exact
data-prediction step weights $z_\theta$ over the whole $\lambda$-interval, and DDIM approximates that
weighted average by the single start value. Take a deliberately curved clean-image trajectory
$z_0(\lambda)=\sin\lambda+0.3\lambda^2$ over an interval of width $h$ starting at $\lambda_0=0.4$, and
compare two estimates of its true interval mean: the endpoint value $z_0(\lambda_0)$ (what DDIM uses) and
the trapezoidal average $\tfrac12(z_0(\lambda_0)+z_0(\lambda_0+h))$ (what Heun uses). Halving $h$ from
$0.4$ down to $0.025$, the endpoint error is $2.35\times10^{-1},\,1.17\times10^{-1},\,5.84\times10^{-2},
\,2.91\times10^{-2},\,1.45\times10^{-2}$ — a clean rate-$1$ decay, $O(h)$. The trapezoidal-average error is
$5.0\times10^{-4},\,4.0\times10^{-4},\,1.4\times10^{-4},\,3.9\times10^{-5},\,1.0\times10^{-5}$ — climbing
to rate $\approx1.9$, i.e. $O(h^2)$, and already two-to-four orders of magnitude smaller in absolute terms.
So the averaging genuinely picks up the linear bend of the trajectory that the chord ignores; it converts
an $O(h)$ error in the interval-averaged clean image into $O(h^2)$. That is the mechanism, measured, and it
is exactly the curvature DDIM was paying for.

I can also check the order claim from the quadrature side, since the trapezoidal rule is what I am really
invoking. The local error of the trapezoidal rule for an integral over a step $h$ is $-(1/12)h^3 z_0''$
plus higher order. Integrate $e^\lambda$ over $[0,h]$ (exact value $e^h-1$) with a single trapezoid and
read off the residual: at $h=0.4$ it is $6.54\times10^{-3}$ against $(1/12)h^3=5.33\times10^{-3}$ (ratio
$1.23$); at $h=0.2$, $7.38\times10^{-4}$ vs $6.67\times10^{-4}$ ($1.11$); at $h=0.1$, $8.76\times10^{-5}$
vs $8.33\times10^{-5}$ ($1.05$); at $h=0.05$, $1.07\times10^{-5}$ vs $1.04\times10^{-5}$ ($1.03$); at
$h=0.025$, $1.32\times10^{-6}$ vs $1.30\times10^{-6}$ ($1.01$). The ratio marches to $1$ as $h$ halves and
the residual drops by $\approx8\times$ per halving — the $h^3$ signature, with the leading constant really
$1/12$. So the trapezoidal step's local error is $O(h^3)$, hence $O(h^2)$ globally: a genuine second-order
correction over the interval.

I want to be careful that this is the *right* form for this harness, because it is deliberately simpler
than the fully general singlestep solver. The general construction places the intermediate point at a
tunable fraction $r_1$ of the interval in $\lambda$ and weights the two predictions by $1/(2r_1)$ with
exponential-integrator coefficients $e^{-h}-1$. Here I am taking the intermediate point to be the
*endpoint itself* ($r_1=1$ in effect, the trapezoidal Heun rather than the midpoint variant) and
weighting the two clean estimates equally, $\tfrac12$ and $\tfrac12$, with the plain $\sqrt{\bar\alpha}$
coefficients rather than $\lambda$-space $\mathrm{expm1}$ factors. This is the honest fill given what the loop
exposes: `self.alpha(t)` hands me $\bar\alpha$ directly, Tweedie hands me $z_0$ directly, and the
arithmetic mean of two endpoint $z_0$ estimates is exactly the trapezoidal second-order correction in
data space. It is not the log-SNR singlestep with a midpoint and $\mathrm{expm1}$ coefficients — I am
*not* importing that machinery, because the substrate's vocabulary is $(\bar\alpha,z_0,\epsilon_{uc})$,
not $(\lambda, h, \phi)$, and a Heun average in $z_0$ is the natural, correct second-order step in that
vocabulary. The trapezoidal and midpoint rules are both $O(h^2)$ quadratures, as the two order checks
above confirm; what I lose by not tuning $r_1$ or the $\mathrm{expm1}$ constants is a slightly larger error
*constant*, not the order. For a ten-step solver that is an acceptable trade for a clean fill, and given
the guidance-amplification argument I would rather have a robust, minimally-tuned second-order step than a
finely-tuned one whose extra sensitivity I would then have to defend at large $s$.

One subtlety the doubled grid forces: the last step. With ten boundaries, the final one lands at the
end of the chain and there is no further point to evaluate a slope at — and more practically, taking a
second model call at the very last step would either overrun the budget or have nowhere useful to
correct toward. So the final step degrades gracefully to first order: just the Euler (DDIM) move,
$z=z^{\text{euler}}$, no averaging. Every interior step is second-order Heun; the last is first-order
DDIM. This is the standard "lower-order final" stabilizer, and it costs nothing — at the very end the
trajectory is nearly straight (low noise, the clean estimate barely moves), so a first-order step there
is almost lossless.

I should keep the budget arithmetic exact, because it is the whole reason the method has this shape. Ten
boundaries: nine of them are second-order (two calls each $=18$) and the last is first-order (one call).
That is $9\times2+1=19$ NFE, at or just under the twenty ceiling depending on how the harness counts the
final partial step — comfortably inside twenty either way. The point is that halving the grid to `[::2]`
with two calls per interior step is precisely what keeps a second-order singlestep method inside twenty
NFE; the unhalved version would be $20\times2=40$ and disqualified on the spot. There is no cheaper way to
fit a two-call step into this budget than to spend it on half as many, coarser steps. One correctness point
the halving buys for free: the intermediate re-evaluation is at timestep $t-2s$, which is an *actual* level
on the original thousand-step grid (the one the harness's stride lands on), so `predict_noise` is called at
a valid, trained noise level rather than an interpolated one — the Euler predictor $z^{\text{euler}}$ is a
genuine latent at a genuine level, and the model is queried where it was trained to answer.

There is a cost inside this I want to name precisely, because it is the tax the singlestep choice pays and
it is exactly what the FID has to overcome. Each interior step spends two full UNet evaluations but only
*one* of them lands on the trajectory: the intermediate $z^{\text{euler}}$ evaluation exists solely to
supply the slope, and after the finite difference is taken it is discarded — it never becomes a sampled
point. So of my twenty calls, ten are "productive" (they advance the state) and ten are "overhead" (they
buy derivatives that are used once and thrown away). That is a steep tax at a tight budget, and it is the
precise reason the grid had to halve. The bet is that ten productive steps *with* a curvature correction
beat twenty productive steps *without* one — that spending half my calls on slope estimation buys back
more than the coarser grid loses. If it does not, the lever is obvious: find the same derivative without
the throwaway call. But I cannot settle that on the whiteboard; the ten-vs-twenty trade is genuinely
empirical, and it is what I am putting to the FID.

So the delta from DDIM is concrete and it is a single structural change with a cost attached: where the
first rung took twenty straight-line steps, I now take ten steps that each evaluate the model twice — once
to predict, once at the predicted endpoint — and average the two clean-image estimates so the step
follows the *bend* of the trajectory instead of a chord. The renoising stays CFG++ (`noise_uc`), the
clean estimate stays guided, and everything else in the substrate is untouched.

Reading the DDIM numbers, here is what I expect this to fix and where I am unsure. On all three
variants the second-order correction should pull FID down, because the trajectory curvature I am now
capturing — the $O(h)\to O(h^2)$ improvement in the interval-averaged clean image I measured above — is
exactly the error DDIM ignored. The biggest absolute improvement should be on **SDXL**, where DDIM was
worst (51.52) precisely because first-order coarseness costs most when there is the most structure to
resolve — that 23-point gap over SD v2.0 is the most room to close, so if the curvature correction is
doing its job, that is where it has the most to close. And I can make that a sharper, falsifiable
prediction than "SDXL most": if the mechanism really is curvature — error proportional to how much the
trajectory bends, which is what left DDIM worst where there was most structure — then the *absolute drops*
should track the DDIM badness ordering itself, largest where DDIM was largest. DDIM ranked SDXL (51.52) >
SD v1.5 (34.23) > SD v2.0 (28.41), so I would predict the FID drops to fall in that same order: SDXL
biggest, SD v1.5 next, SD v2.0 smallest. If instead the drops came out even across variants, or largest on
an easy variant, that would say the improvement is not curvature-driven and I have the mechanism wrong. SD
v1.5 and SD v2.0 should both come down, then, but by less in absolute terms than SDXL. The risk, and the reason this is a real bet rather than a
sure thing, is the halved grid: I have traded twenty first-order steps for ten second-order ones, and if
the doubled step size $h$ is large enough that the $O(h^2)$ error over ten coarse steps does not beat the
$O(h)$ error over twenty fine ones, FID could fail to improve or even regress on the variant where it
bites most. If that happens — if SDXL in particular does not move, or moves the wrong way — the diagnosis
for the next rung is already written: the cure is to get the high-order correction *without*
paying for it in steps — to buy back the derivative from calls I am already spending rather than from a
throwaway one, so the grid need not halve. But the expectation I am committing to is that ten second-order Heun steps land all
three variants below their DDIM FIDs, with the SDXL improvement the largest of the three.
