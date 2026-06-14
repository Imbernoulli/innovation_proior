The first-order run told me exactly where the budget is being wasted, and it told me in FID. DDIM came
in at 34.23 on SD v1.5, 28.41 on SD v2.0, and 51.52 on SDXL — usable images on every variant (it never
diverged, which is the one thing first order reliably buys), but those FIDs are high, and the *shape*
of the gap is the tell. SD v2.0, the easiest variant, sits lowest; SDXL, with the largest latent and
the most structure to resolve in twenty coarse steps, is worst by a wide margin — over 51, nearly
double SD v2.0. That is precisely the signature I expected from a first-order solver: the error scales
with the curvature of the denoiser trajectory between adjacent noise levels, and where there is more
fine structure to resolve, twenty straight-line steps leave the most on the table. So the failure is
not instability and it is not a guidance problem — it is *step-efficiency*. Each DDIM step holds the
clean-image estimate $z_{0|t}$ constant across the interval and moves on a straight chord; the true
trajectory bends, and I am paying the full first-order truncation error twenty times over. The fix is
not to ask for more calls — I do not have them — but to extract the *bend* of the trajectory from the
calls I already spend, i.e. to go to second order.

Let me set up what "second order" even means for this update before I commit, because there is a budget
trap waiting. The diffusion ODE, written on the clean-image (data) prediction, is semi-linear: the
exact step from $s$ to $t$ carries a linear factor exactly and only ever approximates an integral of
$z_\theta(\lambda)$ in the half-log-SNR variable $\lambda$. A first-order solver freezes $z_\theta$ at
the left endpoint — that is DDIM, the $k=1$ member, exactly the line I just ran. To get $k=2$ I need an
estimate of the *first derivative* of $z_\theta$ along the trajectory, and a derivative needs a second
sample. There are two ways to get it. The multistep way reuses the $z_\theta$ I computed at the
previous step for free — no extra call — but it depends on history and its local error constant is set
by the spacing of past steps. The singlestep way takes a fresh evaluation at an intermediate point
strictly inside the current interval and uses the finite difference of the two clean-image predictions
as the derivative; it costs an extra call per step but stands alone, so each step's error constant is
governed entirely by the within-step geometry. The harness leaves me the whole `sample` body, so I can
build either. I will build the singlestep version, because at this rung I want the cleanest, most
history-independent second-order step I can write — and because the substrate already hands me, at each
timestep, exactly the two ingredients a singlestep Heun step needs: a clean-image estimate at the
current level and the machinery to take a provisional step and re-evaluate.

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
$\bar z_0=\tfrac12(z_{0|t}+z_{0|t}^{(2)})$, and then the *actual* step uses this averaged clean estimate,
$z_{t-2s}=\sqrt{\bar\alpha_{t-2s}}\,\bar z_0+\sqrt{1-\bar\alpha_{t-2s}}\,\epsilon_{uc}$, again renoised
with the unconditional noise to stay in the CFG++ convention the substrate fixes.

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
vocabulary. The two agree to leading order in $h$ (the trapezoidal and midpoint rules are both
$O(h^2)$); what I lose by not tuning $r_1$ or the $\mathrm{expm1}$ constants is a slightly larger error
*constant*, not the order. For a ten-step solver that is an acceptable trade for a clean fill.

One subtlety the doubled grid forces: the last step. With ten boundaries, the final one lands at the
end of the chain and there is no further point to evaluate a slope at — and more practically, taking a
second model call at the very last step would either overrun the budget or have nowhere useful to
correct toward. So the final step degrades gracefully to first order: just the Euler (DDIM) move,
$z=z^{\text{euler}}$, no averaging. Every interior step is second-order Heun; the last is first-order
DDIM. This is the standard "lower-order final" stabilizer, and it costs nothing — at the very end the
trajectory is nearly straight (low noise, the clean estimate barely moves), so a first-order step there
is almost lossless.

I should also keep the budget arithmetic exact, because it is the whole reason the method has this
shape. Ten interior+final boundaries: nine of them are second-order (two calls each = eighteen) and the
last is first-order (one call). That is nineteen, not twenty — and depending on how the harness counts
the final partial step it lands at or just under the NFE = 20 ceiling. The point is that halving the
grid to `[::2]` with two calls per interior step is precisely what keeps a second-order singlestep
method inside twenty NFE. If I had not halved, I would be at forty and disqualified.

So the delta from DDIM is concrete and it is a single structural change with a cost attached: where the
first rung took twenty straight-line steps, I now take ten steps that each evaluate the model twice — once
to predict, once at the predicted endpoint — and average the two clean-image estimates so the step
follows the *bend* of the trajectory instead of a chord. The renoising stays CFG++ (`noise_uc`), the
clean estimate stays guided, and everything else in the substrate is untouched.

Reading the DDIM numbers, here is what I expect this to fix and where I am unsure. On all three
variants the second-order correction should pull FID down, because the trajectory curvature I am now
capturing is exactly the error DDIM ignored. The biggest absolute improvement should be on **SDXL**,
where DDIM was worst (51.52) precisely because first-order coarseness costs most when there is the most
structure to resolve — if the curvature correction is doing its job, the largest gap is where it has the
most to close. SD v1.5 (34.23) and SD v2.0 (28.41) should also come down, though by less in absolute
terms. The risk, and the reason this is a real bet rather than a sure thing, is the halved grid: I have
traded twenty first-order steps for ten second-order ones, and if the doubled step size $h$ is large
enough that the $O(h^2)$ error over ten coarse steps does not beat the $O(h)$ error over twenty fine
ones, FID could fail to improve or even regress on the variant where it bites most. If that happens —
if SDXL in particular does not move, or moves the wrong way — the diagnosis for the next rung is already
written: the cure is a method that gets the high-order correction *without* paying for it in steps,
i.e. a multistep solver that reuses past predictions at one call per step, keeping all the fine steps.
But the expectation I am committing to is that ten second-order Heun steps land all three variants below
their DDIM FIDs, with the SDXL improvement the largest of the three.
