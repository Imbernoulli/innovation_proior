The cosine run told me, in numbers, exactly which half of my bet was wrong. Direction-only matching
held up at the small and base widths — 0.5613 and 0.5475 detection AP, a real signal, nothing
catastrophic — but at the large width it fell to 0.4664, a drop of roughly nine AP points off the
small model and the only size where the loss visibly underperforms. That is precisely the place I said
to watch: the large model has the most channels for magnitude information to live in and the most
predictor capacity to match scale had the loss asked it to, and it is exactly there that cosine leaves
the most value on the table. The pattern across sizes — small ≳ base > large, monotone degradation
*with width* — is the signature of a loss that is blind to something the bigger model has more of. The
thing it is blind to is magnitude. So the diagnosis is sharp and it is not a learning-rate or a
collapse problem: the predictor was graded only on the *direction* of the latent feature vectors and
never pushed to get their *strength* right, and the detection probe — which reads activation magnitude
as part of scoring "is there a digit here" — inherits a representation whose scale was never trained.
The premise that the Variance–Covariance regularizer already owns magnitude was the error: it bounds
each feature dimension's spread away from zero, it does not pin the *scale agreement* between the
predicted and target maps, and that residual gap is what the large model can fill but the cosine loss
forbids it from filling.

So the move is forced: put magnitude back into the loss. I want a cost that is small only when the
predicted feature vector matches the target in *both* direction and length, at every spatial-temporal
location. The most direct such cost is the squared Euclidean distance between the two channel vectors —
sum the squared per-coordinate residuals across channels, which is `‖predicted − state‖²`, and that
quantity is zero only when the two vectors coincide exactly, magnitude included. Where cosine factored
out the norms with its `1/‖v‖` normalization, squared distance keeps them: a prediction pointing the
right way but half as long now carries real cost, and the gradient pulls the prediction toward the
target along the full residual vector, not just its perpendicular component. That is the property the
large-model failure says I need.

Let me make sure squared distance is the *right* magnitude-sensitive cost and not just *a* one, because
there are alternatives and I want to know why I am choosing this. The residual at each coordinate is
`r = state − predicted`, and I need a per-coordinate function `ρ(r)` of it, summed or averaged over the
map. Three things I want from `ρ`. It should be blind to the sign of `r` — overshooting a target
coordinate by `δ` should cost the same as undershooting by `δ`, since the encoder's latent has no
privileged direction of error — so `ρ` should be even. It should be smooth through `r = 0`, so the
gradient eases to zero as the prediction lands on the target rather than rattling at a constant
magnitude near the optimum; this is exactly the quadratic-basin behavior cosine *lacked* near its
minimum, where the orthogonal-projection gradient stayed finite. And it should grow with `|r|` so that
making the loss small genuinely means making the residual small. The simplest even, smooth,
increasing-in-`|r|` function is `r²` itself: derivative `2r`, a linear pull toward the target that
vanishes smoothly at zero and a flat quadratic bowl at the optimum. So `ρ(r) = r²` summed over the map
is the squared Euclidean distance, and it has the smooth quadratic basin I want and the full-residual
gradient that trains magnitude.

There is a probabilistic reading that tells me squared error is not merely convenient but the *right*
default, and it matters here because it says what error model I am implicitly assuming. If I posit that
the residual coordinates are independent zero-mean noise with a common scale and ask for the
maximum-likelihood match, the answer depends on the noise density: for Gaussian residuals the MLE is
the one that minimizes the sum of squared residuals — squared error *is* the Gaussian maximum-likelihood
fit, and it reduces, in the one-coordinate case, to the plain mean of the targets. So choosing squared
distance is choosing to treat the prediction error as roughly Gaussian, which is the natural default
when I have no specific reason to expect heavy tails. (That "no heavy tails" assumption is itself a bet
— the next thing I would test if squared error has its own failure mode is whether a few large
residuals from the autoregressive roll-out are dominating the gradient — but for now the Gaussian
default is the principled place to land after cosine's magnitude-blindness, and it is the loss the
VICReg invariance term defaults to anyway.) The point is that squared error is not a third arbitrary
option next to cosine; it is the canonical magnitude-sensitive feature-matching loss, the one the JEPA
lineage starts from, and the obvious correction to a loss that discarded magnitude.

Now the reduction — sum or mean over the `[B, C, T, H, W]` coordinates — and here cosine's behavior
across the three widths is directly instructive. If I *sum* the squared residuals, the loss scales with
the total number of coordinates `B·C·T·H·W`, which means it grows with the channel count `C` and the
spatial resolution, both of which change across the small/base/large configurations. A summed loss
would hand the large model a numerically larger prediction term than the small model purely because it
has more channels, coupling the effective learning rate to model width — exactly the kind of
size-dependent behavior I am trying to *remove*, since the same code must serve all three sizes and be
judged on each. Averaging instead of summing makes the prediction term a *per-coordinate* quantity of
order `ρ(typical residual)`, comparable across widths and stable against batch size and resolution. So
mean reduction, and every coordinate weighted equally — which is precisely the equal-precision,
no-special-channels assumption I made when I chose `ρ` even and uniform. This is the structural fix for
the across-width sensitivity I read off cosine: a per-coordinate mean keeps the loss scale-matched to
the model, so the large width is graded on the same footing as the small one rather than being handed a
differently-scaled objective.

I should also state plainly why I expect this to beat cosine rather than just differ from it. Cosine
and squared error agree on *direction* — both are minimized when the prediction points at the target —
but squared error additionally pins *magnitude*, and the cosine failure was specifically a magnitude
failure that grew with width. Squared error trains the predictor to match the strength of the target
latents, so the detection probe inherits a representation whose activation magnitudes are correct, not
just whose directions are. And the gradient structure flips cosine's weakness: where cosine's
directional pull *weakened* for short feature vectors (the `1/‖p‖` prefactor), squared error's gradient
`2(predicted − state)` is the full residual regardless of vector length, so it trains hard even when
features start small — it grows magnitude where cosine could not. The smooth quadratic basin also means
the predictor settles cleanly onto the target instead of rattling near a kinked minimum. Every place
cosine left value uncaptured, squared error captures it, and it does so without any new
hyperparameter — just the elementwise squared residual, mean-reduced. The full scaffold module is in
the answer.

What I expect against cosine's numbers, falsifiably. The large width is the decisive test: cosine's
0.4664 there is the magnitude-blindness showing up most, so if my diagnosis is right, squared error
should *recover* the large model first and most — I expect it to climb back toward (and likely past) the
small/base range rather than sagging below it, closing or reversing the monotone-with-width decline.
Small and base should hold at least as high as cosine's 0.5613 / 0.5475 and probably higher, since
magnitude is information the probe can use at every width; but the headline number to fix is the large
model — a squared-error loss that does *not* lift the large width above 0.4664 would falsify the claim
that cosine's failure was magnitude, and would point me instead at the roll-out's residual
distribution. The chain in one breath: cosine held at small/base but collapsed at large because it was
blind to magnitude the bigger model has more of → put magnitude back with the squared Euclidean
distance, the even-smooth-growing per-coordinate cost whose Gaussian-MLE reading makes it the
principled default and whose full-residual gradient trains scale where cosine's perpendicular,
`1/‖p‖`-damped gradient could not → mean-reduce over `[B,C,T,H,W]` so the loss is per-coordinate and
scale-matched across the three widths, removing the across-width sensitivity cosine showed →
expecting the large model to recover above 0.4664 and the whole size profile to flatten, with the
residual-tail behavior of squared error itself as the next thing to interrogate if a few large
roll-out residuals turn out to dominate.
