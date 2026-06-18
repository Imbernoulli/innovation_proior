Let me start from what the latent world model actually asks of its final activation, because the
choice is not cosmetic — the latent is the only thing the model has. There is no decoder. The encoder
sends an observation to a latent, the dynamics net predicts the next latent from the current latent and
the action, and reward and value heads read off the latent. The losses that define the latent are
self-referential: a self-predictive consistency term pushes the predicted next latent toward the
encoder's own embedding of the next observation, `‖d(z,a) − sg(h(s'))‖²` with a stop-gradient on the
target; a reward regression; and a bootstrapped value loss whose target `q = r + γ Q̄(z', p(z'))` is
itself a function of the very latents the network produces. So the latent is pinned by nothing external,
and that is precisely where instability lives. A squared-error-in-feature-space consistency loss has a
cheap, useless way to go down that has nothing to do with learning dynamics: change the *scale* of the
representation. If the last linear map can inflate or deflate the magnitude of the latent freely, the
losses get a free parameter that moves the landscape around without encoding anything — and because the
value target reads the network's own latents, this closes a feedback loop with a runaway direction. The
exploding gradients people see on harder tasks are that magnitude blowing up. So the final activation's
first job is structural: remove the magnitude degree of freedom, bound the latent.

Why doesn't the LayerNorm already inside the MLP stacks save me? Because LayerNorm normalizes the
*intermediate* pre-activations flowing through each layer — it fixes the per-layer mean and variance of
the river — but the network's *output*, the final latent fed to dynamics, reward, and value, is produced
by a last linear map *after* the last LayerNorm, and that map can have arbitrarily large weights.
LayerNorm controls the river, not the mouth. The thing I actually consume is unbounded. So I need a
constraint on the *final* layer's output, a final activation that defines the geometry of the latent
itself. That is the editable slot the task hands me: `CustomSimNorm.forward`, inserted at the output of
both the encoder and the dynamics MLP. The contract is narrow — take `(*batch, 128)`, return the same
shape — and everything else is frozen, so my whole design space is "what map do I apply to the raw
latent."

The simplest disciplined answer in the "normalize the magnitude" family is to reach for the cheapest
normalization I trust. I have a long-standing intuition about LayerNorm: when I train normalized
networks I get a big per-step improvement, but I pay for it on every forward pass, and a good chunk of
the per-step gain gets handed back in wall-clock. So I have asked before: am I paying for something I do
not need? LayerNorm bundles two distinct operations. One: subtract the mean — recenter the vector to
zero mean. Two: divide by the standard deviation — rescale it to unit spread. They feel like one move,
"standardize," but they are two, and they buy *different* things. Subtracting the mean buys
re-centering invariance: add the same constant to every entry and the centered vector is unchanged.
Dividing by σ buys re-scaling invariance: scale every entry by α and `(a−μ)/σ` is unchanged. Two
operations, two invariances, cleanly separated.

The mean is the suspicious one. Stabilizing training is fundamentally about controlling *spread* —
keeping activations and gradients from blowing up or vanishing. Subtracting the mean recenters the
vector but does nothing to its variance: `var(a−μ) = var(a)`. Recentering moves the cloud; it does not
shrink or grow it. The thing that actually controls spread — that pulls the magnitude to a fixed scale
regardless of how big the weights got — is the *division by a scale statistic*. So if I have to bet
which of the two invariances is doing the stabilizing work, I bet on re-scaling and treat re-centering
as dead weight. That is exactly the degree of freedom I diagnosed as the source of the latent's runaway:
its overall scale. The re-scaling normalization is precisely the operation that pins it. So in *this*
model the part of LayerNorm I do not need is the part I was already suspicious of, and the part I do need
is the part that solves the world-model's actual problem.

If I drop the mean, what do I divide by? I cannot use σ, because `σ = √((1/n)Σ(aᵢ−μ)²)` is *defined*
through μ — it is the spread *around the mean*. If I refuse to compute μ, I need a measure of scale that
never references it. The natural candidate is the spread around *zero* instead of around the mean: the
root-mean-square, `RMS(a) = √((1/n)Σ aᵢ²)`. This is the quadratic mean of the entries — how big they
are on average, measured from the origin — and the normalization becomes `āᵢ = aᵢ / RMS(a) · gᵢ` with a
learned gain `g`. No μ anywhere, no subtraction, one statistic. Check the degenerate case: if the
entries happen to have zero mean, `σ = RMS(a)` and the rule coincides exactly with LayerNorm — so this
is not a wild departure, it is the same layer with re-centering switched off. And check that it does the
one job I need: `RMS(αa) = |α|·RMS(a)`, so `αa / RMS(αa) = a / RMS(a)` — re-scaling invariance, the
load-bearing one, survives. Re-centering invariance is the only thing discarded, and I just argued it is
the weaker suspect. This is the right tool for bounding a self-predictive latent's magnitude with the
least machinery.

Now I have to fit it to *this* task's edit surface, and this is where the implementation deviates from
the textbook RMSNorm in a way I want to be honest about. The default fill is SimNorm, which reshapes the
128-dim latent into 16 groups of `simnorm_dim = 8` and softmaxes within each group. The whole harness —
the consistency loss, the heads — is built around a latent that has this group structure as its native
shape. The cleanest way to slot RMSNorm into the same slot, and the version that is a true
apples-to-apples swap against the default, is to keep the *same partition* and apply RMS normalization
*within each group of 8* rather than over the full 128-vector. So: reshape to `(*batch, 16, 8)`, compute
the root-mean-square over the last axis (the 8 entries of each group), divide, and multiply by a
learnable gain of size 8 shared across groups, then reshape back. This is **group-wise** RMSNorm, not
the canonical full-vector RMSNorm, and the difference is deliberate: it normalizes each of the 16 groups
to a common magnitude independently, mirroring SimNorm's group structure so the comparison isolates
"softmax-on-a-simplex vs. RMS-rescale" rather than confounding it with "16 groups vs. one block." I keep
the gain per group element (size 8) so each coordinate within a group can be re-weighted, the same role
the learned gain plays in LayerNorm.

A subtlety in the numerics: I put the `eps` *inside* the square root, `rms = √(mean(x²) + eps)`, rather
than clamping the norm from below afterward. With groups of only 8 entries the mean-square can get small,
and adding eps under the root keeps the division well-conditioned without ever producing a zero
denominator; it also keeps the gradient of the sqrt finite at the origin. I use `eps = 1e-8`, small
enough not to bias the scale when the group has real magnitude. The gain initializes to ones so the layer
starts as a pure RMS rescale and learns any per-coordinate emphasis from there.

Let me be clear-eyed about what this buys and what it does *not*, because the gap is the whole reason
this is the first rung and not the last. RMSNorm bounds the *spread* of each group — after the divide,
each group has root-mean-square one (up to the gain) — so the magnitude degree of freedom that drives the
runaway loop is controlled, and gradients should stay tame. That is real and it is the point. But
RMSNorm does *nothing else*. It does not induce sparsity: every coordinate is generically nonzero, the
group is a bounded but dense blob. It does not induce any competition between coordinates: each entry is
rescaled by the same scalar, so there is no pressure to prioritize a few directions. And the learnable
gain can partially re-inflate the scale per coordinate, loosening the bound I just imposed. For a latent
that has to support stable bootstrapped value learning, a bounded-but-shapeless code is exactly the weak
representation I worried about in the lineage: the value head reads a dense vector with no structure to
exploit. I expect that to show up not on the easy tasks — where almost any bounded latent saturates the
reward — but on the one task whose dynamics are rich enough that the latent geometry matters.

So here is where I think this lands and what I am watching, stated as falsifiable expectations. On
walker-walk and cartpole-swingup I expect RMSNorm to sit right at the top of the reward range, near
whatever the other normalizations achieve — these tasks do not stress the latent, so bounding the
magnitude is enough and the lack of structure costs nothing. The honest test is cheetah-run, where the
running gait is a richer, more dynamic behavior and the value head has to read fine structure out of the
latent. There I expect group-wise RMSNorm to be the *weakest* of the disciplined normalizations,
precisely because it bounds magnitude without shaping the code — a dense rescaled latent gives the value
head less to work with than a sparse, competitive one would. If that is what the numbers say — saturated
on the two easy tasks, visibly lower on cheetah-run — then the diagnosis for the next rung writes
itself: bounding the magnitude is necessary but not sufficient, and the missing ingredient is
*structure* — a competition between coordinates that biases the latent toward a sparse, overcomplete
code the readouts prefer. That is exactly the move from a plain rescale to a within-group softmax. But I
do not get to assume that; I run RMSNorm first and let the cheetah-run number tell me how much the
missing structure actually costs. The full scaffold module — the group-wise RMS rescale with the
per-group gain and the eps-under-root — is in the answer.
