The action method is the whole point, and I have a parallel-decoding trunk handing me, at each of the
K·D action slots, a final-layer hidden state conditioned on vision, language, and proprioception. The
cheapest decoder I could write would read each timestep's bundle of D hidden states, run them through
a small MLP, emit a 7-D action, and train it against the logged demonstration with a regression loss.
One trunk pass, a negligible head, full continuous precision. So before I reach for anything heavier I
should be honest about what's wrong with that point estimator, because if nothing is wrong I have no
business paying for a more expensive method — and the answer to "what's wrong" is exactly what makes me
start the ladder here, at the most expressive head I can build on this scaffold.

Stare at the regression objective. Minimizing the mean absolute error over the data picks, for each
observation, the conditional *median* of the logged actions; minimizing squared error picks the
conditional *mean*. Either way the head outputs a single vector per observation. That is fine when, for
a given thing the robot sees, there is essentially one right action. But manipulation demonstrations
are not like that. At a fork an expert sometimes reaches left, sometimes right; sometimes regrasps now,
sometimes a beat later; the logged actions at nearly identical observations form a genuinely
*multimodal* distribution. A point estimator handed a bimodal target sits at the average of the modes,
and the average of "swerve left" and "swerve right" is "drive straight into the obstacle." This is not
a tuning artifact I can anneal away — it is structural: the head emits one vector and the world wants a
distribution. So the real task is not "predict the action," it is "represent p(action chunk |
observation) richly enough to be multimodal," and then sample one action from it. That reframing is why
I open the ladder with a distribution-modeling head rather than the point head: if multimodality is the
deciding factor, the most expressive decoder should be the strongest, and I want to find out.

So I need a head that models a whole conditional distribution over the action chunk. What fits a
complicated, possibly multimodal continuous distribution? A mixture density network would predict the
means, variances, and weights of a Gaussian mixture, but I have to fix the component count up front,
mixtures are notoriously fiddly (mode collapse, dying components), and a small fixed mixture is a
clumsy way to express the geometry of valid action chunks across a task suite. An energy-based head —
learn an energy E(o,a), set p(a|o) ∝ exp(−E) — is genuinely expressive and has been used for
imitation, but sampling means an inner optimization or MCMC at inference, training needs negative
samples and is finicky, and I would be bolting a second inference loop onto a 7B trunk. I want
something that trains with a plain regression-flavored loss — no adversary, no partition function, no
negative samples — yet still samples from a multimodal distribution.

There is a construction that fits that description, and it is worth rebuilding from the ground up. Take
a clean action chunk x₀ and deliberately wreck it with Gaussian noise, a little at a time, over a long
chain: q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ) xₜ₋₁, βₜ I) for a schedule of small βₜ. After enough steps the
chunk is indistinguishable from pure noise. The move is this: if I can learn to undo one step of that
corruption — to go from xₜ back toward xₜ₋₁ — then I can start from pure noise x_T ~ N(0,I) and walk
the chain backwards all the way to a clean sample. The forward corruption is fixed and trivial; all the
learning lives in the reverse step. And because each generation starts from a fresh noise draw,
different seeds land in different basins — the reverse chain is a stochastic map from noise to data, so
it covers multiple modes without my ever naming them. That is precisely the multimodality the point
head could not represent, and it falls out of the construction for free.

I do not want to actually iterate the corruption chain during training, so let me make the forward
process usable. Write αₜ = 1 − βₜ and ᾱₜ = ∏ₛ αₛ. Each step scales by √αₜ and injects variance βₜ.
Composing two steps and using that independent Gaussians add in variance, the chain telescopes to a
one-shot marginal: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε, ε ~ N(0,I). I can jump straight to any noise level in one
shot — no chain to unroll. ᾱₜ runs from ≈1 (barely noised) down to ≈0 (pure noise).

What should the reverse step predict? Model p_θ(xₜ₋₁ | xₜ) as Gaussian — justified because for small βₜ
the true reverse conditional is approximately Gaussian — so I need its mean. The clean route is the
posterior q(xₜ₋₁ | xₜ, x₀), tractably Gaussian because everything is linear-Gaussian, whose mean is a
known combination of xₜ and x₀. But x₀ is exactly what I do not have at sampling time. The
reparameterization that unlocks it: the marginal says x₀ = (xₜ − √(1−ᾱₜ) ε)/√ᾱₜ, so the only unknown
in x₀ given xₜ is the noise ε. Instead of predicting x₀ or the posterior mean, predict **ε** with a
network ε_θ(xₜ, t). Substituting back, each timestep's term of the variational bound collapses to a
weighted ‖ε − ε_θ‖², and dropping the per-timestep weight (the practical choice that trains best)
leaves one clean loss: L = E ‖ε − ε_θ(√ᾱₜ x₀ + √(1−ᾱₜ) ε, t)‖². Sample a clean chunk, a noise tensor,
a timestep; form the noised chunk in one shot; ask the network for the noise; take the MSE. No
adversary, no partition function, no negative samples — a regression loss that fits a full distribution.

Now sampling. The literal ancestral reverse chain takes all T steps because it is the exact reversal of
a length-T Markov chain. For a robot controller that is a latency problem: T forward passes through a 7B
trunk per chunk. The handle is that the training objective only ever constrained the marginals
√ᾱₜ x₀ + √(1−ᾱₜ) ε; it never demanded the Markov forward chain. A family of non-Markovian forward
processes share those marginals and hence the same trained ε_θ, and for that family the generative step
factors into two pieces: first estimate the clean chunk, x̂₀ = (xₜ − √(1−ᾱₜ) ε_θ)/√ᾱₜ, then re-noise it
to the next lower level, xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1−ᾱₜ₋₁−σ²) ε_θ + σ z. With σ = 0 the step is
*deterministic* — "estimate x₀, jump to the next level" — and I can run it on a sparse subsequence of
timesteps. That is the DDIM-style sampler I will use, so inference latency is in principle tunable; here
the protocol fixes the step count.

The schedule βₜ needs care, and the obvious linear ramp is wrong for my data. The action chunk is
low-dimensional and lives in [−1,+1]; a linear βₜ drives ᾱₜ to nearly zero well before t = T, so the
last big fraction of the chain operates on what is already pure noise — wasted steps where there is no
signal left to denoise, with abrupt noise-level jumps in the middle. I want ᾱₜ to change slowly near
both ends and roughly linearly in the middle, so every reverse step does useful work. The squared-cosine
schedule defines ᾱₜ directly on the cumulative product: ᾱₜ = f(t)/f(0), f(t) = cos²(((t/T+s)/(1+s))·π/2),
keeping ᾱ near 1 at the start and descending smoothly with a gentle approach to 0. For small-range
action data this matters more than for images, so squared-cosine it is.

Now the part specific to *this* scaffold, which is not in any generic recipe: where does the observation
conditioning go, and what is ε_θ actually attached to? In a from-scratch action-diffusion policy the
noise predictor is its own network and the observation is piped in through a dedicated path. I have no
such luxury and want none: my observation is already fused — vision, language, proprioception — inside
the 7B trunk, and the only handle I have on it is the trunk's hidden states at the action slots. The
trunk *is* the conditioning network. So I must feed the current noisy action chunk and the diffusion
timestep into the trunk so that its returned hidden states are conditioned on (observation, noisy
action, t), then read the noise prediction off those hidden states.

The noisy chunk first. The trunk lays out each action timestep as ACTION_DIM=7 input slots — one per
dimension, because the base model tokenized each 7-D action into 7 separate tokens and I keep that
layout so I never touch the trunk's sequence handling. A chunk of K timesteps is K·7 scalar slots, and
my noisy chunk is exactly K·7 reals. So I treat each noisy *scalar* as a one-dimensional token and
project it into the trunk's embedding width: a tiny `NoisyActionProjector`, Linear(1→d) → GELU →
Linear(d→d), broadcast over all K·7 scalars. Per-scalar (input dim 1) and not per-action (dim 7),
because the slot layout is one slot per dimension and I want the noisy action aligned slot-for-slot with
how the trunk already attends to actions. The timestep next: the network must know how noisy the input
is, so I sinusoidally encode the integer timestep into the embedding space and hand it to the trunk as
one extra token through the `diffusion_timestep_embeddings` argument the runtime exposes.

One training forward then looks like: sample t and ε, form xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε in one shot via the
scheduler's `add_noise`, project xₜ's K·7 scalars to action-token features, encode t, and call
`runtime.forward(action_token_features=..., diffusion_timestep_embeddings=...)`. The trunk fuses these
with the fixed context under bidirectional attention and returns final hidden states; I select the
action-slot hidden states — at train time by the `current | next` action mask the runtime exposes, at
eval time by the contiguous K·7 span after the prompt — and reshape to (B, K·7, d). To predict noise I
rearrange (B, K·7, d) into (B, K, 7·d) — gathering a timestep's 7 per-dimension hidden states into one
feature — and run a `NoisePredictionModel`, a pre-LayerNorm MLP-residual net taking 7·d in and emitting
7 per timestep; pre-LN residual blocks keep gradients stable while this small net co-trains with the
giant trunk. The loss is the MSE between this prediction and the ε I actually added. At inference I run
the reverse rollout: start from N(0,I) of shape (1, K, 7), and for each scheduled t encode t, project
the current noisy chunk, forward the trunk, extract action-slot hidden states, predict the noise, and
let the scheduler take one denoising step. After the last step the chunk is a clean normalized action.

I should name the cost I am accepting and the protocol fact that decides this rung's fate. Inference is
one trunk forward *per diffusion step* — the runtime fixes 50 inference steps — so a single action chunk
costs ~50 passes through a 7B model, versus one pass for a point head. That is the price of representing
a distribution instead of a vector, and it is the only thing the eval timers should show as dramatically
larger here. More important than latency, though, is *convergence*. The diffusion objective only ever
sees a randomly-noised chunk at a random timestep and asks for the noise; the trunk and the denoiser
must jointly learn, across the whole noise schedule, how to recover clean actions — a far harder fitting
problem than regressing the action directly, and one that the published recipe trains for 100K-250K
steps to make competitive. This task fixes the budget at 6000 steps to stay inside 5 H200h. So the
honest expectation that makes diffusion the weakest rung is this: at 6000 steps the denoiser is
*under-trained*. The reverse process will not yet have learned to walk noise back to coherent action
chunks, so the sampled actions will be close to noise — the policy will mostly fail to complete tasks,
success near the floor on every subset, with eval times several times larger than any single-pass head
because of the 50-step rollout. I expect this to be the lowest rung not because diffusion is a worse
*idea* — in the long-budget regime it is competitive, and its multimodality is a real advantage the
point heads lack — but because it is the idea that most needs training time, and this protocol does not
give it. That diagnosis is exactly what should push the next rung toward a head that learns fast enough
to actually fit the demonstrations inside 6000 steps, even at the cost of giving up multimodality.
