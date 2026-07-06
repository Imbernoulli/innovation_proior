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

Let me make the failure mode quantitative rather than leave it as a slogan, because I want to know how
bad the averaging really is before I pay for a distribution. Suppose at some observation the expert's
logged action on one axis is drawn half the time from a tight cluster near +0.2 and half from a tight
cluster near +0.8 — a plausible "grip early vs grip late" fork on the normalized [−1,+1] scale. The L1
head lands on the median, which for a balanced bimodal target sits in the empty valley at ≈+0.5; the L2
head lands on the mean, also ≈+0.5. Neither +0.5 was ever a valid action — the gripper is neither open
nor closed, the arm neither committed nor waiting — and across a chunk of 8 timesteps and 7 axes those
midpoint errors compound into a trajectory that satisfies no episode. The damage scales with how far
apart the modes are: the point estimate's error is roughly the half-separation of the modes, 0.3 here,
which dwarfs the finest control resolution the actuators actually resolve. So the multimodality tax is
not a rounding effect I can shrug off; it is a large, structural bias precisely at the decision points
that decide task success. That is the thing worth a heavier head — if such a head can actually be fit.

So I need a head that models a whole conditional distribution over the action chunk. What fits a
complicated, possibly multimodal continuous distribution? A mixture density network would predict the
means, variances, and weights of a Gaussian mixture, but I have to fix the component count up front, and
the geometry works against me: my target is a 56-dimensional chunk (K·D = 8·7), so an m-component
diagonal mixture head must emit m·(56 means + 56 log-variances + 1 weight) = 113m numbers, and a
handful of demonstrations per task cannot supervise many components without collapse — mixtures are
notoriously fiddly, with dying components and mode collapse, and a small fixed mixture is a clumsy way
to express the geometry of valid action chunks across a task suite. An energy-based head — learn an
energy E(o,a), set p(a|o) ∝ exp(−E) — is genuinely expressive and has been used for imitation, but
sampling means an inner optimization or MCMC at inference: even a stingy Langevin sampler wants tens of
gradient-of-energy evaluations per action, and each such evaluation is a pass through a 7B-conditioned
energy, so I would be bolting a second inference loop — comparable in cost to the diffusion rollout I am
about to accept — onto the trunk, plus training on negative samples and a partition function I cannot
compute. I want something that trains with a plain regression-flavored loss — no adversary, no partition
function, no negative samples — yet still samples from a multimodal distribution.

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
I should confirm the composition telescopes before I lean on it, so take two steps concretely. From
x₀, x₁ = √α₁ x₀ + √β₁ ε₁ and x₂ = √α₂ x₁ + √β₂ ε₂ = √(α₁α₂) x₀ + √α₂√β₁ ε₁ + √β₂ ε₂. The two injected
noises are independent Gaussians, so their variances add: α₂β₁ + β₂ = α₂(1−α₁) + (1−α₂) = 1 − α₁α₂. The
mean coefficient is √(α₁α₂) = √ᾱ₂ and the total noise variance is 1 − ᾱ₂ — with α₁ = 0.99, α₂ = 0.98
that is mean coefficient 0.98499 and noise variance 0.02980, and 1 − (0.99)(0.98) = 0.02980 to the
digit. So the chain telescopes to a one-shot marginal: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε, ε ~ N(0,I). I can jump
straight to any noise level in one shot — no chain to unroll. ᾱₜ runs from ≈1 (barely noised) down to
≈0 (pure noise).

What should the reverse step predict? Model p_θ(xₜ₋₁ | xₜ) as Gaussian — justified because for small βₜ
the true reverse conditional is approximately Gaussian — so I need its mean. The clean route is the
posterior q(xₜ₋₁ | xₜ, x₀), tractably Gaussian because everything is linear-Gaussian, whose mean is the
known combination μ̃ = (√ᾱₜ₋₁ βₜ /(1−ᾱₜ)) x₀ + (√αₜ (1−ᾱₜ₋₁)/(1−ᾱₜ)) xₜ. But x₀ is exactly what I do
not have at sampling time. The reparameterization that unlocks it: the marginal says
x₀ = (xₜ − √(1−ᾱₜ) ε)/√ᾱₜ, so the only unknown in x₀ given xₜ is the noise ε. Instead of predicting x₀
or the posterior mean, predict **ε** with a network ε_θ(xₜ, t). Substituting back, each timestep's term
of the variational bound collapses to a weighted ‖ε − ε_θ‖², and dropping the per-timestep weight (the
practical choice that trains best) leaves one clean loss: L = E ‖ε − ε_θ(√ᾱₜ x₀ + √(1−ᾱₜ) ε, t)‖².
Sample a clean chunk, a noise tensor, a timestep; form the noised chunk in one shot; ask the network
for the noise; take the MSE. No adversary, no partition function, no negative samples — a regression
loss that fits a full distribution.

There is a real fork in *what* the network predicts, and I should settle it on the merits rather than by
default, because it changes the loss geometry. Three targets are equivalent in principle — the clean
chunk x₀, the noise ε, or the posterior mean directly — but they weight the objective differently across
the schedule. Predicting x₀ makes the loss ‖x₀ − x̂₀‖² easy at high noise (ᾱ ≈ 0, where any guess near
the data mean is fine) and hard at low noise, so the network spends its capacity on the trivial end.
Predicting ε inverts that: because x̂₀ = (xₜ − √(1−ᾱₜ) ε_θ)/√ᾱₜ, a fixed error in ε_θ is amplified by
1/√ᾱₜ when reconstructing the clean chunk, so at low noise (ᾱ near 1) the ε-loss and the x₀-loss nearly
coincide, while at high noise the ε target stays an honest unit-variance Gaussian the network can always
attempt. Empirically that "predict ε and drop the per-timestep weight" combination is the one that
trains stably, and it also happens to be the parameterization whose per-step loss is a plain unweighted
MSE against a N(0,I) target — a clean, well-conditioned regression at every noise level. So ε-prediction
it is, and the choice is not cosmetic: it is what keeps the gradient magnitudes comparable across the 50
levels I am already spreading my 6000 steps across.

Now sampling. The literal ancestral reverse chain takes all T steps because it is the exact reversal of
a length-T Markov chain. For a robot controller that is a latency problem: T forward passes through a 7B
trunk per chunk. The handle is that the training objective only ever constrained the marginals
√ᾱₜ x₀ + √(1−ᾱₜ) ε; it never demanded the Markov forward chain. A family of non-Markovian forward
processes share those marginals and hence the same trained ε_θ, and for that family the generative step
factors into two pieces: first estimate the clean chunk, x̂₀ = (xₜ − √(1−ᾱₜ) ε_θ)/√ᾱₜ, then re-noise it
to the next lower level, xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1−ᾱₜ₋₁−σ²) ε_θ + σ z. With σ = 0 the step is
*deterministic* — "estimate x₀, jump to the next level" — and I can run it on a sparse subsequence of
timesteps. That is the DDIM-style sampler I will use, so inference latency is in principle tunable; here
the protocol fixes the step count at 50.

The schedule βₜ needs care, and the obvious linear ramp is wrong for my data. The action chunk is
low-dimensional and lives in [−1,+1]; a linear βₜ drives ᾱₜ to nearly zero well before t = T, so the
last big fraction of the chain operates on what is already pure noise — wasted steps where there is no
signal left to denoise, with abrupt noise-level jumps in the middle. I want ᾱₜ to change slowly near
both ends and roughly linearly in the middle, so every reverse step does useful work. The squared-cosine
schedule defines ᾱₜ directly on the cumulative product: ᾱₜ = f(t)/f(0), f(t) = cos²(((t/T+s)/(1+s))·π/2),
with s = 0.008, keeping ᾱ near 1 at the start and descending smoothly with a gentle approach to 0. Let
me actually tabulate it for T = 50 to check it does what I want, not just assert it. At t = 0, ᾱ ≈
1.00000; at t = 10, ᾱ ≈ 0.899; at t = 25, ᾱ ≈ 0.494; at t = 40, ᾱ ≈ 0.094; at t = 49, ᾱ ≈ 0.00097; at
t = 50, ᾱ = 0. The signal-to-noise ratio ᾱ/(1−ᾱ) reads 8.9 at t = 10, 0.98 at t = 25, and 0.10 at
t = 40 — spanning nearly two orders of magnitude and passing through unity right at the midpoint. That
is exactly the property I wanted: no cluster of steps stranded at SNR ≈ 0 doing nothing, and the
crossover between "mostly signal" and "mostly noise" sitting in the middle of the chain where the
sparse DDIM subsequence spends its evaluations. For small-range action data this matters more than for
images, so squared-cosine it is.

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
layout so I never touch the trunk's sequence handling. A chunk of K timesteps is K·7 = 56 scalar slots,
and my noisy chunk is exactly K·7 reals. So I treat each noisy *scalar* as a one-dimensional token and
project it into the trunk's embedding width: a tiny `NoisyActionProjector`, Linear(1→d) → GELU →
Linear(d→d), broadcast over all 56 scalars. Per-scalar (input dim 1) and not per-action (dim 7),
because the slot layout is one slot per dimension and I want the noisy action aligned slot-for-slot with
how the trunk already attends to actions. The timestep next: the network must know how noisy the input
is, so I sinusoidally encode the integer timestep into the embedding space and hand it to the trunk as
one extra token through the `diffusion_timestep_embeddings` argument the runtime exposes.

One training forward then looks like: sample t and ε, form xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε in one shot via the
scheduler's `add_noise`, project xₜ's 56 scalars to action-token features, encode t, and call
`runtime.forward(action_token_features=..., diffusion_timestep_embeddings=...)`. The trunk fuses these
with the fixed context under bidirectional attention and returns final hidden states; I select the
action-slot hidden states — at train time by the `current | next` action mask the runtime exposes, at
eval time by the contiguous 56-slot span after the prompt — and reshape to (B, 56, d). To predict noise
I rearrange (B, 56, d) into (B, K, 7·d) — gathering a timestep's 7 per-dimension hidden states into one
feature — and run a `NoisePredictionModel`, a pre-LayerNorm MLP-residual net taking 7·d in and emitting
7 per timestep; pre-LN residual blocks keep gradients stable while this small net co-trains with the
giant trunk. The loss is the MSE between this prediction and the ε I actually added.

I should trace the shapes end to end once, because a silent reshape bug here would poison the whole
rung and look like "diffusion just doesn't work." The noisy chunk enters as (B, 8, 7). Flatten to
(B, 56, 1) and the projector lifts the trailing 1 to d = 4096, giving action-token features
(B, 56, 4096); the timestep embedding is (B, 1, 4096), one extra token. The trunk returns hidden states
over the full sequence; masking to the 56 action slots and reshaping gives (B, 56, 4096). Rearranging to
gather each timestep's 7 slots yields (B, 8, 7·4096) = (B, 8, 28672), which is exactly the
`NoisePredictionModel` input width (hidden_dim·ACTION_DIM = 4096·7 = 28672), and its output is
(B, 8, 7). Reshaped to the noise tensor's (B, 8, 7), it lines up with ε element-for-element, so the MSE
is well-posed. The dimensions close, which is the cheap check that the conditioning path is wired the
way the derivation intends. At inference I run the reverse rollout: start from N(0,I) of shape
(1, 8, 7), and for each scheduled t encode t, project the current noisy chunk, forward the trunk,
extract action-slot hidden states, predict the noise, and let the scheduler take one denoising step.
After the last step the chunk is a clean normalized action.

Before I accept the cost, let me size the from-scratch machinery I am asking 6000 steps to fit, because
that number is the crux of this rung's fate. The `NoisyActionProjector` is Linear(1→4096) with 8,192
parameters plus Linear(4096→4096) with 4096·4096 + 4096 ≈ 16.78M — about 16.8M all told. The
`NoisePredictionModel` alone has an input projection Linear(28672→4096) worth 28672·4096 + 4096 ≈
117.4M parameters, and with two residual feedforward blocks (~33M each at this width) and the output
head it lands north of 180M. The sinusoidal time encoder is parameter-free. So I am asking the optimizer
to fit on the order of 200M *randomly initialized* parameters — plus the LoRA rank-32 adapters on the
trunk — from scratch, against a noise-prediction target, inside the budget. That is the liability I am
buying in exchange for multimodality.

One more scaffold-specific decision hides in how I read the noise off the trunk, and the tempting
shortcut is worth killing explicitly. I could ignore the per-slot layout, pool the 56 action-slot hidden
states into a single vector, and feed a small standalone denoiser that conditions on that pooled summary
plus t — a clean separation that treats the trunk purely as an observation encoder. But that throws away
the one thing the parallel-decode trunk is built to give me: a *per-slot* representation under
bidirectional attention, where slot (k, d) already knows about every other action dimension and timestep
in the chunk. Pooling collapses that into one 4096-vector and forces the tiny standalone denoiser to
re-derive the intra-chunk structure the 7B trunk just computed for free. So I keep the noise prediction
attached slot-for-slot: I inject the noisy chunk *as* the action-token features so the trunk re-attends
to it, and I read the noise back from the same 56 slots. The cost is that the trunk must be in the loop
at every one of the 50 reverse steps — which is exactly the latency I flagged — but the alternative
buys latency back only by discarding the conditioning quality that is the whole reason to build on this
trunk instead of a from-scratch policy. It is the right trade for a rung whose entire premise is
"exploit the trunk's fused representation," even knowing the 50-pass rollout is what will dominate the
eval timers.

There is also a coupling I should note between the two step counts, because it removes a latency escape I
might have hoped for. DDIM's whole point is that I can *subsample* the schedule — train on 50 levels, run
inference on, say, 10 — and pay a tenth of the passes. But the protocol pins both
`FIXED_NUM_DIFFUSION_STEPS_TRAIN` and `FIXED_NUM_DIFFUSION_STEPS_INFERENCE` to 50, so I visit every
trained level at inference and get no acceleration. That is arguably the honest setting for a fair
comparison — it isolates the head's quality from sampler tricks — but it means the 50× decode tax is
locked in, not something I can trade away, and it is the reason diffusion's eval cost is structural here
rather than a knob I forgot to turn.

I should name the cost I am accepting and the protocol fact that decides this rung's fate. Inference is
one trunk forward *per diffusion step* — the runtime fixes 50 inference steps — so a single action chunk
costs ~50 passes through a 7B model, versus one pass for a point head. That is the price of representing
a distribution instead of a vector, and it is the only thing the eval timers should show as dramatically
larger here. More important than latency, though, is *convergence*, and here the budget arithmetic is
brutal in a way worth spelling out. The training config is batch size 8 with gradient accumulation 4, so
the effective batch is 32, and 6000 optimizer steps means 6000·32 = 192,000 sample-gradient exposures
total. A point head would spend every one of those 192,000 exposures regressing the *exact* clean target
it must reproduce at eval. The diffusion head cannot: each exposure samples one random timestep out of
50, so the reverse map at any given noise level is supervised by only ~192,000/50 ≈ 3,840 exposures —
the gradient signal is diluted roughly fifty-fold across the noise schedule, on top of the objective
being noise-prediction rather than the action itself. And the published regime that makes an
action-diffusion policy competitive trains for 100K–250K steps; 6,000 is 6% of the low end and 2.4% of
the high end. So the honest expectation that makes diffusion the weakest rung is this: at 6000 steps the
denoiser is *under-trained*. The reverse process will not yet have learned to walk noise back to
coherent action chunks, so the sampled actions will be close to noise — the policy will mostly fail to
complete tasks, success near the floor on every subset, with eval times several times larger than any
single-pass head because of the 50-step rollout. I expect this to be the lowest rung not because
diffusion is a worse *idea* — in the long-budget regime it is competitive, and its multimodality is a
real advantage the point heads lack — but because it is the idea that most needs training time, and this
protocol does not give it. That diagnosis is exactly what should push the next rung toward a head that
learns fast enough to actually fit the demonstrations inside 6000 steps, even at the cost of giving up
the distribution-modeling I have been chasing here.
