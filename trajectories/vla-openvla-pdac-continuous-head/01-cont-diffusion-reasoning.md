The action method is the whole point, and I have a parallel-decoding trunk handing me, at each of the
K·D action slots, a final-layer hidden state conditioned on vision, language, and proprioception. The
cheapest decoder I could write would read each timestep's bundle of D hidden states, run them through a
small MLP, emit a 7-D action, and train it against the logged demonstration with a regression loss. One
trunk pass, a negligible head, full continuous precision. So before I reach for anything heavier I should
be honest about what's wrong with that point estimator — if nothing is wrong I have no business paying
for a more expensive method, and the answer is exactly what makes me open the ladder at the most
expressive head I can build on this scaffold.

Stare at the regression objective. Minimizing the mean absolute error over the data picks, for each
observation, the conditional *median* of the logged actions; minimizing squared error picks the
conditional *mean*. Either way the head outputs a single vector per observation. That is fine when, for a
given thing the robot sees, there is essentially one right action. But manipulation demonstrations are
not like that. At a fork an expert sometimes reaches left, sometimes right; sometimes regrasps now,
sometimes a beat later; the logged actions at nearly identical observations form a genuinely *multimodal*
distribution. A point estimator handed a bimodal target sits at the average of the modes, and the average
of "swerve left" and "swerve right" is "drive straight into the obstacle." This is not a tuning artifact
I can anneal away — it is structural: the head emits one vector and the world wants a distribution.
Concretely, if the logged action on some axis is half the time near +0.2 and half near +0.8, both the
median and the mean land near the empty valley at +0.5 — a value never actually valid, off by the
half-separation of the modes, and across 8 timesteps and 7 axes those midpoint errors compound into a
trajectory no episode endorses. So the real task is not "predict the action," it is to represent
p(action chunk | observation) richly enough to be multimodal and sample one action from it — which is why
I open with a distribution-modeling head: if multimodality is the deciding factor, the most expressive
decoder should be the strongest.

So I need a head that models a whole conditional distribution over the action chunk. A mixture density
network predicts the means, variances, and weights of a Gaussian mixture, but I must fix the component
count up front, and the geometry works against me: the target is a 56-dimensional chunk (K·D = 8·7), so
an m-component diagonal mixture must emit m·(56 + 56 + 1) = 113m numbers, and a handful of demonstrations
per task cannot supervise many components without the usual mode collapse and dying components. An
energy-based head — learn E(o,a), set p(a|o) ∝ exp(−E) — is genuinely expressive, but sampling means an
inner optimization or MCMC: even a stingy Langevin sampler wants tens of gradient-of-energy evaluations
per action, each a pass through the 7B-conditioned energy, so I would be bolting a second inference loop
onto the trunk, plus training on negatives and a partition function I cannot compute. I want something
that trains with a plain regression-flavored loss — no adversary, no partition function, no negatives —
yet still samples from a multimodal distribution.

There is a construction that fits. Take a clean action chunk x₀ and deliberately wreck it with Gaussian
noise, a little at a time: q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ) xₜ₋₁, βₜ I) for a schedule of small βₜ. After
enough steps the chunk is pure noise. If I can learn to undo one step — go from xₜ back toward xₜ₋₁ —
then I can start from x_T ~ N(0,I) and walk the chain backwards to a clean sample. The forward corruption
is fixed and trivial; all the learning lives in the reverse step. And because each generation starts from
a fresh noise draw, different seeds land in different basins — the reverse chain is a stochastic map from
noise to data that covers multiple modes without my ever naming them. That is precisely the multimodality
the point head could not represent.

I do not want to iterate the corruption chain during training. Write αₜ = 1 − βₜ and ᾱₜ = ∏ₛ αₛ.
Composing two steps, x₂ = √(α₁α₂) x₀ + √α₂√β₁ ε₁ + √β₂ ε₂; the two injected noises are independent, so
their variances add: α₂β₁ + β₂ = α₂(1−α₁) + (1−α₂) = 1 − α₁α₂, while the mean coefficient is
√(α₁α₂) = √ᾱ₂. So the chain telescopes to a one-shot marginal: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε, ε ~ N(0,I). I
can jump straight to any noise level — ᾱₜ runs from ≈1 (barely noised) down to ≈0 (pure noise).

What should the reverse step predict? Model p_θ(xₜ₋₁ | xₜ) as Gaussian — for small βₜ the true reverse
conditional is approximately Gaussian — so I need its mean. The clean route is the posterior
q(xₜ₋₁ | xₜ, x₀), whose mean is a known combination of xₜ and x₀; but x₀ is exactly what I lack at
sampling time. The reparameterization that unlocks it: the marginal gives x₀ = (xₜ − √(1−ᾱₜ) ε)/√ᾱₜ, so
the only unknown in x₀ given xₜ is the noise ε. Predict **ε** with a network ε_θ(xₜ, t); substituting
back, each timestep's term of the variational bound collapses to a weighted ‖ε − ε_θ‖², and dropping the
per-timestep weight leaves L = E ‖ε − ε_θ(√ᾱₜ x₀ + √(1−ᾱₜ) ε, t)‖². Sample a clean chunk, a noise
tensor, a timestep; form the noised chunk in one shot; ask for the noise; take the MSE. No adversary, no
partition function, no negatives — a regression loss that fits a full distribution. The choice of ε over
the equivalent targets (x₀, or the posterior mean) is not cosmetic: predicting x₀ makes the loss trivial
at high noise, where any guess near the data mean suffices, so the network spends capacity on the easy
end, while ε stays an honest unit-variance target the network can attempt at every level, keeping the
gradient magnitudes comparable across the 50 levels I am spreading 6000 steps over.

Now sampling. The literal ancestral reverse chain takes all T steps — T forward passes through a 7B trunk
per chunk, a latency problem. But the training objective only ever constrained the marginals; it never
demanded the Markov forward chain. A family of non-Markovian forward processes share those marginals and
the same trained ε_θ, and for that family the step factors: estimate x̂₀ = (xₜ − √(1−ᾱₜ) ε_θ)/√ᾱₜ, then
re-noise to the next lower level, xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1−ᾱₜ₋₁−σ²) ε_θ + σ z. With σ = 0 the step is
deterministic — "estimate x₀, jump to the next level" — and can run on a sparse subsequence of timesteps.
That is the DDIM sampler; in principle it makes inference latency tunable, but here the protocol fixes the
step count at 50.

The schedule βₜ needs care, and the obvious linear ramp is wrong here. The action chunk is
low-dimensional and lives in [−1,+1]; a linear βₜ drives ᾱₜ to nearly zero well before t = T, stranding
the last big fraction of the chain on what is already pure noise — wasted steps, with abrupt noise-level
jumps in the middle. I want ᾱₜ to change slowly near both ends and roughly linearly in between so every
reverse step does useful work. The squared-cosine schedule sets ᾱₜ directly on the cumulative product:
ᾱₜ = f(t)/f(0), f(t) = cos²(((t/T+s)/(1+s))·π/2), s = 0.008, so the signal-to-noise ratio ᾱ/(1−ᾱ) passes
through unity near the midpoint of the chain rather than collapsing early and leaving a cluster of steps
stranded at SNR ≈ 0. For small-range action data this matters more than for images, so squared-cosine it
is.

Now the part specific to *this* scaffold: where does the observation conditioning go, and what is ε_θ
attached to? In a from-scratch action-diffusion policy the noise predictor is its own network with a
dedicated observation path. I have no such luxury and want none: my observation is already fused —
vision, language, proprioception — inside the 7B trunk, and my only handle on it is the trunk's hidden
states at the action slots. The trunk *is* the conditioning network. So I feed the current noisy chunk
and the diffusion timestep into the trunk so its returned hidden states are conditioned on (observation,
noisy action, t), then read the noise off those hidden states.

The trunk lays out each action timestep as ACTION_DIM=7 input slots — one per dimension, because the base
model tokenized each 7-D action into 7 separate tokens and I keep that layout so I never touch the
trunk's sequence handling. A chunk of K timesteps is K·7 = 56 scalar slots, and my noisy chunk is exactly
K·7 reals. So I treat each noisy *scalar* as a one-dimensional token and project it into the embedding
width with a tiny `NoisyActionProjector`, Linear(1→d) → GELU → Linear(d→d), broadcast over all 56
scalars — per-scalar, not per-action, so the noisy action aligns slot-for-slot with how the trunk already
attends to actions. The timestep the network must know: I sinusoidally encode it and hand it in as one
extra token through the `diffusion_timestep_embeddings` argument the runtime exposes.

One training forward: sample t and ε, form xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε via the scheduler's `add_noise`,
project the 56 scalars to action-token features (B, 56, 4096), encode t to (B, 1, 4096), and call
`runtime.forward(action_token_features=..., diffusion_timestep_embeddings=...)`. The trunk fuses these
with the fixed context under bidirectional attention and returns hidden states; I select the action slots
— at train time by the `current | next` action mask, at eval time by the contiguous 56-slot span after
the prompt — reshape to (B, 56, 4096), and rearrange to (B, 8, 7·4096) = (B, 8, 28672), gathering each
timestep's 7 per-dimension states into one feature. That width is exactly the `NoisePredictionModel`
input (hidden_dim·ACTION_DIM); it is a pre-LayerNorm MLP-residual net emitting 7 per timestep, so its
output lines up with ε element-for-element and the loss is the MSE against the ε I added. At inference the
reverse rollout starts from N(0,I) of shape (1, 8, 7) and, for each scheduled t, encodes t, projects the
current chunk, forwards the trunk, predicts the noise, and lets the scheduler take one denoising step —
after the last step the chunk is a clean normalized action.

The cost I am asking 6000 steps to absorb is worth sizing, because that number decides this rung's fate.
The `NoisyActionProjector`'s Linear(4096→4096) is ≈16.8M; the `NoisePredictionModel`'s input projection
Linear(28672→4096) alone is ≈117M, and with two residual blocks and the output head it lands north of
180M. So I am asking the optimizer to fit on the order of 200M *randomly initialized* parameters — plus
the LoRA rank-32 adapters on the trunk — from scratch against a noise-prediction target, inside the
budget. That is the liability I buy in exchange for multimodality.

One scaffold decision I decline deliberately: I could pool the 56 action-slot hidden states into a single
vector and feed a standalone denoiser conditioned on that summary plus t, treating the trunk as a pure
observation encoder. But that discards the one thing the parallel-decode trunk exists to give me — a
per-slot representation under bidirectional attention, where slot (k, d) already knows every other
dimension and timestep in the chunk. Pooling would force the tiny denoiser to re-derive the intra-chunk
structure the 7B trunk just computed for free. So I keep the noise prediction attached slot-for-slot:
inject the noisy chunk *as* the action-token features so the trunk re-attends to it, and read the noise
back from the same 56 slots. The price is that the trunk is in the loop at every one of the 50 reverse
steps — exactly the latency I flagged — but that is the right trade for a rung whose premise is to
exploit the trunk's fused representation.

DDIM's whole point is that I could subsample — train on 50 levels, run inference on ten — and pay a tenth
of the passes. But the protocol pins both `FIXED_NUM_DIFFUSION_STEPS_TRAIN` and
`FIXED_NUM_DIFFUSION_STEPS_INFERENCE` to 50, so I visit every trained level at inference and get no
acceleration. That isolates the head's quality from sampler tricks, but it also locks in the 50× decode
tax rather than leaving it a knob I can turn.

So the cost is one trunk forward per diffusion step — ~50 passes per chunk against one for a point head —
and that is the only thing the eval timers should show as dramatically larger. More decisive is
convergence. The effective batch is 32 (batch 8 × grad-accum 4), so 6000 steps is 192,000
sample-gradient exposures. A point head would spend every one of them regressing the exact clean target
it must reproduce at eval. The diffusion head cannot: each exposure samples one of 50 timesteps, so any
given noise level sees only ~3,840 exposures — the gradient signal diluted roughly fiftyfold across the
schedule, on top of the target being noise rather than the action itself. And the published regime that
makes an action-diffusion policy competitive trains 100K–250K steps; 6000 is a few percent of that. So
the honest expectation that makes diffusion the weakest rung: at 6000 steps the denoiser is
*under-trained*, the reverse process has not learned to walk noise back to coherent chunks, the sampled
actions stay close to noise, and success sits near the floor on every subset — with eval times several
times larger than any single-pass head because of the 50-step rollout. Not because diffusion is a worse
*idea* — in the long-budget regime it is competitive, and its multimodality is a real advantage the point
heads lack — but because it is the idea that most needs training time, and this protocol withholds it.
That is exactly what should push the next rung toward a head that learns fast enough to fit the
demonstrations inside 6000 steps, even at the cost of the distribution-modeling I have been chasing here.
