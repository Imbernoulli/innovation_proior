The deterministic mean encoder landed exactly where its premise predicted it would, and the numbers say
which premise broke. On the dense low-dimensional `point-robot` it is fine — mean −13.9, tight across seeds
(−13.18, −14.17, −14.33) — because there the goal really is readable from a handful of dense transitions
and "one transition reveals the task" roughly holds; the standard deviation across seeds is about 0.5, a
coefficient of variation under 4%, which is what a healthy family looks like. But `cheetah-vel` is the
worst of the three baselines so far at −84.7, and the per-seed spread is the tell: −64.2, −81.6, and a
−108.4 seed that is more than a third worse than the best. Put a number on it — the seed standard deviation
is about 18 on a mean of −85, a coefficient of variation near 22%, roughly six times point-robot's ~4%.
That is not noise around a good solution; that is an encoder that sometimes locks onto a usable
target-velocity representation and sometimes does not, with nothing to make it reliable — exactly the
"unpinned representation" I flagged, quantified. And the tail is doing most of the damage: drop the −108.4
seed and the other two average −72.9, so more than a third of the gap to a respectable cheetah score is a
single unlucky run, not a uniformly weak method. A fix that only firms up the tail — without touching the
median — would already move the mean a long way here. And `sparse-point-robot` is the sharpest failure: 0.30
mean, with two of three seeds at *exactly* 0.0 and the third at 0.90 — under this benchmark's own
convention a 0 means the goal was never reached in the budget. So two of three sparse seeds never found the
goal at all, and the one that did barely registered a single graze. That is the fixed-point-of-failure I
described walking in: an encoder fed all-zero-reward context, producing a `z` the policy cannot act on, and
no mechanism to break the loop by finding the reward once.

Both failures point at the same two holes in the mean encoder, and I can name them precisely now that I
have the numbers. First, the mean over per-transition embeddings throws away *order*. On `cheetah-vel` the
target velocity is only weakly visible in any single `(s, a, r)` — it is the *trend* of the reward as speed
changes across a sequence that pins it down — and a permutation-invariant bag of transitions has no way to
read a trend. That is why the cheetah encoding is unreliable at a 22% coefficient of variation: a
contrastive loss that only enforces "these blocks are the same task, those are different" never forces `z`
to encode the underlying target the policy needs, and a mean cannot recover it from the sequence even if it
wanted to. Second, and this is what kills sparse, the contrastive signal needs *contrast*, and on
`sparse-point-robot` almost every context transition has reward 0 and is therefore nearly identical across
tasks — there is almost nothing for the distance-metric loss to separate tasks *by*. The two dead sparse
seeds are exactly that.

So the diagnosis is concrete, and I can lay it out as a two-axis design table rather than a single
hunch, because that is what stops me from fixing one hole and leaving the other open. One axis is
*aggregation*: mean (order-blind) versus something sequential. The other is the *encoder's training
signal*: the distance-metric geometry (same/different) versus something that forces `z` to encode the
task's actual reward structure. FOCAL sits at (mean, distance-metric), and each single-axis fix repairs
only one failure. If I keep the mean and merely add a reward-prediction objective, I give `z` content on
sparse — predicting where the +1 lives is real contrast — but I have done nothing for cheetah, because the
mean still destroys the sequence before any decoder sees `z`, so the velocity *trend* is gone upstream of
the fix. If instead I keep the distance-metric loss and merely add recurrence, I let a GRU read the cheetah
trend, but on sparse the recurrent encoder is still fed an all-zero-reward context that a same/different
loss has nothing to separate — the dead seeds stay dead. Only the *diagonal* move — sequential aggregation
*and* a reward-structure objective — addresses both holes at once, and that is what I build. Let me take
the two changes in turn, because each one reverses a specific commitment from the last rung and I want the
reversal to be earned, not casual.

Take the sequence problem first, because it changes the primitive. I argued for permutation-invariance at
the previous rung from "each transition independently reveals the task," but the cheetah failure shows the
premise is too strong here: when the task is only weakly visible per transition, *what I should do next*
depends on the whole sequence so far, not on an order-free set. The right primitive for online, ordered
inference is recurrence. Plain set-attention is permutation-invariant — it reintroduces the order-blindness
I am trying to escape unless I bolt positional encodings on; a Transformer would read order but is far
heavier and shines on long-range dependencies across long sequences, whereas here the sequences are short
(capped at path length) and the dependency I need is the *running* summary of a single trajectory, exactly
what a recurrent hidden state carries. Recurrence also matches deployment for free: at test time
transitions arrive one at a time and I fold each into the belief as it comes, one GRU step, where a
Transformer would re-attend over the whole growing context every step. So the GRU is the cheapest
primitive that reads order and updates online. I replace the mean encoder with a recurrent one: embed each transition `(o, a,
r)` with small feature extractors, run a GRU whose hidden state carries the running summary of experience,
and read out the task latent from the hidden state. One GRU step is one belief update, folding in one
transition at a time, which is exactly the online structure the rollout protocol needs — `update_context`
appends a transition, `adapt` runs the GRU over the accumulated context and reads off `z` from the final
hidden state. The harness's `infer_posterior` gets the same treatment: reset the GRU, run it over the
sampled context block, take the last step's readout. A recurrent encoder is not free. The context arrives
as `(num_tasks, seq_len, feat_dim)`; a pre-MLP lifts each transition to width 200, the GRU runs over the
sequence, and the `μ`/`logvar` heads read the last step to `(num_tasks, latent_dim)` — the posterior after
all `seq_len` transitions have been folded in, in order. On cost, the GRU plus pre-MLP and reward decoder
is on the order of `3·10⁵` parameters, roughly four times the `~9·10⁴`-parameter mean encoder. Under this
benchmark's short budget (~twenty iterations, an hour of wall time) that heavier encoder is a real risk of
being slower to train or overfitting — the thing to watch on the one family the mean encoder already
handled. Crucially, on this benchmark I should keep the context the GRU sees
*chronological and within a single trajectory* — if I let it stitch together independent trajectories
without resetting the hidden state at episode boundaries, the running summary gets polluted by
discontinuities, so I cap the context length at the path length so each task's context comes from one
coherent rollout. That is a harness-specific care the mean encoder never needed, because a mean does not
care about order; a GRU does. Concretely: if the diagnostic content of a cheetah context is a monotone
*trend* — reward climbing as speed passes through the target and falling past it — a mean `(1/T)Σ φ(cₜ)`
maps the increasing sequence and its exact reverse to the *same* `z`, annihilating the trend before the
loss sees it, while a GRU's last hidden state is not symmetric in its inputs and separates them. That is
the one expressive gap the recurrence buys, and it is exactly where cheetah's target velocity hides.

Now the richer training signal, and this is where I have to think about what objective actually forces `z`
to carry reward structure. The previous encoder was trained by a distance-metric loss — pure geometry,
"keep tasks apart" — which is necessary (the continuity argument from the last rung still holds: distinct
tasks must be separable for the value functions to exist) but not sufficient, because geometry alone never
says *what* about the task `z` should encode. What I want is a *generative* signal: make `z` good at
predicting the thing that differs across tasks. In this family the reward function is what differs, so I
attach a **reward decoder** — a small network that, given a transition `(s, a)` and the task latent `z`,
predicts the reward — and train the encoder so that the latent it produces lets the decoder reconstruct
rewards accurately. This is the move that would have saved both failures, and I can see the forcing concretely rather than
just asserting it. On `cheetah-vel` the reward is essentially velocity-matching, `r ≈ −|v(s) − v*|` where
`v(s)` is the current forward velocity, already present in `obs`, and `v*` is the task's target. The
decoder sees `(obs, action, z)` and must fit `r` across a whole batch of a single task under one `z`;
since `obs` already carries `v(s)`, the only free quantity the decoder can use to place the reward's peak
is `z`, so minimizing the reconstruction error *forces* `z` to carry a code for `v*` — the exact quantity
the mean encoder left unpinned. On `sparse-point-robot` the decoder must output near-0 for the many
zero-reward rows and near-1 for the rare hit rows; since the `(obs, action)` distribution looks the same
across tasks, the only way to separate the two is to read the goal location out of `z`, so `z` is forced
to encode where the +1 lives — precisely the task-distinguishing bit the contrast had nothing to grab.

Let me make the latent stochastic while I am at it, because the recurrence readout naturally produces a
distribution and I want the regularization that comes with it. The GRU reads out a Gaussian posterior over
`z` — a mean and a log-variance — and I sample `z` by reparameterization so gradients flow. The
stochasticity buys two things. It lets me put an information bottleneck on the latent in the form of a KL
to a unit-Gaussian prior, `KL(q(z|c) ‖ N(0,I)) = −½ Σ(1 + logσ² − μ² − σ²)`, which squeezes `z` to the
minimal reward-relevant content and discourages it from memorizing training-task idiosyncrasies under the
short budget. At initialization the heads output `μ ≈ 0`, `σ² ≈ 1`, so `KL ≈ 0` — the belief starts *at*
the prior, where an uninformed posterior should sit; as `z` acquires content `μ` grows away from zero and
`σ²` shrinks below one and the KL climbs, taxed by `λ`. With `reward_pred_weight = 1.0` against
`kl_lambda = 0.1`, the reconstruction term outweighs the bottleneck roughly ten to one early on, so `z`
first *acquires* task content and only then gets squeezed toward minimality — the right order, rather than
a bottleneck that strangles the latent before it has learned anything. So the encoder's loss is an
ELBO-flavored pair: the reward-prediction reconstruction term plus
the KL, `L_enc = λ·KL(q(z|c) ‖ N(0,I)) + reward_pred_loss`. The reward decoder and the GRU encoder train
together under this loss; the decoder is used only at training time to shape `z` and is dropped at
evaluation, where `adapt` just runs the GRU to get the belief and the policy acts on it.

Here I have to be careful about which version of this reconstruction I actually implement, because the
clean formulation is more elaborate than what this harness can carry, and I should match the harness rather
than import machinery it cannot support. The clean version reconstructs the *whole* trajectory — past and
future — from a latent inferred off each *prefix*, summing the ELBO over every context length with each
KL's prior set to the *previous* posterior, so inference becomes an online Bayes filter and the latent is
forced to be a task descriptor that generalizes to unseen states. I tried the per-prefix, partial-belief
version of that here and it does not converge on `cheetah-vel`, and I can see why from the structure of the
sum. A per-prefix ELBO weights all context lengths `k = 1 … T` roughly equally, but a `z` inferred from
only the first few transitions — where the target velocity is barely observable from a single `(s, a)` —
cannot predict rewards at all, so every short-prefix term is essentially ungradientable noise. At least
half the prefixes have fewer than `T/2` transitions, and on cheetah those carry almost no velocity signal,
so more than half of the ELBO terms are near-noise and the encoder stalls: the usable gradient from the
long, well-informed prefixes is drowned out by the mass of hopeless short ones. So I fall back to the
design that actually trains under this budget: a *single* posterior `z` read off the last GRU step (not a
sum over prefixes, so the decoder is always working from the best-informed belief, the one with all `T`
transitions folded in), the KL taken to the fixed unit Gaussian (not chained to the previous posterior),
and the reward decoder asked to predict the rewards of the *SAC training batch* under that single `z`,
rather than reconstruct a held-out future of the encoder's own trajectory. I also drop the transition
decoder entirely — only the reward head — because in these families the reward carries the task identity
and the transition reconstruction is the expensive, less-informative half. So what I am keeping from the
clean idea is the recurrent ordered encoder, the stochastic latent with a KL bottleneck, and a
reward-reconstruction auxiliary loss; what I am dropping, because the harness and the budget cannot carry
it, is the per-prefix summed ELBO, the future reconstruction, the belief-chaining prior, and the transition
head. The reduced version still does the one thing the previous rung could not: it forces `z` to encode the
reward structure of the task, sequentially.

One more structural point, and it is the same one the previous rung had: how the encoder relates to the
value gradients. The clean formulation detaches the latent from the RL loss and trains the VAE with its own
optimizer and buffer, for speed and to stop the two objectives interfering — and conditions the policy on
the *distribution* `(μ, logσ²)` rather than a sample, which is what gives Bayes-optimal-style behavior
instead of posterior sampling. This harness does neither, and I follow the harness: the policy conditions
on a *sampled* `z` (the same SAC-on-`z` interface the scaffold provides), and the encoder optimizer is
stepped together with the critic, so the encoder receives the Bellman gradient in addition to its own
reconstruction-plus-KL loss (the `z` fed to the Q-heads is not detached; it is detached only in the value
and policy losses). So the encoder here is shaped by three signals at once — reward reconstruction, the KL
bottleneck, and the critic — not by an isolated VAE. That is more entangled than the clean version, and I
flag it as the thing I would revisit, but it is what the scaffold's update structure supports, and it is a
strict enrichment over the previous rung's geometry-plus-critic signal: I have *added* the reward-prediction
term and swapped the bag-of-transitions encoder for a sequential one. The full scaffold module is in the
answer.

Now the expectations against the FOCAL numbers I opened with. The reward-prediction signal and the
sequential encoder are aimed at the two failures. On `cheetah-vel` I expect the encoder to *firm up*:
reconstructing the velocity reward should force a consistent target-velocity representation across seeds,
so the −108 tail seed disappears and the coefficient of variation falls from ~22% toward the single-digit
range point-robot shows; and since the two good seeds already average −72.9, removing the tail should pull
the mean up toward there. On `sparse-point-robot` I expect the reward decoder to rescue the dead seeds —
forcing `z` to predict where the +1 lives should turn the two 0.0 seeds nonzero and lift the mean well
above 0.30, a success only if it clears the low-single-digits with no seed stuck at exactly zero. The risk
is `point-robot`: FOCAL was already healthy at −13.9, and the heavier three-signal encoder (four times the
parameters, same ~20-iteration budget) could overfit or train slower, so it might land a little worse;
against that its sequential readout is strictly more expressive, so I watch the direction rather than
assume one. The honest open question is `sparse-point-robot` in absolute terms: even with a reward decoder
this rung has no *exploration* mechanism — a single stochastic `z` is sampled once, and the only
within-episode randomness left is undirected per-step action noise. So it may *sharpen task inference*
without fixing the deeper "you have to reach the goal once to learn anything" problem: if sparse improves
but stays low with a wide luck-of-the-rollout spread, then a single sampled `z` plus undirected noise still
gives no directed way to reach the rare reward — the crack the climb from here has to answer to.
