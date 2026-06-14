The deterministic mean encoder landed exactly where its premise predicted it would, and the numbers say
which premise broke. On the dense low-dimensional `point-robot` it is fine — mean −13.9, tight across seeds
(−13.18, −14.17, −14.33) — because there the goal really is readable from a handful of dense transitions
and "one transition reveals the task" roughly holds. But `cheetah-vel` is the worst of the three baselines
so far at −84.7, and the per-seed spread is the tell: −64.2, −81.6, and a −108.4 seed that is more than a
third worse than the best. That is not noise around a good solution; that is an encoder that sometimes
locks onto a usable target-velocity representation and sometimes does not, with nothing to make it
reliable. And `sparse-point-robot` is the sharpest failure: 0.30 mean, with two of three seeds at *exactly*
0.0 and the third at 0.90 — under this benchmark's own convention a 0 means the goal was never reached in
the budget. So two of three sparse seeds never found the goal at all, and the one that did barely registered.

Both failures point at the same two holes in the mean encoder, and I can name them precisely now that I
have the numbers. First, the mean over per-transition embeddings throws away *order*. On `cheetah-vel` the
target velocity is only weakly visible in any single `(s, a, r)` — it is the *trend* of the reward as speed
changes across a sequence that pins it down — and a permutation-invariant bag of transitions has no way to
read a trend. That is why the cheetah encoding is unreliable: a contrastive loss that only enforces "these
blocks are the same task, those are different" never forces `z` to encode the underlying target the policy
needs, and a mean cannot recover it from the sequence even if it wanted to. Second, and this is what kills
sparse, the contrastive signal needs *contrast*, and on `sparse-point-robot` almost every context
transition has reward 0 and is therefore nearly identical across tasks — there is almost nothing for the
distance-metric loss to separate tasks *by*. The two dead sparse seeds are exactly that: an encoder fed a
context that carries no task-distinguishing information, producing a `z` the policy cannot act on, and no
mechanism anywhere to *go find* the rare reward in the first place.

So the diagnosis is concrete: I need an encoder that (a) respects the *sequence* of experience instead of
treating context as an unordered bag, and (b) is shaped by a signal richer than "same/different task" —
something that forces `z` to encode the actual reward structure of the task, which is what would have
saved both cheetah and sparse. Let me build toward exactly that.

Take the sequence problem first, because it changes the primitive. I argued for permutation-invariance at
the previous rung from "each transition independently reveals the task," but the cheetah failure shows the
premise is too strong here: when the task is only weakly visible per transition, *what I should do next*
depends on the whole sequence so far, not on an order-free set. The right primitive for online, ordered
inference is recurrence. So I replace the mean encoder with a recurrent one: embed each transition `(o, a,
r)` with small feature extractors, run a GRU whose hidden state carries the running summary of experience,
and read out the task latent from the hidden state. One GRU step is one belief update, folding in one
transition at a time, which is exactly the online structure the rollout protocol needs — `update_context`
appends a transition, `adapt` runs the GRU over the accumulated context and reads off `z` from the final
hidden state. The harness's `infer_posterior` gets the same treatment: reset the GRU, run it over the
sampled context block, take the last step's readout. Crucially, on this benchmark I should keep the context
the GRU sees *chronological and within a single trajectory* — if I let it stitch together independent
trajectories without resetting the hidden state at episode boundaries, the running summary gets polluted by
discontinuities, so I cap the context length at the path length so each task's context comes from one
coherent rollout. That is a harness-specific care the mean encoder never needed, because a mean does not
care about order; a GRU does.

Now the richer training signal, and this is where I have to think about what objective actually forces `z`
to carry reward structure. The previous encoder was trained by a distance-metric loss — pure geometry,
"keep tasks apart" — which is necessary (the continuity argument from the last rung still holds: distinct
tasks must be separable for the value functions to exist) but not sufficient, because geometry alone never
says *what* about the task `z` should encode. What I want is a *generative* signal: make `z` good at
predicting the thing that differs across tasks. In this family the reward function is what differs, so I
attach a **reward decoder** — a small network that, given a transition `(s, a)` and the task latent `z`,
predicts the reward — and train the encoder so that the latent it produces lets the decoder reconstruct
rewards accurately. This is the move that would have saved both failures: on `cheetah-vel`, reconstructing
the velocity-matching reward *forces* `z` to encode the target velocity, the very quantity the mean encoder
could not read off the sequence; on `sparse-point-robot`, predicting the reward forces `z` to encode where
the +1 lives, which is precisely the task-distinguishing bit the contrast had nothing to grab.

Let me make the latent stochastic while I am at it, because the recurrence readout naturally produces a
distribution and I want the regularization that comes with it. The GRU reads out a Gaussian posterior over
`z` — a mean and a log-variance — and I sample `z` by reparameterization so gradients flow. The
stochasticity buys two things. It lets me put an information bottleneck on the latent in the form of a KL
to a unit-Gaussian prior, `KL(q(z|c) ‖ N(0,I)) = −½ Σ(1 + logσ² − μ² − σ²)`, which squeezes `z` to the
minimal reward-relevant content and discourages it from memorizing training-task idiosyncrasies under the
short budget. And it is the natural object to put a variational reconstruction objective on. So the
encoder's loss is an ELBO-flavored pair: the reward-prediction reconstruction term plus the KL,
`L_enc = λ·KL(q(z|c) ‖ N(0,I)) + reward_pred_loss`. The reward decoder and the GRU encoder train together
under this loss; the decoder is used only at training time to shape `z` and is dropped at evaluation, where
`adapt` just runs the GRU to get the belief and the policy acts on it.

Here I have to be careful about which version of this reconstruction I actually implement, because the
clean formulation is more elaborate than what this harness can carry, and I should match the harness rather
than import machinery it cannot support. The clean version reconstructs the *whole* trajectory — past and
future — from a latent inferred off each *prefix*, summing the ELBO over every context length with each
KL's prior set to the *previous* posterior, so inference becomes an online Bayes filter and the latent is
forced to be a task descriptor that generalizes to unseen states. I tried the per-prefix, partial-belief
version of that here and it does not converge on `cheetah-vel`: a `z` inferred from only the first few
context transitions, where the target velocity is barely observable from a single `(s, a)`, cannot predict
rewards well enough for the per-step ELBO to give a usable gradient, and the encoder stalls. So I fall back
to the design that actually trains under this budget: a *single* posterior `z` read off the last GRU step
(not a sum over prefixes), the KL taken to the fixed unit Gaussian (not chained to the previous posterior),
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

Now the falsifiable expectations against the FOCAL numbers I opened with. The reward-prediction signal and
the sequential encoder are aimed squarely at the two failures, so I expect `cheetah-vel` to *firm up*: the
−108 seed should disappear and the spread should tighten, because reconstructing the velocity reward forces
a consistent target-velocity representation regardless of which seed I am on — I expect the cheetah mean to
move from −84.7 toward the low-to-mid −80s with much smaller per-seed variance. On `sparse-point-robot` I
expect the reward decoder to rescue the dead seeds: forcing `z` to predict where the +1 lives should turn
the two 0.0 seeds into nonzero returns and lift the mean well above FOCAL's 0.30 — I would call this rung a
success on sparse only if the mean clears ~1.5 and no longer has seeds stuck at exactly zero. The risk I am
carrying is `point-robot`: FOCAL was already healthy there at −13.9, and a heavier, recurrent, three-signal
encoder could overfit or simply be slower to train under the tiny budget, so I would not be shocked if the
GRU *helps* point-robot too (its sequential readout is strictly more expressive) and I will watch whether
it lands at or below −12. The honest open question is `sparse-point-robot` in absolute terms: even with a
reward decoder, this rung still has no *exploration* mechanism — nothing that makes the agent commit to
going somewhere to find the rare reward — so a single stochastic `z` may sharpen task inference without
fixing the deeper "you have to reach the goal once to learn anything" problem. If sparse improves but stays
low, that is the diagnosis for the next rung already written down: I will need a latent whose *uncertainty*
the agent can act on — a probabilistic belief that supports committing to a task hypothesis for a whole
episode — which is exactly the exploration mechanism a single point-estimate `z` cannot provide.
