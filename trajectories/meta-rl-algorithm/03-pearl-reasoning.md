The recurrent reward-decoder encoder did exactly the two things I built it to do, and the numbers show
both — and they also show the one thing it could not do. On `point-robot` it improved on FOCAL's −13.9 to
−10.1, tight across seeds (−9.78, −11.00, −9.45): the sequential readout is strictly more expressive and the
dense low-dim task rewards that. On `cheetah-vel` it firmed up the spread I worried about — −76.6, −89.7,
−85.4, no −108 outlier — so the reward-reconstruction signal did force a more consistent target-velocity
representation, though the mean (−83.9) barely moved from FOCAL's −84.7 because the budget caps how good any
single-`z` encoder gets here. And on `sparse-point-robot` it rescued the dead seeds exactly as predicted:
the two 0.0 seeds became 1.40 and 1.17, the third 3.39, lifting the mean from 0.30 to 1.99. So the reward
decoder works: forcing `z` to predict where the +1 lives turned an encoder that saw no contrast into one
that finds the goal *some* of the time.

But look harder at sparse, because that is where the ceiling is. 1.99 mean with a best seed of only 3.39 is
still low in absolute terms — the agent reaches the goal occasionally, not reliably, and the spread (1.17 to
3.39) says it is luck-of-the-rollout whether a given seed's exploration happens to stumble onto the reward
enough times to learn from it. I named this risk at the end of the last rung and the data confirms it: the
single stochastic `z` *sharpened task inference* (the reward decoder gave it something to encode) but it did
*not* give the agent an *exploration* mechanism. The latent is sampled, fed to the policy, and the only
randomness the policy then has within an episode is its per-step action noise — which is time-invariant and
undirected, so it jitters around and never *commits* to going somewhere to check whether the goal is there.
On a sparse task you have to reach the goal at least once to get any signal at all, and undirected jitter
across a half-circle of possible goals will not reliably do that in this budget. That is the deeper problem
the reward decoder cannot touch, and it is the diagnosis that sets up this rung: I need a latent whose
*uncertainty* the agent can act on — a belief it can sample a *hypothesis* from and pursue coherently for a
whole episode.

So let me rebuild the inference side around uncertainty as the first-class object. The previous encoder read
out a Gaussian but used it as a point estimate dressed up — sample once, condition, done — and trained it
with a reward-reconstruction loss that shaped *what* `z` encodes without making the *uncertainty* itself
meaningful or usable. I want the opposite: a genuine *belief* over the task, a posterior `q(z|c)` whose
variance means "how unsure am I," and an exploration mechanism that exploits exactly that variance. The
mechanism I am reaching for is posterior sampling — classical Bayesian RL's trick: keep a posterior over
which MDP you are in, sample one MDP-hypothesis, act optimally for it for a whole episode (temporally
extended, coherent exploration, not per-step jitter), observe, update the posterior, sample again. As the
belief narrows, behavior shifts from exploring to exploiting on its own. That is precisely the "commit to a
hypothesis and pursue it" I just said sparse needs — and it only works if `z` is *probabilistic* and held
*fixed for the episode*. A point estimate has no hypothesis to commit to.

For that I need three things to line up: the posterior must be a real probabilistic object, it must be
permutation-invariant over the context (back to that argument in a moment), and it must sharpen with
evidence the way a Bayesian belief does. Start with how to fuse the context into a belief. Each transition
`c_n` is, by the Markov property, an independent sample of the same reward and dynamics — so the context is
an unordered *set*, and the belief given all of them is proportional to the *product* of per-transition
factors. (I drop the GRU here, and that is a deliberate reversal: I added recurrence at the last rung to read
a *trend*, but a probabilistic belief that sharpens correctly wants independent evidence *fused*, not
sequentially summarized, and the product fusion is exactly that.) Let each transition emit a Gaussian factor
`N(μ_n, σ_n²)` — its own little vote on the task with its own spread — and fuse them by the product of
Gaussians. The product is itself Gaussian with a closed form: precisions add, `1/σ² = Σ_n 1/σ_n²`, and the
mean is the precision-weighted average `μ = σ²·Σ_n μ_n/σ_n²`. Read what that means and it is exactly the
belief I want: confident factors (small `σ_n²`) pull harder, and every additional transition *increases* the
precision, i.e. *shrinks* the variance — the belief sharpens with evidence, which is Bayesian filtering
falling straight out of the product. And the product is symmetric in its factors, so the encoder is
permutation-invariant by construction. So the encoder MLP now outputs, per transition, a mean and a
(pre-softplus) variance — `2·latent_dim` outputs — and the agent fuses them into `(z_means, z_vars)` and
samples `z` by reparameterization.

Now the training signal for this encoder, and here I make the second reversal from the last rung. I trained
the GRU with a reward-reconstruction loss; that was the move that gave `z` content. But for a posterior I
want a different regularizer — the information bottleneck. Put a KL from the belief to a unit-Gaussian prior,
`β·KL(q(z|c) ‖ N(0,I))`, into the objective. This is doing double duty: it is the VAE-style regularizer that
keeps the belief near the prior, *and*, read as a variational bound on the mutual information `I(z;c)`, it is
an information bottleneck that squeezes `z` to the minimal task-relevant statistic — which matters enormously
under this tiny budget, where a fat unconstrained `z` would memorize training-task idiosyncrasies and not
transfer to the held-out tasks. The prior `N(0,I)` is also the thing I sample from at the very start of an
episode when I have no context — pure prior-conditioned exploration. So the KL gives me the bottleneck *and*
the exploration prior in one term. And what makes `z` *good for control*, beyond being minimal? I train the
encoder from the *critic*: the Bellman gradient of the Q-loss flows into the encoder, so `z` is shaped to be
exactly the task summary that makes the value function accurate — which is what the policy actually needs to
act well, more directly than reward reconstruction (which spends capacity predicting reward magnitudes the
controller does not directly consume). So `z` carries gradients into the encoder through the Q-loss and the
KL, and is detached in the value and policy losses — the same structure the previous rungs used, but now the
encoder's *own* loss is the KL bottleneck rather than a reconstruction term, and the reward decoder is gone
entirely.

There is one more piece, and it is the part that earns the "off-policy meta-RL" name and that I have been
quietly relying on without isolating: *which data trains which part*. Meta-learning needs the
adaptation-data distribution to match between meta-train and meta-test — at test time the encoder infers from
on-policy exploration data, so during training I must not feed it ancient off-policy transitions or I train
it on the wrong distribution. But that matching constraint binds *only* on the encoder's input, not on the
control updates: the actor and critic just need good value estimates and do not care whether the transitions
came from a stale policy. So I decouple the samplers. The actor and critic learn from decorrelated minibatches
drawn from the *entire* replay buffer (where all the sample-efficiency lives), while the encoder's context is
drawn by a *separate* sampler from *recently collected* data — the harness's separate encoder buffer, cleared
per task each iteration so the context stays recent — and from a batch distinct from the RL batch. The
collection itself mixes prior-conditioned exploration (sample `z` from `N(0,I)`, gather data before the agent
knows the task) with posterior-conditioned exploration (re-infer `q(z|c)` after some context accumulates),
so the encoder sees both prior- and posterior-conditioned data, exactly the distribution it faces at test.
This decoupling is what was implicitly carrying the previous rungs and what I now make load-bearing: it is
the operational form of "the encoder's data ≠ the policy's data," and it is what lets off-policy efficiency
coexist with the distribution match the encoder needs.

Putting it together in this task's edit surface: the agent holds the permutation-invariant product-of-Gaussians
encoder (MLP → `2·latent_dim`, softplus on the variance, fuse, reparameterized sample) and the scaffold's
`z`-conditioned SAC heads; `infer_posterior` fuses the context into `(z_means, z_vars)` and samples; `adapt`
runs it on accumulated context; `compute_kl_div` is the KL to `N(0,I)`. The algorithm's meta-gradient step
samples context from the recent encoder buffer (distinct from the off-policy RL batch from the full buffer),
backprops the KL into the encoder, backprops the Q-loss into the Q-nets *and* the encoder and steps them
together, then the value loss and policy loss on detached `z`, with a soft target-value update. Test-time
evaluation is the posterior-sampling loop the protocol already implements: clear context, sample `z` from the
prior, roll out, accumulate, re-infer, repeat, then act deterministically with the inferred `z`. This is a
faithful fill — unlike the previous two rungs I am not dropping or reducing any of the method's machinery,
because all of it (the probabilistic belief, the bottleneck, the posterior-sampling exploration, the
decoupled samplers) fits the harness directly. The full scaffold module is in the answer.

Now the falsifiable expectations against the VariBAD numbers I opened with. The whole point of this rung is
the sparse task, so that is the sharp prediction: posterior sampling gives the agent temporally-extended,
hypothesis-committed exploration that undirected action jitter never could, so `sparse-point-robot` should
jump clearly above VariBAD's 1.99 — I would call this rung a real success on sparse only if the mean roughly
doubles, into the mid-single-digits, and I expect the best seed to push well past VariBAD's 3.39 (a seed near
or above ~9 would be the signature of an agent that actually commits to sweeping the half-circle). The
information-bottleneck KL should also help generalization to held-out tasks under the tiny budget, so I
expect `cheetah-vel` to *improve*, not just firm up: dropping the reconstruction term in favor of a
critic-trained, bottlenecked `z` should let the high-dimensional encoding land lower than VariBAD's −83.9 —
I would look for the low-to-mid −60s, a clear step. The honest risk is `point-robot`: VariBAD's recurrent
encoder won it (−10.1), and a permutation-invariant product-of-Gaussians belief discards the sequential
structure that helped there, so I would not be surprised if PEARL *regresses* on the easy dense task — say
into the −14 to −16 range — while winning the two hard ones. That trade is the explicit bet of this rung:
give up a little on the easiest, low-dimensional family where any encoder suffices, to win decisively on the
two families that actually exercise meta-RL — high-dimensional encoding and sparse-reward exploration —
because the probabilistic belief and the posterior-sampling it enables are aimed precisely at those. If the
measured ordering is "loses point-robot, wins cheetah-vel and sparse-point-robot clearly," the climb has
landed where the design said it would.
