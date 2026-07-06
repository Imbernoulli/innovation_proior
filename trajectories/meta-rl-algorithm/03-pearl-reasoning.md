The recurrent reward-decoder encoder did exactly the two things I built it to do, and the numbers show
both — and they also show the one thing it could not do. On `point-robot` it improved on FOCAL's −13.9 to
−10.1, tight across seeds (−9.78, −11.00, −9.45): the sequential readout is strictly more expressive and the
dense low-dim task rewards that, a clean 27% gain. On `cheetah-vel` it firmed up the spread I worried about
— −76.6, −89.7, −85.4, no −108 outlier — so the reward-reconstruction signal did force a more consistent
target-velocity representation. But look at *how* it firmed up. The seed standard deviation fell from
FOCAL's ~18 to about 5.5, a 3.3× reduction, and the coefficient of variation from ~22% to ~6.5% — the tail
seed is gone, exactly as predicted — yet the *mean* barely moved, −83.9 against −84.7, about one percent.
That combination is the informative one: firming up the variance bought me almost nothing on the average,
which means the average was never being held down by the unreliable tail. It was being held down by a
ceiling that a *single-`z`* encoder, mean or recurrent, simply hits under this budget. Reconstructing
reward magnitudes pins the representation, but the pinned representation still tops out around −84. So the
cheetah lesson is not "make `z` firmer" — I already did — it is "make `z` a different kind of object,"
because firmness was not the binding constraint.

And on `sparse-point-robot` it rescued the dead seeds exactly as predicted: the two 0.0 seeds became 1.40
and 1.17, the third 3.39, lifting the mean from 0.30 to 1.99, a 6.6× jump. So the reward decoder works:
forcing `z` to predict where the +1 lives turned an encoder that saw no contrast into one that finds the
goal *some* of the time. But look harder at sparse, because that is where the ceiling is. 1.99 mean with a
best seed of only 3.39 is still low in absolute terms, and the spread is the tell in a different way than
cheetah's was: 1.17 to 3.39 is a factor of ~2.9 between worst and best seed, wide, on a task where you
have to *reach* the goal before you can learn anything. If task inference were the bottleneck the seeds
would cluster once the decoder had done its work; a spread this wide on a must-reach-goal task reads as
luck-of-the-rollout — whether a given seed's exploration happened to stumble onto the reward enough times
to learn from it. I named this risk at the end of the last rung and the data confirms it: the single
stochastic `z` *sharpened task inference* (the reward decoder gave it something to encode) but it did *not*
give the agent an *exploration* mechanism. The latent is sampled once, fed to the policy, and the only
randomness the policy then has within an episode is its per-step action noise — which is time-invariant and
undirected, so it jitters around and never *commits* to going somewhere to check whether the goal is there.
On a sparse task you have to reach the goal at least once to get any signal at all, and undirected jitter
across a half-circle of possible goals will not reliably do that in this budget. So the two families that
matter both point the same way: cheetah says the single-`z` object has a ceiling that firmness cannot lift,
and sparse says the missing thing is directed, committed exploration. That is the diagnosis that sets up
this rung.

Before I commit to a mechanism I should lay the real options on the table, because "add exploration" has
more than one honest reading. One option is to bolt an intrinsic-reward bonus onto the reward stream — a
curiosity or count-based term that pays the agent for visiting novel states. Walk it a few steps and it
does not fit: it drives *undirected* coverage of the state space, not coverage of the *task hypotheses*, so
on a half-circle of goals it would wander broadly rather than sweeping toward candidate goal locations;
it adds a per-environment shaping coefficient that is fragile to tune under a twenty-iteration budget; and
it does not sit cleanly on the frozen SAC-on-`z` substrate, which conditions on a task variable, not on a
modified reward. A second option is the clean Bayes-optimal reading: keep the recurrent encoder but
condition the policy on the *distribution* `(μ, σ)` and let it act optimally under uncertainty. But the
scaffold conditions on a *sampled* `z` vector, not a belief, so this means changing the value heads' input
contract; more to the point, a policy that hedges over the current belief every step is still doing
*per-step* averaging, not the *episode-length commitment* sparse needs, and I already found at the last
rung that the belief-chaining version of this does not even train under the budget. That leaves the third
option, and it is the one that turns the belief's *uncertainty* into the exploration mechanism directly
rather than adding one alongside it. So let me rebuild the inference side around uncertainty as the
first-class object.

The mechanism I am reaching for is posterior sampling — classical Bayesian RL's trick: keep a posterior
over which MDP you are in, sample one MDP-hypothesis, act optimally for it for a whole episode (temporally
extended, coherent exploration, not per-step jitter), observe, update the posterior, sample again. As the
belief narrows, behavior shifts from exploring to exploiting on its own. That is precisely the "commit to a
hypothesis and pursue it" I just said sparse needs — sample a goal on the half-circle, drive straight to
it for the episode, and if it was wrong the next sample tries elsewhere — and it only works if `z` is
*probabilistic* and held *fixed for the episode*. A point estimate has no hypothesis to commit to; the
previous rung read out a Gaussian but used it as a point estimate dressed up — sample once, condition, done
— and trained it with a reward-reconstruction loss that shaped *what* `z` encodes without making the
*uncertainty* itself meaningful or usable. I want the opposite: a genuine *belief* over the task, a
posterior `q(z|c)` whose variance means "how unsure am I," and an exploration mechanism that exploits
exactly that variance.

For that I need three things to line up: the posterior must be a real probabilistic object, it must be
permutation-invariant over the context, and it must sharpen with evidence the way a Bayesian belief does.
Start with how to fuse the context into a belief. Each transition `c_n` is, by the Markov property, an
independent sample of the same reward and dynamics — so the context is an unordered *set*, and the belief
given all of them is proportional to the *product* of per-transition factors. I drop the GRU here, and
that is a deliberate reversal I owe an argument, because I added recurrence at the last rung on purpose. The
resolution is that the two rungs are optimizing different things and the reversal is not a contradiction:
recurrence bought me a way to read a *trend*, but what the trend actually was, on cheetah, was *more
evidence accumulated over the sequence* — and a probabilistic belief that sharpens correctly wants that
same evidence *fused as exchangeable samples*, not sequentially summarized. The Markov property says the
transitions are exchangeable given the task, so imposing an order on them (as a GRU must) is imposing
spurious structure on evidence that has none; the product fusion accumulates exactly the same evidence
while respecting the exchangeability and, crucially, carrying a *calibrated* variance the GRU's readout
never had. So I am not giving up cheetah's evidence-accumulation by dropping the GRU — I am getting it in
the form that also gives me the uncertainty I now need. Let each transition emit a Gaussian factor
`N(μ_n, σ_n²)` — its own little vote on the task with its own spread — and fuse them by the product of
Gaussians. The product is itself Gaussian with a closed form: precisions add, `1/σ² = Σ_n 1/σ_n²`, and the
mean is the precision-weighted average `μ = σ²·Σ_n μ_n/σ_n²`.

Read what that closed form means, and check it against cases I already know the answer to, because a
formula I can't sanity-check is a formula I don't trust. Take `N` equally-confident factors, each with
`σ_n² = 1`. Then `1/σ² = Σ_n 1 = N`, so `σ² = 1/N` and `μ = (1/N)Σ_n μ_n` — the belief mean is just the
sample mean of the factor means and its variance shrinks as `1/N`. That is *exactly* the Bayesian posterior
for a Gaussian mean under `N` unit-variance observations, the textbook result, so the product fusion is not
some ad-hoc pooling — it is the correct filter in the case I can verify by hand. Numerically, `N = 64`
context transitions take the belief std from the prior's `1` down to `1/√64 = 0.125`, an eight-fold
sharpening, and every extra transition tightens it further: the belief *sharpens with evidence*, which is
the whole point. Now the asymmetric case, which is where sparse gets rescued: let one transition be highly
diagnostic, `σ_n² = 0.01`, and the rest uninformative, `σ² = 1`. The precision sum is dominated by
`1/0.01 = 100`, swamping the ~63 unit contributions, so `μ ≈ μ` of the confident factor and `σ² ≈ 0.01` —
a single confident vote dominates the belief. On sparse the confident vote is precisely the rare transition
where the +1 fired: the moment the agent grazes the goal once, that transition's factor pins the posterior
onto the goal location, and the many zero-reward transitions barely move it. That is the mechanism that
makes the rare reward *count*, and it is a property the mean and the GRU both lacked — they weighted every
transition the same. It is worth being explicit that a naive alternative — emit per-transition means and
just *average* them — would not do this: an unweighted average of 64 factor means dilutes the one
diagnostic vote by a factor of 64, drowning the +1's information in the sea of zero-reward transitions.
The precision weighting is the whole difference: it is what lets a confident factor speak loudly and an
uncertain one stay quiet, and it is inseparable from representing the belief as a *distribution* with a
variance rather than as a point. So the probabilistic object is not decoration; it is what the sparse task
was missing. And the product is symmetric in its factors, so the encoder is permutation-invariant
by construction. So the encoder MLP now outputs, per transition, a mean and a (pre-softplus) variance —
`2·latent_dim` outputs — and the agent fuses them into `(z_means, z_vars)` and samples `z` by
reparameterization. Let me confirm the shapes close: the context is `(num_tasks, N, context_dim)`, the
encoder maps the last axis to `2·latent_dim` giving `(num_tasks, N, 10)`, I split into `mu` and softplus'd
`sigma²` each `(num_tasks, N, 5)`, and for each task I fuse the `N` factors along the transition axis down
to a single `(5,)` mean and variance, stacking back to `(num_tasks, 5)`; a clamp at `σ² ≥ 10⁻⁷` keeps a
degenerate zero-variance factor from sending the precision to infinity. The dimensions line up.

Now the training signal for this encoder, and here I make the second reversal from the last rung. I trained
the GRU with a reward-reconstruction loss; that was the move that gave `z` content. But for a posterior I
want a different regularizer — the information bottleneck. Put a KL from the belief to a unit-Gaussian prior,
`β·KL(q(z|c) ‖ N(0,I))`, into the objective. This is doing double duty. Read as a variational quantity, the
expected KL is an upper bound on the mutual information `I(z;c)` (with the prior playing the role of the
variational marginal), so pushing it down *caps* how much the latent can encode about the context — a
genuine information bottleneck that squeezes `z` to the minimal task-relevant statistic. That matters
enormously under this tiny budget: there are only 30 to 40 training tasks per family, few enough that an
unconstrained `z` could simply memorize a lookup from training-task index to an arbitrary code, which would
fit the training tasks perfectly and transfer to the held-out tasks not at all; the bottleneck forecloses
that by forcing `z` to keep only bits that generalize. The prior `N(0,I)` is also the thing I sample from
at the very start of an episode when I have no context — pure prior-conditioned exploration. So the KL
gives me the bottleneck *and* the exploration prior in one term, and I can trace the sparse story
end-to-end through it: at episode start, no context, so `z ∼ N(0,I)` is a random hypothesis; the policy
drives coherently toward the point on the half-circle that hypothesis encodes; if that is near the true
goal the +1 fires, its confident factor dominates the product and pins the posterior (the asymmetric case
above); subsequent episodes re-sample from the now-sharpened belief and converge onto the goal; if it was
wrong, the next prior sample simply tries elsewhere. That is directed, committed, self-correcting
exploration, and it falls out of the belief plus the prior with no separate bonus. This also tells me how
to set `β` per family, rather than leaving it a single global constant. On the dense envs the belief gets
informative context every step, so a light bottleneck (`β = 0.1`) is enough to keep `z` from memorizing
and I do not want to over-tax a latent that has real reward structure to carry. On sparse the exploration
prior is doing the load-bearing work, and I want early-episode behavior to stay *genuinely* prior-sampled —
broad hypothesis coverage across the half-circle — rather than collapsing prematurely onto a half-learned
belief built from a handful of all-zero transitions; a stronger pull toward the prior (`β = 1.0`) holds the
belief wide until a real +1 arrives to sharpen it, which is exactly when I want the collapse to happen. So
the same KL term is tuned to serve the bottleneck on the dense envs and the exploration prior on the sparse
one. And what makes `z`
*good for control*, beyond being minimal? I train the encoder from the *critic*: the Bellman gradient of
the Q-loss flows into the encoder, so `z` is shaped to be exactly the task summary that makes the value
function accurate — which is what the policy actually needs to act well, more directly than reward
reconstruction (which spends capacity predicting reward magnitudes the controller does not directly
consume, and which — the cheetah ceiling suggests — is part of why the single-`z` mean stayed stuck). So
`z` carries gradients into the encoder through the Q-loss and the KL, and is detached in the value and
policy losses — the same structure the previous rungs used, but now the encoder's *own* loss is the KL
bottleneck rather than a reconstruction term, and the reward decoder is gone entirely.

There is one more piece, and it is the part that earns the "off-policy meta-RL" name and that I have been
quietly relying on without isolating: *which data trains which part*. Meta-learning needs the
adaptation-data distribution to match between meta-train and meta-test — at test time the encoder infers from
on-policy exploration data, so during training I must not feed it ancient off-policy transitions or I train
it on the wrong distribution. But that matching constraint binds *only* on the encoder's input, not on the
control updates: the actor and critic just need good value estimates and do not care whether the transitions
came from a stale policy. So I decouple the samplers. The actor and critic learn from decorrelated minibatches
(batch 256) drawn from the *entire* replay buffer, where all the sample-efficiency lives, while the encoder's
context (embedding batch ~64) is drawn by a *separate* sampler from *recently collected* data — the harness's
separate encoder buffer, cleared per task each iteration so the context stays recent — and from a batch
distinct from the RL batch. The per-iteration clear is the operational teeth: it guarantees the encoder only
ever sees this iteration's collection, never stale history, so its training input matches the on-policy
distribution it will face at test. The collection itself mixes prior-conditioned exploration (sample `z`
from `N(0,I)`, gather data before the agent knows the task) with posterior-conditioned exploration (re-infer
`q(z|c)` after some context accumulates), so the encoder sees both prior- and posterior-conditioned data,
exactly the distribution it faces at test. This decoupling is what was implicitly carrying the previous
rungs and what I now make load-bearing: it is the operational form of "the encoder's data ≠ the policy's
data," and it is what lets off-policy efficiency coexist with the distribution match the encoder needs.

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
or above ~9 would be the signature of an agent that actually commits to sweeping the half-circle, since only
an agent that reliably reaches the goal early and then exploits can rack up that many in-goal steps). The
information-bottleneck KL should also help generalization to held-out tasks under the tiny budget, so I
expect `cheetah-vel` to *improve*, and here I can be sharper than "firm up" because VariBAD already told me
firming up is not enough: its variance cut of 3.3× moved the mean by one percent, so the ceiling was never
the variance. Dropping the reconstruction term in favor of a critic-trained, bottlenecked `z` attacks the
ceiling itself — the representation is now shaped by what the value function needs rather than by reward
magnitudes it does not — so I expect a genuine *mean* step, into the low-to-mid −60s, not just another
variance reduction around −84. The honest risk is `point-robot`: VariBAD's recurrent encoder won it (−10.1),
and a permutation-invariant product-of-Gaussians belief discards the sequential structure that helped there,
so I would not be surprised if this rung *regresses* on the easy dense task — say into the −14 to −16 range
— while winning the two hard ones. That trade is the explicit bet of this rung: give up a little on the
easiest, low-dimensional family where any encoder suffices, to win decisively on the two families that
actually exercise meta-RL — high-dimensional encoding and sparse-reward exploration — because the
probabilistic belief and the posterior-sampling it enables are aimed precisely at those. If the measured
ordering is "loses point-robot, wins cheetah-vel and sparse-point-robot clearly," the climb has landed where
the design said it would.
