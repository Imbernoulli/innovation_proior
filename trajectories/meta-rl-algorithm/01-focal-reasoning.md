The scaffold hands me a SAC actor-critic that already conditions on a task variable `z`; what it does
*not* hand me is anything that makes `z` mean something. The default `z` is a dummy zero, `infer_posterior`
and `adapt` are no-ops, and `train_iteration` takes no gradients — so the policy is the same policy on
every task and there is no meta-learning at all. The frozen substrate already narrows which family of
meta-RL I can even build. RL² would make the *policy* recurrent and train the whole recurrence with
ordinary RL on trial return — but that is an on-policy object, and this substrate reads decorrelated
minibatches out of a replay buffer, which destroys exactly the temporal credit assignment the recurrence
depends on. MAML / ProMP would meta-learn an initialization and adapt in an inner loop — but there is no
inner-loop machinery here and both of MAML's loops are on-policy. What the substrate actually exposes is a
`z`-slot on the actor and both critics plus an encoder hook (`infer_posterior`/`adapt`), so it commits me
to the *context-encoder* family: learn a map from collected transitions to a `z` that identifies the task,
trained by a meta-gradient step. I want the *simplest* such encoder that could work, to set a floor the
next rung has to beat — fewest moving parts in the inference path, and a training signal *decoupled* from
the value learning so I can reason about it in isolation.

Start from what the latent variable even is. The tasks in this family share the state and action space
and differ only in their reward (and, on some families, transition) functions. Take the clean case of
deterministic dynamics — which is what these continuous-control families essentially are — and suppose
the tasks satisfy what I'll call task-transition correspondence: two tasks agree on transition and
reward everywhere if and only if they are the same task. Then the pair `(P, R)` *identifies* the task,
and under determinism each `(s, a)` has a unique outcome `(s', r)`, so every task is a function
`f_T(s, a) = (s', r)`. The whole family `{f_T}` lives on the transition space `S × A × S × ℝ`, which is
exactly the context space the harness samples from. I can be concrete about the sizes the harness hands
me: on `point-robot` and `sparse-point-robot` a transition `(o, a, r)` is `2 + 2 + 1 = 5` numbers, and on
`cheetah-vel` it is `20 + 6 + 1 = 27` — so "embed the function `f_T`" means map points of a 5- or
27-dimensional space to a `latent_dim = 5` vector. The encoder isn't inferring a fuzzy belief about a
hidden quantity; it is embedding the function `f_T` from samples of it. And here is the consequence that
makes the simple design legitimate: because `(P, R)` pin the task pointwise, a *single* transition tuple
in principle already constrains which task I'm in — `f_T` is determined by its values, and one value is a
real constraint. I do not need to integrate evidence across many transitions to slowly become confident.
Two things drop straight out. The encoder should be **permutation-invariant** — order cannot matter if
each transition independently reveals the task — and it should be **deterministic** — there is no
irreducible uncertainty to represent, because the task is recoverable, not guessed. No probabilistic
posterior, no information bottleneck, no learned belief distribution over the task. The permutation-invariant
aggregation that costs the least is the one prototypical networks use for class prototypes: embed every
transition with a shared MLP and take the **mean** of the per-transition embeddings. So the encoder is a
width-200 depth-3 MLP from one transition `(o, a, r)` to a `latent_dim`-vector, and `z` for a task is the
mean over its context rows. On `cheetah-vel` that MLP is roughly `27·200 + 200·200·(depth−1) + 200·5`
weights, on the order of `10⁵` parameters — an order of magnitude below the `net_size = 300` policy and
each critic. The encoder is deliberately the cheapest network in the agent, which is what "thinnest floor"
should mean numerically, not just rhetorically. That mean map is `infer_posterior`, and `adapt` just
calls it on the accumulated online context.

"Deterministic mean encoder" only says how I *combine* embeddings; it says nothing about what makes the
embedding *good*. If I let the encoder learn purely through the SAC critic's Bellman gradients — the way
a value-trained context encoder would — I have a problem I can see coming. So let me ask whether the
geometry of `z` even needs a direct objective, because if it doesn't I shouldn't add one. The value
function `Q(s, a, z)` is a continuous network: for close `z₁, z₂` it is *forced* to output close
Q-values. Make that quantitative — a critic with Lipschitz constant `L` in its `z`-argument can express
at most `L·δ` of a value gap between two tasks whose embeddings sit `δ` apart. Now take two genuinely
different tasks whose true Q-values at some `(s, a)` differ by `ΔQ` — built from different rewards and
dynamics, `ΔQ` is not small — but whose embeddings happen to land within `δ` of each other. Unless
`L·δ ≥ ΔQ`, the conditioned value functions for those two tasks are *unrepresentable*: the network
physically cannot output two well-separated values at two nearly-identical inputs. So the separation `δ`
in latent space is a hard budget the critic has to spend, and if the encoder lets distinct tasks'
embeddings sit near each other, control fails downstream no matter how good the actor-critic is. The
geometry of `z` is therefore not cosmetic; keeping distinct tasks far apart in latent space is a
*precondition* for the value functions to exist. I need an objective that lives on the embedding space
itself and clusters same-task transitions while separating different-task ones — and, importantly, I want
it decoupled from the value learning so I can trust it.

The textbook objective for "cluster same, separate different" is the contrastive loss: a quadratic
attractor `1{same}·‖q_i−q_j‖²` pulling same-task pairs together, and a margin hinge
`1{diff}·max(0, m−‖q_i−q_j‖)²` pushing different-task pairs apart up to distance `m`. If I just reach for
that, it degenerates — the clusters do not cleanly separate; I get blobs holding transitions from several
tasks at once. The why matters, because it dictates the fix. Two failures compound. First, the margin
hinge is a spring that acts *only within radius m* and whose force is *weakest exactly where I need it
most*: its gradient magnitude with respect to distance is `2(m−d)`, which is *bounded* (at most `2m` at
contact) and roughly flat, so a pair of embeddings sitting right on top of each other feels essentially
the same push as a pair already half-separated. At the start, all embeddings are bunched near the origin
— the latent is tanh-bounded, the encoder randomly initialized — so every pairwise distance is tiny, and
this bounded spring has no special urgency there. Second, and deeper, the attractive squared-distance
term is, algebraically, *variance maximization*. I can just do the sum on paper: for scalar embeddings,
`Σ_{i,j}(x_i−x_j)² = Σ_{i,j}(x_i² − 2x_i x_j + x_j²) = 2N Σ x_i² − 2(Σ x_i)²`, and `2N²·Var(X)
= 2N²[(1/N)Σx_i² − x̄²] = 2N Σx_i² − 2(Σx_i)²` — the two expressions are identical, so summing squared
pairwise distances *is* `2N²·Var`, per axis. Variance is a single global scalar, and many embedding
distributions share it — including the degenerate one that piles half the mass at `+1` and half at `−1`
on each axis. On the tanh-bounded interval `[−1,1]` that two-point distribution has the maximal variance
(`= 1`, since a bounded variable maximizes spread at the extremes), and with `latent_dim = 5` the corners
`{−1,+1}⁵` are `2⁵ = 32` points all at that same maximal spread. So the optimizer can cram all 40
`sparse-point-robot` training tasks onto those 32 corners — by pigeonhole it *must* collide at least eight
tasks onto shared corners — and the variance-maximizing objective reads the *same* global scalar whether
the 40 tasks land on 40 distinct points or pile onto two. It is completely indifferent to the collision.
That is precisely the merged-blob failure, made arithmetic: a global statistic cannot enforce a
per-pair separation.

So I know what I actually need: a repulsion between different-task pairs that is *strong at short range
and fades as distance grows* — the opposite profile of the margin spring, and one that cannot be
satisfied by a global statistic. The function with that profile is a *negative* power of distance:
`β / (‖q_i−q_j‖ⁿ + ε)` blows up as the distance goes to zero and relaxes once the pair is well
separated. Replacing the margin term with this gives the deep-metric-learning loss
`L = 1{same}·‖q_i−q_j‖² + 1{diff}·β/(‖q_i−q_j‖ⁿ + ε)`. Compare the two repulsion force profiles (with
`β = 1`, `ε = 10⁻³`, `m = 1`, `n = 2`; the inverse-square force is `2βd/(d²+ε)²`). At `d = 0.1`, two
clusters badly overlapping, the margin pushes with `2(1−0.1) = 1.8` while the inverse-square pushes with
`≈1653`, a thousandfold stronger exactly where I need separation most; at `d = 0.9` it is `0.2` versus
`≈2.7`; and past `d = m = 1` the margin is *dead* at `0` while the inverse-square still pushes `≈2`. The
profiles are opposite where it matters — margin weakest at overlap, inverse-power strongest there — and
the `ε` floor just caps the contact force so gradients stay finite. Physically it is a system of like
charges in a bounded box: with `n = 1` it is the Coulomb potential and the charges pile up at the corners,
and the tanh-bounded latent cube `(−1,1)^l` is the conducting box that lets the repulsion settle into a
well-separated equilibrium instead of flinging everything to infinity. I take `n = 2` — sharper than
Coulomb, concentrating effort on the closest offending pairs — which coincides with the Cauchy
graph-embedding objective; `β` matches the attractive and repulsive scales, `ε` floors the denominator.
Now the variance trick cannot game the loss: the offending pairs *inside* a merged pile are at small
distance where the `1/d²` term is screaming, so every pair of distinct clusters is forced apart while the
same-task quadratic pulls each task tight — tight clusters scattered to the extremes, the geometry the
continuity argument demanded.

Now I have to decide how this encoder objective relates to the value-learning gradients, and the harness
makes the choice for me in a way I should be explicit about. The clean version of this method, in the
setting it was conceived for, also carries a *behavior-regularization* machinery — a divergence between
the learned policy and the data's behavior policy, estimated by a dual-form-KL discriminator, folded into
the critic target and the actor — because that setting is *fully offline* (no environment interaction at
train or test) and offline bootstrapping over out-of-distribution actions diverges without a tether to
the data support. But this harness is **online**: `collect_data` interacts with the environment every
iteration, so the replay buffer is filled by the current policy and the out-of-distribution-value
problem the behavior regularizer fights does not bite here. So I drop the behavior regularizer entirely —
no discriminator, no value penalty, no adaptive `α` — and keep only the two genuinely load-bearing pieces
for this setting: the deterministic mean encoder and the inverse-power DML loss, dropped onto the
scaffold's plain SAC. The harness makes the contrastive loss convenient — `sample_context_from_buffer`
returns a per-task context block, so I split each task's context in half, mean-encode each half into two
embeddings sharing the task's label, and apply the squared distance to same-task pairs and the
inverse-power to different-task pairs across the meta-batch. With `meta_batch = 16` this gives `32`
embeddings: `16` same-task pairs attract and `~480` different-task pairs repel, so negatives outnumber
positives roughly `30 : 1` and with the `1/d²` blow-up dominate the early gradient — the right order of
operations, repulsion first flinging distinct tasks apart, then the quadratic attractor tightening each
task's two halves into a point. That distance-metric loss is the encoder's *primary* shaping signal. One
gradient-flow point where I deviate from the fully-decoupled offline version and follow the scaffold's own
update structure: the contrastive loss is backpropped into the encoder, but the encoder optimizer is
stepped *together with* the critic, and the `z` that feeds the two Q-heads is **not** detached — so the
Bellman gradient also reaches the encoder, as it would for a critic-trained context encoder (`z` is
detached only in the value and policy losses). So the encoder is shaped by the DML loss *plus* the critic;
what I keep from the clean version is the contrastive objective and the deterministic mean aggregation,
not the gradient-level isolation. The SAC update underneath is the standard twin-Q / value /
squashed-Gaussian-actor step, reward scaled with the soft target value net slowly tracked. The literal
scaffold fill is in the answer.

The exponent `n` is a real knob, and the force ratio pins it. The repulsion scales as `1/dⁿ⁺¹`, so
between the closest offending pair at `d` and the next at `2d` the force ratio is `2ⁿ⁺¹`: `4` at `n = 1`,
`8` at `n = 2`, `32` at `n = 4`. Large `n` is winner-take-all — it separates one collision at a time and
can leave a second merged pair untouched; small `n` spreads the force so thin a badly overlapping pair
barely gets more attention than a well-separated one, the margin-hinge disease again. `n = 2` gives the
closest pair clear priority (`8×` the next) without going winner-take-all, so several collisions repair in
parallel. The same-task and different-task terms act on disjoint pair-sets — the two halves of one task
feel only the quadratic attractor, distinct tasks feel only the repulsion — so each task collapses to a
point while distinct tasks are shoved apart, with no deadlock.

So the design closes: a deterministic, mean-aggregating context encoder shaped primarily by an
inverse-power distance metric loss that forces distinct task clusters to the corners of the bounded latent
cube, feeding `z` into a plain online SAC (with the critic gradient also reaching the encoder). Why start
the ladder here? Because it is the thinnest encoder that respects the one hard constraint — distinct tasks
must be separable in `z` for the value functions to exist — with a contrastive training signal I can
reason about directly, free of the probabilistic-belief apparatus that the recoverable-task argument says
is unnecessary.

Where does this crack? The whole construction rests on "a single transition reveals the task." On
`cheetah-vel`, where the target velocity is only weakly visible in any one `(s, a, r)`, a
permutation-invariant mean throws away the temporal structure that would let me read the target off a
*sequence* — and a contrastive loss that only enforces "same task / different task" never teaches `z` to
encode the underlying quantity the policy needs to set its running speed. So the cheetah representation is
unpinned: nothing forces `z` to *be* the target velocity, only to be far from other tasks' `z`, and a
bounded random-init encoder can satisfy "far from the others" in ways that carry the speed and ways that
do not. I expect cheetah to be the weak point and an *unreliable* one — mediocre returns with a wide
spread across seeds, because whether a run's encoder stumbles onto a usable target-velocity code is left
to chance. On `sparse-point-robot` the trouble is worse and self-reinforcing: the reward is `+1` only
within radius `ρ` of the goal and `0` elsewhere, so under an untrained policy almost every context row is
`(o, a, 0)`, near-identical across tasks because the reward coordinate — the only one carrying the goal's
identity — is `0` for all of them. The contrastive loss has almost no contrast to separate tasks by,
produces a `z` that barely varies with the task, the policy acts task-agnostically and rarely reaches any
goal, which keeps the reward `0`, which keeps `z` uninformative — a fixed point of failure, with no
exploration mechanism anywhere that commits to *going somewhere* to break it. So I expect sparse near zero
(the goal essentially never reached), and only the dense low-dimensional `point-robot` — goal readable
from a few dense transitions, every reward informative — to look healthy. Two limitations the climb from
here has to answer: an order-blind mean cannot read the cheetah target that lives in the *sequence*, and
nothing here drives the agent to *find* the sparse reward it has never seen.
