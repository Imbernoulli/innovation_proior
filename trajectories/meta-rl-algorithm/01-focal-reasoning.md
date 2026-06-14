The scaffold hands me a SAC actor-critic that already conditions on a task variable `z`; what it does
*not* hand me is anything that makes `z` mean something. The default `z` is a dummy zero, `infer_posterior`
and `adapt` are no-ops, and `train_iteration` takes no gradients — so the policy is the same policy on
every task and there is no meta-learning at all. The first real thing I have to design, then, is the
context encoder: the map from collected transitions to a `z` that tells the policy which task it is in.
I want the *simplest* encoder that could possibly work, so that whatever I build next has a clean floor
to beat. Simplicity here has a precise meaning — the fewest moving parts in the inference path, and a
training signal for the encoder that is *decoupled* from the value learning so I can reason about it in
isolation. That points me at a deterministic encoder shaped directly by a distance objective, and I'll
spend the rest of this working out why that is both the natural starting point and where it should crack.

Start from what the latent variable even is. The tasks in this family share the state and action space
and differ only in their reward (and, on some families, transition) functions. Take the clean case of
deterministic dynamics — which is what these continuous-control families essentially are — and suppose
the tasks satisfy what I'll call task-transition correspondence: two tasks agree on transition and
reward everywhere if and only if they are the same task. Then the pair `(P, R)` *identifies* the task,
and under determinism each `(s, a)` has a unique outcome `(s', r)`, so every task is a function
`f_T(s, a) = (s', r)`. The whole family `{f_T}` lives on the transition space `S × A × S × ℝ`, which is
exactly the context space the harness samples from. The encoder isn't inferring a fuzzy belief about a
hidden quantity; it is embedding the function `f_T` from samples of it. And here is the consequence that
makes the simple design legitimate: because `(P, R)` pin the task pointwise, a *single* transition tuple
in principle already constrains which task I'm in — `f_T` is determined by its values, and one value is a
real constraint. I do not need to integrate evidence across many transitions to slowly become confident.
Two things drop straight out. The encoder should be **permutation-invariant** — order cannot matter if
each transition independently reveals the task — and it should be **deterministic** — there is no
irreducible uncertainty to represent, because the task is recoverable, not guessed. No probabilistic
posterior, no information bottleneck, no product-of-Gaussians belief. The permutation-invariant
aggregation that costs the least is the one prototypical networks use for class prototypes: embed every
transition with a shared MLP and take the **mean** of the per-transition embeddings. So the encoder is a
width-200 depth-3 MLP from one transition `(o, a, r)` to a `latent_dim`-vector, and `z` for a task is the
mean over its context rows. That is `infer_posterior`, and `adapt` just calls it on the accumulated
online context.

"Deterministic mean encoder" only says how I *combine* embeddings; it says nothing about what makes the
embedding *good*. If I let the encoder learn purely through the SAC critic's Bellman gradients — the way
a value-trained context encoder would — I have a problem I can see coming. So let me ask whether the
geometry of `z` even needs a direct objective, because if it doesn't I shouldn't add one. The value
function `Q(s, a, z)` is a continuous network: for close `z₁, z₂` it is *forced* to output close
Q-values. Now take two genuinely different tasks whose embeddings happen to land close together. The
network must give them nearly equal Q-values — but their *true* Q-values, built from different rewards
and dynamics, are not close at all. A single continuous approximator simply cannot output two
well-separated values at two nearly-identical inputs. So if the encoder lets distinct tasks' embeddings
sit near each other, the conditioned value functions for those tasks become unrepresentable, and control
fails downstream no matter how good the actor-critic is. The geometry of `z` is therefore not cosmetic;
keeping distinct tasks far apart in latent space is a *precondition* for the value functions to exist. I
need an objective that lives on the embedding space itself and clusters same-task transitions while
separating different-task ones — and, importantly, I want it decoupled from the value learning so I can
trust it.

The textbook objective for "cluster same, separate different" is the contrastive loss: a quadratic
attractor `1{same}·‖q_i−q_j‖²` pulling same-task pairs together, and a margin hinge
`1{diff}·max(0, m−‖q_i−q_j‖)²` pushing different-task pairs apart up to distance `m`. If I just reach for
that, it degenerates — the clusters do not cleanly separate; I get blobs holding transitions from several
tasks at once. The why matters, because it dictates the fix. Two failures compound. First, the margin
hinge is a spring that acts *only within radius m* and whose force is *weakest exactly where I need it
most*: at the start, all embeddings are bunched near the origin (the latent is tanh-bounded, the encoder
randomly initialized), so every pairwise distance is tiny, and the hinge's gradient with respect to
distance, `−2(m−d)`, is bounded and has no special urgency at small `d`. Second, and deeper, the
attractive squared-distance term is, algebraically, *variance maximization*: summing squared pairwise
distances over a set equals `2N²·Var(X)`. Variance is a single global scalar, and many embedding
distributions share it — including the degenerate one that piles half the mass at `+1` and half at `−1`
on each axis, which can cram several tasks into each pile while still having huge variance. The optimizer
happily spreads mass to the extremes and merges distinct tasks, because the global statistic it is
maximizing says nothing about whether *every* pair of distinct clusters is separated. That is precisely
the merged-blob failure.

So I know what I actually need: a repulsion between different-task pairs that is *strong at short range
and fades as distance grows* — the opposite profile of the margin spring, and one that cannot be
satisfied by a global statistic. The function with that profile is a *negative* power of distance:
`β / (‖q_i−q_j‖ⁿ + ε)` blows up as the distance goes to zero and relaxes once the pair is well
separated. Replacing the margin term with this gives the deep-metric-learning loss
`L = 1{same}·‖q_i−q_j‖² + 1{diff}·β/(‖q_i−q_j‖ⁿ + ε)`. Now the different-task term is a genuine
repulsive *potential*, not a capped spring: two distinct-task embeddings sitting on top of each other
feel an enormous push apart, and crucially the potential cannot be gamed by the variance trick, because
the offending pairs *inside* a merged pile are at small distance and the `1/dⁿ` term is screaming at them.
Every pair of distinct clusters is forced apart. Physically it is a system of like charges in a bounded
box: with `n = 1` it is literally the Coulomb potential, the charges migrate to the boundary and pile up
at the corners, and the tanh-bounded latent cube `(−1,1)^l` is the conducting box that lets the repulsion
settle into a well-separated equilibrium instead of flinging everything to infinity. The same-task
quadratic still pulls each task into a tight cluster, so I get tight clusters scattered to the extremes —
exactly the geometry the continuity argument demanded. For the power I take `n = 2` (the inverse-square,
sharper than Coulomb, concentrating effort on the closest offending pairs), which coincides with the
Cauchy graph-embedding objective whose whole point is preserving local topology better than the quadratic
form; `β` just matches the attractive and repulsive scales and `ε` floors the denominator.

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
inverse-power to different-task pairs across the meta-batch. That distance-metric loss is the encoder's
*primary* shaping signal. I should be precise about the gradient flow, though, because here I deviate from
the fully-decoupled offline version and follow the scaffold's PEARL-shaped update instead: the contrastive
loss is backpropped into the encoder, but the encoder optimizer is stepped *together with* the critic, and
the `z` that feeds the two Q-heads is **not** detached — so the Bellman gradient also reaches the encoder,
exactly as it would for a critic-trained context encoder. The `z` is detached only where it feeds the
value head and the policy. So the encoder is shaped by the DML loss *plus* the critic, not by the DML loss
alone; the design choice I am keeping from the clean version is the contrastive objective and the
deterministic mean aggregation, not the gradient-level isolation. The SAC update underneath is the standard
twin-Q / value / squashed-Gaussian-actor step — reward scaled, the soft target value net slowly tracked,
`z` detached in the value and policy losses — exactly as the building blocks provide. The literal scaffold
fill is in the answer.

So the design closes: a deterministic, mean-aggregating context encoder shaped primarily by an
inverse-power distance metric loss that forces distinct task clusters to the corners of the bounded latent
cube, feeding `z` into a plain online SAC (with the critic gradient also reaching the encoder). Why start
the ladder here? Because it is the thinnest encoder that respects the one hard constraint — distinct tasks
must be separable in `z` for the value functions to exist — with a contrastive training signal I can
reason about directly, free of the probabilistic-belief apparatus that the recoverable-task argument says
is unnecessary.

Now I should be honest about where I expect this to crack, because that is what will point at the next
rung. The whole construction rests on "a single transition reveals the task." On `cheetah-vel`, where
the task is a target velocity that is only weakly visible in any one `(s, a, r)` and the reward is dense
but ambiguous, a permutation-invariant mean of per-transition embeddings throws away the temporal
structure that would let me read the target off a *sequence* — and a contrastive loss that only enforces
"these blocks are the same task, those are different" never teaches `z` to encode the underlying quantity
the policy actually needs to set its running speed. I expect the cheetah encoding to be the weak point:
the conditioned policy should land mediocre returns there. On `sparse-point-robot` the construction is in
even more trouble — the reward is +1 only near the goal and 0 everywhere else, so almost every context
transition carries reward 0 and is identical across tasks, the contrastive signal has almost nothing to
separate tasks *by*, and there is no exploration mechanism here that commits to going somewhere to *find*
the goal in the first place. A deterministic `z` with no notion of "I am uncertain, let me probe" cannot
drive the temporally-extended exploration a sparse task needs. So I expect `sparse-point-robot` to be
where this floor fails hardest — returns near zero, meaning the goal is essentially never reached. The
dense, low-dimensional `point-robot` is the one place the assumptions roughly hold (the goal is readable
from a few dense transitions), so it should be the only family where this rung looks healthy. If that
split is what I measure — fine on dense low-dim, weak on high-dim encoding, near-zero on sparse — the
diagnosis writes the next step for me: I need an encoder that respects the *sequence* of experience
rather than treating context as an unordered bag, and eventually a representation that can carry
*uncertainty* so the agent can explore the sparse task at all.
