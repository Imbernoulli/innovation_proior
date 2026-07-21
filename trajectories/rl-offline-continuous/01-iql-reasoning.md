I am standing on the scaffold with its `train` doing nothing, and I have to pick the first real rung.
Let me name the exact disease the scaffold is silent about, because the whole choice of where to start
follows from it. I have a fixed dataset D of transitions (s, a, r, s', a'), no environment, and I want a
policy better than whoever collected D. The standard tool is approximate dynamic programming: minimize
the TD error toward r + γ max_{a'} Q(s', a') with a target critic, and act greedily. Online that is
fine. Offline it is poison, and I want to be precise about why, because every design decision is about
dodging this one thing. The bootstrap target contains a maximization (or, in continuous control, an
actor that ascends Q) over actions a' at the next state. That ranges over actions that never appear in D
at s'. On those out-of-distribution actions the critic has no data; it extrapolates, and the
extrapolation is almost always *upward*, because the max actively hunts for whichever action the network
happens to over-value. The inflated target backs up through the Bellman recursion, the policy steers
toward the over-valued actions, and — with no new transitions to ever correct the mistake — the error
compounds into divergence. This is not a small bias; it is the reason the scaffold's default twin-critic
actor-critic, run naively on a static buffer, would simply fall over.

I want the "almost always upward" made quantitative, because it sets how hard I have to fight. Suppose
the critic reports q* + ε with ε zero-mean approximation noise of spread σ over the action argument. The
bootstrap reads the critic not at one action but at the argmax, so it samples the *upper order statistic*
of that noise: the expected maximum of k independent zero-mean errors grows like σ√(2 ln k), which for a
few tens of effective action directions is nearly 2σ of pure upward bias at every backup, before the
reward has said anything. And it compounds — a per-step optimistic bias δ sums through the geometric
series to δ/(1 − γ), a hundred-fold amplification at γ = 0.99. So the target is not to *reduce* the OOD
query but to *eliminate* it from value learning if I can; a hundred-fold multiplier leaves no room to
compromise at the floor.

So why begin the ladder with implicit Q-learning specifically, and not a behavior-constrained or a
conservative method? Because I want the *cleanest* possible answer to the OOD-query problem as my floor:
a method whose value learning never evaluates the critic at a single action outside the dataset, so I can
attribute any later gains to mechanisms beyond "don't touch unseen actions." The whole zoo of fixes —
generative-model constraints, conservative penalties, divergence regularizers — all still, somewhere in
the loop, query a learned Q at an action that was not in the data: the generative-model constraints emit
sampled actions that can fall OOD, the conservative penalty has to sample OOD actions precisely to push
them down. The OOD query is still in the loop, and by the multiplier above, still dangerous. There is one
family that removes the query *entirely*, and that is where I want my floor. Both live alternatives keep
the query. A behavior-regularized actor-critic preserves the ∇_aQ ascent and penalizes divergence from
π_β, but the ascent still evaluates Q at the actor's action — the multiplier stays armed — and it needs a
fitted behavior model I would tune blind. A conservative value penalty attacks overestimation directly
but introduces an offline-unvalidatable temperature and *courts* the very OOD query it fears, resting on a
knife-edge cancellation. Neither can be my *clean* baseline.

The cleanest in-sample objective is the SARSA target: bootstrap with the dataset's own next action a',
not a max. The scaffold even hands me `next_actions` in the batch for exactly this. Minimizing
(r + γ Q(s', a') − Q(s, a))² over (s, a, s', a') ∈ D never touches an OOD action — every a and a' came
from D. But MSE fits Q to the *mean* of its targets, so the fixed point is Q^{π_β}: the value of the
behavior policy. That is pure policy *evaluation*, with no improvement, and it does no iterated dynamic
programming, so it cannot stitch — it cannot carry the value of a good downstream state backward across
transitions that belong to *different* dataset trajectories. On a task whose dataset is a pile of
suboptimal fragments that only collectively imply a good policy, plain SARSA collapses. That matters
here, because one of my three evaluation datasets is exactly a stitching task: Maze2d rewards reaching a
goal, and the medium dataset is full of partial trajectories that have to be sewn together. So I am
caught between a safe-but-improvement-free SARSA and the dangerous max.

What I actually want in the backup is not the mean over a' and not the unrestricted max, but the value
of the *best in-support action* — a max restricted to actions the behavior policy could produce at s'.
That restriction is the whole game: it improves (so iterating it does real dynamic programming and can
stitch), but it never reaches an OOD action (so it stays safe). I cannot compute it directly, because
taking a max over in-support actions would mean enumerating or sampling actions and querying Q at each —
and the moment I sample an action and query Q I am back to evaluating Q outside the data. I need the max
over in-support actions *without ever querying Q at any specific a'*.

Here is the reframe that makes it computable. Fix a state s. As a' ranges over the behavior distribution
at s, the quantity Q(s, a') is a *random variable*, its randomness coming from the action. SARSA's MSE
estimates its *mean*. What I want is the upper edge of its support. The statistic that climbs into the
upper tail of a random variable, estimable by a one-line reweighting of MSE on in-sample data, is the
**expectile**. The τ-expectile minimizes the asymmetric squared loss |τ − 1(u<0)|·u²: for a positive
residual (a sample above the estimate) the weight is τ, for a negative one it is 1 − τ. At τ = 0.5 both
weights are ½ and this is plain MSE, so the 0.5-expectile is the mean. For τ > 0.5 the samples *above*
the estimate are weighted more, so the estimate is pulled *up*, and as τ → 1 it approaches the supremum
of the support. That is exactly the in-support max I wanted, expressed as a regression I can run with SGD
on dataset actions only.

A toy pins the dial down. Take {0, 1, 2} equally likely; the τ-expectile m solves
τ·Σ_{x>m}(x−m) = (1−τ)·Σ_{x<m}(m−x). At τ = 0.5, with m in (1, 2): 0.5(2−m) = 0.5(2m−1) gives m = 1, the
mean — SARSA recovered. At τ = 0.7: 0.7(2−m) = 0.3(2m−1) → 1.7 = 1.3m → m ≈ 1.31, about 31% of the way
from the mean toward the maximum of 2. And τ → 1 sends the right side to zero, so m → 2, the supremum. So
τ = 0.5 is honest evaluation, τ near 1 is the in-support argmax, and τ = 0.7 sits deliberately between —
optimistic but not reckless — which is why I treat τ as the safety–improvement knob and set it below 1 on
purpose.

But I must take the expectile over actions *only*. The naive move — put the asymmetric loss directly on
the TD residual r + γ Q(s', a') − Q(s, a) — is wrong, and the reason is the load-bearing subtlety of the
whole method. That target carries two sources of randomness: the action a' (which I *want* to be
optimistic about — the best a' is the improvement signal) and the stochastic transition s' (which I
emphatically do *not* want to be optimistic about — being optimistic over the dynamics rewards a target
that is high merely because the dice landed on a lucky next state, and that optimism compounds into a
wildly over-valued function). So I separate the two. A value network V(s) takes the upper expectile over
*actions* with the transition fixed: regress V(s) asymmetrically against the target critic's Q(s, a)
over dataset (s, a). Then Q is backed up onto r + γ V(s') by an *ordinary* MSE that averages honestly
over the next-state transition. The division of labor is the point: V does the optimistic in-support
action selection; Q does the honest mean over dynamics; both losses touch only dataset (s, a, s'), and
no policy and no OOD action appear anywhere in value training. This is provably a spectrum — at τ = 0.5
it is SARSA (policy evaluation of π_β), and as τ → 1 it is Q-learning restricted to in-support actions
(true multi-step DP up to the in-support optimum) — so τ is the dial between safety and improvement, and
I will treat it as a hyperparameter rather than slamming it to 1, because large τ leans on extreme upper
residuals: near τ = 1 the effective weight on the handful of samples above the current estimate becomes
overwhelming and the fit rides on the noisiest tail of the target-critic values, so the variance of V
blows up exactly when I most rely on it. Setting τ = 0.7 keeps the up-weighting a modest 0.7-to-0.3, a
factor of about 2.3 in favor of the upper residuals — enough to improve over the mean, as the toy showed,
without betting the fit on a few outliers.

Now I map this onto the scaffold's edit surface, because that is what actually runs, and the scaffold
constrains me in specific ways I should honor rather than fight. The fixed loop hands me a six-tensor
batch, exposes `next_actions`, gives me `soft_update` for Polyak targets, and locks hidden width at 256.
It also fixes `discount = 0.99`, `tau = 5e-3`, lr `3e-4`, and — important — it does **not** offer me a
reward-preprocessing knob that I will use here; I leave rewards as the dataset gives them, so my floor is
the unadorned IQL with no D4RL reward rescaling. I build three function approximators inside
`OfflineAlgorithm`: a twin critic (two `Critic` instances plus Polyak targets, optimized jointly), a
state-value `ValueFunction` V, and a Gaussian `Actor`. I deliberately make the actor a *state-independent*
log-std Gaussian — the mean is an MLP with a Tanh output and the log-std is a single learned parameter
vector — because policy extraction will be advantage-weighted maximum likelihood and I only need clean
log-probabilities of dataset actions, not a full reparameterized squashing. State-dependent variance
would let the policy claim near-zero uncertainty on a lucky action and blow up its log-prob weight; a
single global log-std removes that failure mode and costs only `action_dim` extra parameters, which
against the two 256×256 hidden layers in the actor trunk (about 65k parameters each once you count the
256×256 weight and its bias) is negligible — the choice is essentially free in the budget and strictly
safer. To respect the parameter cap and match the method's reference shape, the critic and value nets are
two-hidden-layer 256-unit MLPs, which is the minimum depth that lets a Q over concatenated (s, a) express
the action-conditioned value the expectile needs to see.

The per-step update has a fixed order dictated by the dependencies. First I snapshot next_v = V(s')
under no-grad, because the Q update needs V at the next state and I want it stable for this step. Then
the V update: target_q = min over the two *target* critics of Q(s, a) (clipped double-Q, to stop the V
regression from chasing an inflated single critic), advantage = target_q − V(s), and V is stepped on the
asymmetric L2 of that advantage with iql_tau = 0.7 — the value of τ that, on locomotion, sits high
enough to improve over π_β without the variance blow-up of τ near 1. Then the Q update: targets =
r + (1 − done)·γ·next_v, and both online critics are regressed to it by MSE, after which I Polyak-update
both target critics with tau = 5e-3. The honest MSE here is correct *because* V(s') has already done the
optimistic action selection; what remains is to average γV(s') over the transition, which is exactly what
a mean does. I use the *target* critics inside the V regression but the *online* critics as the thing the
MSE fits, so the two value nets lean on each other through a slow Polyak link rather than a tight loop:
V trusts a target-network Q that moves at rate tau = 5e-3, and Q trusts a detached next_v — no gradient
path closes on itself within a single step, which is what keeps this three-network dance from oscillating.

The clipped double-Q inside the V target deserves one more beat, because it is the second overestimation
brake and it interacts with the expectile. If I regressed V against a single critic, the expectile would
climb the upper tail of *that critic's* value — and a single critic, trained by bootstrapped regression,
is exactly the object that develops isolated upward bulges. So the expectile would faithfully find and
amplify the critic's own error, which is the opposite of what I want. Taking min over the two target
critics first caps each target-Q at the more pessimistic of two independently-initialized estimators
before the expectile ever sees it: concretely, if at some (s, a) critic one reads 5.0 and critic two
reads 3.2, the min hands the V regression 3.2, so a bulge that only one critic invented is discarded
rather than climbed. The min supplies pessimism over *estimator error*; the expectile supplies optimism
over *the action distribution*; stacking them, I am optimistic about which in-support action is best but
pessimistic about whether any single critic's number can be trusted — the two biases point in opposite
directions on purpose and neither is allowed to run away.

One dimension trap is worth defusing before it corrupts every loss. In the Q update, rewards and dones
arrive as (B, 1) while next_v is (B,), so rewards + (1−done)·γ·next_v would broadcast to (B, B) and the
MSE would silently average a quarter-million cross terms. Squeezing rewards and dones to (B,) makes every
quantity (B,) at the point of reduction — the target aligns one-to-one with each critic's output, and the
detached advantage from the V step multiplies the per-sample log-prob elementwise before the mean.

That leaves the policy, and it must obey the same commandment: never query Q at an unseen action. So I
cannot do argmax or a DDPG-style ∇_a Q ascent, both of which evaluate Q at the policy's possibly-OOD
action. I extract by advantage-weighted regression, and I want the derivation on paper because it is what
guarantees the extraction stays in-support. Pose policy improvement as a KL-constrained problem: maximize
E_{a∼π}[A(s,a)] subject to KL(π ‖ π_β) ≤ ε. Forming the Lagrangian with multiplier λ and taking the
functional stationarity condition in π gives the closed form π*(a|s) ∝ π_β(a|s)·exp(A(s,a)/λ) — the
behavior policy tilted by the exponentiated advantage, never placing mass where π_β placed none, so it is
in-support *by construction*. I cannot represent π* directly, so I project it onto my parametric actor by
minimizing KL(π* ‖ π_θ), which reduces to a weighted maximum likelihood: the loss exp(β·(Q − V))·(−log
π(a|s)) averaged over dataset (s, a), with β = 1/λ the inverse temperature. This is pure reweighting of
observed actions by how advantaged they are, querying no OOD action, with the stay-near-π_β constraint
baked into the fact that only logged actions ever get a gradient. β = 3.0 is the inverse temperature for
locomotion (β → 0 is behavior cloning, β → ∞ greedily concentrates on the single best-advantaged action);
a handful of transitions can have huge advantages whose exp weights would swamp the loss, so I clamp
exp(βA) at 100, capping any one transition at 100× the weight of an average one. And I cosine-anneal the actor
learning rate over the full 1e6 steps, so the advantage-weighting can be aggressive early — when the
critic is still learning and large corrections help — and settle to a near-zero learning rate late, when
I want the policy to stop chasing the last flickers of a noisy advantage. The actor's `.act` is what
evaluation rolls out — I take the distribution mean at eval time, because the log-std was only ever there
to define the likelihood for the weighted regression, not to inject noise into a deployed policy.

Where do I expect this floor to land, and where is it fragile? On HalfCheetah and Walker2d — dense-reward
locomotion where the medium dataset is a single coherent mediocre policy — the in-sample value learning
should produce a stable improvement over the behavior policy without divergence, sitting comfortably above
random but not dramatically above the data, because τ = 0.7 is deliberately conservative and the advantage
weighting stays close to π_β; HalfCheetah especially has little headroom to disagree over, so I expect it
nearly pinned to the data with a tight seed spread. Maze2d — the stitching task — is the genuine worry.
IQL *can* stitch in principle (the expectile backup is real DP), but its improvement is gated by τ: at 0.7
the in-support max is only weakly approximated (the toy said about 31% of the way from mean to max), so on
a task that needs aggressive value propagation across fragments the conservative expectile may leave a lot
on the table, and the advantage-weighted extraction (which never extrapolates beyond dataset actions) caps
how far the goal-reaching policy can deviate from the meandering data. So I expect a comparatively weak,
higher-variance Maze2d where the conservatism bites — a weak value signal propagated across fragments is
exactly what lands in different places on different seeds. That gap, if it appears, is the opening for a
next rung to exploit the critic directly rather than hedge with an expectile. The full module — Gaussian
`Actor`, 2×256 `Critic` and `ValueFunction`, and the V → Q → policy update order — is in the answer.
