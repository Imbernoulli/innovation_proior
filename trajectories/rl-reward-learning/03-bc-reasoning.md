AIRL did beat GAIL on every environment, and the margins fell exactly where I predicted — but the
absolute numbers carry a sharper lesson than I expected. HalfCheetah went 1646 → 2514, a lift of a factor
$2514/1646\approx1.53$ — real but modest, the non-terminal body needing the terminal-state fix least, just
as I expected the smallest margin there. Hopper went 25.7 → 840, a factor of $\approx33$, and Walker2d
77.8 → 358, a factor of $\approx4.6$ — large gains off the floor, and the ordering (Hopper margin far
larger than Walker margin far larger than HalfCheetah margin) is exactly the inverse-of-collapse signature
I said would confirm that done-aware shaping and the generator-independent discriminator rescued the
terminating bodies. So the structured reward worked as the theory said. And yet look *inside* the means,
because the mean is hiding the real story. Hopper's 840 is $\{153, 169, 2198\}$: two seeds sitting at
$\sim160$ — a body that survives a little longer than GAIL's but is still, in return terms, barely alive
— and a single seed at $2198$ that found a real gait and ran. The mean is dragged up by one lucky seed;
the median is $169$. Walker2d's 358 is $\{290, 654, 129\}$ — all three low, none anywhere near a real
walking return, the "best" seed at $654$ still a fraction of a competent walk. So the honest read is:
AIRL's terminating-body success is $2198/169\approx13$-to-one seed-dependent on Hopper and essentially
absent on Walker. The adversarial frame is *less* fragile than GAIL's — it lifted every environment — but
it is still fragile: on the terminating bodies it mostly produces a barely-alive policy, with the
occasional lucky seed masking that in the mean. The structure helped; it did not make the min-max game
reliable on clean demonstrations.

That is the diagnosis that decides the next rung, so let me name the root liability precisely, because it
is shared by both rungs below. GAIL and AIRL both learn a reward by an *adversarial* objective, which
means the reward is *non-stationary by construction*: the discriminator moves every round, so the reward
the policy ascends and the value function it fits are both chasing a target that keeps shifting. On a
clean, tight expert distribution the discriminator's pull toward separation is strong (I saw its accuracy
climb toward $1$ on GAIL), and that pull is enough that two out of three AIRL seeds get stuck in a
degenerate basin before a usable, stable reward emerges. I have now spent two rungs adding machinery to
tame that non-stationarity — input normalization, output normalization, $16\times$ bumped inner updates,
done-aware shaping — and the terminating bodies still mostly die: Hopper's two dead seeds and Walker's
three low seeds are the residue that no amount of discriminator-side engineering removed. The honest read
of those seeds is that the remaining failure is not in the reward *structure* — AIRL's reward is
well-conditioned and the one good Hopper seed proves the structure can carry a real gait — but in the
*adversarial loop itself*. So the move is not to add more structure to the discriminator. It is to ask
whether I need the discriminator — the moving reward, the policy-density evaluation, the whole min-max game
— at all.

Let me walk the design space honestly, because "drop the adversary" is a big step and there are
intermediate options I should rule out on paper first. Option one: keep AIRL and add reward-replay or a
target-discriminator to slow the non-stationarity, the way off-policy RL stabilizes a moving target. But
the evidence says the instability is not merely *fast* movement I can damp — it is that on clean, separable
demos the discriminator reaches a confident basin that traps two-thirds of seeds, and slowing the approach
to that basin (I already tried, via the inner-loop and normalization machinery) postpones rather than
prevents the trap. Option two: ensemble the discriminator across seeds to average out the bad basins. That
improves the *mean* by construction but does nothing for a single run's reliability, and the task scores
per-seed returns; it is a reporting trick, not a fix. Option three, the one I will take: step back to what
the task actually provides and ask whether the reward-learning detour is necessary at all.

What does the task actually give me and actually score? I have a fixed pile of expert $(s,a,s')$
transitions from a *competent* MuJoCo agent — clean, on-distribution, dense — and I am scored on the
policy's true-reward return after training. The adversarial methods spend their entire budget learning a
*reward* so that PPO can then learn a *policy* — two coupled, non-stationary optimizations stacked on top
of each other. But the expert demonstrations already contain, directly, the one thing I ultimately need:
for each state the expert visited, the action a competent policy took. The most direct possible use of that
signal is to skip reward learning entirely and fit the policy to the expert actions by supervised learning.
That is behavioral cloning — and the lineage flagged its weakness up front (covariate shift, compounding
$\varepsilon T^2$ error off the expert distribution), which is exactly why the ladder started with the
occupancy-matching methods that were *supposed* to beat it. But I now have *measured* evidence that those
methods, in this harness, are dominated by adversarial instability rather than by the covariate-shift
advantage they were supposed to have. So the question is empirical, and it is worth running BC as the top
rung precisely to settle it: on these clean, dense MuJoCo demos with a fixed PPO substrate, does plain
cloning's *stability* beat the adversarial methods' *principle*?

Let me reason about *why* cloning could win here, because it is not obvious from the lineage — the lineage
says BC should lose. Recall the covariate-shift penalty I sized at rung 1: BC's regret grows like
$\varepsilon T^2$, and with $T\approx1000$ that $T^2=10^6$ multiplier is what makes a small per-step error
$\varepsilon$ compound into a fall. Two things blunt that penalty on *this* task. First, the demos are
dense — tens of thousands of transitions — and the experts are competent, so the expert state distribution
covers the locomotion manifold well and the per-step disagreement $\varepsilon$ a supervised fit achieves
on in-distribution states is small; a smaller $\varepsilon$ shrinks the $\varepsilon T^2$ product directly,
and the learner rarely has to extrapolate far off the manifold where $\varepsilon$ would balloon. Second,
and decisively, the *alternatives* are not paying a small covariate-shift cost — they are paying a large
adversarial-instability cost, and I have the numbers: two of three AIRL Hopper seeds dead at $\sim160$,
all three Walker seeds low. Trading a structural-but-bounded error (compounding drift, kept small by dense
clean demos) for a stable supervised objective with no min-max game and no non-stationary reward can be a
net win when the demos are this clean. Cloning has no discriminator to saturate, no reward to drift, no
policy-density to evaluate, nothing to balance — it is a single stationary supervised loss on a fixed
dataset. On clean demos that stability is worth more than the occupancy-matching guarantee, *if* the
experiment bears it out.

Now I have to land BC in *this* scaffold, and here the scaffold forces a version of cloning that is quite
different from the classical road-following cloner — and I have to derive the harness's actual version, not
the textbook one. The classical formulation laid out a whole apparatus: a population-coded output bank read
by center of mass for fine continuous control, synthetic perspective-transformed views to manufacture the
off-center recovery examples the expert never demonstrates, pure-pursuit relabeling of those views, and a
replay buffer with a mean-steering-toward-straight eviction rule to lock in left-right symmetry. *None of
that exists here.* The scaffold gives me no road geometry to synthesize recovery views from, no
population-coded output — the policy is a fixed Gaussian actor-critic with a mean head and a
state-independent log-std — and no mechanism to relabel manufactured states. The harness exposes exactly
one thing the classical cloner did not have and that I should use: the policy's *full action distribution*.
So the task's BC is not "regress the action to a point"; it is *maximum-likelihood cloning of the Gaussian
policy* — minimize the negative log-probability of the expert's actions under $\pi(a\mid s)$. Writing that
loss out for the Gaussian actor, $-\log\pi(a\mid s)=\tfrac12\sum_i\big[(a_i-\mu_i(s))^2/\sigma_i^2+
\log(2\pi\sigma_i^2)\big]$: minimizing it drives the mean head $\mu(s)$ toward the expert action (the
squared term, an MSE weighted by inverse variance) *and* fits the log-std $\log\sigma$ (the normalizer
term), because pushing $\sigma$ too small blows up the $(a-\mu)^2/\sigma^2$ penalty on any residual error
while pushing it too large blows up the $\log\sigma$ term — the two terms balance at the $\sigma$ that
matches the actual action spread. So a single NLL objective trains both the mean and the variance at once
and gives proper action coverage rather than a brittle point estimate that a plain MSE-on-the-mean would
produce. That is the tuned-library BC objective, and it is what the harness's fill implements.

The mechanics the scaffold dictates, derived from its rigid contract. The reward net is *unused* — BC does
not learn a reward — so `RewardNetwork` is a dummy returning zeros, and `compute_reward` returns zeros for
every transition. That has a load-bearing consequence I should make explicit: with the learned reward
identically zero, the fixed running-reward-normalization sees a constant stream (it normalizes zeros to
zeros, modulo the $\varepsilon$ in its denominator) and the PPO update computes advantages from a
value-target that is everywhere zero — so the secondary PPO step contributes essentially no gradient and,
critically, injects *none* of the non-stationarity that destabilized the adversarial methods. PPO is
present but inert; it cannot fight the cloning. The actual learning happens in `update()`, and to train the
*policy* there I need the policy reference *and* its optimizer — the scaffold hands both in through
`set_policy(policy, optimizer)`, which the BC fill stores (unlike AIRL, which kept only the reference; BC
keeps the optimizer because *it*, not the fixed PPO step, does the gradient updates on the policy). Inside
`update()` I sample expert $(s,a)$ minibatches, evaluate $\log\pi(a\mid s)$ via the policy's
`get_action_and_value(obs, expert_acts)` — which returns the log-prob of the *expert* action under the
current Gaussian — and minimize $-\mathbb{E}[\log\pi(a\mid s)]$, with a tiny entropy *penalty*
($-0.001\cdot\text{entropy}$, i.e. a small push *against* collapsing the policy's variance to zero) to
keep the action distribution from degenerating to a spike. Why a penalty and not the usual entropy
*bonus*? A bonus would inflate $\sigma$ and hurt the clone's precision; here I want the opposite guard —
the coefficient is deliberately tiny ($10^{-3}$) so it does not fight the NLL's own $\sigma$-fitting, it
only stops a pathological collapse to zero variance that would make the log-prob and its gradient explode.

The step-budget choice is worth pinning down against the adversarial rungs. I run several BC gradient
steps per `update()` call — the fill uses `n_bc_steps=20` — so cloning dominates any residual influence of
the secondary PPO step, and I clip the policy grad-norm at $0.5$ and step the policy optimizer directly.
Compare the adversarial rungs' $16$ effective discriminator steps per round ($4\times4$): I am spending a
comparable per-round gradient budget, but on a *stationary* supervised loss over a *fixed* dataset rather
than on a moving discriminator over a shifting policy distribution — same amount of computation, but every
one of those $20$ steps pulls toward the same fixed target instead of chasing one that moves. That is the
whole trade in one line: I keep the compute, I drop the non-stationarity. The expert demos again store no
special structure I need beyond $(obs, acts)$; I do not even touch $s'$ or dones. The grad-norm clip at
$0.5$ is not incidental either: the NLL gradient scales like $(a-\mu)/\sigma^2$, so early in training when
$\mu$ is far from the expert action and $\sigma$ is still near its initialization, the per-step gradient
can be large and a single unclipped step could overshoot the mean past the expert and destabilize the
log-std fit; clipping the norm to $0.5$ caps that step size so the $20$ inner steps make steady, bounded
progress toward the expert action rather than oscillating around it. It is the supervised analogue of the
trust region PPO gives the adversarial rungs — cheap here because the target is fixed.

The dummy reward net still has to satisfy the contract, so let me confirm it is legal and cheap. It must
expose `forward(state, action, next_state) -> (batch,)` and the algorithm must set `self.reward_net`, so I
keep a `RewardNetwork` that holds a single unused `nn.Linear(1,1)` — a handful of parameters — and whose
`forward` returns `torch.zeros(batch)`. Against the $\approx1.05\times$-largest-baseline parameter cap this
is trivially inside budget: three parameters versus the tens of thousands the adversarial reward nets spent,
so BC is the *smallest* reward footprint on the ladder even though it produces the policy I actually care
about. The asymmetry is the point — BC moves all of its capacity into the policy (which is fixed-size and
not mine to grow) and spends essentially nothing on the reward object the other rungs poured their budget
into.

It is worth checking the terminal-body argument from the policy side, because that is where I am claiming
the decisive win and I should make sure the mechanism is real and not wishful. On Hopper and Walker2d the
episode ends when the torso pitches past a threshold; survival is a matter of producing, at each state, an
action close enough to a competent one that the body stays inside the viable set. AIRL's failure there was
that two of three seeds never grew a reward whose gradient reliably pointed at "stay upright," so the
policy wandered out of the viable set within a few steps ($\sim160$-return seeds). A cloned Gaussian does
not have to *discover* that gradient at all: for every expert state it is trained directly toward the
expert's action, and because the expert stays upright the entire demonstration, in-distribution the clone
inherits an upright-keeping action at every visited state. The residual risk is purely covariate shift —
the clone drifts to a state the expert never showed and emits a poor action there — but on dense demos
that manifold is well covered, so the drift is slow and the $\varepsilon T^2$ budget is spent gradually
rather than blown in the first few steps. That is exactly the difference between "barely alive, occasional
lucky seed" and "consistently alive across seeds" that I expect to see: AIRL's terminating-body survival
was gated on the adversarial reward *emerging* (a coin-flip across seeds), whereas BC's survival is gated
only on the demos covering the states the clone visits (reliable when the demos are dense and clean). If
the reasoning is right the seed spread on Hopper and Walker should tighten relative to AIRL's
two-dead-one-alive scatter, not just the mean rise.

So the falsifiable expectations, read against the two rungs below. If the diagnosis is right — that the
adversarial methods are dominated by min-max instability on clean demos, not by their occupancy-matching
principle — then BC should *beat AIRL on the aggregate*, and it should beat it most clearly where AIRL's
instability bit hardest: the terminating bodies. I expect BC's Hopper and Walker2d to land in the
low-thousands and the low-to-mid thousands respectively — not the $\{153, 169, 2198\}$ two-dead-one-alive
pattern AIRL gave, but consistently-alive cloned gaits across seeds, because a supervised fit to a
competent expert keeps the body upright without needing a reward to emerge first. On HalfCheetah, where
AIRL already reached $2514$ and the body never terminates, I expect BC to be *competitive but not
dramatically ahead* — possibly even with one runaway seed (cloning a fast cheetah can overshoot the
expert's speed) and one weak seed (cloning can also underfit a high-dimensional gait), so a wide spread
around the low-$2000$s that might even trail AIRL's mean slightly. The clean signature that would confirm
the whole ladder's story: BC strongest overall, winning Hopper and Walker2d decisively while roughly tying
or modestly trailing AIRL on the non-terminal HalfCheetah — i.e. the simplest method beating the
principled ones precisely because the principle's cost (instability) outweighs its benefit (occupancy
matching) on clean demonstrations. If instead BC underperforms AIRL across the board, then the
covariate-shift penalty really does dominate here and the adversarial frame was right after all — but given
the two-thirds-dead AIRL seeds on the terminating bodies, I expect the opposite, and that is the bar BC
must clear as the top of this ladder.
