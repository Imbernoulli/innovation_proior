AIRL did beat GAIL on every environment, and the margins fell exactly where I predicted — but the
absolute numbers carry a sharper lesson than I expected. HalfCheetah went 1646 → 2514, a real but modest
lift (the non-terminal body needed the terminal-state fix least). Hopper went 25.7 → 840 and Walker2d
77.8 → 358 — large *relative* gains off the floor, confirming that done-aware shaping and the
generator-independent discriminator rescued the collapse. So the structured reward worked as theory said.
And yet look at the variance: Hopper's 840 mean is {153, 169, 2198} — two seeds still essentially dead,
one seed that found a good gait and ran. Walker2d's 358 is {290, 654, 129} — all low, none near a real
walking return. The adversarial frame is *less* fragile than GAIL's, but it is still fragile: on the
terminating bodies AIRL mostly produces a barely-alive policy, with the occasional lucky seed masking it
in the mean. The structure helped; it did not make the min-max game reliable on clean demonstrations.

That is the diagnosis that decides the next rung. Both GAIL and AIRL inherit the same root liability — an
adversarial reward that is *non-stationary by construction*: the discriminator moves every round, so the
reward the policy ascends and the value function it fits are chasing a target that keeps shifting, and on
a clean, tight expert distribution the discriminator's pull is strong enough that two out of three seeds
get stuck in a degenerate basin before a usable reward emerges. I have spent two rungs adding machinery to
tame that non-stationarity — input normalization, output normalization, bumped inner updates, done-aware
shaping — and the terminating bodies still mostly die. The honest read of the AIRL seeds is that the
remaining failure is not in the reward *structure* but in the *adversarial loop itself*. So the move is
not to add more structure to the discriminator. It is to ask whether I need the discriminator — the
environment interaction, the moving reward, the policy-density evaluation — at all.

Step back to what the task actually provides and what it actually scores. I have a fixed pile of expert
$(s,a,s')$ transitions from a *competent* MuJoCo agent — clean, on-distribution, dense. And I am scored
on the policy's true-reward return after training. The adversarial methods spend their entire budget
learning a *reward* so that PPO can then learn a *policy* — two coupled, non-stationary optimizations
stacked on top of each other. But the expert demonstrations already contain, directly, the one thing I
ultimately need: for each state the expert visited, the action a competent policy took. The most direct
possible use of that signal is to skip reward learning entirely and fit the policy to the expert actions
by supervised learning. That is behavioral cloning — and the lineage flagged its weakness up front
(covariate shift, compounding $\varepsilon T^2$ error off the expert distribution), which is exactly why
the ladder started with the occupancy-matching methods that were supposed to beat it. But I now have
*measured* evidence that those methods, in this harness, are dominated by adversarial instability rather
than by the covariate-shift advantage they were supposed to have. So the question is empirical: on these
clean, dense MuJoCo demos with a fixed PPO substrate, does plain cloning's stability beat the adversarial
methods' principle? The whole point of running BC as the top rung is to settle that.

Let me reason about *why* cloning could win here, because it is not obvious from the lineage. BC's
covariate-shift penalty bites when the learner drifts off the expert's state distribution into states it
never trained on. Two things blunt that penalty on this task. First, the expert demos are dense — tens of
thousands of transitions — and the experts are competent, so the expert state distribution covers the
locomotion manifold well; the learner does not have to extrapolate far. Second, and decisively, the
*alternatives* are not paying a small covariate-shift cost — they are paying a large adversarial-instability
cost, two-thirds of their seeds stuck in degenerate basins. Trading a structural-but-bounded error
(compounding drift) for a stable supervised objective with no min-max game and no non-stationary reward
can be a net win when the demos are clean. Cloning has no discriminator to saturate, no reward to drift,
no policy-density to evaluate, nothing to balance — it is a single stationary supervised loss. On clean
demos that stability is worth more than the occupancy-matching guarantee, *if* the experiment bears it out.

Now I have to land BC in *this* scaffold, and here the scaffold forces a version of cloning that is quite
different from the classical road-following cloner — and I have to derive the harness's actual version, not
the textbook one. The classical formulation laid out a whole apparatus: a population-coded output bank
read by center of mass for fine continuous control, synthetic perspective-transformed views to manufacture
the off-center recovery examples the expert never demonstrates, pure-pursuit relabeling of those views, and
a replay buffer with a mean-steering-toward-straight eviction rule to lock in left-right symmetry. *None of
that exists here.* The scaffold gives me no road geometry to synthesize recovery views from, no
population-coded output — the policy is a fixed Gaussian actor-critic with a mean head and a
state-independent log-std — and no mechanism to relabel manufactured states. The harness exposes exactly
one thing the classical cloner did not have and that I should use: the policy's *full action distribution*.
So the task's BC is not "regress the action"; it is *maximum-likelihood cloning of the Gaussian policy* —
minimize the negative log-probability of the expert's actions under $\pi(a\mid s)$, which trains both the
actor mean and the log-std at once and gives proper action coverage rather than a brittle point estimate.
That is the imitation-library BC objective, and it is what the harness's fill implements.

The mechanics the scaffold dictates, derived from its rigid contract. The reward net is *unused* — BC
does not learn a reward — so `RewardNetwork` is a dummy returning zeros, and `compute_reward` returns
zeros for every transition. That means the PPO loop still runs but ascends a near-zero (post-normalization,
arbitrary) reward — PPO is *secondary*; it must not be allowed to fight the cloning. The actual learning
happens in `update()`, and to train the *policy* there I need the policy reference and its optimizer — the
scaffold hands both in through `set_policy(policy, optimizer)`, which the BC fill stores (unlike AIRL,
which kept only the reference; BC keeps the optimizer because *it*, not the fixed PPO step, does the
gradient updates on the policy). Inside `update()` I sample expert $(s,a)$ minibatches, evaluate
$\log\pi(a\mid s)$ via the policy's `get_action_and_value(obs, expert_acts)` — which returns the log-prob
of the expert action under the current Gaussian — and minimize the negative log-likelihood
$-\mathbb{E}[\log\pi(a\mid s)]$, with a tiny entropy *penalty* ($-0.001\cdot\text{entropy}$, i.e. a small
push *against* collapsing the policy's variance to zero) to keep the action distribution from degenerating.
I run several BC gradient steps per `update()` call (the fill uses 20) so cloning dominates the secondary
PPO updates, clip the policy grad-norm at 0.5, and step the policy optimizer directly. The expert demos
again store no special structure I need beyond $(obs, acts)$.

One scaffold subtlety worth naming because it is load-bearing for *why* BC stays stable here: because the
learned reward is identically zero, the fixed running-reward-normalization and the PPO update operate on a
constant signal and contribute essentially nothing — they cannot inject the non-stationarity that
destabilized the adversarial methods. BC's only optimization is the stationary supervised NLL on a fixed
dataset. There is no game to lose.

So the falsifiable expectations, read against the two rungs below. If the diagnosis is right — that the
adversarial methods are dominated by min-max instability on clean demos, not by their occupancy-matching
principle — then BC should *beat AIRL on the aggregate*, and it should beat it most clearly where AIRL's
instability bit hardest: the terminating bodies. I expect BC's Hopper and Walker2d to land in the
low-thousands and the low-to-mid thousands respectively — not the {153, 169, 2198} two-dead-one-alive
pattern AIRL gave, but consistently-alive cloned gaits, because a supervised fit to a competent expert
keeps the body upright without needing a reward to emerge first. On HalfCheetah, where AIRL already reached
2514 and the body never terminates, I expect BC to be *competitive but not dramatically ahead* — possibly
even with one runaway seed (cloning a fast cheetah can overshoot) and one weak seed (cloning can also
underfit a high-dimensional gait), so a wide spread around the low-2000s. The clean signature that would
confirm the whole ladder's story: BC strongest overall, winning Hopper and Walker2d decisively while
roughly tying or modestly trailing AIRL on the non-terminal HalfCheetah — i.e. the simplest method beating
the principled ones precisely because the principle's cost (instability) outweighs its benefit
(occupancy matching) on clean demonstrations. If instead BC underperforms AIRL across the board, then the
covariate-shift penalty really does dominate here and the adversarial frame was right after all — but
given the two-thirds-dead AIRL seeds, I expect the opposite, and that is the bar BC must clear as the top
of this ladder.
