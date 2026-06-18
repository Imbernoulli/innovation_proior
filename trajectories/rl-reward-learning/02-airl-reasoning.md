The GAIL numbers came back exactly along the fault line I predicted, and the shape of the failure tells
me precisely what to fix. HalfCheetah limped to a mean of 1646 (1381 / 1602 / 1956 across seeds) — alive,
because that body has no terminal state, so even a partly-collapsed gait keeps accumulating return over a
fixed-length episode. But Hopper cratered to a mean of 25.7 (4 / 37 / 36) and Walker2d to 77.8 (72 / 9 /
152) — both essentially zero, both with the seed-to-seed jitter of a body that falls almost immediately.
That is the saturation mechanism made visible: on these clean, tight expert demonstrations the
discriminator *wins* the min-max game, its logits saturate, the reward $-\log(1-D)$ flattens, and on the
terminating bodies the policy gets no stable signal to stay upright, so episodes end on the first stumble
and the return collapses. HalfCheetah survives only because its non-terminal structure forgives a bad
gait. The diagnosis is not "tune the discriminator harder" — I already bumped the inner updates and the
batch, and added input normalization, and Hopper still died. The diagnosis is that the *reward itself*
carries nothing usable once the game saturates: GAIL's discriminator at its optimum is $\tfrac12$
everywhere, an unstructured object from which no reward can be extracted, so when it saturates short of
that optimum the policy is left ascending an arbitrary, shaped-by-accident signal.

So the next rung has to make the recovered reward *structured* — a real reward function, not just a
classifier score that happens to be high near the expert. Let me reason from what GAIL leaves on the
table to what that structure should be.

Start from the same adversarial frame, because occupancy matching is still the right idea — it killed
BC's compounding error. The GAN reaches its optimum when the policy matches the expert, at which point a
*free* discriminator is $\tfrac12$ everywhere and tells me nothing. But I do not have to use a free
discriminator. In the GAN the optimal discriminator is $D^*=p_{\text{data}}/(p_{\text{data}}+q)$, and
here I *know* $q$ — the generator density is my policy, which I can evaluate as $\log\pi(a\mid s)$.
So I plug $q$ in and let the discriminator model only the data density, in the Boltzmann form. The
discriminator becomes $D=\exp f/(\exp f+\pi)$, i.e. a sigmoid whose logit is $f(s,a,s')-\log\pi(a\mid
s)$: a learned reward term minus the *filled-in* log policy density. Two payoffs. First, the optimal
discriminator is now independent of the generator — it is optimal exactly when $\exp f$ matches the data
density up to the partition constant — which makes the adversarial game far more stable than GAIL's,
because the discriminator is no longer chasing a moving target. That alone should help on the bodies
where GAIL's instability was fatal. Second, and the whole point: I can read a reward back out of $f$. A
free discriminator at the optimum is $\tfrac12$ and yields nothing; this *structured* one keeps $f$ as a
reward.

What reward does it recover? At the GAN optimum the policy matches the expert, $D=\tfrac12$, and
$\exp f/(\exp f+\pi_E)=\tfrac12$ forces $f^*=\log\pi_E(a\mid s)$. Under the maximum-entropy model the log
of the optimal policy is the advantage, so $f^*=A^*(s,a)$ — the recovered reward is the expert's
advantage. That is already strictly more than GAIL gives me. But the advantage is an *entangled* object:
$A^*=Q^*-V^*=r+\gamma V^*(s')-V^*(s)$ under deterministic dynamics, which is the true reward shaped by the
value function $V^*$. The value function is baked in. For the task as posed — recover a reward, train a
policy, score it in the *same* environment — entanglement is not fatal: re-optimizing the advantage in the
training MDP reproduces the expert. But entanglement is exactly why GAIL's reward was so brittle. The raw
$f$, free to be any function, can pour all its expressive power into matching the advantage's
value-function component and very little into the part that actually distinguishes expert behavior — and
when training saturates, that ill-conditioned, value-dominated signal is what the policy is stuck with.
If I give the network an explicit place to *put* the value-function shaping, the remainder is forced to be
a cleaner reward, and the whole object is better-conditioned for the policy to optimize.

The structure to impose is the only policy-invariant degree of freedom there is: potential-based shaping.
The transform $r+\gamma\Phi(s')-\Phi(s)$ leaves the optimal policy unchanged for any potential $\Phi$, and
without knowing the dynamics it is the *only* class of reward transformations that is policy-invariant. So
I carve the discriminator's $f$ into a reward term and a potential-shaping term, each its own network:
$f(s,a,s')=g(s,a)+\gamma\,h(s')-h(s)$. Whatever shaping the optimization wants to apply, it dumps into
$h$; $g$ is left to be the unshaped reward. Working the algebra at the optimum (with the chaining lemma
under decomposable dynamics) gives $h^*=V^*$ up to a constant and $g^*=r$ up to a constant — $h$ soaks up
exactly the value-function shaping that made the advantage entangled, and $g$ comes out clean. This is the
AIRL discriminator, and the structure is what should stabilize training on the terminating bodies:
$h$ absorbing the value gradient means $g$ does not have to, so the reward handed to the policy is
better-behaved than GAIL's value-dominated mush.

Now I have to land this in *this* scaffold, and the scaffold forces several concrete departures from the
clean derivation that I must respect line by line.

First, the discriminator logit needs $\log\pi(a\mid s)$, which means the module must read the policy. The
scaffold hands the policy in through `set_policy(policy, optimizer)` if I define it — but the policy
*learner* is the fixed PPO loop, so I take the policy *reference* and ignore the optimizer (I do not train
the policy; PPO does). In `update()` I evaluate $\log\pi(a\mid s)$ under no-grad on both expert and policy
batches and subtract it from $f$ to form the discriminator logit; expert label $1$, policy label $0$, BCE.
That is the structured discriminator, not a free classifier — the $-\log\pi$ term is the whole reason the
optimal discriminator is generator-independent.

Second — the terminal-state subtlety, which is sharp on exactly the bodies GAIL killed. The shaping
$\gamma h(s')-h(s)$ preserves the optimal policy only if I am honest about terminal states: at an
episode's final transition there is no genuine $s'$, and the real value function sets a terminal state's
future value to zero. If I let $h(s')$ fire on a terminal "next state" I add a phantom potential that
breaks policy-invariance — and with Hopper and Walker2d *terminating on falling*, this is not a corner
case, it is most of the interesting transitions. So I zero the shaping when the transition is terminal:
$\gamma(1-\text{done})\,h(s')-h(s)$. The reward net's `raw_f` takes a `done` argument and multiplies
$h(s')$ by $(1-\text{done})$. This done-aware shaping is the single most important reason AIRL should
rescue the terminating bodies that GAIL collapsed on — it keeps $f$ a *valid* potential-shaped reward
across variable-length episodes, instead of an invalid one that paid the policy for phantom future value
right at the moment the body fell.

Third, the normalization layering, which is dictated by the fixed loop and is subtle. The substrate
applies its *own* running mean/std normalization to the buffer rewards before the PPO update — fixed, not
editable. AIRL's raw shaped $f$ can have a large, drifting scale (it is a difference of three network
outputs), and if I feed that raw value into the fixed buffer normalization, the running stats chase a
moving target and either saturate or obliterate the signal. So I add a *second* RunningNorm on the reward
net's *output*: `_out_rms` whitens $f$ so the value entering the fixed template-level normalization is
already roughly unit-variance, making that fixed step near-identity rather than destructive. `compute_reward`
returns the *normalized* shaped $f$ (under no-grad, with `done` unavailable at rollout time so the terminal
correction is applied only during `update()` on the discriminator side). And as in GAIL I keep a RunningNorm
on the *obs inputs* (`_obs_rms`, refreshed each round from the freshest policy rollout) so the
discriminator cannot cheat on raw observation scale. Three normalizations now coexist: obs-input (mine),
reward-output (mine), buffer-reward (the fixed loop's) — and the middle one exists specifically to keep
the third from collapsing the signal.

Fourth, the budget knobs again. Same constraint as GAIL: `irl_batch_size` and `n_irl_updates_per_round`
are fixed, too few against a fast PPO policy, so I bump `_inner_updates=4` and `_batch_mult=4` inside
`update()`, resampling fresh expert and policy minibatches each inner step, and refresh `_out_rms` from the
concatenated raw $f$ each step. One honest concession the math does not force but the data does: the expert
demos store no `done` flags, so I assume expert transitions are non-terminal (correct for HalfCheetah,
which never terminates; mildly wrong for Hopper/Walker terminal states, but the reference imitation library
also lacks expert dones by default). I pass `expert_done_zeros` for the expert side and the real
`policy_dones` for the generator side. The architecture: $g$ is an MLP $[s,a]\to256\to256\to1$ and $h$ is
an MLP $[s]\to256\to256\to1$ — the largest reward net on the ladder, which is exactly why the scaffold's
parameter budget was sized at $1.05\times$ the AIRL net.

Now the falsifiable expectations, read directly against GAIL's numbers. The structured, done-aware reward
should help *most* exactly where GAIL failed worst — the terminating bodies — because that is where the
phantom-terminal-potential and the saturated unstructured reward did the damage. So I expect Hopper and
Walker2d to climb decisively off GAIL's near-zero floor (25.7 and 77.8): not to the expert, but into the
hundreds-to-low-thousands, with the generator-independent optimal discriminator and the done-aware shaping
giving the policy a stable signal to keep the body upright long enough to accumulate return. On
HalfCheetah, GAIL already survived at 1646 thanks to non-termination, so AIRL's terminal-state fix buys it
less there; I expect a modest improvement (the stabler game and the better-conditioned reward should still
push it up, into the low-to-mid 2000s, but the gap over GAIL should be smaller than on the terminating
bodies). The clean signature that would confirm the diagnosis: AIRL beats GAIL on *every* environment, and
the *margin* is largest on Hopper and Walker2d and smallest on HalfCheetah — the inverse of where GAIL's
terminal-driven collapse hit hardest. If instead AIRL also collapses on the terminating bodies, then the
problem is not the reward structure at all but the adversarial frame itself on clean demos — and the next
rung would have to abandon adversarial reward learning entirely. That is the bar AIRL must clear, and it is
exactly the question step 3 will answer.
