The GAIL numbers came back exactly along the fault line I predicted, and the shape of the failure tells
me precisely what to fix. HalfCheetah limped to a mean of 1646 (1381 / 1602 / 1956 across seeds) — alive,
because that body has no terminal state, so even a partly-collapsed gait keeps accumulating return over a
fixed-length episode. But Hopper cratered to a mean of 25.7 (4 / 37 / 36) and Walker2d to 77.8 (72 / 9 /
152) — both essentially zero. Let me read the arithmetic rather than eyeball it, because the pattern is
the diagnosis. HalfCheetah beats Hopper by a factor of $1646/25.7\approx64$ and Walker2d by
$1646/77.8\approx21$: two orders of magnitude of gap between the non-terminating body and the terminating
ones. And within the terminating bodies the seeds are not merely low, they are *ragged* — Hopper's three
seeds span $4$ to $37$, a factor of nine apart around a mean of $26$; Walker's span $9$ to $152$, a factor
of seventeen. That raggedness is the fingerprint of episodes that end on the first stumble: a body that
falls at step 5 versus step 40 produces returns that differ by an order of magnitude while both are, in
absolute terms, dead. If the policy had learned *anything* stable the seeds would cluster; instead they
scatter across the small returns you get from a handful of steps before termination. That is the
saturation mechanism made visible: on these clean, tight expert demonstrations the discriminator *wins*
the min-max game, its logits saturate, the reward $-\log(1-D)$ flattens toward the $0.13$-and-falling
values I traced for policy transitions, and on the terminating bodies the policy gets no gradient telling
it which transitions kept the torso up, so episodes end on the first stumble and the return collapses.
HalfCheetah survives only because its non-terminal structure integrates even that flattened reward over a
full-length episode. The diagnosis is not "tune the discriminator harder" — I already bumped the inner
updates to $16\times$ and added input normalization, and Hopper still died at $26$. The diagnosis is that
the *reward itself* carries nothing usable once the game saturates: a free discriminator at its optimum is
$\tfrac12$ everywhere, an unstructured object from which no reward can be extracted, so when it saturates
short of that optimum the policy is left ascending an arbitrary, shaped-by-accident signal.

So the next rung has to make the recovered reward *structured* — a real reward function, not just a
classifier score that happens to be high near the expert. Before reaching for new machinery, let me check
the design space, because there are cheaper moves than changing the discriminator and I should rule them
out on paper. Option one: keep GAIL and lean harder on the knobs I already have — more inner updates, a
bigger effective batch, a slower discriminator learning rate to delay saturation. But the arithmetic
above already falsifies this: I ran $16\times$ the discriminator signal and the terminating bodies still
died, and the failure is not that the discriminator is *undertrained* — it is that its optimum is
structureless, so delaying arrival at that optimum only postpones the collapse rather than preventing it.
Option two: change the divergence — a Wasserstein critic instead of the JS classifier, which does not
saturate to $\tfrac12$ and keeps a gradient even when the distributions are far apart. Tempting, but it
buys the wrong thing: a non-saturating *critic* still yields a critic score, not a reward with any
guaranteed relationship to the environment's true reward, and I would still be handing PPO an object I
cannot interpret across terminal transitions. Option three, the one I will take: keep the adversarial
frame (occupancy matching is still the right idea — it killed BC's compounding error) but change what the
discriminator *is* so that a genuine reward falls out of it. Let me reason from what GAIL leaves on the
table to what that structure should be.

The GAN reaches its optimum when the policy matches the expert, at which point a *free* discriminator is
$D^*=p_E/(p_E+q)=\tfrac12$ everywhere and tells me nothing. But I do not have to use a free discriminator.
In the GAN the optimal discriminator is $D^*=p_E/(p_E+q)$, and here I *know* $q$ — the generator density
is my own policy, which I can evaluate as $\log\pi(a\mid s)$. So I plug $q$ in and let the discriminator
model only the data density, in the Boltzmann form $D=\exp f/(\exp f+\pi)$. Let me confirm this is just a
constrained sigmoid: its logit is $\log\frac{D}{1-D}=\log\frac{\exp f/(\exp f+\pi)}{\pi/(\exp f+\pi)}=
\log\frac{\exp f}{\pi}=f-\log\pi(a\mid s)$. So $D=\sigma\big(f(s,a,s')-\log\pi(a\mid s)\big)$ — a sigmoid
whose logit is a learned reward term minus the *filled-in* log policy density. Two payoffs. First, the
optimal discriminator is now independent of the generator: $D$ is optimal exactly when $\exp f$ matches the
data density up to the partition constant, a condition that does not mention $\pi$, so as the policy moves
the *target* for $f$ does not, and the discriminator stops chasing a moving object. That directly attacks
the non-stationarity that made GAIL's value function chase a shifting reward — and it should help most
where GAIL's instability was fatal. Second, and the whole point: I can read a reward back out of $f$,
whereas a free discriminator at the optimum is $\tfrac12$ and yields nothing.

What reward does it recover? At the GAN optimum the policy matches the expert, $D=\tfrac12$, and
$\exp f/(\exp f+\pi_E)=\tfrac12$ forces $\exp f=\pi_E$, i.e. $f^*=\log\pi_E(a\mid s)$. Under the
maximum-entropy model the optimal policy is $\pi_E(a\mid s)\propto\exp A^*(s,a)$, so its log is the
advantage up to a state-only normalizer: $f^*=A^*(s,a)$. That is already strictly more than GAIL gives me.
But the advantage is an *entangled* object: $A^*=Q^*-V^*=r+\gamma V^*(s')-V^*(s)$ under deterministic
dynamics, which is the true reward shaped by the value function $V^*$. The value function is baked in. For
the task as posed — recover a reward, train a policy, score it in the *same* environment — entanglement is
not fatal: re-optimizing the advantage in the training MDP reproduces the expert. But entanglement is
exactly why GAIL's reward was so brittle. The raw $f$, free to be any function, can pour all its
expressive capacity into matching the advantage's value-function component and very little into the part
that actually distinguishes expert behavior — and when training saturates, that ill-conditioned,
value-dominated signal is what the policy is stuck with. If I give the network an explicit place to *put*
the value-function shaping, the remainder is forced to be a cleaner reward, and the whole object is
better-conditioned for the policy to optimize.

The structure to impose is the only policy-invariant degree of freedom there is: potential-based shaping.
Let me verify the invariance claim myself so I know I am not throwing away signal, because it is the whole
license for the decomposition. Take any potential $\Phi(s)$ and form the shaped reward
$r'=r+\gamma\Phi(s')-\Phi(s)$. Its optimal action-value satisfies $Q'^*(s,a)=Q^*(s,a)-\Phi(s)$ (the
telescoping shaping terms collapse the discounted sum into a single boundary $-\Phi(s)$), hence
$V'^*(s)=V^*(s)-\Phi(s)$, and the advantage is $A'^*=Q'^*-V'^*=(Q^*-\Phi)-(V^*-\Phi)=Q^*-V^*=A^*$ —
unchanged. So shaping shifts $Q$ and $V$ together and leaves the greedy policy fixed for *any* $\Phi$;
without knowing the dynamics it is the only class of reward transformations with that property. So I carve
the discriminator's $f$ into a reward term and a potential-shaping term, each its own network:
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

Second — the terminal-state subtlety, which is sharp on exactly the bodies GAIL killed, and I can size it
from the GAIL numbers. The shaping $\gamma h(s')-h(s)$ preserves the optimal policy only if I am honest
about terminal states: at an episode's final transition there is no genuine $s'$, and the real value
function sets a terminal state's future value to zero. If I let $h(s')$ fire on a terminal "next state" I
add a phantom potential $\gamma h(s')$ right where the body fell — I *pay the policy for the imagined
future value of a state that has no future*. Consider how many transitions this corrupts. A Hopper episode
that dies at step $L$ contributes one terminal transition out of $L$, a fraction $1/L$. GAIL's Hopper died
so fast that its returns were in the single-to-low-double digits, meaning $L$ was on the order of a few to
a few tens of steps — so the terminal transition was a *sizable fraction*, sometimes a tenth or more, of
every episode the policy saw, and every one of them carried a phantom potential that told the policy the
fall had future value. This is not a corner case on the terminating bodies; it is a systematic
mis-shaping concentrated exactly at the falls I am trying to prevent. So I zero the shaping when the
transition is terminal: $\gamma(1-\text{done})\,h(s')-h(s)$. The reward net's `raw_f` takes a `done`
argument and multiplies $h(s')$ by $(1-\text{done})$. This done-aware shaping is the single most important
reason AIRL should rescue the terminating bodies that GAIL collapsed on — it keeps $f$ a *valid*
potential-shaped reward across variable-length episodes, instead of an invalid one that paid the policy
for phantom future value at the moment of the fall.

Third, the normalization layering, which is dictated by the fixed loop and is subtle. The substrate
applies its *own* running mean/std normalization to the buffer rewards before the PPO update — fixed, not
editable. AIRL's raw shaped $f$ can have a large, drifting scale: it is a difference of three network
outputs, $g(s,a)+\gamma h(s')-h(s)$, so its magnitude can wander into the tens as $g$ and $h$ grow during
training, and unlike GAIL's bounded $-\log(1-D)\in(0,\infty)$ it is not intrinsically scale-controlled. If
I feed that raw value into the fixed buffer normalization, the running stats chase a moving target and
either saturate or obliterate the signal. So I add a *second* RunningNorm on the reward net's *output*:
`_out_rms` whitens $f$ to roughly unit variance, so the value entering the fixed template-level
normalization is already $\sim\mathcal{N}(0,1)$ and that fixed step becomes near-identity (whitening an
already-whitened signal changes almost nothing) rather than destructive. `compute_reward` returns the
*normalized* shaped $f$ under no-grad — and here the contract bites once more: `compute_reward` is called
at rollout time with no `done` available, so the terminal correction cannot be applied there and is instead
enforced during `update()` on the discriminator side, where the policy dones are passed in. And as in GAIL
I keep a RunningNorm on the *obs inputs* (`_obs_rms`, refreshed each round from the freshest policy
rollout) so the discriminator cannot cheat on raw observation scale. Three normalizations now coexist:
obs-input (mine, anti-cheating), reward-output (mine, scale-taming), buffer-reward (the fixed loop's) —
and the middle one exists specifically to keep the third from collapsing the signal.

Fourth, the budget knobs again. Same constraint as GAIL: `irl_batch_size` and `n_irl_updates_per_round`
are fixed, too few against a fast PPO policy, so I bump `_inner_updates=4` and `_batch_mult=4` inside
`update()` — the same $16\times$ effective discriminator signal — resampling fresh expert and policy
minibatches each inner step, and refresh `_out_rms` from the concatenated raw $f$ each step. One honest
concession the math does not force but the data does: the expert demos store no `done` flags, so I assume
expert transitions are non-terminal (exactly correct for HalfCheetah, which never terminates; mildly wrong
for the Hopper/Walker terminal states in the demos, but the tuned reference configs also lack expert dones
by default). I pass `expert_done_zeros` for the expert side and the real `policy_dones` for the generator
side. The architecture: $g$ is an MLP $[s,a]\to256\to256\to1$ and $h$ is an MLP $[s]\to256\to256\to1$.
Let me count parameters against the cap, since this is now two nets instead of one. On HalfCheetah, $g$
has input width $17+6=23$, so $23\cdot256+256=6144$ plus $65792$ plus $257$, total $72{,}193$; $h$ has
input width $17$, so $4608+65792+257=70{,}657$; together $\approx142{,}850$, roughly $1.87\times$ GAIL's
single $76{,}545$-parameter discriminator. This is the largest reward net on the ladder so far, which is
exactly why the scaffold's parameter budget is sized at $1.05\times$ this net — I sit right at the cap, and
the decomposition into two heads is what spends it.

Let me verify the done-aware fix does what I claim on a transition I can trace by hand, because the whole
rescue hinges on it and it is cheap to check the sign. Take a two-state fragment where the potential net
has learned something sensible: $h(s)$ large and positive for an upright torso (high future value) and
$h(s_{\text{fallen}})$ near zero. Now the policy takes a bad action and the body falls at this step, so
the transition is terminal, $\text{done}=1$, and the "next state" $s'$ is a nominal fallen configuration
with, say, $h(s')\approx0$ but which the untrained $h$ might just as easily read as *positive* because it
has never been told this state has no future. Without the mask, $f=g(s,a)+\gamma h(s')-h(s)$: if $h(s')$
comes out positive the shaping term $\gamma h(s')$ *adds* reward to the very transition where the body
died — the policy is paid, at the moment of the fall, for the imagined future of a state that has none, so
falling can look no worse (or even better) than staying up. With the mask, $\gamma(1-\text{done})h(s')=0$
and the transition reduces to $f=g(s,a)-h(s)$: it loses the large positive $-h(s)$ boundary term (an
upright state's potential *subtracted*), i.e. leaving a high-value upright state to fall is correctly
scored as a large negative shaping contribution. So the mask flips the sign of the incentive at exactly
the terminal transition — from "falling has phantom future value" to "leaving a good state to fall costs
you its potential." That is precisely the gradient a Hopper or Walker policy needs to stop toppling on the
first stumble, and it is absent from the unmasked (and from GAIL's) reward.

The $-\log\pi(a\mid s)$ term deserves one scale check too, since I am subtracting it from $f$ inside the
logit and a wildly-scaled subtrahend could swamp the learned reward. Under the Gaussian actor,
$\log\pi(a\mid s)=-\tfrac12\sum_i\big[(a_i-\mu_i)^2/\sigma_i^2+\log(2\pi\sigma_i^2)\big]$ over the
action dimensions. For a competent expert whose actions sit near the policy mean early on, the squared
term is small and the magnitude is dominated by the $-\tfrac12\sum_i\log(2\pi\sigma_i^2)$ normalizer,
which for a $3$-to-$6$-dimensional action with order-unity $\sigma$ is a handful — not hundreds. So
$-\log\pi$ contributes an $O(1)$-to-$O(10)$ offset to the logit, comparable to $f$ itself, exactly the
regime where it meaningfully constrains the discriminator (making it generator-independent) without
drowning the learned reward term. It is evaluated under no-grad so no gradient flows into the policy from
the discriminator side; the policy only ever moves through PPO on the reward I serve.

Now the falsifiable expectations, read directly against GAIL's numbers. The structured, done-aware reward
should help *most* exactly where GAIL failed worst — the terminating bodies — because that is where the
phantom-terminal-potential and the saturated unstructured reward did the damage. So I expect Hopper and
Walker2d to climb decisively off GAIL's near-zero floor ($25.7$ and $77.8$): not to the expert, but into
the hundreds-to-low-thousands, with the generator-independent optimal discriminator and the done-aware
shaping giving the policy a stable enough signal to keep the body upright long enough to accumulate return.
On HalfCheetah, GAIL already survived at $1646$ thanks to non-termination, so AIRL's terminal-state fix
buys it less there; I expect a modest improvement (the stabler game and the better-conditioned reward
should still push it up, into the low-to-mid $2000$s, but the gap over GAIL should be a fraction of what I
expect on the terminating bodies). The clean signature that would confirm the diagnosis: AIRL beats GAIL
on *every* environment, and the *margin* is largest on Hopper and Walker2d and smallest on HalfCheetah —
the inverse of where GAIL's terminal-driven collapse hit hardest. If instead AIRL also collapses on the
terminating bodies, then the problem is not the reward structure at all but the adversarial frame itself
on clean demos — and the next rung would have to abandon adversarial reward learning entirely. That is the
bar AIRL must clear, and it is exactly the question step 3 will answer.
