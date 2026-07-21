The default fill is plain DQN, and it is the right floor to start from because it is the most generic
thing the scaffold allows: a single linear head on the fixed 84-dimensional features, regressed toward
`r + γ max_{a'} Q(s',a';θ⁻)` with a squared error. Everything downstream — the encoder, the uniform
replay, the epsilon schedule, the hard target sync — is frozen, so the only lever I have is what sits on
top of those 84 features and how it is trained. Before I reach for a fancier target or a heavier loss, I
want to ask the cheapest possible question: is the *architecture* of that head wasting the data
I already have? Because if it is, I can fix it without changing the algorithm at all — same target, same
loss, same replay — and a free win with no algorithmic risk is worth taking first.

So let me stare at what the head is being asked to do. It takes the shared 84-dim feature and emits
`|A|` numbers, one per action, and the agent acts greedily on them. One thing nags. In most states of a
control task the action barely matters. Think about LunarLander hovering with the lander roughly upright
and centered: firing the left thruster, firing the right one, or doing nothing all lead to nearly the
same near-term future and nearly the same return. The only moments where the choice of action carries
real return information are the few where the lander is tilting toward a crash or drifting off the pad
and exactly one action saves it. So across the bulk of the state space, the `|A|` numbers the head
outputs are all nearly equal, and the only thing that actually varies and matters is their common
level — *how good is it to be in this state at all*.

But the single linear head has no way to know that. It estimates each of the `|A|` outputs more or less
independently, as `|A|` separate linear functions of the same features, and it has to keep them mutually
consistent everywhere. Worse — and this is the part that really bothers me about the *update*, not just
the representation — look at what one TD step touches. I sample a transition `(s,a,r,s',done)` from the
buffer; in `update` the loss is `(td_target − Q(s,a))²`, and `Q(s,a)` is read by `gather` on the single
taken action `a`. The gradient flows back only into the column of the head that produces `Q(s,·)` for
that one action. The outputs for the *other* actions in `s` get no direct signal from this transition.
So the "how good is this state" information — which is shared across all `|A|` actions and is exactly
what a bootstrapping algorithm leans on at every state, because the target is built from a `max` over
next-state values — is only ever nudged through whichever single action happened to be sampled. The most
important quantity in the whole network is getting the most diluted update.

Let me put a number on "most diluted," because the magnitude of the effect decides whether this is worth
chasing at all. The loop samples a uniform batch of 128 transitions and trains once every ten environment
steps. In plain DQN each transition's gradient enters exactly one column of the head — the column for
its sampled action — so across a batch the `|A|` columns split the 128 gradients between them, roughly
`128/|A|` apiece once the replay is well mixed. The state-value level the bootstrap actually leans on is
not even a named parameter; it is smeared across all `|A|` columns, so it only moves when the columns
move, and each column moved with a fraction of the batch. On LunarLander with `|A| = 4` that is about 32
informative gradients per column per batch; on Acrobot with `|A| = 3`, about 43; on CartPole with
`|A| = 2`, about 64. Now count what a dedicated value stream would get: the chain rule below gives
`∂Q_j/∂V = 1` for whichever action `j` was sampled, so `V` receives a direct gradient on all 128
transitions, every batch. The effective update count on the value estimate jumps from `~128/|A|` to
`128` — a 4× enrichment on LunarLander, 3× on Acrobot, 2× on CartPole. Two mechanistic predictions fall
out of that arithmetic: the gain should scale with `|A|` (more actions sharing one value) and with the
fraction of states where the action is nearly irrelevant (more value to share). LunarLander maximizes
both, which is why it is the natural place to look for the effect.

The value of a state and the relative merit of the actions within it are really two different objects,
and the single head conflates them. There is a name for the split. Define `A(s,a) = Q(s,a) − V(s)`, the
**advantage** — Baird had this in '93. `V(s)` is how good the state is; `A(s,a)` is how much better or
worse action `a` is than the state's baseline. Two identities fall right out and they are worth writing
because they tell me how to make the split well-posed. If `V(s) = E_{a∼π}[Q(s,a)]`, then
`E_{a∼π}[A(s,a)] = 0` — the advantages average to zero under the policy. And for a deterministic greedy
policy with `a* = argmax_a Q(s,a)`, `V(s) = Q(s,a*) = max_a Q(s,a)`, so `A(s,a*) = 0` and every other
advantage is non-positive. So I have two candidate "anchors" for pinning the decomposition: the mean and
the max.

The decomposition is exactly what I want, so let me make the head produce it directly. Keep the fixed
encoder — those 84 features are common to both objects and I cannot touch them anyway — and replace the
one linear head with two: a **value stream**, a single linear map `84 → 1` producing `V(s)`, and an
**advantage stream**, a linear map `84 → |A|` producing `A(s,·)`. Then recombine them into a single `Q`
output, because the whole appeal here is that I keep the exact input/output interface of an ordinary
Q-network: `forward(obs)` still returns `(batch, |A|)` Q-values, so the frozen evaluation harness still
argmaxes over them, the same max-target TD loss in `update` still drives it, and I have changed nothing
about the algorithm. If instead I exposed two heads with two separate losses, I would lose that
drop-in property; so the recombination must live inside the forward pass.

The obvious recombination, straight from `A = Q − V`, is to add them back: `Q(s,a) = V(s) + A(s,a)`,
broadcasting the scalar `V` across the `|A|` advantage entries. Clean — but let me sanity-check it before
I get attached, because the loss only ever sees `Q`. The TD loss is `(y − Q(s,a))²` and `y` itself is
built from `Q` values; `V` and `A` are internal and nothing pins them individually. Take any constant `c`
and set `V'(s) = V(s) + c`, `A'(s,a) = A(s,a) − c`. Then `V' + A' = V + A = Q`, identically. The loss is
unchanged. So there is an entire one-parameter family of `(V, A)` pairs producing the same `Q`, and
gradient descent has no reason to prefer the one where `V` is the true value: `V` could drift to `V + 100`
with `A` absorbing the `−100` and `Q` would not care. The decomposition is **unidentifiable**, and the
redundant degree of freedom is not harmless — two streams trading a constant back and forth is exactly
the kind of slack that makes optimization wander. The naive sum is dead; I have to stop `V` from hiding
an arbitrary offset inside `A`.

What pins it is one extra normalization that removes the shared offset before forming `Q`. Go back to the
identities. The **max** anchor: subtract the max advantage,
`Q(s,a) = V(s) + (A(s,a) − max_{a'} A(s,a'))`. Check it. Since `V` and `max A` are both constants across
actions, `argmax_a Q = argmax_a A`, so the greedy action is `a* = argmax A`; plug it in and
`Q(s,a*) = V + (A(s,a*) − max A) = V`. So `V` is forced to equal `max_a Q` and the centered advantage is
zero at the best action and non-positive elsewhere — clean textbook semantics. And the bad trade is gone:
under `A' = A − c`, `max A' = max A − c`, so `A' − max A' = A − max A` is unchanged while `V' = V + c`,
giving `Q' = Q + c ≠ Q` — the loss now sees the offset immediately. The max pins it.

But the max anchor has a training problem I should think through. The subtracted quantity `max_{a'} A`
tracks whichever action currently has the largest advantage, and during learning the identity of that
argmax flips around as the estimates wobble. Every time the leading action changes, the reference point
jumps discontinuously and the entire advantage vector has to re-center against the new maximum. The
advantages are being asked to compensate for any change in the *optimal* action's advantage — a moving,
jumpy target. That is exactly the sort of instability I am trying to *remove*, not add.

So use the smoother anchor — the other identity, `E_a[A] = 0`. To avoid depending on a changing policy
distribution, take the simplest policy-independent convention over the finite action set: subtract the
**uniform mean**, `Q(s,a) = V(s) + (A(s,a) − (1/|A|) Σ_{a'} A(s,a'))`. Does it kill the same bad trade?
Under `A' = A − c`, the mean shifts by `−c` too, so `A' − mean A' = A − mean A` is unchanged while
`V' = V + c`, so `Q' = Q + c ≠ Q`. Yes — mean-subtraction pins `V` against the centered advantage just
as the max did. What does it cost? At the greedy action `Q(s,a*) = V + (A(s,a*) − mean A) ≠ V` in general,
so `V` is no longer exactly `max_a Q`; it is the *mean* of the Q-values over actions, and the centered
advantage is `Q(s,a) − mean_{a'} Q(s,a')`. I lose the exact max-value interpretation. What I buy is
stability: the mean of `|A|` numbers is a smooth, slowly moving quantity that does not lurch when the
argmax flips, so the advantages only have to track the mean rather than chase the optimal action's
advantage. I take that trade — exact semantics for optimization stability — which is also the right call
here precisely because the whole point of this step is a *free architectural* win that does not destabilize
the algorithm. Crucially, subtracting any per-state constant never changes the rank order of the
actions, so `argmax_a Q` is identical to the naive sum and identical to `argmax_a A`: the greedy and
epsilon-greedy policies are exactly preserved. The aggregation is purely a training-time offset-control
device; it touches what is learned, not what is acted.

Now the payoff shows up in gradient flow, which was the original complaint. Write the sampled action as
`j` and `Q_j = V + A_j − (1/n) Σ_k A_k` with `n = |A|`. For any TD loss `ℓ(Q_j, y)` with
`δ = ∂ℓ/∂Q_j`, the chain rule gives `∂Q_j/∂V = 1`, `∂Q_j/∂A_j = 1 − 1/n`, and `∂Q_j/∂A_k = −1/n` for
`k ≠ j`. So `V(s)` receives the **full** TD signal on *every* sampled transition regardless of which
action was taken — the shared state value is now learned from all the data instead of being smuggled
through one action's column. The advantage gradients sum to zero across the vector —
`Σ_k ∂Q_j/∂A_k = (1 − 1/n) + (n−1)·(−1/n) = 0` — so the advantage stream can only reshape the readout,
never lift or lower its level; that level is `V`'s alone. In states where the actions are equivalent, the
advantage stream can settle to ≈0 across the board after centering and the network gets the value right
through `V` rather than fitting `|A|` separate near-equal numbers. That is precisely the redundancy I
started from, removed by construction.

Now ground this in *this* task's edit surface, because here is where it differs from the generic Atari
version of the idea, and I want the differences explicit. There is no conv torso to share and no
trunk-gradient pathology to fix: the shared trunk is the **fixed MLP encoder** (`obs_dim → 120 → 84`),
which I am forbidden to touch, so I cannot move the split earlier into a conv stack and there is no
`1/√2` feature-gradient rescale to apply — both streams are single linear maps off the same frozen
84-dim feature, and that is the whole architecture. Count the parameters exactly, because the runtime
check will reject anything that smells like added encoder capacity rather than an algorithmic change. The
default linear head is `84·|A| + |A|` (weights plus biases). The dueling pair is a value head
`84·1 + 1 = 85` plus an advantage head `84·|A| + |A|`, which is identical to the default head — so the
entire cost of the split is the 85-parameter value stream, and it is *independent of* `|A|`. Set that
against the frozen encoder, `obs_dim·120 + 120 + 120·84 + 84`: that is 11 244 parameters on LunarLander
(`obs_dim = 8`), 11 004 on Acrobot (6), and 10 764 on CartPole (4), all dominated by the shared
`120·84 + 84 = 10 164` second layer. So the split adds 85 against roughly 11 000 — under one percent —
and, crucially, it adds no nonlinearity and no width anywhere. That is exactly the profile the parameter
check is meant to wave through: whatever gain arrives is provably a reorganization of how the same 84
features are read out, not smuggled capacity. The
bootstrap target stays the scaffold's **plain DQN max target** — `r + γ max_{a'} Q(s',a';θ⁻)` — *not*
the double-DQN selection/evaluation split that the generic dueling recipe is usually paired with, because
the buffer, the target-sync cadence, and the update structure are the frozen DQN loop and I am only
allowed to change the head and the loss; decoupling overestimation is a separate concern. The
loss remains the squared TD error. The one stabilizer I do keep, and it is faithful to the generic
recipe, is a **global gradient-norm clip at 10** after `backward()`: the two-stream head plus the
bootstrapped max target can occasionally produce a large update, and clipping tames it cheaply; it is the
only optimization change and it does not interact with the frozen loop. Epsilon-greedy `select_action`
is unchanged — the centered `Q` has the same argmax as the default. So the literal edit is: split the
linear head into `value_head: 84 → 1` and `advantage_head: 84 → |A|`, return
`V + A − A.mean(dim=1, keepdim=True)` from `forward`, and add the norm clip to `update`; everything else
is the default DQN fill. (The full scaffold module is in the answer.)

Two other changes are reachable on this edit surface, and both are worse. A *nonlinear* head — a small
MLP `84 → 64 → |A|` in place of the linear map — strictly adds readout capacity but does nothing about
the complaint: a bigger head still emits `|A|` numbers whose shared level is only nudged through the
sampled action's column, so the gradient-dilution survives intact and I would be spending parameters to
fit the wrong thing better, inviting the very capacity check I want to pass. Swapping the plain max for
the double-DQN selection/evaluation split, `y = r + γ Q(s', argmax_{a'} Q(s',a';θ); θ⁻)`, attacks a real
DQN pathology — overestimation from a max over noisy values — but that is a bias in the *level* of the
target, orthogonal to how the head *reads out* the value once given; folding it in here would blur which
change paid, and overestimation is a separate concern. The head split is the one change that maps
one-to-one onto the defect I diagnosed — the diluted, shared state value.

What do I expect, and how would it fail? CartPole's default DQN already saturates the 500 cap, so the
split can only match it — the test is whether it *holds* 500 across seeds rather than adding variance.
Acrobot's reward is a dense negative time-to-goal where the action matters at almost every step, so the
value/advantage split has little redundancy to exploit and I expect a roughly on-par result. LunarLander
is the interesting one: long hovering stretches where the action barely matters, interleaved with the
few decisive correction moments, are exactly the structure the value stream is built to exploit, so this
is where I most expect a gain — *if* one materializes. But I am wary: LunarLander's long horizons,
deceptive crash basin, and misleading shaped reward make it high-variance, and a single linear value
stream off a frozen 84-dim feature is not much capacity to nail the state value across that landscape;
the mean-anchor also means `V` is only the mean Q, so where one action is sharply better the centered
advantage carries most of the signal and the architectural benefit shrinks. If LunarLander comes back
low or wildly seed-dependent, the reading is clean: reorganizing the readout was not the binding
constraint here, and whatever holds LunarLander down is not in the architecture of the readout at all.
