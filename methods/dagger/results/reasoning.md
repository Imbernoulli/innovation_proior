Let me start from what actually goes wrong, because the failure mode is concrete and I have watched it happen. I want a controller, and I have an expert `π*` who demonstrates good behavior. The obvious thing — the thing everyone does — is to log the expert's trajectories, collect every `(state, action)` pair, and fit a classifier or regressor to predict the expert's action from the state. Pure supervised learning. In the i.i.d. story it should work: if my fitted policy disagrees with the expert only `ε` of the time on the states the expert visits, surely it's a good policy. But when I deploy it, it isn't. Pomerleau saw this with ALVINN steering a car: the net, driving for itself, drifts a little off the center of the road, and now it's looking at a scene no human demonstrator ever produced, because a good human driver never lets the car get that far off-center — so there's no "steer hard back to the middle" example in the training set, the net guesses, guesses wrong, drifts more, and the whole thing falls apart. The errors don't just happen; they *compound*.

So let me make this precise, because I suspect the compounding isn't a tuning failure but something structural. The quantity I actually care about is the cost of the learned policy under *its own* induced state distribution. Write `d^t_π` for the distribution of states at time `t` when I run `π` from the start, and `d_π = (1/T) Σ_{t=1}^T d^t_π` for the average state distribution over a horizon of `T` steps. With a task cost `C(s,a)` bounded in `[0,1]` and `C_π(s) = E_{a∼π(s)}[C(s,a)]`, the total cost-to-go is `J(π) = Σ_{t=1}^T E_{s∼d^t_π}[C_π(s)] = T·E_{s∼d_π}[C_π(s)]`. In imitation I usually can't see `C`; I see the expert, and I minimize a surrogate `ℓ(s,π)` — say the expected 0-1 disagreement with `π*` at `s`. The supervised approach fits

  `π̂_sup = argmin_{π ∈ Π} E_{s ∼ d_{π*}}[ ℓ(s,π) ]`,

i.e. it minimizes loss under `d_{π*}`, the *expert's* state distribution. But I deploy `π̂` and it generates `d_{π̂}`, a *different* distribution. That mismatch is the whole disease, and I want the exact price of it. Suppose `ℓ` upper-bounds the 0-1 loss and `E_{s∼d_{π*}}[ℓ(s,π̂)] = ε`. Picture the run: at each step there's about an `ε` chance `π̂` disagrees with what the expert would do. The first time it disagrees, it can land in a state `π*` would never have reached — off the manifold of demonstrations — and on that off-distribution tail I have *no* guarantee at all; the policy can pay the maximal cost `1` for every one of the remaining steps. So one early mistake, probability `~ε`, can cost up to `~T`. Over `T` chances to make that first mistake, the expected extra cost is on the order of `T · T · ε`. That gives me

  `J(π̂_sup) ≤ J(π*) + T² ε`.

And is the `T²` real or just a loose bound from my hand-waving above? My counting argument ("`T` chances, each costing up to `T`") could easily have over-counted; before I trust the quadratic I want a construction where I can read off the exact mistake count. Kääriäinen's sequence-prediction example gives one: predict the next output from the previous *correct* output with per-step error `ε`, and the expected number of mistakes over length `T` is `M(T,ε) = T/2 − (1−(1−2ε)^{T+1})/(4ε) + 1/2`. I should not just trust that this "behaves like `Θ(T²ε)`" — let me actually expand it for small `ε`. The binomial coefficient `C(T+1,2) = T(T+1)/2`. Write `(1−2ε)^{T+1} = 1 − 2ε(T+1) + 4ε²·T(T+1)/2 − O(ε³)`. Then `1 − (1−2ε)^{T+1} = 2ε(T+1) − 4ε²·T(T+1)/2 + O(ε³)`, and dividing by `4ε`,

  `(1−(1−2ε)^{T+1})/(4ε) = (T+1)/2 − ε·T(T+1)/2 + O(ε²)`.

Substituting back, the `T/2`, the `−(T+1)/2`, and the `+1/2` cancel down to zero (`T/2 − (T+1)/2 + 1/2 = 0`), and what survives is `M(T,ε) = ε·T(T+1)/2 + O(ε²)`. So in the small-`ε` regime the *leading term is exactly* `ε T(T+1)/2`, which is `Θ(T²ε)`. Let me sanity-check the algebra against the closed form numerically. At `T = 32, ε = 10^{-5}`: the formula gives `M = 5.279×10^{-3}`, and `ε T(T+1)/2 = 10^{-5}·32·33/2 = 5.280×10^{-3}` — agreement to the part I kept. At `T = 8, ε = 10^{-3}` the formula gives `0.03583` against `ε T(T+1)/2 = 0.036`, and the gap is the `O(ε²)` term I dropped (here `εT² = 0.064`, so the correction is genuinely second-order, not a sign that the leading term is wrong). The ratio `M/(εT²)` runs `0.75, 0.62, 0.56, 0.53, 0.51` for `T = 2,4,8,16,32` — drifting toward `1/2` exactly as `ε T(T+1)/2 ≈ εT²/2` predicts. So the quadratic is not an artifact of my loose counting; a concrete problem hits `~εT²/2` mistakes. And there's an imitation example with cost exactly `(1−εT)J(π*) + T² ε`. The bound is *tight*: no amount of cleverness inside the supervised box gets me below quadratic in `T`. That settles which knob matters — the quadratic isn't because my classifier is bad; it's because I trained it on the wrong distribution. The classifier never saw the states its own mistakes create, so it never learned to recover, so its mistakes compound. The fix has to change *which states I train on*, not which classifier I use. Wall, but an informative one: I need the training distribution to be the policy's own.

The obstacle is a chicken-and-egg problem. I want `argmin_{π∈Π} E_{s∼d_π}[ℓ(s,π)]`, the loss under the policy's *own* distribution. But `d_π` depends on `π`, the very thing I'm solving for, and the dynamics are unknown and complicated so I can't compute `d_π` in closed form — I can only *sample* it by running `π`. Worse, because the input distribution `d_π` rides on `π`, the objective is non-convex even when `ℓ(s,·)` is convex in `π` for each fixed `s`: moving `π` moves the distribution I'm averaging over, not just the integrand. So I can't just gradient-descend the thing directly.

Let me look at what people have tried to get around this, because two existing approaches already attack the distribution mismatch, and seeing exactly where each one stalls will tell me what I'm missing. The first is forward training. The idea: if the problem is that `π̂` is tested on states it wasn't trained on, then train it on the right states by construction. Build a *non-stationary* policy — one classifier `π_t` per time step — and train them in order, `t = 1, 2, …, T`. At step `t`, I've already fixed `π_1,…,π_{t-1}`, so I can run them to sample `d^t`, the actual distribution of states at time `t` that my deployed policy will face, and I fit `π_t` to mimic `π*` on exactly that `d^t`. Now each piece is trained on its own test distribution. Does the bound improve? Let me redo the telescoping carefully, because I'll reuse this machinery later. Consider the policy `π_{1:k}` that runs my learned `π` for the first `k` steps and then switches to the expert `π*`. Then

  `J(π) = J(π*) + Σ_{t=0}^{T-1} [ J(π_{1:T-t}) − J(π_{1:T-t-1}) ]`,

a telescoping sum where the leftmost term is `J(π_{1:T}) = J(π)` and the rightmost is
`J(π_{1:0}) = J(π*)`. Each difference `J(π_{1:T-t}) − J(π_{1:T-t-1})` is the cost of running `π`
one extra step before reverting to `π*`, which I can write per time-step: defining the
cost-to-go `Q^{π*}_{T-t+1}(s,a)` as the `(T-t+1)`-step cost of taking action `a` in `s` then
following `π*`,

  `J(π) = J(π*) + Σ_{t=1}^{T} E_{s∼d^t_π}[ Q^{π*}_{T-t+1}(s,π) − Q^{π*}_{T-t+1}(s,π*) ].`

Now suppose the per-state cost-to-go gap from one off-policy action is bounded,
`Q^{π*}_{T-t+1}(s,a) − Q^{π*}_{T-t+1}(s,π*) ≤ u` for all `a`, `t`. The difference inside the
expectation is nonzero only when `π` and `π*` pick different actions, and the probability of
that is upper-bounded by `ℓ(s,π)`; when it happens the increase is at most `u`. So

  `J(π) ≤ J(π*) + u Σ_{t=1}^{T} E_{s∼d^t_π}[ℓ(s,π)] = J(π*) + u T ε`,

with `ε = E_{s∼d_π}[ℓ]`. Linear in `T` now — provided `u` is small. And `u` *is* small in the
cases I care about: if the cost is exactly the 0-1 imitation loss then `u ≤ 1` (the expert pays
zero cost-to-go from anywhere), and if `π*` recovers quickly from disturbances — say the chain
under `π*` mixes rapidly — then `u = O(1)` regardless. Good. So forward training genuinely
beats the quadratic. But — and this is the wall — it trains and stores `T` separate policies,
in strict sequence, and I *cannot stop early*: there is no policy for steps beyond where I've
gotten to. For a robot with `T` in the thousands, or a task whose horizon isn't even
well-defined, this is hopeless. I throw it out as a practical method but keep the bound; that
`uTε` is exactly the kind of guarantee I want, and the telescoping argument is going to come
back.

The second existing approach tries to fix forward training's impracticality by going back to a
*stationary* policy, built up as a stochastic mixture — SMILe, and its cousins SEARN and
Conservative Policy Iteration. The instinct here is the CPI one: don't change the executed
policy abruptly, or the state distribution lurches and everything you estimated goes stale;
change it *slowly*. So SMILe starts from `π_0` that always queries and executes the expert.
At iteration `n`, it trains a fresh `π̂_n` to mimic the expert under the trajectories `π_{n-1}`
induces, then nudges: `π_n = π_{n-1} + α(1−α)^{n-1}(π̂_n − π_0)`. Read that as a mixture
bookkeeping move — it adds probability `α(1−α)^{n-1}` of executing the new `π̂_n` at any step
and removes that much probability of deferring to the queried expert, so after `n` iterations
the probability of still calling the expert is `(1−α)^n`, decaying geometrically. Stop at `N`,
strip the residual expert weight, return the renormalized mixture
`π̃_N = (π_N − (1−α)^N π_0)/(1−(1−α)^N)`. Choosing `α = O(1/T²)` and `N = O(T² log T)` gets
near-linear regret. So it's stationary and horizon-independent — better than forward training on
practicality. But look at what I'm left holding: a *stochastic* policy, a distribution over a
whole grab-bag of `π̂_n`'s of wildly varying quality. At deployment it flips a coin and might
execute one of the early, terrible ones. The controller is erratic and, in a feedback system,
can be outright unstable — one bad sampled action drops me into a bad region and the next coin
flip doesn't necessarily rescue me. And with the standard small mixing rate it still asks for
`O(T² log T)` iterations.
SEARN and CPI share the same shape — keep folding a new policy into a growing mixture — and the
same defect. Wall: I don't want a soft mixture of many policies, most of them bad. I want one
clean, deterministic, deployable controller. So I have two partial solutions: forward training
gives me a great bound but a non-stationary policy I can't run; SMILe gives me a stationary
policy I can run but a stochastic, unstable one and a terrible iteration count.

Let me strip both down to what they got *right* and see what's actually forced. The lesson of
the `T² ε` bound is unambiguous: train on the states the policy *itself* visits, not the
expert's. The lesson of forward training: if you do that, you can get the `uTε` linear bound.
The lesson of SMILe/CPI: you have to collect data under a policy that's *close to something
sensible* and let it drift, because the first few learned policies are garbage and visit
useless states. So whatever I build collects states by running *some* policy in the system,
labels those states with the *expert's* action (the expert is the supervisor — it tells me what
the right action was at the state I actually reached, which is exactly the recovery behavior the
supervised approach never sees), trains a supervised policy on that, and iterates.

The question is what to do with the data across iterations. SMILe *mixes policies* — it never
really commits to one. Let me try the opposite: don't mix the policies, *accumulate the data*.
At iteration `i`, run the current learned policy `π̂_i`, collect the states it visits, label
each with `π*`'s action, and — instead of training a new policy on only this fresh batch and
blending it in — throw the new `(state, expert-action)` pairs onto a growing aggregate dataset
`D` and retrain a single policy from scratch on *all of `D`*. One policy, deterministic,
retrained each round on the union of every state any iteration has ever visited.

Why would that converge to something good? Let me think about what the aggregate is doing.
Round 1, I have only expert states (since I have no learned policy yet, or I run the expert) —
that's just behavior cloning, and I get a mediocre `π̂_2`. Round 2, I run `π̂_2`, it drifts into
states the expert never showed, I collect *those* states and ask the expert what to do there,
and add them to `D`. Now `D` contains both the expert's nice states and the learner's
off-distribution states with correct labels. Retrain on all of it. Round 3, the new policy
drifts somewhere slightly different, I collect *those* states, add them. Over the iterations I'm
*building up the set of inputs the policy is likely to encounter at deployment*, each labeled
with the right answer. The training distribution is chasing the policy's own induced
distribution, which is exactly the thing the `T²ε` bound told me I needed.

Stare at "retrain on all the data seen so far" for a second. At iteration `i` I'm picking the
policy that does best on the union of all previous rounds' losses — the best policy *in
hindsight* over everything I've seen. I've seen that rule before: that's *Follow-The-Leader*, the
canonical online-learning algorithm. So maybe this aggregate-and-refit loop is secretly an online
learning algorithm — if I treat each iteration as one online example whose loss is the loss under
that iteration's state distribution, then "refit on all data so far" is literally FTL on that
sequence. Is that just a superficial naming match, or does it buy me anything? FTL has a known
property: it is *no-regret* on strongly convex losses, meaning its average loss over the rounds
converges to the loss of the single best fixed policy in hindsight over all those distributions.
Let me chase what that would give me here. If FTL on my per-iteration losses is no-regret, then by
definition `(1/N)Σ_i ℓ_i(π̂_i)` is close to `min_π (1/N)Σ_i ℓ_i(π)`. The average is at least the
min, so *some* `π̂_i` has loss near that best-in-hindsight value on the distribution that produced
its data. And if I additionally assume that best-in-hindsight policy is itself good — an
error-reduction assumption: for any input distribution there's *some* `π ∈ Π` with small loss —
then that `π̂_i` is good under its own distribution, which is exactly the object I couldn't
optimize directly. So the naming match isn't superficial: aggregation is doing real work, because
it converts the non-convex `E_{s∼d_π}[ℓ(s,π)]` I was stuck on into a sequence of ordinary fixed
losses fed to a no-regret learner. I'll have to make each of these "close to" and "some `π̂_i`"
steps precise below — but this is the lever worth pursuing.

But I should keep SMILe's good instinct too — early learned policies are bad and waste a lot of
data visiting irrelevant states. So let me allow, *optionally*, mixing the expert into the
data-collecting policy: at iteration `i`, collect with `π_i = β_i π* + (1−β_i) π̂_i`, a coin that
with probability `β_i` executes the expert's action and otherwise the learner's. Set `β_1 = 1`
so the very first round is pure expert (which conveniently means I don't even have to specify an
initial `π̂_1` before I have any data), and let `β_i` decay after that — `β_i = p^{i-1}` for some
`p < 1`, exponential decay just like SMILe's `(1−α)^n`. The simplest parameter-free choice is
`β_i = I(i = 1)`: pure expert in round 1, pure learner thereafter. I'll need to figure out the
exact requirement on `{β_i}` from the analysis, but intuitively the expert-mixing has to *fade*,
because in the end I want the guarantee to be about the *learned* policy's own distribution, not
about a policy that's still secretly leaning on the expert.

The pieces now force a loop:

  initialize `D ← ∅`, `π̂_1 ←` anything in `Π`;
  for `i = 1` to `N`:
    `π_i = β_i π* + (1−β_i) π̂_i`;
    sample `T`-step trajectories with `π_i`;
    `D_i = { (s, π*(s)) : s` visited by `π_i }`;
    `D ← D ∪ D_i`;
    train `π̂_{i+1}` on `D`;     // Follow-The-Leader: best on all data so far
  return the best `π̂_i` on validation.

No free parameters except the supervised learner inside, and a number of iterations that I'll
show scales near-linearly with the horizon. Now I have to actually prove it works — and I want to
prove it for *any* no-regret online learner choosing the `π̂_i`'s, not just FTL, because then the
whole thing is a clean *reduction of imitation learning to no-regret online learning*, and I get
continuous-action problems for free (FTL/online-convex methods handle real-valued prediction,
unlike the classification-only reductions SMILe/SEARN rest on).

Set up the online learning instance precisely. The loss the online learner suffers at round `i`
must be the loss under the *current* policy's state distribution — that's the only choice that
makes the regret comparison say something about distributions-under-the-policy:

  `ℓ_i(π) = E_{s ∼ d_{π_i}}[ ℓ(s,π) ].`

The online learner is choosing the learned policies `π̂_i`, while `π_i` is the data-collecting
mixture that defines the loss distribution. So the regret statement I need is
`(1/N) Σ_{i=1}^N ℓ_i(π̂_i) − min_{π∈Π} (1/N) Σ_{i=1}^N ℓ_i(π) ≤ γ_N` with `γ_N → 0`, and for
strongly convex `ℓ` (which I can arrange, e.g. regularized losses) FTL gives `γ_N = Õ(1/N)`.

Let me first get the part the online reduction gives directly: a learned policy with small loss
on the distribution used to collect its data. Define
`ε_N = min_{π∈Π} (1/N) Σ_{i=1}^N E_{s∼d_{π_i}}[ℓ(s,π)]`, the loss of the best single policy in
hindsight across all the iteration distributions. Then

  `min_i E_{s∼d_{π_i}}[ℓ(s,π̂_i)] ≤ (1/N) Σ_i ℓ_i(π̂_i) ≤ γ_N + min_{π∈Π} (1/N) Σ_i ℓ_i(π) = γ_N + ε_N.`

The first inequality is just "the min is below the average," the second is the no-regret
definition. So *some* learned policy `π̂_i` has small loss on the state distribution generated at
round `i`. If there were no expert mixing, `π_i = π̂_i` and I would already have a policy good
under its own distribution.

With expert mixing, the guarantee is still under `d_{π_i}`, and `π_i = β_i π* + (1−β_i)π̂_i`
still leans on the expert while data is collected. Deployment uses the pure learned `π̂_i`, so I
have to control how far apart `d_{π_i}` and `d_{π̂_i}` are, which is the next wall. Lemma: I claim
`‖d_{π_i} − d_{π̂_i}‖_1 ≤ 2 T β_i`. Let me derive it.
Over a `T`-step trajectory, `π_i` executes `π̂_i` at *every* step — i.e. never touches the expert
— with probability `(1−β_i)^T` (each step independently keeps the learner with probability
`1−β_i`). Condition on the complementary event "`π_i` called the expert at least once," and call
the resulting state distribution `d`. Then I can split

  `d_{π_i} = (1−β_i)^T d_{π̂_i} + (1−(1−β_i)^T) d`,

because with probability `(1−β_i)^T` the whole trajectory was pure-learner (distribution exactly
`d_{π̂_i}`) and otherwise it's `d`. Subtract `d_{π̂_i}`:

  `‖d_{π_i} − d_{π̂_i}‖_1 = (1−(1−β_i)^T) ‖d − d_{π̂_i}‖_1 ≤ 2 (1−(1−β_i)^T),`

using that the `ℓ_1` distance between two distributions is at most `2`. Now I need to bound
`1−(1−β_i)^T`. The clean inequality is `(1−β)^T ≥ 1 − βT` for `β ∈ [0,1]` (Bernoulli /
convexity — `(1−β)^T` is convex in `β` and lies above its tangent line at `β=0`, which is
`1−βT`). Hence `1−(1−β_i)^T ≤ T β_i`, and

  `‖d_{π_i} − d_{π̂_i}‖_1 ≤ 2 T β_i.`

Worth noting this only beats the trivial bound `‖·‖_1 ≤ 2` when `β_i ≤ 1/T` — for large `β_i`
the two distributions can be totally different, which makes sense (if you call the expert most of
the time, your state distribution is the expert's, not the learner's). So the early, large-`β`
iterations don't get a useful bound; I'll handle them by counting them separately.

Now convert the per-step loss under `d_{π_i}` to one under `d_{π̂_i}`. A loss bounded by
`ℓ_max` integrated against two distributions differing by `‖·‖_1` is controlled by the loose
bound `|E_p[ℓ] − E_q[ℓ]| ≤ ℓ_max ‖p−q‖_1`. The distance itself is no larger than `2`, and the lemma
gives `2Tβ_i`, so I can write the distribution penalty as `2ℓ_max min(1,Tβ_i)`. Thus

  `E_{s∼d_{π̂_i}}[ℓ(s,π̂_i)] ≤ E_{s∼d_{π_i}}[ℓ(s,π̂_i)] + 2 ℓ_max · min(1, T β_i),`

where the `min(1, Tβ_i)` uses the trivial bound for the big-`β` rounds and `Tβ_i` otherwise.
Assume the `β_i` are non-increasing, and let `n_β` be the largest index with `β_n > 1/T` (the
rounds where the expert is mixed in enough that the distributions can be far apart). Then average
over `i` and chain with the no-regret bound:

  `min_{π̂ ∈ π̂_{1:N}} E_{s∼d_{π̂}}[ℓ(s,π̂)]`
  `≤ (1/N) Σ_i E_{s∼d_{π̂_i}}[ℓ(s,π̂_i)]`
  `≤ (1/N) Σ_i [ E_{s∼d_{π_i}}[ℓ(s,π̂_i)] + 2 ℓ_max min(1, Tβ_i) ]`
  `≤ γ_N + min_{π∈Π}(1/N)Σ_i ℓ_i(π) + (2 ℓ_max/N)[ n_β + T Σ_{i=n_β+1}^N β_i ]`
  `= ε_N + γ_N + (2 ℓ_max/N)[ n_β + T Σ_{i=n_β+1}^N β_i ].`

The middle step splits the `min(1, Tβ_i)` sum: the first `n_β` rounds contribute `1` each (total
`n_β`), the rest contribute `Tβ_i`. There it is — a bound on one of the *learned* policies under
*its own* distribution:

  `E_{s∼d_{π̂}}[ℓ(s,π̂)] ≤ ε_N + γ_N + (2 ℓ_max/N)[ n_β + T Σ_{i=n_β+1}^N β_i ].`

Now I can read off exactly what the `β` schedule has to satisfy and *why*. The extra penalty is
`(2ℓ_max/N)[n_β + T Σ_{i>n_β} β_i]`, and I need it to vanish as `N → ∞`. Take the exponential
schedule `β_i = (1−α)^{i-1}`. Then `n_β` (where `(1−α)^{n-1}` crosses `1/T`) is about
`(log T)/α`, a constant in `N`, and `T Σ_{i>n_β} β_i ≤ T · (1−α)^{n_β}/α ≤ T · (1/T)/α = 1/α`,
also constant in `N`. So `[n_β + T Σ_{i>n_β}β_i] ≤ (1/α)[log T + 1]`, and the whole penalty is
`O((log T)/(Nα))`, negligible once `N = Õ(T)`. More generally the requirement is just
`\bar β_N = (1/N) Σ_{i=1}^N β_i → 0` — the *average* expert-mixing fraction must go to zero, so
that asymptotically I'm collecting under the learner, not the expert. This is *why* `β` must
fade: if it didn't, `d_{π_i}` would stay pinned near the expert's distribution and the bound
would never become about `π̂`'s own distribution. And since I need `N = Õ(T)` anyway just to make
`γ_N` small (FTL's `γ_N = Õ(1/N)`), the expert-mixing penalty is free — it's already negligible
at the iteration count I have to pay regardless. The parameter-free `β_i = I(i=1)` is the extreme
case `n_β ≤ 1` with a one-time `2ℓ_max/N` penalty that's manifestly `O(1/N)`. Good.

Let me also verify the no-regret claim from the other direction — that the sequence of mixed
`π_i` is no-regret whenever the inner learner picking `π̂_i` is, because that's what justifies
calling this a reduction rather than a one-off trick. Take nonnegative convex `ℓ_i` with
`ℓ_i(π*) ≤ C` for all `i`. By convexity of `ℓ_i` and `π_i = β_i π* + (1−β_i)π̂_i`,

  `ℓ_i(π_i) ≤ β_i ℓ_i(π*) + (1−β_i) ℓ_i(π̂_i) ≤ β_i C + ℓ_i(π̂_i).`

Average over `i`:

  `(1/N) Σ_i ℓ_i(π_i) ≤ C \bar β_N + (1/N) Σ_i ℓ_i(π̂_i) ≤ C \bar β_N + γ_N + min_{π∈Π}(1/N)Σ_i ℓ_i(π),`

the last step using the no-regret guarantee of the inner learner on the `π̂_i`. So the
mixed-policy procedure's own average regret is `≤ γ_N + C \bar β_N`, which `→ 0` provided
`γ_N → 0` and `\bar β_N → 0`. With a `β` schedule whose partial sums are `O(log N)`, and strongly
convex losses, the aggregate learner's regret is itself `Õ(1/N)`. Confirmed: any no-regret
learner plugged in as the policy-chooser makes the whole imitation procedure no-regret — and FTL
(retrain on the aggregate) is just the most natural such learner.

Now stitch the surrogate-loss guarantee back to actual task cost. If `ℓ` upper-bounds the 0-1
loss against `π*`, I can feed the per-distribution
surrogate guarantee into the forward-training telescoping bound I derived earlier. That bound was
`J(π) ≤ J(π*) + uTε` for a policy with surrogate loss `ε` under its own distribution and
cost-to-go gap `u`. So combining,

  `J(π̂) ≤ J(π*) + u T (ε_N + γ_N) + O(1)`,

and choosing `N = Õ(uT)` makes `uTγ_N = O(1)`, leaving `J(π̂) ≤ J(π*) + uTε_N + O(1)`. Linear in
`T` (times the recoverability constant `u`), which is exactly the regime forward training reached
— but now with a *single stationary deterministic* policy and an iteration count I can stop
whenever I like. That's the payoff: I kept forward training's bound and SMILe's
horizon-independence, and dropped SMILe's stochastic-mixture instability.

One more thing I've been glossing: all of the above assumed the online learner sees the loss on
the *true* distribution `d_{π_i}`. In practice each round I only sample `m` trajectories with
`π_i` and observe the empirical loss on that finite dataset `D_i`. I need to bound the true loss
under the policy's own distribution by the regret on the *sampled* losses. Let
`ε̂_N = min_{π∈Π} (1/N) Σ_i E_{s∼D_i}[ℓ(s,π)]` be the best-in-hindsight *training* loss, and let
the online learner guarantee regret `γ_N` on the empirical losses. The gap between empirical and
true is a sum of bounded martingale differences: define `Y_{ij}` as the difference between the
expected per-step loss of `π̂_i` under `d_{π_i}` and its average per-step loss on the `j`-th
sampled trajectory at iteration `i`. Each `Y_{ij}` is zero-mean (the sampled trajectory is drawn
from `d_{π_i}`), bounded in `[−ℓ_max, ℓ_max]`, and — ordered
`Y_{11},…,Y_{1m},Y_{21},…,Y_{Nm}` — forms a martingale difference sequence (each term's
conditional mean given the past is zero, since `π̂_i` and `d_{π_i}` are determined before the
`j`-th trajectory is drawn). I can't use plain Hoeffding because the `π̂_i` across rounds are
*dependent* — each is trained on the accumulated data — so I reach for Azuma–Hoeffding, which is
exactly the martingale version:

  `(1/(mN)) Σ_{i=1}^N Σ_{j=1}^m Y_{ij} ≤ ℓ_max √( 2 log(1/δ) / (mN) )` with probability `≥ 1−δ`.

Threading this through the same chain as before,

  `E_{s∼d_{π̂}}[ℓ(s,π̂)] ≤ ε̂_N + γ_N + (2 ℓ_max/N)[ n_β + T Σ_{i=n_β+1}^N β_i ] + ℓ_max √( 2 log(1/δ)/(mN) )`

with probability `≥ 1−δ`. For the generalization term `ℓ_max √(2log(1/δ)/(mN))` to be on the
order of `1/T` and so negligible over `T` steps when the surrogate is the cost, I need the total
trajectory count `mN` on the order of `T² log(1/δ)` for constant `ℓ_max` — more explicitly,
making the term at most `1/T` requires `mN ≥ 2ℓ_max²T² log(1/δ)`. With `m = O(1)` trajectories per
iteration, that is `N = O(T² log(1/δ))` iterations. For an arbitrary task cost carried through the
`uT` telescoping bound, the same term has to be on the order of `1/(uT)`, giving the
`u²T² log(1/δ)` trajectory scale. The quadratic here is purely an artifact of
Azuma–Hoeffding ignoring strong convexity; leaning on the strongly-convex fast-rate results
(Kakade & Tewari 2009) I'd conjecture `Õ(T log(T/δ))` total trajectories for the surrogate case,
or the analogous `u`-scaled version for task cost. Either way it's a polynomial in `T`, not the
exponential blowup the naive supervised approach was heading toward in the worst case.

Let me also be honest about the cost of this route versus the prior reductions: I needed the
inner learner to be *no-regret* — which for the clean `Õ(1/N)` rates means strongly convex (or
regularized) surrogate losses. That's a stronger assumption than the pure error-reduction
(classification-only) that SMILe and SEARN get away with. But it's a *common* assumption, it's
satisfied by ordinary regularized classifiers/regressors, and in exchange it handles
continuous-valued actions natively and gives me the single deterministic policy I wanted. A fair
trade.

Now let me write the actual procedure as code, because the whole point is that it drops into the
interactive imitation harness I already have — roll out a policy, query the expert at the visited
states, refit a supervised policy on a dataset. The "retrain on the aggregate" line *is*
Follow-The-Leader; the `β`-mix is the optional expert-in-the-loop. In a token-level distillation
trainer, the same objects become: prompts and generated prefixes are states, the student is the
learner, the teacher is the expert, the `lmbda` coin is the learner-side mixing probability
(`1−β_i` in the notation above) that chooses whether I train on student-generated states rather
than the dataset batch, and the imitation label is the teacher's top-1 token on those states,
masked with the same `-100` convention as the trainer. I should not reimplement the optimizer step
there; the existing trainer already owns accumulation, scaling, and stepping. I only swap the token
loss, and I leave the trainer's state-mixing order intact: optional teacher generations first, then
the `lmbda` coin that replaces the batch with student generations, then the usual trainer step calls
my loss.

```python
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers.trainer import Trainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.gkd_trainer import GKDTrainer
from trl.trainer.utils import empty_cache


def dataset_aggregation(expert, env, horizon, n_iters, beta_schedule=None, m=1):
    """Turn interactive access to an expert into one stationary deterministic policy
    whose loss is good under its OWN induced state distribution. Each round is one
    online-learning example; retraining on the accumulated data is Follow-The-Leader."""

    # beta_i: probability of executing the EXPERT (not the learner) while collecting.
    # Default = parameter-free indicator: pure expert round 1, pure learner after.
    # Requirement from the analysis: average mixing (1/N) sum(beta_i) -> 0, so the
    # collected states converge to the LEARNER's own distribution (else the bound
    # never becomes about the policy we deploy).
    if beta_schedule is None:
        beta_schedule = [1.0] + [0.0] * (n_iters - 1)   # beta_i = I(i == 1)

    dataset = []                       # the aggregate D = union of all D_i
    policy = None                      # pi_hat_1: unused in round 1 since beta_1 = 1
    candidates = []

    for i in range(n_iters):
        beta = beta_schedule[i]
        # pi_i = beta * expert + (1 - beta) * pi_hat_i : per-step coin between the two.
        def mixed_act(s, policy=policy, beta=beta):
            if policy is None or np.random.rand() < beta:
                return expert.act(s)               # query + execute the expert
            return policy.act(s)                   # execute the current learner

        # Roll out the MIXED policy to collect states the deployed policy will face;
        # label EACH visited state with the EXPERT's action (the recovery signal that
        # plain behavior cloning never sees), then aggregate.
        for _ in range(m):                          # m trajectories this iteration
            s = env.reset()
            for _ in range(horizon):
                a_star = expert.act(s)              # expert label at the visited state
                dataset.append((s, a_star))         # D_i added into the aggregate D
                s = env.step(mixed_act(s))

        # Follow-The-Leader: refit a single supervised policy on ALL data so far.
        states = np.stack([s for s, a in dataset])
        actions = np.stack([a for s, a in dataset])
        policy = SupervisedPolicy()
        policy.train(states, actions)               # minimize surrogate loss on D
        candidates.append(policy)

    # Return the best learned policy on a validation rollout (any one in the sequence
    # carries the guarantee; pick the empirically best).
    return min(candidates, key=lambda p: validation_cost(p, expert, env, horizon))


class Top1DaggerGKDTrainer(GKDTrainer):
    @staticmethod
    def top1_teacher_loss(student_logits, teacher_logits, labels=None, reduction="batchmean"):
        """Expert action = teacher argmax at each completion position; -100 masks prompts/pad."""
        target_tokens = teacher_logits.argmax(dim=-1)
        if labels is not None:
            target_tokens = target_tokens.masked_fill(labels == -100, -100)

        flat_targets = target_tokens.reshape(-1)
        flat_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            flat_targets,
            ignore_index=-100,
            reduction="none",
        )

        if labels is not None:
            valid = flat_targets != -100
            flat_loss = flat_loss[valid]
            denom = valid.sum().clamp_min(1).to(flat_loss.dtype)
        else:
            denom = torch.tensor(max(student_logits.size(0), 1), device=flat_loss.device, dtype=flat_loss.dtype)

        if reduction == "batchmean":
            return flat_loss.sum() / denom
        if reduction == "sum":
            return flat_loss.sum()
        if reduction == "mean":
            return flat_loss.mean() if flat_loss.numel() else flat_loss.sum()
        return flat_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Same completion-token alignment as the trainer path; only the token loss changes."""
        student_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        prompt_lengths = inputs["prompts"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        labels = inputs["labels"][:, prompt_lengths:]
        loss = self.top1_teacher_loss(student_logits, teacher_logits, labels, reduction="batchmean")
        empty_cache()
        return (loss, student_outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """State mixing copied from GKDTrainer: teacher seq_kd first, lmbda student replacement second."""
        if self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        return Trainer.training_step(self, model, inputs, num_items_in_batch)
```
