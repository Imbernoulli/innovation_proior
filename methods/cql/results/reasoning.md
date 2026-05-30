OK, let me think this through from scratch. I have a fixed dataset of transitions `(s, a, r, s')`,
logged by some behavior policy `π_β`, and I am not allowed to touch the environment again. I want
the best policy I can squeeze out of this data. And the embarrassing fact is that if I just run a
standard off-policy algorithm — fit `Q` to the Bellman target, push the policy toward high-`Q`
actions — the thing falls apart. The Q-values explode, the policy goes to garbage. So before I
invent anything I want to understand *exactly why*, because the failure mode is going to dictate
the fix.

Here is the loop. Policy evaluation fits `Q(s,a)` toward `r(s,a) + γ E_{a'~π}[Q(s',a')]`. Policy
improvement moves `π` toward actions that maximize `Q`. Now stare at the target. The next-state
action `a'` is sampled from `π`, the policy I am currently improving. But `Q` was only ever
trained on `(s,a)` pairs where `a ~ π_β`. So `Q(s', a')` for an `a'` that `π_β` never takes is
just... whatever the network extrapolates. There's no data there. And because `π` is *defined* to
chase large `Q`, it will happily drift to exactly those out-of-distribution actions where `Q`
came out spuriously high. That spurious value goes straight into the Bellman target, the next
fit raises `Q` there to match, the policy chases it harder, and the whole thing bootstraps upward.

In online RL this can't persist: the agent would actually take the action, the environment would
return the true (low) value, and the target would get corrected. That feedback loop is doing
silent error-correction all the time and I never appreciated it. Offline, that loop is cut. There
is no mechanism, anywhere in the algorithm, that ever says "no, that OOD action is not actually
worth 500." The error is a fixed point of the procedure, not a transient.

Let me also be precise about which shift is the problem, because it changes the fix. The states
`s` and `s'` in every backup come from the dataset. So I never query `Q` at an OOD *state* — the
state marginal is fine. What I do query is `Q(s', a')` at an OOD *action*. It's purely action
distribution shift. Good — that tells me whatever correction I apply only has to worry about
actions, at states I actually have.

So what have people tried? The instinct in the literature is: keep the policy close to the data,
so it never asks about OOD actions in the first place. BCQ fits a generative model of `π_β` and
only lets the policy pick actions near samples from it. BEAR constrains the policy to the
*support* of `π_β` with a sampled MMD penalty, and stacks an ensemble of Q-functions to be
conservative about uncertainty. The divergence-penalty family (penalize KL or Wasserstein from
`π_β`) is the same spirit with a different distance. They all (a) need to *estimate* `π_β`, which
is painful when the data comes from a mixture of sources, and (b) — this is the part that nags me
— they constrain the *policy* but do nothing to the *Q-function itself*.

And there's a measurement that makes me think constraining the policy isn't even sufficient. Take
a narrow dataset — say it's all from a near-deterministic expert. Define the per-iteration gap
`Δ̂^k = E_{s,a~D}[ max_{a'~Unif} Q̂^k(s,a') − Q̂^k(s,a) ]`: how much better some random OOD action
*looks*, under the current Q, than the action that was actually in the data. If the Q-function
were sane, in-data actions should look at least as good as random ones, so this should be `≤ 0`.
But for a policy-constraint method it comes out *positive and rising* over training. Even though
the constraint is correcting which actions get backed up, the *neural Q-function* still reports
high values at OOD actions, because function approximation couples values across `(s,a)` — pushing
`Q` up where there's data drags it up at nearby OOD actions too, and on a low-entropy dataset that
coupling is severe. And right when `Δ̂^k` keeps climbing, the policy "unlearns" and collapses. So
the disease lives in the Q-function. Constraining the policy is treating a symptom.

That reframes the whole problem for me. Don't fence the policy in. Fix the values. If I could get
a `Q` that is *deliberately pessimistic about actions it has no business trusting* — and pessimistic
in a controlled, provable way, not just "add noise" — then optimizing a policy against it is safe:
the policy can chase the max all it wants, but the max won't be at a hallucinated OOD action,
because I've pushed those down myself. I want a `Q` that *lower-bounds* the true value. If my
estimate is a guaranteed underestimate, I can never be fooled into thinking a bad action is good.

Now, how do I *make* `Q` a lower bound without throwing away the Bellman structure that makes it a
value function at all? The natural move: keep fitting the Bellman error, but add a term that simply
*pushes Q down*. Push it down where? Under some distribution `μ(a|s)` that I get to choose — and
since I never query OOD states, I'll keep `μ`'s state marginal equal to the data's, `μ(s,a) =
d^β(s) μ(a|s)`, and only let `μ` choose *actions*. So minimize `E_{s~D, a~μ}[Q]` alongside the
Bellman fit. With a tradeoff knob `α`:

  Q̂^{k+1} = argmin_Q  α · E_{s~D, a~μ}[Q(s,a)]  +  ½ E_{s,a,s'~D}[ (Q(s,a) − B̂^π Q̂^k(s,a))² ].

Does this actually give a lower bound? Let me work it in the tabular case where I can represent
every `Q(s,a)` independently, so I can just take the derivative entry by entry. The penalty term
`α E_{s~D,a~μ}[Q]` differentiates, at a fixed `(s,a)`, to `α · d^β(s) μ(a|s)`. The Bellman term
`½ E_{s,a~D}[(Q − B̂^π Q̂^k)²]` puts weight `d^β(s) π_β(a|s)` on `(s,a)` and differentiates to
`d^β(s) π_β(a|s)·(Q(s,a) − B̂^π Q̂^k(s,a))`. Setting the sum to zero and cancelling `d^β(s)`:

  α μ(a|s) + π_β(a|s) (Q(s,a) − B̂^π Q̂^k(s,a)) = 0
  ⟹ Q̂^{k+1}(s,a) = B̂^π Q̂^k(s,a) − α · μ(a|s) / π_β(a|s).

There it is. Every iterate is the Bellman target *minus* a non-negative bump `α μ/π_β` (everything
positive: `α>0`, `μ≥0`, and I'm assuming `π_β>0` on the support). So `Q̂^{k+1} ≤ B̂^π Q̂^k`
pointwise — I underestimate the target at every step. Now push it to the fixed point. At
convergence `Q̂^π = B̂^π Q̂^π − α μ/π_β`, and since `B̂^π Q = R̂ + γ P^π Q`,

  Q̂^π = (I − γ P^π)^{-1} [ R̂ − α μ/π_β ] = Q̂^π_{Bellman} − α (I − γP^π)^{-1} [μ/π_β].

The matrix `(I − γP^π)^{-1} = Σ_t (γP^π)^t` has all non-negative entries, and `μ/π_β ≥ 0`, so the
correction I'm subtracting is non-negative everywhere. Pointwise lower bound. If the empirical
Bellman operator were exact this is already done for any `α > 0`.

But it isn't exact — `B̂^π` is a single-sample backup. So there's a gap between `B̂^π` and the true
`B^π`. How big? Under the standard concentration assumptions on the reward and the dynamics —
`|r̂ − r| ≤ C_{r,δ}/√|D(s,a)|` and `‖T̂ − T‖_1 ≤ C_{T,δ}/√|D(s,a)|`, the kind of bound used in the
exploration literature — I can bound, with probability `≥ 1−δ`,

  |B̂^π Q − B^π Q|(s,a) ≤ ( C_{r,δ} + γ C_{T,δ}·2R_max/(1−γ) ) / √|D(s,a)|  =: C_{r,T,δ} R_max / ((1−γ)√|D(s,a)|),

where I bounded `Q ≤ 2R_max/(1−γ)` to control the transition term. So `B̂^π Q̂^k` could overshoot
the true `B^π Q̂^k` by up to that much. Carry it through the same fixed-point algebra and the
relation becomes

  Q̂^π(s,a) ≤ Q^π(s,a) − α [(I − γP^π)^{-1} μ/π_β](s,a) + [(I − γP^π)^{-1} C_{r,T,δ}R_max/((1−γ)√|D|)](s,a).

Now the lower bound is no longer automatic: the last (positive) sampling-error term could outweigh
the (negative) penalty term. But I get to choose `α`. I just need the penalty to dominate the
error everywhere, i.e.

  α · min_{s,a} [μ(a|s)/π_β(a|s)] ≥ max_{s,a} C_{r,T,δ}R_max/((1−γ)√|D(s,a)|),

so `α ≥ max_{s,a} C_{r,T,δ}R_max/((1−γ)√|D(s,a)|) · (min_{s,a} μ/π_β)^{-1}` makes
`Q̂^π(s,a) ≤ Q^π(s,a)` for all `s∈D, a`. And nicely, the threshold *shrinks* as `√|D(s,a)|` grows
— with enough data, a tiny `α` suffices, and in the noiseless limit any `α>0` works. So far so good:
a provable pointwise lower bound on `Q^π`.

But wait — stare at that penalty for a second. It subtracts `α μ/π_β` from *every* action, including
the actions that *are* in the data. That feels wasteful. I'm being conservative about actions I
actually have evidence for. And I don't even need a pointwise lower bound on `Q`. What does the
rest of the algorithm actually consume? Policy evaluation and improvement only ever use the *value
of the policy*, `V^π(s) = E_{a~π}[Q(s,a)]`. I only need *that* to be a lower bound, not `Q` at
every individual action. If I'm only on the hook for `E_π[Q] ≤ V^π`, I have slack — I can give back
some value on the in-data actions and still keep the expectation under `π` below the truth.

How do I give value back on in-data actions? Add a term that *maximizes* `Q` under the data
distribution. So now I push down under `μ` and push *up* under `π_β`:

  Q̂^{k+1} = argmin_Q  α ( E_{s~D, a~μ}[Q] − E_{s~D, a~π_β}[Q] )  +  ½ E_{s,a,s'~D}[ (Q − B̂^π Q̂^k)² ].

Redo the tabular derivative. The push-down contributes `+α d^β μ(a|s)`, the push-up contributes
`−α d^β π_β(a|s)`, the Bellman term as before. Setting to zero and cancelling `d^β(s)`:

  α(μ(a|s) − π_β(a|s)) + π_β(a|s)(Q − B̂^π Q̂^k) = 0
  ⟹ Q̂^{k+1}(s,a) = B̂^π Q̂^k(s,a) − α ( μ(a|s)/π_β(a|s) − 1 ).

Now the bump is `α(μ/π_β − 1)`, which can be *negative* — wherever `μ(a|s) < π_β(a|s)` I'm actually
*adding* to `Q`. So I've thrown away the pointwise lower bound: there will be `(s,a)` with
`Q̂^{k+1} > Q^{k+1}`. (Concretely you can hand-build a little three-state, two-action MDP and a
behavior policy that makes this happen.) That's fine — I gave up pointwise on purpose. The question
is whether the *value* still underestimates. Take the expectation under `π`, and set `μ = π`
(I'm evaluating `π`, so push down under `π` itself):

  V̂^{k+1}(s) = E_{a~π}[Q̂^{k+1}(s,a)] = B^π V̂^k(s) − α E_{a~π}[ π(a|s)/π_β(a|s) − 1 ].

So the per-iteration value correction is `α D_CQL(s)` with `D_CQL(s) := Σ_a π(a|s)(π(a|s)/π_β(a|s) − 1)`.
Is that non-negative? If it is, I underestimate the value at every step. Let me expand it by adding
and subtracting `π_β` in the leading factor:

  D_CQL(s) = Σ_a π(π/π_β − 1)
           = Σ_a (π − π_β + π_β)(π/π_β − 1)
           = Σ_a (π − π_β)·(π − π_β)/π_β  +  Σ_a π_β(π/π_β − 1)
           = Σ_a (π − π_β)²/π_β  +  Σ_a (π − π_β).

The second sum is `Σπ − Σπ_β = 1 − 1 = 0`. The first sum is a sum of squares over a positive
denominator, so `≥ 0`, and it's `= 0` iff `π = π_β`. So `D_CQL(s) ≥ 0` always. The value is
underestimated each step, and at the fixed point

  V̂^π(s) = V^π(s) − α [ (I − γP^π)^{-1} E_π[ π/π_β − 1 ] ](s),

again with the non-negative `(I − γP^π)^{-1}` making the correction non-negative. So `V̂^π(s) ≤
V^π(s)`. And this bound is *tighter* than the pointwise version: there I subtracted `α(I−γP^π)^{-1}
μ/π_β`; here I subtract `α(I−γP^π)^{-1}(π/π_β − 1)` — the extra `−1` means I take back less value.
Exactly the slack I wanted. Adding the sampling-error term back, the `α` threshold becomes

  α ≥ max_{s,a∈D} C_{r,T}R_max/((1−γ)√|D(s,a)|) · max_{s∈D} [ Σ_a π(a|s)(π(a|s)/π_β(a|s) − 1) ]^{-1},

and again it decays as the dataset grows.

Now I want to pin down something that's bugging me: why push *up* under `π_β` specifically? I chose
it because it's the data distribution, but is it *forced*, or could I push up under some other `ν`?
Let me generalize: replace the push-up `E_{π_β}[Q]` with `E_ν[Q]` for an arbitrary `ν(a|s)`. The
tabular iterate becomes `Q̂^{k+1} = B̂^π Q̂^k − α(μ − ν)/π_β`, and with `μ = π` the per-step value
penalty is `π^T (π − ν)/π_β`. I want this `≥ 0` *for every possible target policy* `π` — otherwise
some `π` breaks the bound. So I should ask which `ν` survives the worst case:

  max_ν  min_π  Σ_a π(a|s)·(π(a|s) − ν(a|s))/π_β(a|s),   s.t. Σπ = 1, Σν = 1, π,ν ≥ 0.

Solve the inner `min_π` first, with a Lagrange multiplier `η` for `Σπ = 1` (and, since a Boltzmann
`π` is full-support, the positivity multipliers vanish by KKT). The objective in `π` is
`Σ_a π²/π_β − Σ_a π ν/π_β`; its gradient in `π(a|s)` is `2π(a|s)/π_β(a|s) − ν(a|s)/π_β(a|s) + η = 0`.
Solving and using `Σπ = 1` to fix `η` gives `π*(a|s) = ½ν(a|s) + ½π_β(a|s)`. Plug that back in and
the inner value becomes, after simplification, `Σ_a π_β(½ − ν/(2π_β))(½ + ν/(2π_β))`, which is a
function only of `ν`. Maximizing *that* over `ν` (with `Σν=1`, `ν≥0`) — it's a downward
concave-in-the-ratio expression maximized when `ν/π_β = 1`, i.e. `ν = π_β`, where its value is
exactly `0`. So `ν = π_β` is the *only* choice for which the penalty is guaranteed `≥ 0` against an
adversarial `π`; for any `ν ≠ π_β`, there's a `π` that makes the penalty negative and breaks the
lower bound. So pushing up under the behavior distribution isn't a convenient choice — it's the
necessary one. Good, that settles it.

Now the practical headache: I've been writing the push-down distribution `μ` as if I get to fix it,
and I set `μ = π` for the evaluation bound. But in a real algorithm `π` keeps changing as I improve
it, and re-running full off-policy evaluation for each policy iterate is expensive. I'd rather have
`μ` *track* whatever the current `Q` thinks is best — push down hardest on exactly the actions the
Q-function is currently most tempted to overrate. That suggests not fixing `μ` at all, but
*maximizing* over it, with a regularizer `R(μ)` to keep it sane:

  min_Q max_μ  α ( E_{s~D, a~μ}[Q] − E_{s~D, a~π_β}[Q] )  +  ½ E_D[ (Q − B̂^π Q̂^k)² ]  +  R(μ).

This is a whole family — pick `R` and you get a member. What's the natural `R`? I want `μ` to
concentrate on high-`Q` actions but not collapse to a spike, so penalize it for being too peaked —
add its entropy, `R(μ) = H(μ)`. Then the inner problem at each state is

  max_μ  E_{a~μ}[Q(s,a)] + H(μ)   s.t.  Σ_a μ(a|s) = 1, μ ≥ 0.

Lagrangian: `Σ_a μ(Q − log μ) + λ(1 − Σμ)`. Differentiate in `μ(a)`: `Q(a) − log μ(a) − 1 − λ = 0`,
so `μ*(a) ∝ exp(Q(s,a))` — a Boltzmann distribution over actions. Plug it back. The maximized inner
value is `E_{μ*}[Q] + H(μ*) = Σ_a μ*(Q − log μ*)`, and since `log μ*(a) = Q(a) − log Z` with
`Z = Σ_a exp Q(a)`, we get `Σ_a μ*(Q − Q + log Z) = log Z = log Σ_a exp Q(s,a)`. The whole
max-over-`μ` push-down term collapses to a clean `log-sum-exp`. So the objective becomes

  min_Q  α E_{s~D}[ log Σ_a exp Q(s,a) − E_{a~π_β}[Q(s,a)] ]  +  ½ E_D[ (Q − B̂^π Q̂^k)² ].

This is exactly the soft-maximum of `Q` minus the value on the data. And it's self-targeting: the
`log-sum-exp` is dominated by whatever action currently has the largest `Q`, so the gradient pushes
down hardest precisely on the action the Q-function is most overrating right now, while the
`E_{π_β}[Q]` term holds up the in-data actions. That's the gap-expanding behavior I was chasing,
falling straight out of an entropy regularizer. Call this variant CQL(H).

I could also regularize toward a prior `ρ` by using `R(μ) = −D_KL(μ ‖ ρ)`. The same Lagrangian computation gives
`μ*(a) ∝ ρ(a) exp(Q(s,a))`. With `ρ = Uniform` this is CQL(H) again. With `ρ = π̂^{k-1}` (the
previous policy) the push-down term becomes an exponentially-weighted average of `Q` over actions
drawn from the previous policy instead of a `log-sum-exp` over all actions — and I expect this to
matter in high-dimensional action spaces, where estimating `log Σ_a exp Q` by sampling has nasty
variance. Call that CQL(ρ). And there's a variance-regularized cousin if I set `R` from a
distributionally-robust-optimization identity, penalizing `var_a(Q)` — but the entropy one is the
clean default.

The fixed-`μ` argument is not enough unless the adaptive family keeps the same pessimistic behavior
while the policy is moving. With `π_{Q̂^k}(a|s) ∝ exp(Q̂^k(s,a))` being the soft-optimal policy for the
current `Q`, the per-step value of the *actual* next policy `π̂^{k+1}` decomposes as

  E_{π̂^{k+1}}[Q̂^{k+1}] = E_{π̂^{k+1}}[B^π Q̂^k]
        − α underbrace{ E_{π_{Q̂^k}}[ π_{Q̂^k}/π_β − 1 ] }_{(a), the underestimation if π̂^{k+1}=π_{Q̂^k}}
        + α Σ_a underbrace{(π_{Q̂^k} − π̂^{k+1})}_{(b)} · π_{Q̂^k}/π_β.

Term (a) is the underestimation I'd get if the policy were exactly the soft-optimal one; it's `≥ 0`
by the `D_CQL ≥ 0` argument. Term (b) is the slop from `π̂^{k+1}` not being exactly `π_{Q̂^k}`,
bounded by `D_TV(π_{Q̂^k}, π̂^{k+1}) · max_a π_{Q̂^k}/π_β`. As long as (a) dominates (b) — i.e. the
policy moves slowly, `D_TV ≤ ε` small, with `E_{π_Q}[π_Q/π_β − 1] ≥ max_a(π_Q/π_β)·ε` — each
iteration underestimates, and by induction `V̂^{k+1} ≤ V^{k+1}`. So I need the policy to change
*slowly* relative to `Q`. That immediately tells me to use a much smaller learning rate on the
actor than on the critic — keep `ε` tiny so the bound holds.

Now the property I actually care about is the gap. Does the backup widen the difference between
in-data and OOD Q-values beyond what the true Q has? The iterate is `Q̂^{k+1} = B^{π^k} Q̂^k −
α_k(μ_k − π_β)/π_β`. Take its expectation under `π_β`: the penalty contributes
`−α_k · π_β^T (μ_k − π_β)/π_β = −α_k Σ_a (μ_k − π_β) = −α_k·0 = 0` — the `π_β` in numerator and
denominator cancel and the difference of two densities sums to zero. So pushing up under `π_β`
introduces *zero* expected shift at in-data actions. But under `μ_k` the penalty contributes
`−α_k μ_k^T(μ_k − π_β)/π_β = −α_k Δ̂^k` with `Δ̂^k ≥ 0` (same square-sum argument as `D_CQL`). So

  E_{π_β}[Q̂^{k+1}] − E_{μ_k}[Q̂^{k+1}] = ( E_{π_β}[B Q̂^k] − E_{μ_k}[B Q̂^k] ) + α_k Δ̂^k,

and subtracting the same difference computed under the *true* iterate `Q^{k+1}`, the residual error
terms can be cancelled by choosing `α_k` large enough (concretely `α_k` above a threshold involving
`D_TV(π_β, μ_k)·R_max/(1−γ)` divided by `Δ̂^k`, plus a sampling-error margin). With that choice,

  E_{π_β}[Q̂^{k+1}] − E_{μ_k}[Q̂^{k+1}] > E_{π_β}[Q^{k+1}] − E_{μ_k}[Q^{k+1}].

CQL *expands* the gap between in-distribution and OOD actions relative to the truth. That's the
direct fix for the diagnostic I started from: whatever erroneous high value function approximation
wants to put at an OOD action, I can choose `α` to push it back below the in-data actions. And it's
why I don't need an explicit policy constraint — the gap-expansion implicitly keeps the induced
policy `∝ exp Q̂^k` close to `π_β`. No behavior-policy model required, which is the thing every
baseline was paying for.

Does this whole procedure optimize anything coherent, or am I just bolting a penalty on? Look at the
fixed point of the modified evaluation: `Q̂^π` solves a Bellman equation in the empirical MDP `M̂`
but with the reward replaced by `r(s,a) − α(π/π_β − 1)`. So maximizing the policy value under `Q̂^π`
is identical to

  π* = argmax_π  J(π, M̂) − α/(1−γ) · E_{s~d^π_{M̂}}[ D_CQL(π, π_β)(s) ],

i.e. maximize empirical-MDP return while paying a penalty `D_CQL = Σ_a π(π/π_β − 1)` for straying
from `π_β`. A well-defined penalized objective — and the penalty I never wrote down explicitly, it
emerged from the gap-expansion. From here a safe-improvement statement follows in the style of the
SPIBB/CPI analyses: relate `J(·, M̂)` to `J(·, M)` via the same concentration bounds (the return
difference between empirical and true MDP is controlled by `√|A|/√|D(s)| · √(D_CQL + 1)`, using
`‖Δ d^π‖_1` bounds), and you get

  J(π*, M) ≥ J(π_β, M)
             − 2(C_{r,δ}/(1−γ) + γR_max C_{T,δ}/(1−γ)²) E[√|A|/√|D(s)| √(D_CQL+1)]
             + α/(1−γ) E[D_CQL(π*, π_β)].

Equivalently the slack `ζ` is the sampling-error term minus the conservative empirical gain, and
that gain is non-negative. So `π*` is a `ζ`-safe improvement over the behavior policy, and as the
dataset grows the sampling-error term shrinks, so smaller `α` suffices.
That's the guarantee I wanted: at least as good as the data, with high probability, and provably
better when the penalized empirical gain outweighs the sampling error.

One loose end I waved past: I derived everything assuming `Q` is tabular and exactly representable.
With function approximation the iterate isn't a free per-entry minimizer. Take the linear case
`Q = w^T Φ(s,a)`. Setting the gradient in `w` of the modified objective to zero gives a normal
equation `(Φ^T D Φ) w^{k+1} = Φ^T D (B^π Q̂^k) − α_k Φ^T diag[d^β(μ − π_β)]`, with `D =
diag(d^β π_β)` — the first term is the ordinary LSTD-Q iterate, the second is the penalty. The
value under `π` then equals the LSTD-Q value minus `α_k · π^T P_Φ [(π − π_β)/π_β]`, where
`P_Φ = Φ(Φ^T D Φ)^{-1}Φ^T D` is the projection onto the feature span. I need that penalty
non-negative. Minimizing `f(π) = π^T P_Φ[(π − π_β)/π_β]` over `π` (Lagrangian, set gradient to
zero) gives a stationary `π*` with `(P_Φ + P_Φ^T)[π*/π_β] = (P_Φ + P_Φ^T)1`, at which `f(π*) = 0`;
so `f(π) ≥ 0` for all `π`. Hence the CQL value lower-bounds the LSTD-Q value for any `α ≥ 0`. To get
a bound on the *true* tabular value (LSTD-Q can itself overestimate when the truth isn't in the
feature class), pick `α_k` to also swallow that projection error — numerator
`D^T[Φ(Φ^TDΦ)^{-1}Φ^T − I](B^π Q̂^k)` over denominator `D^T[Φ(Φ^TDΦ)^{-1}Φ^T](D[(π−π_β)/π_β])` —
and induct. If the true value *is* representable the numerator is zero and any `α>0` works. The
neural case reduces to this: under an NTK linearization, a single gradient step makes the features
`∇_θ Q̂^k`, and the one-step update contains the NTK factor `M^kD` with
`M^k = (∇_θ Q̂^k)^T ∇_θ Q̂^k` in the same penalty position as the linear projection. The same
non-negative penalty argument plus an `α_k` chosen to cancel the unpenalized
overestimation gives the lower bound. So the guarantees survive linear and (NTK) non-linear
approximation, with `α_k` doing the work of compensating for representation error.

Time to make it real. I'll build on an entropy-regularized off-policy actor-critic (a SAC-style
agent: stochastic tanh-Gaussian actor, twin critics with target networks, automatic temperature)
and a Q-learning version (QR-DQN) for discrete actions, because the change is tiny — swap the
critic's plain Bellman loss for the CQL(H) objective. The critic loss is the usual twin-Q TD error
*plus* `α_cql · ( log-sum-exp_a Q(s,a) − Q(s, a_data) )` averaged over the batch. In discrete
actions I compute `log Σ_a exp Q(s,a)` exactly with `logsumexp` over the action dimension. In
continuous actions I can't enumerate, so I estimate the soft-maximum by importance sampling: draw
`N` actions from a uniform `Unif(a)`, `N` from the current policy at `s`, and `N` from the current
policy at `s'`, correct each by its log-density, and `logsumexp` the concatenation. The exact
mixture weights and `1/N` factors are additive constants inside the logarithm after sampling; for
the critic gradient they can be dropped, and for the Lagrange version they are absorbed into the
target gap. The data term is just `Q(s, a)` on the dataset action.

For `α_cql`: a fixed value is brittle across datasets, so I'll also offer the Lagrange form. Turn
the penalty into a constraint with a budget `τ` and let a dual variable adapt it:
`min_Q max_{α≥0} α(diff − τ)`, where `diff = E_s[log-sum-exp_a Q − E_{π_β}[Q]]`. If the in-data-vs-OOD
gap exceeds `τ`, `α` climbs and penalizes harder; if it's below, `α` relaxes toward zero. This
auto-tunes the conservatism from a quantity I can read off the dataset alone — no online validation,
which I don't have. The actor is unchanged SAC (`max E[Q − α_ent log π]`), except — per the
slow-policy-change requirement from the across-iterations bound — I run it at a much smaller
learning rate than the critic (e.g. `3e-5` for the policy vs `3e-4` for `Q`), keeping `ε` small so
each update stays a lower bound. No behavior-policy estimator anywhere; the conservatism lives
entirely in the critic.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self):
        return self.value

class ContinuousCQL:
    """SAC critic with the CQL(H) regularizer added. Only the critic loss differs from SAC."""

    def __init__(self, actor, critic_1, critic_2, target_critic_1, target_critic_2,
                 discount=0.99, cql_n_actions=10, cql_temp=1.0, cql_alpha=5.0,
                 cql_lagrange=False, cql_target_action_gap=-1.0,
                 cql_importance_sample=True, cql_clip_diff_min=-np.inf,
                 cql_clip_diff_max=np.inf, backup_entropy=False, qf_lr=3e-4):
        self.actor = actor
        self.critic_1, self.critic_2 = critic_1, critic_2
        self.target_critic_1, self.target_critic_2 = target_critic_1, target_critic_2
        self.discount, self.backup_entropy = discount, backup_entropy
        self.cql_n_actions, self.cql_temp, self.cql_alpha = cql_n_actions, cql_temp, cql_alpha
        self.cql_lagrange, self.cql_target_action_gap = cql_lagrange, cql_target_action_gap
        self.cql_importance_sample = cql_importance_sample
        self.cql_clip_diff_min, self.cql_clip_diff_max = cql_clip_diff_min, cql_clip_diff_max
        if cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=qf_lr)

    def _critic_regularizer(self, obs, actions, next_obs, q1_data, q2_data):
        # This is the empty critic-correction slot from the offline actor-critic scaffold.
        B, adim = actions.shape[0], actions.shape[-1]
        rand_a = actions.new_empty((B, self.cql_n_actions, adim)).uniform_(-1, 1)
        cur_a, cur_logp = self.actor(obs, repeat=self.cql_n_actions)
        nxt_a, nxt_logp = self.actor(next_obs, repeat=self.cql_n_actions)
        cur_a, cur_logp = cur_a.detach(), cur_logp.detach()
        nxt_a, nxt_logp = nxt_a.detach(), nxt_logp.detach()

        q1_rand, q2_rand = self.critic_1(obs, rand_a), self.critic_2(obs, rand_a)
        q1_cur, q2_cur = self.critic_1(obs, cur_a), self.critic_2(obs, cur_a)
        q1_nxt, q2_nxt = self.critic_1(obs, nxt_a), self.critic_2(obs, nxt_a)

        cat_q1 = torch.cat([q1_rand, q1_nxt, q1_cur], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_nxt, q2_cur], dim=1)
        if self.cql_importance_sample:
            rand_log_density = np.log(0.5 ** adim)
            cat_q1 = torch.cat([q1_rand - rand_log_density,
                                q1_nxt - nxt_logp,
                                q1_cur - cur_logp], dim=1)
            cat_q2 = torch.cat([q2_rand - rand_log_density,
                                q2_nxt - nxt_logp,
                                q2_cur - cur_logp], dim=1)

        q1_ood = torch.logsumexp(cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        q2_ood = torch.logsumexp(cat_q2 / self.cql_temp, dim=1) * self.cql_temp
        q1_diff = torch.clamp(q1_ood - q1_data, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        q2_diff = torch.clamp(q2_ood - q2_data, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime().exp(), min=0.0, max=1e6)
            min_q1 = alpha_prime * self.cql_alpha * (q1_diff - self.cql_target_action_gap)
            min_q2 = alpha_prime * self.cql_alpha * (q2_diff - self.cql_target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            (-(min_q1 + min_q2) * 0.5).backward(retain_graph=True)  # dual ascent on alpha_prime
            self.alpha_prime_optimizer.step()
        else:
            min_q1 = self.cql_alpha * q1_diff
            min_q2 = self.cql_alpha * q2_diff
        return min_q1 + min_q2

    def _critic_loss(self, obs, actions, next_obs, rewards, dones, ent_alpha):
        # --- standard twin-Q TD loss (the Bellman-fit half of the objective) ---
        q1 = self.critic_1(obs, actions)
        q2 = self.critic_2(obs, actions)
        next_a, next_logp = self.actor(next_obs)
        target_q = torch.min(self.target_critic_1(next_obs, next_a),
                             self.target_critic_2(next_obs, next_a))
        if self.backup_entropy:
            target_q = target_q - ent_alpha * next_logp          # SAC soft backup
        td_target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.discount * target_q.detach()
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        cql_penalty = self._critic_regularizer(obs, actions, next_obs, q1, q2)
        return td_loss + cql_penalty              # Bellman fit + conservative critic correction

    def _actor_loss(self, obs, ent_alpha):        # unchanged SAC; run at a SMALL learning rate
        a, logp = self.actor(obs)
        q = torch.min(self.critic_1(obs, a), self.critic_2(obs, a))
        return (ent_alpha * logp - q).mean()
```

For discrete actions the only change is that `log Σ_a exp Q(s,a)` is exact — `torch.logsumexp` over
the action dimension of the Q-head — and the backup uses `B*` (the max) instead of `B^π`; CQL drops
on top of QR-DQN with the same `diff = logsumexp − Q(s, a_data)` penalty.

So the causal chain, end to end: offline, the Bellman max bootstraps OOD-action values that the
data never corrects, so estimates blow up; constraining the policy treats the symptom because the
*Q-function* itself misvalues OOD actions under function-approximation coupling; so instead I make
`Q` a provable lower bound by adding a push-down term `E_μ[Q]`, which alone gives a pointwise bound
but over-penalizes; adding a push-up term `E_{π_β}[Q]` recovers a *tighter* bound on the policy
*value* (and `π_β` is the unique distribution that keeps it a bound); maximizing over `μ` with an
entropy regularizer turns the push-down into a self-targeting `log Σ_a exp Q`; the resulting backup
is gap-expanding, so it lower-bounds value across iterations (given slow policy updates, hence the
small actor learning rate), optimizes a well-defined penalized empirical-MDP objective, and yields
a safe improvement over the behavior policy — all with a ~20-line critic-loss change on top of SAC
or QR-DQN and no behavior-policy model.
