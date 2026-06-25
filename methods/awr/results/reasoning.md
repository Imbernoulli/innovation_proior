Let me start from what actually makes reinforcement learning miserable to use, because that's the thing I want to fix. When I train a model with supervised learning, my life is simple: I write down a loss, I call a regression routine, and I get back a stable, convergent fit. When I train a policy with reinforcement learning, I am suddenly babysitting target networks, double critics, clipped surrogates, importance ratios, trust-region solvers, entropy coefficients — and any one of them, set slightly wrong, quietly poisons the whole run. The asymmetry bothers me. The thing I most want is an RL algorithm whose *inner loop is just supervised regression* — squared error for one piece, weighted log-likelihood for another, nothing that can diverge — that is still good enough to compete on continuous control. And I want it to do the thing supervised learning can't and on-policy RL won't: reuse data. Data from earlier versions of the policy sitting in a replay buffer, or even a frozen dataset somebody else collected, should be usable, not thrown away after a single gradient step.

So let me lay out the tools I have and figure out exactly where each one stops short, because the gaps are going to tell me what to build. Plain policy gradients differentiate the expected return `J(pi) = E_{tau~p_pi}[ sum_t gamma^t r_t ]` directly. Conceptually clean, but the estimator is high-variance and the whole thing is on-policy: the gradient is an expectation under the *current* policy's trajectory distribution, so a batch collected by an older policy is formally invalid and gets discarded. That's the sample-efficiency wall. On the other side, Q-learning and its actor-critic descendants reuse a buffer through Bellman backups, but they're fragile — they need target networks, double-Q, clipping, exploration noise, and even then they collapse. And there's a specific failure I keep seeing on off-policy or static data: the Bellman backup evaluates the critic at the next action the *current* policy would take, `Q(s', a')` with `a' ~ pi`, and when that `a'` is outside the actions present in the data, the critic's value there is pure extrapolation. The bootstrap propagates that error, and the policy learns to chase phantom high values. It's telling that on static datasets, plain behavioral cloning — which never queries an action it hasn't seen — often beats these methods. The lesson I take is: don't bootstrap a value at out-of-distribution actions; if I can train the policy only on actions that actually appear in the data, I sidestep the entire failure.

Could I just importance-sample to make old data valid? Reweight returns by the policy ratio `pi/mu`? It's unbiased, sure, but the variance of that ratio blows up the moment `pi` and `mu` drift apart, and I'm explicitly imagining a buffer full of data from *many* past policies, some quite stale. Importance weights across that spread would be unusable. Wall. So plain IS is out as the foundation.

Now there's one family that already has the property I'm most attached to — the inner loop being supervised regression — and that's reward-weighted regression. The idea, from Peters and Schaal, is to recast policy search as expectation-maximization: pretend the reward is an (improper) likelihood, and then improving the policy becomes weighted maximum likelihood. The E-step weights each observed action by the exponential of its reward; the M-step fits the policy to those actions by weighted regression. Concretely, each iteration solves

  pi_{k+1} = argmax_pi  E_{s ~ d_{pi_k}, a ~ pi_k} [ log pi(a|s) * exp( (1/beta) R_{s,a} ) ],

with `R_{s,a}` the return and `beta` a temperature. I love the shape of this — it's literally a weighted log-likelihood, the most stable, most convergent objective I know, and it's a couple of lines on top of a regression routine. So why isn't this the answer already? Three things. First, the sampling policy in the expectation is `pi_k`, the current policy, so it's on-policy: data is gathered fresh and dropped. Second, and this nags at me more, the weight is `exp((1/beta) R_{s,a})` — an exponential of the *raw return*, with nothing subtracted. That feels brittle, but I want to be careful and actually pin down *how*, because my first guess is wrong. My initial worry was that adding a constant to every reward — shifting all returns up by, say, `+10` — would wreck the update. Let me check that on the smallest possible case: one state, two actions with returns `R = 2.0` (good) and `R = 1.0` (bad), `beta = 1`. Raw weights `exp(2), exp(1) = 7.39, 2.72`, which normalize within the state to `0.731, 0.269`. Now shift both returns up by `+10`: raw weights become `162755, 59874` — enormous — but they normalize to `0.731, 0.269`, *identical*. Shift by `+100`: still `0.731, 0.269`. So a pure offset cancels in the per-state normalizer; the within-state distribution is offset-invariant. My naive complaint was wrong, and it's worth knowing that before I build on it.

The real problem shows up only when I remember the policy is *one shared network* fit by *one pooled regression* across all states. Take two states: at `s1` the returns are `100` and `101`, at `s2` they are `0` and `1` — both have the same structure, the good action one unit better. Raw-return weights: at `s1`, `exp(101), exp(100) ≈ 7.3e43, 2.7e43`; at `s2`, `exp(1), exp(0) = 2.72, 1.0`. The shared maximum-likelihood fit sums `log pi * weight` over *all* samples, and `s1`'s absolute weights are `~e^100` times larger than `s2`'s, so `s1`'s samples utterly dominate and `s2`'s good-vs-bad distinction (`2.72` vs `1.0`) is invisible. The update depends on the absolute *level* of the return at each state, which carries no information about which action is better *there*. And it's worse than just imbalance: at the sharp temperature this method actually uses later (`beta = 0.05`), `exp(100/0.05)` simply overflows a float — `exp(2/0.05)` is already `2.4e17`. Third, the empirical fact: with neural-network policies, RWR has been shown to underperform badly — well below the trust-region and off-policy methods of the day. So the regression-as-RL skeleton is right, but exponentiating the bare return is the weak joint.

Stare at that cross-state scale problem, because it's the crux. What I actually care about at a state `s` is not "was the return large" but "was *this* action better than what the policy would typically do here." That's the advantage, `A(s,a) = R_{s,a} - V(s)`, the return minus the baseline value of the state. Let me run the same two-state example with `A` instead of `R`. At `s1`, `V = 100.5` (the mean return), so the advantages are `+0.5, -0.5`; at `s2`, `V = 0.5`, advantages again `+0.5, -0.5`. The weights `exp(0.5), exp(-0.5) = 1.65, 0.61` are now *identical and O(1) at both states* — the good-over-bad signal is the same magnitude everywhere, and the pooled regression learns "pick good over bad" at `s2` just as strongly as at `s1`. The baseline centers each state's weights so all states contribute on the same scale, and the global-offset case from before is subsumed: shifting all rewards by `+100` shifts every `V` by `+100` too, so the advantages — and the weights — are exactly unchanged. So my instinct is: I want RWR's weighted-regression update, but with the weight an exponential of the *advantage*, not the return. The question is whether I can get there *principledly* — derive it, so the `beta` and the exact form aren't hand-waves — rather than just patching a baseline in by analogy. Because if I just assert "subtract V," I haven't earned the exponential form or the temperature.

Let me see if the right starting objective forces the advantage out on its own. RWR (and the related relative-entropy method, REPS) both maximize the expected return `J(pi)`. What if instead I maximize the expected *improvement* over the policy that collected my data? Call the data-collecting policy `mu`. Define

  eta(pi) = J(pi) - J(mu).

This is the natural thing to want anyway — I have data from `mu`, I want a `pi` that's *better than* `mu`. And there's a classical identity for exactly this, from Kakade and Langford: the difference in returns of two policies equals the advantage of one accumulated under the other's visitation. Let me write it and check it, because it's doing real work. With `A^mu(s,a) = R^mu_{s,a} - V^mu(s)` the advantage with respect to the sampling policy `mu`,

  eta(pi) = J(pi) - J(mu) = E_{tau ~ p_pi}[ sum_{t=0}^inf gamma^t A^mu(s_t, a_t) ].

Let me convince myself this is right rather than take it on faith. Expand the advantage as a one-step Bellman residual of `mu`'s value: `A^mu(s_t,a_t) = r(s_t,a_t) + gamma V^mu(s_{t+1}) - V^mu(s_t)`. Now sum `gamma^t` times that along a trajectory drawn from `pi`:

  E_{tau~p_pi}[ sum_t gamma^t ( r_t + gamma V^mu(s_{t+1}) - V^mu(s_t) ) ].

The `gamma^t gamma V^mu(s_{t+1})` term and the `-gamma^{t+1} V^mu(s_{t+1})` term from the next step telescope; for bounded values the tail term vanishes as the horizon goes to infinity, so everything cancels except the very first `-V^mu(s_0)` and the running reward sum. It collapses to

  E_{tau~p_pi}[ -V^mu(s_0) + sum_t gamma^t r_t ] = -E_{s_0}[V^mu(s_0)] + J(pi) = -J(mu) + J(pi),

since `E_{s_0}[V^mu(s_0)]` is just `J(mu)`. The identity holds. Good — and notice what it bought me: the moment I phrase the goal as *improvement* rather than raw return, the advantage appears automatically, with the baseline `V^mu` already inside it. The baseline isn't a variance-reduction trick I bolt on; it's what "improvement over `mu`" *means*. That's the principled route to the thing I wanted.

Now rewrite the trajectory expectation as a state expectation. Pushing the `gamma^t` into the state-visitation counts,

  eta(pi) = E_{s ~ d_pi(s)} E_{a ~ pi(a|s)} [ A^mu(s, a) ],   d_pi(s) = sum_t gamma^t p(s_t = s | pi),

the unnormalized discounted state distribution of `pi`. And here's the wall this hits, the same one Kakade-Langford and the trust-region line ran into: the expectation is over `d_pi`, the states the *new* policy visits — which depends on the very `pi` I'm optimizing, and which I can't sample without running `pi`. I can't optimize an objective whose sampling distribution moves with the optimization variable. So I do what the trust-region derivation does: approximate `d_pi` by `d_mu`, the state distribution I *do* have data from. Define the surrogate

  hat_eta(pi) = E_{s ~ d_mu(s)} E_{a ~ pi(a|s)} [ A^mu(s, a) ].

When is this legitimate? It matches `eta(pi)` to first order at `pi = mu`, and it stays a good approximation as long as `pi` doesn't wander too far from `mu` — the difference is controlled by how different their state distributions are, which is in turn bounded by the KL divergence between `pi` and `mu`. So the surrogate is only trustworthy inside a neighborhood of `mu`, which means I'm not allowed to maximize it freely — I have to maximize it *subject to staying close to `mu` in KL*. The approximation I made to get a tractable objective is exactly what forces a trust-region constraint. That gives me a clean constrained problem:

  maximize_pi   integral d_mu(s) integral pi(a|s) [ R^mu_{s,a} - V^mu(s) ] da ds
  subject to    integral d_mu(s) D_KL( pi(.|s) || mu(.|s) ) ds <= epsilon,
                integral pi(a|s) da = 1  for all s.

The KL constraint does double duty: it keeps the surrogate honest, and it keeps `pi` close to the *data distribution*, which — remembering the out-of-distribution lesson — means `pi` won't run off toward actions the data never contained. (Strictly I'd want the KL bounded at every state pointwise, but that's infinitely many constraints; relaxing to the expected KL over `d_mu` is the tractable version.)

Now I solve it. This is a constrained variational problem over the function `pi(a|s)`, so I form the Lagrangian, with `beta` the multiplier on the KL constraint and `alpha_s` the per-state multipliers enforcing normalization:

  L(pi, beta, alpha) = integral d_mu(s) integral pi(a|s)[ R^mu_{s,a} - V^mu(s) ] da ds
        + beta ( epsilon - integral d_mu(s) D_KL(pi(.|s) || mu(.|s)) ds )
        + integral alpha_s ( 1 - integral pi(a|s) da ) ds.

I want the optimal `pi`, so I take the functional derivative of `L` with respect to `pi(a|s)` at a particular `(s,a)` and set it to zero. The first term contributes `d_mu(s)(R^mu_{s,a} - V^mu(s))` — it's linear in `pi(a|s)`. The KL term: `D_KL(pi||mu) = integral pi log(pi/mu) da = integral pi log pi - integral pi log mu`, so its derivative with respect to `pi(a|s)` is `log pi(a|s) - log mu(a|s) + 1` (the `+1` from differentiating `pi log pi`), and it enters `L` with coefficient `-beta d_mu(s)`. The normalization term contributes `-alpha_s`. Putting it together:

  dL/dpi(a|s) = d_mu(s)( R^mu_{s,a} - V^mu(s) ) - beta d_mu(s)( log pi(a|s) - log mu(a|s) + 1 ) - alpha_s = 0.

Solve for `log pi(a|s)`:

  log pi(a|s) = (1/beta)( R^mu_{s,a} - V^mu(s) ) + log mu(a|s) - 1 - alpha_s / (beta d_mu(s)).

Exponentiate:

  pi(a|s) = mu(a|s) exp( (1/beta)( R^mu_{s,a} - V^mu(s) ) ) * exp( -1 - alpha_s/(beta d_mu(s)) ).

The last factor doesn't depend on `a` — it's a per-state constant, and it's pinned by the normalization constraint `integral pi(a|s) da = 1`. So it has to be the reciprocal of the partition function. The optimal policy is

  pi*(a|s) = (1/Z(s)) mu(a|s) exp( (1/beta)( R^mu_{s,a} - V^mu(s) ) ),
  Z(s) = integral mu(a'|s) exp( (1/beta)( R^mu_{s,a'} - V^mu(s) ) ) da'.

There it is — and look what fell out. The optimal policy is the data policy `mu` *reweighted by the exponentiated advantage*, with the temperature being exactly the KL multiplier `beta`. The exponential isn't a heuristic I picked; it's the closed-form solution of the KL-constrained improvement problem. And the multiplier `beta` controls the trade-off precisely as a temperature should: small `beta` makes the exponential sharp (aggressive, trusts the advantage estimate, big step away from `mu`), large `beta` flattens it toward `mu` (conservative). This is the same Gibbs/Boltzmann shape that the relative-entropy method gets from its dual — `pi proportional to (data) * exp(weight/temperature)` — except its "weight" was a one-step Bellman error tied to a feature-matching value function and a messy dual you minimize with BFGS, and here the weight is the return-based advantage and the value function is going to be, I hope, an ordinary regression.

But `pi*` is a *non-parametric* object — a reweighting of `mu` defined pointwise. My policy is a neural network. I need to project `pi*` onto the manifold of parameterized policies. The natural projection is to minimize the KL from `pi*` to my parametric `pi`, averaged over the states I have:

  argmin_pi  E_{s ~ d_mu(s)} [ D_KL( pi*(.|s) || pi(.|s) ) ].

Expand that KL: `D_KL(pi* || pi) = integral pi* log pi* da - integral pi* log pi da`. The first term doesn't involve my parameters, so minimizing the KL is maximizing `E_{s~d_mu} integral pi*(a|s) log pi(a|s) da` — a cross-entropy, fitting `pi` to the distribution `pi*`. Now substitute `pi*(a|s) = (1/Z(s)) mu(a|s) exp((1/beta)A^mu)` and turn the integral over `a` weighted by `mu(a|s)` into an expectation over `a ~ mu`:

  argmax_pi  E_{s ~ d_mu(s)} E_{a ~ mu(a|s)} [ (1/Z(s)) exp( (1/beta)( R^mu_{s,a} - V^mu(s) ) ) * log pi(a|s) ].

The `1/Z(s)` is a positive per-state constant that doesn't depend on the action. If I were fitting an independent conditional distribution at each state, multiplying every sample from that state by a positive constant would not change the per-state optimizer at all. With a shared neural network it would reweight states, so dropping it is a state-weighting approximation: I keep the action-dependent weighting I can estimate from samples and avoid estimating a per-state partition function. That leaves

  argmax_pi  E_{s ~ d_mu(s)} E_{a ~ mu(a|s)} [ log pi(a|s) * exp( (1/beta)( R^mu_{s,a} - V^mu(s) ) ) ].

That is *exactly* RWR's weighted-regression update — `E[ log pi(a|s) * weight ]` over samples drawn from the data — but with the weight being the exponentiated *advantage* instead of the exponentiated return. Which is precisely the modification my pain analysis pointed at, now derived rather than asserted: the improvement objective put the baseline `V^mu` inside the weight, the KL-constrained Lagrangian produced the exponential and the temperature, and the projection turned it back into a stable weighted maximum-likelihood fit. The inner loop is supervised regression, and it trains `pi` only on actions `a ~ mu` that are actually in the data — so the out-of-distribution catastrophe never arises, because I never evaluate `pi` or a value at an action I didn't observe.

There's a subtlety in the choice of projection direction I should pin down, because it's load-bearing and easy to get backwards. I minimized `D_KL(pi* || pi)` — the *forward* KL, with `pi*` (the data-weighted target) first. That direction is what turns the projection into "fit `pi` to samples drawn from `mu`, reweighted," i.e. a sample-based weighted MLE that needs only samples, never the density `mu(a|s)` itself. Had I used the reverse KL `D_KL(pi || pi*)`, I'd get the `pi`-side expectation `E_{a~pi}[log pi - log pi*]`, which contains `log mu` explicitly — that forces me to *learn an explicit model of the behavior policy `mu`* and plug its density into a penalty. That's the road the explicit-behavior-model methods take, and it's exactly the complexity I'm trying to avoid: fitting `mu` is itself a hard density-estimation problem, especially when `mu` is a moving mixture of many past policies. The forward KL gives me the constraint *implicitly*, for free, just by sampling from the buffer. So forward KL it is — not an arbitrary convention, but the choice that keeps the whole thing to a regression over observed actions.

So now I have an on-policy algorithm: collect data with `pi_k`, fit `V` to its returns, do one weighted-advantage regression step, repeat. But this is still on-policy — `mu = pi_k`, data dropped each iteration — and the entire point was to reuse a buffer of *many* past policies. I need to redo the derivation with `mu` being a mixture, because that's where experience replay actually lives. Let me model the buffer honestly: it holds trajectories from policies `pi_1, ..., pi_k`, and sampling a trajectory from the buffer is sampling policy `pi_i` with probability `w_i` (with `sum_i w_i = 1`). So the buffer's distributions are mixtures:

  mu(s,a) = sum_i w_i d_{pi_i}(s) pi_i(a|s),   d_mu(s) = sum_i w_i d_{pi_i}(s),   mu(a|s) = mu(s,a)/d_mu(s).

Now what's the improvement objective? Improvement over the *whole* buffer, weighted the same way: `eta(pi) = J(pi) - sum_i w_i J(pi_i)`. By linearity, `= sum_i w_i (J(pi) - J(pi_i))`, and each term I already know from Kakade-Langford — `J(pi) - J(pi_i) = E_{s~d_pi, a~pi}[ A^{pi_i}(s,a) ]`. So the exact expression is

  eta(pi) = sum_i w_i E_{s ~ d_pi, a ~ pi} [ A^{pi_i}(s, a) ].

If I now used a single `d_mu` expectation multiplying the unweighted sum of advantages, I would lose which policy actually visited which state. The data in the buffer gives me a sharper surrogate: replace `d_pi` inside each term by that term's own sampling distribution `d_{pi_i}`, then sum the terms,

  hat_eta(pi) = sum_i w_i E_{s ~ d_{pi_i}, a ~ pi} [ A^{pi_i}(s, a) ].

This can also be written as an expectation over `d_mu(s) = sum_j w_j d_{pi_j}(s)` with the advantage at a state equal to the density-weighted average `[sum_i w_i d_{pi_i}(s) A^{pi_i}(s,a)] / [sum_j w_j d_{pi_j}(s)]`. The KL is against the *mixture* conditional `mu(a|s)`, which is exactly what I want: it keeps `pi` from choosing actions that are unlike *all* of the past policies, and because each `pi_i` enters `mu` weighted by its own state density `d_{pi_i}(s)`, `pi` is only constrained to resemble `pi_i` at the states `pi_i` actually visited. Running the identical Lagrangian calculation as before, the optimal policy is again Gibbs-form:

  pi*(a|s) = (1/Z(s)) mu(a|s) exp( (1/beta) * [ sum_i w_i d_{pi_i}(s) ( R^{pi_i}_{s,a} - V^{pi_i}(s) ) ] / [ sum_j w_j d_{pi_j}(s) ] ).

Two ugly things to deal with here. First, the baseline in the exponent is `[ sum_i w_i d_{pi_i}(s) V^{pi_i}(s) ] / [ sum_j w_j d_{pi_j}(s) ]` — a state-density-weighted *average* of the value functions of all the past policies. I am not going to fit a separate `V^{pi_i}` for every policy in the buffer; with only a little data from each, those estimates would be garbage. But I don't have to. Watch what happens if I fit a *single* value function `Vbar(s)` by ordinary squared-error regression onto the observed returns across the *whole* buffer:

  Vbar = argmin_V  sum_i w_i E_{s ~ d_{pi_i}, a ~ pi_i} [ || R^{pi_i}_{s,a} - V(s) ||^2 ].

The minimizer of a squared loss is the conditional mean of the target. Differentiate with respect to `V(s)` and set to zero: at each state `s`, `sum_i w_i d_{pi_i}(s) (E[R^{pi_i}_{s,.}] - V(s)) = 0`, so

  Vbar(s) = [ sum_i w_i d_{pi_i}(s) V^{pi_i}(s) ] / [ sum_j w_j d_{pi_j}(s) ],

which is *exactly* the weighted-average baseline I needed. So a single regression of one value network onto the buffer returns automatically computes the right mixture baseline — no per-policy value functions, no bookkeeping. That's the kind of collapse that tells me the formulation is the natural one.

Second ugly thing: the return in the exponent is the density-weighted average of `R^{pi_i}_{s,a}` values, and each `R^{pi_i}_{s,a}` is itself the expected return after taking `(s,a)` and then continuing with policy `pi_i`. That expectation sits *inside* a nonlinear `exp`. Estimating it directly would require rolling out multiple continuation policies from the same state-action pair — needing a resettable environment, which I don't have. So I make the cheap approximation: use the single observed return target for the `(s,a)` pair actually in the buffer, `R^D_{s,a}`. Because the expectation is inside the exponential, a single-sample estimate is *biased* (Jensen: `E[exp(X)] != exp(E[X])`). That is the bias I accept to keep the update sample-based. With both simplifications, the practical policy update over the buffer `D` is

  argmax_pi  E_{(s,a) ~ D} [ log pi(a|s) * exp( (1/beta)( R^D_{s,a} - Vbar(s) ) ) ],

and the value update is the squared-error regression above — both just sampling minibatches from `D`. Two regressions, one buffer, a fixed temperature. That's the algorithm, and it's about as simple as I dared hope.

Now the practical knobs, each of which I want to justify rather than just set. The first is `beta`. The clean derivation says `beta` is the KL multiplier, and methods in this lineage adaptively solve for it each iteration to hit a target KL `epsilon` — that's another optimization, more code, more fragility. Do I need it? The role of `beta` is to set how aggressively the exponential sharpens the advantage; a single fixed value across training is a fixed trust-region temperature. In the replay implementation I want to match, I standardize advantages before the exponential and set `temp = 1.0`, so the exponent sees a normalized advantage divided by a unit-scale temperature. In the compact on-policy loss slot I have to fill, the loop also hands me normalized advantages, but the slot uses the sharper local setting `0.05`, then normalizes the clipped weights back to mean one so the policy-gradient scale stays steady.

Second, the returns. I could use raw Monte-Carlo returns `sum_t gamma^t r_t` for both the value regression target and the advantage. But MC returns are high-variance — a single noisy trajectory tail swings the target a lot — and that variance flows straight into the value fit and the advantage. The standard lower-variance estimator is the `TD(lambda)` return, which blends the bootstrap `r_t + gamma V(s_{t+1})` with the longer-horizon return via a `lambda` mix, bootstrapping off the value function from the previous iteration. With `lambda = 0.95` it keeps much of the long-horizon return signal while trading some bias for lower variance; at `lambda = 1` the recursion becomes the Monte-Carlo target. Concretely, walking backward through a path of length `L` with `val_t` the value estimates:

  return_{L-1} = r_{L-1} + gamma * val_L
  return_t     = r_t + gamma * ( (1 - lambda) val_{t+1} + lambda * return_{t+1} ),  t < L-1,

and the advantage is then `A = return_t - V(s_t)`.

Third, and this one I learn from the shape of the weight itself: the policy weight is `exp((1/beta) A)`. With `beta` small, a single large positive advantage produces an astronomically large weight, and that one sample dominates the regression. Let me make "dominates" concrete by tracing the weight computation on a minibatch of normalized advantages `A = [2.0, 0.5, 0.0, -0.5, -1.0, -1.0]` at `beta = 0.05`. Unclipped, `exp(A/beta)` gives `[2.4e17, 2.2e4, 1, ~0, ~0, ~0]`, and the share of total weight carried by the single `A=2.0` sample is `2.4e17 / (2.4e17 + ...) = 1.0000` — that one action *is* the entire regression target; every other sample is rounding error. So I cap it: `omega_hat = min( exp((1/beta) A), omega_max )` with `omega_max = 20`. Re-tracing with the clamp, the weights become `[20, 20, 1, 0, 0, 0]`, and the `A=2.0` sample's share drops from `1.0000` to `0.49` — large, but no longer the whole update; the second-best action now counts equally. It's a clip on the weight, the regression analogue of the gradient/ratio clipping that stabilizes the on-policy methods — it prevents a handful of high-advantage outliers from hijacking the update. (Worth noting how readily the cap engages: at `beta = 0.05` the weight hits the ceiling of `20` once `A` exceeds `beta * log 20 = 0.15`, i.e. about a sixth of a standard deviation of normalized advantage, so the clip is a routine part of the update, not a rare safety net.) I also need the exponent to see standardized advantages, so the temperature means the same thing across tasks and batches. In the full replay loop that means explicitly subtracting the buffer advantage mean and dividing by its standard deviation before the exponential; in the compact loss slot, the surrounding loop already hands the loss normalized minibatch advantages. The clipped weights are then renormalized to mean one — tracing it through, `[20, 20, 1, 0, 0, 0]` becomes `[2.93, 2.93, 0.15, 0, 0, 0]`, whose mean is exactly `1.0` — a local step-size stabilizer that keeps the policy-loss scale steady for the shared optimizer, not a new theoretical term.

And experience replay itself is now not just an efficiency hack but a stability knob I can reason about. The buffer is a FIFO queue of, say, the last 50k samples; `mu` is the mixture of policies whose data is still in it. A *larger* buffer holds older data longer, so `mu` drifts more slowly; because the KL constraint ties `pi` to `mu`, a slowly-changing `mu` forces `pi` to change slowly too — more stable, but slower progress. A *smaller* buffer is closer to on-policy: faster, but the trust region snaps around a small dataset and overfits, which destabilizes. So buffer size is a direct stability-versus-speed dial, and I can see *why* from the trust-region penalty, not just observe it. I'll also sample states uniformly from `D` rather than from the discounted `d_mu` the theory asks for — strictly an approximation, but it's standard, simpler, and the discounting mostly reweights early-trajectory states, which uniform sampling handles acceptably.

Let me now write the thing as it would actually drop into a standard actor-critic harness. The policy is a Gaussian whose mean is a fixed MLP and whose log-std is a learned vector; the value is a separate MLP. The harness already gives me, per minibatch, the observations, the actions taken, the old log-probs, the advantages, and the returns (from the `TD(lambda)`/GAE estimator). All I have to design is the loss: a weighted-regression policy loss and a squared-error value loss. The advantage-weighted regression loss is `-(log pi(a|s) * weight)` averaged, with `weight = clamp(exp(A/beta), max=omega_max)`, and the value loss is `0.5 * (V - return)^2`.

```python
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    """Gaussian policy (mean from a fixed MLP, learned log-std) + separate value MLP.
    Only the action/value readout and the per-minibatch loss are the design surface."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        h = 64
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        # Gaussian policy. log pi(a|s) is what the weighted-regression loss fits;
        # the policy is trained only on actions that are passed in (i.e. observed in
        # the data), never on actions the policy is free to invent -> no OOD blow-up.
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """Advantage-weighted regression: a weighted maximum-likelihood policy loss and
    a squared-error value loss -- both ordinary supervised regression."""
    beta = 0.05            # KL multiplier reused as a fixed temperature (no dual solve)
    weight_max = 20.0      # cap on exp((1/beta)*A) so outliers don't hijack the update

    # log pi(a|s) of the OBSERVED actions, plus the value V(s); entropy for monitoring.
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)

    # Diagnostics only (kept for the harness's logging).
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Closed-form Gibbs weight from the KL-constrained improvement problem.
    # mb_advantages is already normalized by the surrounding minibatch loop.
    with torch.no_grad():
        weights = torch.exp(mb_advantages / beta)
        weights = torch.clamp(weights, max=weight_max)
        weights = weights / (weights.sum() + 1e-8) * weights.numel()

    # Policy loss = -E[ log pi(a|s) * weight ]  (weighted MLE = supervised regression)
    pg_loss = -(newlogprob * weights).mean()

    # Value loss = 0.5 * E[ (V(s) - return)^2 ]  (the single mean value fn Vbar by MSE)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```

Let me trace the causal chain back, because I want to be sure each piece earned its place. I started wanting an RL algorithm whose inner loop is plain supervised regression and that can reuse off-policy data — because value-bootstrapping methods are fragile and blow up at out-of-distribution actions, and on-policy gradients throw data away. Reward-weighted regression already had the regression-as-RL skeleton, but it exponentiated the raw return, which is scale-sensitive and high-variance, and it was on-policy. Phrasing the objective as expected *improvement* over the data policy, rather than raw return, made the advantage — return minus a baseline — appear on its own via the performance-difference identity, fixing the scale sensitivity for principled reasons. The improvement objective depends on the new policy's own state visitation, which I can't sample, so I swapped in the data policy's state distribution, and that surrogate is only valid near `mu`, which forced a KL trust-region constraint — which, conveniently, is also what keeps `pi` near the observed data. Solving that KL-constrained problem with a Lagrangian gave a closed-form Gibbs policy: the data policy reweighted by the exponentiated advantage, with the KL multiplier as the temperature `beta`. Projecting that non-parametric optimum onto my network via the *forward* KL turned it back into a weighted maximum-likelihood fit over observed actions — giving the constraint implicitly, with no need to model `mu`'s density. Redoing the derivation for a buffer that mixes many past policies showed the right baseline is a state-density-weighted average of their value functions, which a *single* squared-error value regression over the whole buffer computes automatically; the return in the exponent gets a biased single observed target for tractability. Fixing `beta` instead of solving a dual, using `TD(lambda)` returns to trade some bias for lower variance, clipping the weights so outliers don't explode, and sizing the replay buffer to trade stability against speed are the practical choices, each tied back to a specific failure it prevents. What lands is two supervised regressions on one replay buffer — a value fit and an advantage-weighted policy fit — that handle continuous and discrete actions alike and never touch an action the data didn't contain.
