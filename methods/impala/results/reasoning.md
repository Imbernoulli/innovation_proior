OK, let me think about this from the throughput problem, because that's what's actually killing me. I have an actor-critic agent — a policy π and a value baseline V^π — and on a single domain it already eats something like a billion frames and several days to get good. I want to train one agent on tens of domains at once. At that point the wall-clock is just hopeless: a single GPU, or a single machine's worth of actors, isn't going to feed a deep net fast enough. So before I touch the learning rule at all, I need to ask: what is the most data I can possibly push through this thing per second, and what architecture gets me there?

Look at how A3C scales. Each worker holds its own copy of the parameters, steps its own environment, computes an on-policy n-step advantage gradient against its local copy, and ships the *gradient* back to a central parameter server, asynchronously, Hogwild-style. That's elegant and fault tolerant, but three things bother me when I imagine 1000 of these. First, I'm sending gradients, which are the size of the model — every worker, every few steps, pushes a full parameter-sized blob over the network. Deep nets make that worse and worse. Second, each worker does its own forward and backward on a tiny batch of one trajectory; a GPU hates that, it wants big fused ops. Third, with that many asynchronous workers the gradients going into the parameter server are stale — computed against parameters that have since moved — and the staleness degrades learning.

So what if I flip what gets communicated? Instead of every worker computing gradients, let the workers — call them *actors* — only *generate experience*: step the environment, record states, actions, rewards, and send the raw trajectory to a central *learner*. The learner is the only thing that holds the optimiser and does backprop. Now the actors send observations, which is cheaper than gradients and, crucially, decoupled from model size. The learner receives whole trajectories from many actors at once, stacks them into a big batch, and does one fat GPU update. It can fold the time dimension into the batch dimension for the convnet and the output layer, run them on thousands of timesteps in parallel, and only the recurrent core has to respect time order. That's exactly the shape of computation a GPU is built for. This decoupling is the whole point: acting is embarrassingly parallel across cheap machines, learning is one big accelerated batch.

Let me sanity-check the obvious alternative, which is to keep it synchronous: one learner that steps a *batch* of environments in lockstep, like batched A2C. The trouble is that a synchronous batch step is only as fast as its slowest environment. On Atari, where rendering and game logic are dirt cheap relative to the tensor ops, that's fine. But I care about 3D environments like DeepMind Lab, which are expensive and have high variance in per-step simulation time, plus variable-length episodes that stall on reset. One slow env in the batch and every other env waits. Synchronising every n steps instead of every step doesn't fix it; it just moves the stall. So the batched-synchronous design caps my throughput at the tail of the environment-time distribution. I don't want that. I want actors that never wait on each other — fully asynchronous generation, queued to the learner.

There's a closer relative, GA3C, which already decouples the forward pass (acting) from the backward pass (learning) with dynamic batching on the GPU. Good instinct, same as mine. But it learns *on-policy* on data that the decoupling has made *off-policy*, and it gets instabilities; their patch is to add a small constant to the action probabilities when forming the policy gradient. That's a band-aid over a real wound, and the wound is exactly the thing I'm about to walk into.

Because here's the cost of decoupling. By the time a trajectory reaches the learner and the learner computes its gradient, the learner's policy has already moved — it's done several updates while that trajectory was being generated and queued. So the policy that *generated* the data, call it the behaviour policy μ, lags behind the learner's current policy, call it π, by several updates. The faster and more distributed I make this, the bigger the lag. My data is off-policy, and the gap grows with exactly the scale I'm chasing. If I run an on-policy actor-critic update on it — pretend μ = π — I'm computing a gradient for the wrong distribution. That's the source of GA3C's instability, and I should expect the damage to grow as I add machines. A tiny ε in the action probabilities can keep logs finite, but it does not change which policy generated the sample. So policy-lag is the enemy, and the question becomes: how do I correctly learn the value function and the policy of the *learner's* policy π from trajectories drawn under a *different*, lagged policy μ?

This is off-policy learning, and I know the textbook tool: importance sampling. If I want an expectation under π but I sample under μ, I reweight by the ratio π/μ. For a single step that's `π(a|x)/μ(a|x)`. For a multi-step return I'd reweight the whole thing by the *product* of per-step ratios, `∏_t π(a_t|x_t)/μ(a_t|x_t)`. Unbiased, in principle. But stare at that product. The more off-policy I am — the more μ and π disagree — the wilder the individual ratios, and a product of n of them can blow up exponentially or collapse to near zero. The variance of that estimator grows with the horizon. So raw importance sampling on n-step returns is a non-starter at the lags I expect. I need the correction, but I need to tame its variance, and I need it to target a *state-value* V, because my actor-critic learns V, not a state-action Q.

That last point rules out reaching for Retrace as-is. Retrace is the nicest off-policy multi-step correction I know — it cuts the importance-sampling traces with coefficients `c_i = λ min(1, π/μ)`, truncating each ratio at 1 so the product can't explode, and it's provably safe for any μ and π. But Retrace is built to correct a state-action value Q, and it needs to *learn* Q. My agent has a V. And there's a second nuisance: Retrace doesn't collapse back to the ordinary on-policy n-step return when μ = π — its trace machinery stays switched on. I'd love one update rule that is *literally* the on-policy n-step update when there's no lag, and only deviates when there is. So I'm going to have to build my own correction, on V, keeping Retrace's good idea (truncate the per-step ratios so the product behaves) but designed around the state-value function.

Let me set up notation and start from what I can't argue with. The one-step temporal difference for V is `r_t + γ V(x_{t+1}) − V(x_t)`. On-policy, summing discounted TDs telescopes the interior values and gives the n-step return. The cleanest way to write a value *target* — the thing I regress V(x_s) toward — is "current estimate plus a correction": `v_s = V(x_s) + (sum of corrections)`. If I'm on-policy the correction is just the sum of discounted TDs, `Σ_{t=s}^{s+n-1} γ^{t-s} (r_t + γ V(x_{t+1}) − V(x_t))`, and that telescopes — the `−γ^{t-s}V(x_t)` of term t cancels the `+γ^{(t-1)-s}·γ V(x_t)` of term t−1 — leaving `Σ_{t} γ^{t-s} r_t + γ^n V(x_{s+n})`, the n-step Bellman target, minus the leading `V(x_s)` it adds back. Good. So on-policy my target is "bootstrap n real rewards then V". Now I need to inject the off-policy correction into this sum without letting variance explode and without breaking that on-policy collapse.

Two separate things are wrong off-policy, and they're wrong in two *different* places, so I should correct them with two *different* knobs. Let me find them.

Thing one: each individual TD residual `r_t + γ V(x_{t+1}) − V(x_t)` was generated by taking action a_t under μ, but I want the residual as if a_t came from π. That's a per-step distributional mismatch at time t, and the per-step IS ratio `π(a_t|x_t)/μ(a_t|x_t)` is what corrects it. So I'll multiply each TD residual by some version of this ratio — call the truncated version `ρ_t` — giving a corrected residual `δ_t V = ρ_t (r_t + γ V(x_{t+1}) − V(x_t))`. This ρ_t lives *inside* each TD.

Thing two: the residual at time t only belongs in the target for state x_s if the path from s to t was itself something π would have taken. The earlier actions a_s, …, a_{t-1} were all drawn from μ, and I'm crediting their downstream residual back to x_s. So the *propagation* of a time-t residual back to time s needs its own correction — the product of per-step ratios along the way, `∏_{i=s}^{t-1} π(a_i|x_i)/μ(a_i|x_i)`. That's the trace, and it's where the explosive product lives. Call the truncated per-step factors `c_i`, and the propagation weight is `∏_{i=s}^{t-1} c_i`.

So the target wants to be
`v_s = V(x_s) + Σ_{t=s}^{s+n-1} γ^{t-s} (∏_{i=s}^{t-1} c_i) δ_t V`, with `δ_t V = ρ_t (r_t + γ V(x_{t+1}) − V(x_t))`, and with `∏_{i=s}^{s-1} c_i = 1` so the s-th term has no trace. There it is structurally: ρ_t corrects the residual itself, the product of c_i corrects how far back that residual reaches.

Now the truncations. For the trace, I *must* clip, because that product is the thing that blows up — same lesson as Retrace. So `c_i = min(c̄, π(a_i|x_i)/μ(a_i|x_i))` for some ceiling c̄. For the residual ratio, I'll also clip it, `ρ_t = min(ρ̄, π(a_t|x_t)/μ(a_t|x_t))` with ceiling ρ̄ — but notice ρ_t is *not* multiplied across time, it sits alone inside one TD, so its variance doesn't compound over the horizon the way the trace product does. That asymmetry is going to matter. Let me also pin down: I'll assume `ρ̄ ≥ c̄`. I don't yet see exactly why, but the trace ceiling being no larger than the residual ceiling feels right, and I'll watch for where the proof needs it.

First, the sanity check I demanded: on-policy collapse. If π = μ, every ratio is 1, so `ρ_t = min(ρ̄, 1)` and `c_i = min(c̄, 1)`. If I take c̄ ≥ 1 (and ρ̄ ≥ 1, which holds since ρ̄ ≥ c̄ ≥ 1), then all c_i = 1 and ρ_t = 1, and the target is exactly `V(x_s) + Σ γ^{t-s}(r_t + γV(x_{t+1}) − V(x_t))`, which telescopes to the on-policy n-step Bellman target `Σ γ^{t-s} r_t + γ^n V(x_{s+n})`. So with no lag my rule *is* the ordinary on-policy n-step update — the property Retrace doesn't have. One algorithm for both regimes. Good.

Now the real question, the one that tells me whether this object is even sensible: what does V converge to under this rule? What's the fixed point? Let me define the operator I'm iterating. Take the infinite-horizon version so I don't carry the n around:
`R V(x) = V(x) + E_μ[ Σ_{t≥0} γ^t (c_0 … c_{t-1}) ρ_t (r_t + γ V(x_{t+1}) − V(x_t)) | x_0 = x ]`,
expectation over trajectories from μ. I want the V with `R V = V`, and I want to know it's unique and that iterating R gets me there.

Let me first reorganise R so the V-dependence is visible. Pull out the V terms. The `V(x)` out front, and inside the sum each term t contributes `−γ^t (∏_{s=0}^{t-1} c_s) ρ_t V(x_t)` and `+γ^{t+1} (∏_{s=0}^{t-1} c_s) ρ_t V(x_{t+1})`. The t=0 residual contributes `−ρ_0 V(x_0)`, which combines with the leading `V(x_0)=V(x)` to give `(1 − E_μ ρ_0) V(x)`. Collecting, I can write
`R V(x) = (1 − E_μ ρ_0) V(x) + E_μ[ Σ_{t≥0} γ^t (∏_{s=0}^{t-1} c_s)( ρ_t r_t + γ(ρ_t − c_t ρ_{t+1}) V(x_{t+1}) ) ]`.
Let me check that the V(x_{t+1}) coefficient is right: V(x_{t+1}) gets `+γ` from the residual at time t (coefficient `γ^t (∏_{0}^{t-1}c_s) ρ_t · γ`) and `−1` from the residual at time t+1 (coefficient `γ^{t+1}(∏_{0}^{t}c_s) ρ_{t+1}` with the minus sign). Factor `γ^{t+1}(∏_{0}^{t-1}c_s)` out of both: `γ^{t+1}(∏_{0}^{t-1}c_s)(ρ_t − c_t ρ_{t+1})`. Yes — that matches, since the second term carries the extra `c_t` from extending the product. Good, the rewrite holds.

Now take two value functions V_1, V_2 and subtract — the constants and the rewards cancel, leaving a linear map on the difference:
`R V_1(x) − R V_2(x) = (1 − E_μ ρ_0)(V_1(x) − V_2(x)) + E_μ[ Σ_{t≥0} γ^{t+1}(∏_{s=0}^{t-1}c_s)(ρ_t − c_t ρ_{t+1})(V_1(x_{t+1}) − V_2(x_{t+1})) ]`.
I'd like to write this as a single clean sum over states visited. Reindex the sum so everything is in terms of `V_1(x_t) − V_2(x_t)`. Shift t→t−1 in the sum term (so x_{t+1} becomes x_t), and fold the leading `(1−E_μρ_0)(V_1(x)−V_2(x))` in as the t=0 term, using the conventions `c_{-1} = ρ_{-1} = 1` and `∏_{s=0}^{t-2} c_s = 1` for t = 0 and 1. Then
`R V_1(x) − R V_2(x) = E_μ[ Σ_{t≥0} γ^t (∏_{s=0}^{t-2} c_s)(ρ_{t-1} − c_{t-1} ρ_t)(V_1(x_t) − V_2(x_t)) ]`.
Let me verify the t=0 term reproduces the leading piece: `γ^0 · 1 · (ρ_{-1} − c_{-1} ρ_0)(V_1(x)−V_2(x)) = (1 − ρ_0)(V_1−V_2)`, and in expectation `(1 − E_μ ρ_0)(V_1(x)−V_2(x))`. Matches. So define the weights `α_t = ρ_{t-1} − c_{t-1} ρ_t`.

For this to be a contraction in sup-norm I want: the weights are non-negative (in expectation), and their total is < 1. Then `|R V_1(x) − R V_2(x)|` is a convex-ish combination — a non-negative-weighted sum, with total weight η — of differences `|V_1 − V_2|` at other states, hence `≤ η ‖V_1 − V_2‖_∞`.

Non-negativity first. `E_μ α_t = E_μ[ρ_{t-1} − c_{t-1} ρ_t]`. Here's where `c̄ ≤ ρ̄` earns its keep. Since c_{t-1} and ρ_{t-1} are the *same* ratio clipped at c̄ and ρ̄ respectively with c̄ ≤ ρ̄, we always have `c_{t-1} ≤ ρ_{t-1}`. So `ρ_{t-1} − c_{t-1} ρ_t ≥ c_{t-1} − c_{t-1} ρ_t = c_{t-1}(1 − ρ_t)`. Take expectations: `E_μ α_t ≥ E_μ[c_{t-1}(1 − ρ_t)]`. This last expectation is non-negative only after conditioning on the history through x_t: c_{t-1} is already fixed and non-negative, while `E_μ[ρ_t | x_t] ≤ E_μ[π(a_t|x_t)/μ(a_t|x_t) | x_t] = Σ_a π(a|x_t) = 1`. Hence `E_μ[c_{t-1}(1 − ρ_t)] = E_μ[c_{t-1} E_μ[1 − ρ_t | x_t]] ≥ 0`. So the weights are non-negative in expectation. (That's exactly why I need `ρ̄ ≥ c̄`: it makes `ρ_{t-1} ≥ c_{t-1}` before the conditional-expectation step.)

Now the total weight, η. Sum the coefficients:
`Σ_{t≥0} γ^t E_μ[(∏_{s=0}^{t-2}c_s)(ρ_{t-1} − c_{t-1}ρ_t)]`.
Split it. The first piece is `S := Σ_{t≥0} γ^t E_μ[(∏_{s=0}^{t-2}c_s) ρ_{t-1}]`. The second is `Σ_{t≥0} γ^t E_μ[(∏_{s=0}^{t-1}c_s) ρ_t]` (the `c_{t-1}` joins the product to make `∏_{0}^{t-1}`). That second sum is the same shape as S but shifted: it's `Σ_{t≥0} γ^t E_μ[(∏_{0}^{t-1}c_s)ρ_t] = γ^{-1} Σ_{t≥1} γ^{t} E_μ[(∏_{0}^{t-1}c_s)ρ_t] = γ^{-1}(S − 1)`, where the `−1` peels off the t=0 term of S, which is `γ^0 E[(∏_{0}^{-2}c_s)ρ_{-1}] = 1`. So the total is
`S − γ^{-1}(S − 1) = γ^{-1} − (γ^{-1} − 1) S`.
That's my η: `η = γ^{-1} − (γ^{-1} − 1) S`, with `S = Σ_{t≥0} γ^t E_μ[(∏_{s=0}^{t-2}c_s)ρ_{t-1}]`.

Is η < 1? Note `γ^{-1} − 1 > 0`. And S is at least its first two terms: t=0 gives 1, t=1 gives `γ E_μ ρ_0`, so `S ≥ 1 + γ E_μ ρ_0`. Therefore `η ≤ γ^{-1} − (γ^{-1}−1)(1 + γ E_μ ρ_0)`. Expand: `γ^{-1} − (γ^{-1}−1) − (γ^{-1}−1)γ E_μρ_0 = 1 − (1 − γ) E_μ ρ_0`. (Check the last term: `(γ^{-1}−1)γ = 1 − γ`.) So `η ≤ 1 − (1−γ) E_μ ρ_0`. If I assume there's a `β ∈ (0,1]` with `E_μ ρ_0 ≥ β` — which just says the behaviour policy has *some* mass where π wants to go, so ρ_0 isn't degenerate — then `η ≤ 1 − (1−γ)β < 1`. Contraction. Unique fixed point, and iterating R converges to it geometrically.

Now the payoff question: *what* is that fixed point? Solve `R V = V`, which by the form of R means the expected correction is zero, i.e. `E_μ[ ρ_t (r_t + γ V(x_{t+1}) − V(x_t)) | x_t ] = 0` at the fixed point. Compute that conditional expectation. The key algebraic move: `μ(a|x) min(ρ̄, π(a|x)/μ(a|x)) = min(ρ̄ μ(a|x), π(a|x))`. So
`E_μ[ρ_t(r + γ E_{x'} V(x') − V(x_t)) | x_t] = Σ_a min(ρ̄ μ(a|x_t), π(a|x_t)) [ r(x_t,a) + γ Σ_y p(y|x_t,a)V(y) − V(x_t) ]`.
Now divide and multiply by the normaliser `Σ_b min(ρ̄ μ(b|x_t), π(b|x_t))`. Define
`π_ρ̄(a|x) = min(ρ̄ μ(a|x), π(a|x)) / Σ_b min(ρ̄ μ(b|x), π(b|x))`.
Then the bracketed sum becomes `[ Σ_a π_ρ̄(a|x_t)( r + γ Σ_y p V(y) − V(x_t) ) ] · Σ_b min(ρ̄ μ(b|x_t), π(b|x_t))`. The first factor is the Bellman residual for the policy π_ρ̄ — it's zero exactly when V = V^{π_ρ̄}, the value function of π_ρ̄. So the fixed point is `V^{π_ρ̄}`.

Stare at π_ρ̄. It's a normalised pointwise minimum of `ρ̄ μ` and `π`. If `ρ̄ = ∞` the min is always π, so `π_ρ̄ = π` and the fixed point is `V^π` — the value of the actual target policy, no bias. If ρ̄ is finite, the min caps how far π_ρ̄ can move away from μ at any action where π wants much more mass than `ρ̄ μ` allows; π_ρ̄ is a policy *between* μ and π. As `ρ̄ → 0`, `ρ̄ μ` dominates, the min is `ρ̄ μ` everywhere, normalisation gives back μ, and the fixed point is `V^μ`. So ρ̄ slides the *target of evaluation* continuously from μ (ρ̄→0) to π (ρ̄→∞). The bigger ρ̄, the smaller the off-policy bias.

And here's the thing that just fell out and that I should hold onto: the fixed point depends on ρ̄ *only*. Nowhere in identifying `V^{π_ρ̄}` did c̄ appear — c̄ only entered η, the contraction modulus. So the two knobs really are doing two different jobs, exactly as I split them structurally. ρ̄ — the clip on the residual ratio — picks *which* value function I converge to (the bias, the policy π_ρ̄ being evaluated). c̄ — the clip on the trace product — affects only η, i.e. *how fast* I converge, and it's the knob that controls variance, because the trace is the product that explodes. And reassuringly, for *any* level of variance reduction c̄ I choose, I land on the *same* fixed point. So I can crank c̄ down hard to kill variance without changing what I'm solving for. ρ̄ for bias, c̄ for variance and speed. I get to tune them independently.

Let me also note the variance asymmetry I flagged earlier, now that I see the structure. The ρ_t are not multiplied across time — each sits alone in its own δ_t — so even though raising ρ̄ raises variance, that variance doesn't compound with the horizon. The c_i *are* multiplied, `c_0…c_{t-1}`, so their variance is the dangerous compounding kind, which is exactly why the trace is where truncation is non-negotiable and why c̄ is the real variance lever. Different mechanisms, different roles.

In the tabular case I can say more than "contraction": a stochastic-approximation argument finishes it. If I do online updates `V_{k+1}(x_s) = V_k(x_s) + α_k(x_s) Σ_{t≥s} γ^{t-s}(c_s…c_{t-1})ρ_t(r_t + γV_k(x_{t+1}) − V_k(x_t))`, with every state visited infinitely often and Robbins-Monro stepsizes (`Σ_k α_k = ∞`, `Σ_k α_k^2 < ∞`), then since this is stochastic approximation toward the fixed point of a contraction, `V_k → V^{π_ρ̄}` almost surely. Standard once I have the contraction.

For computing the target in practice I don't want to evaluate that double sum directly. Unfold the definition from the back. Write `v_s − V(x_s) = Σ_{t≥s} γ^{t-s}(c_s…c_{t-1})δ_t V`. Peel off the t=s term, which is just `δ_s V`, and factor `γ c_s` out of everything else: the remaining sum is `γ c_s Σ_{t≥s+1} γ^{t-(s+1)}(c_{s+1}…c_{t-1})δ_t V = γ c_s (v_{s+1} − V(x_{s+1}))`. So
`v_s = V(x_s) + δ_s V + γ c_s ( v_{s+1} − V(x_{s+1}) )`.
A backward recursion: start from the end of the trajectory with the bootstrap value, scan backward accumulating `acc ← δ_t + γ c_t · acc`, then add V(x_s). One pass, cheap, vectorisable over the batch.

Now the policy. I have a value target; I need a policy gradient that uses the same off-policy data. On-policy, `∇V^μ(x_0) = E_μ[ Σ_s γ^s ∇log μ(a_s|x_s) Q^μ(x_s,a_s) ]`, implemented as ascending `∇log μ(a_s|x_s) q_s` with q_s an estimate of Q. But my data is from μ and I'm trying to improve the evaluated policy, which — from the value analysis — is π_ρ̄, not π and not μ. So importance-sample between π_ρ̄ and μ: ascend
`E_{a_s∼μ}[ (π_ρ̄(a_s|x_s)/μ(a_s|x_s)) ∇log π_ρ̄(a_s|x_s) q_s | x_s ]`,
with `q_s` an estimate of `Q^{π_ρ̄}(x_s,a_s)`. The leading IS ratio `π_ρ̄/μ` — recall `π_ρ̄ ∝ min(ρ̄ μ, π)`, so `π_ρ̄/μ ∝ min(ρ̄, π/μ)` — is precisely the truncated ratio ρ_s up to the (state-dependent) normaliser, so in practice the policy-gradient weight on the score is just `ρ_s`, the same truncated residual ratio I already compute. The truncation on ρ_s here also caps the variance of the policy-gradient IS weight, the same way ρ̄ does for the value update.

What do I use for q_s? The tempting choice is `q_s = v_s`, the value target I just built. But check whether it's an unbiased estimate of `Q^{π_ρ̄}(x_s,a_s)` even in the best case where the value function is perfect, V = V^{π_ρ̄}. Unfold v_s at a perfect V:
`E[v_s | x_s, a_s] = V^{π_ρ̄}(x_s) + ρ_s(r_s + γE[V^{π_ρ̄}(x_{s+1})] − V^{π_ρ̄}(x_s)) + γ c_s E[δ_{s+1}V^{π_ρ̄} + …]`.
At the true value the expected future TDs vanish (that's what V^{π_ρ̄} being the fixed point *means*), so the tail dies and
`E[v_s|x_s,a_s] = V^{π_ρ̄}(x_s) + ρ_s(Q^{π_ρ̄}(x_s,a_s) − V^{π_ρ̄}(x_s)) = (1 − ρ_s)V^{π_ρ̄}(x_s) + ρ_s Q^{π_ρ̄}(x_s,a_s)`,
using `r_s + γE[V^{π_ρ̄}(x_{s+1})] = Q^{π_ρ̄}(x_s,a_s)`. That's a *blend* of V and Q, equal to Q only when ρ_s = 1 or V = Q. So v_s is a biased Q-estimate. The fix: use `q_s = r_s + γ v_{s+1}` instead. At a perfect value,
`E[q_s|x_s,a_s] = r_s + γ E[V^{π_ρ̄}(x_{s+1}) + δ_{s+1}V^{π_ρ̄} + γc_{s+1}δ_{s+2}V^{π_ρ̄} + …]`,
and again every expected δ vanishes at the true value, so `E[q_s|x_s,a_s] = r_s + γE[V^{π_ρ̄}(x_{s+1})] = Q^{π_ρ̄}(x_s,a_s)`, exactly unbiased. So `q_s = r_s + γ v_{s+1}` is the right Q-target. And I subtract the baseline V(x_s) to cut variance — the advantage is `q_s − V(x_s) = r_s + γ v_{s+1} − V(x_s)`, which the baseline doesn't bias because in expectation `E_{a∼μ}[ρ_s ∇log π · V(x_s)]` factors V(x_s) out of the action expectation.

So the policy-gradient direction is `ρ_s ∇_ω log π_ω(a_s|x_s) (r_s + γ v_{s+1} − V_θ(x_s))`. One more piece: like A3C, add an entropy bonus `−∇_ω Σ_a π_ω(a|x_s) log π_ω(a|x_s)` to keep the policy from collapsing prematurely onto a narrow set of actions before it's explored. And the value update is gradient descent on the l2 loss to the target, direction `(v_s − V_θ(x_s)) ∇_θ V_θ(x_s)`. The full learner update is the weighted sum of the three: policy gradient, baseline regression, entropy bonus. One small bias caveat I'll keep in mind: this gradient improves π_ρ̄, not π, but if ρ̄ is large enough the gap `V^{π_ρ̄} − V^π` is small and q_s is a good estimate of `Q^π` too, so improving π_ρ̄ effectively improves π.

Let me write the core off-policy correction the way the learner will actually run it — work in log-space for the ratios for numerical stability, take ρ̄ = c̄ = 1 (clip everything at 1, the simplest stable choice), append the bootstrap value at the tail for the n-step truncation, and scan the recursion backward.

```python
import collections
import tensorflow as tf

VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')

def log_probs_from_logits_and_actions(policy_logits, actions):
    # log π(a_t|x_t) for the taken actions; shape [T, B].
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=policy_logits, labels=actions)

def from_importance_weights(log_rhos, discounts, rewards, values,
                            bootstrap_value,
                            clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
    # log_rhos[t] = log( π(a_t|x_t) / μ(a_t|x_t) );  discounts[t] = γ·(not done).
    rhos = tf.exp(log_rhos)
    clipped_rhos = tf.minimum(clip_rho_threshold, rhos)   # ρ_t = min(ρ̄, π/μ): fixed point
    cs = tf.minimum(1.0, rhos)                            # c_t = min(c̄, π/μ): the trace (variance)

    # V(x_{t+1}), with the bootstrap value at the tail for the n-step truncation.
    values_t_plus_1 = tf.concat([values[1:],
                                 tf.expand_dims(bootstrap_value, 0)], axis=0)
    # δ_t V = ρ_t ( r_t + γ V(x_{t+1}) − V(x_t) ).
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # Backward recursion: v_s − V(x_s) = δ_s + γ c_s ( v_{s+1} − V(x_{s+1}) ).
    def scanfunc(acc, seq):
        discount_t, c_t, delta_t = seq
        return delta_t + discount_t * c_t * acc
    vs_minus_v_xs = tf.scan(
        fn=scanfunc, elems=(discounts, cs, deltas),
        initializer=tf.zeros_like(bootstrap_value),
        parallel_iterations=1, back_prop=False, reverse=True)
    vs = vs_minus_v_xs + values                           # v_s

    # PG advantage q_s − V(x_s) = ρ_s ( r_s + γ v_{s+1} − V(x_s) ); note q_s = r_s + γ v_{s+1}.
    vs_t_plus_1 = tf.concat([vs[1:],
                             tf.expand_dims(bootstrap_value, 0)], axis=0)
    clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    # Stop gradients: these are targets, not differentiable through the rollout.
    return VTraceReturns(vs=tf.stop_gradient(vs),
                         pg_advantages=tf.stop_gradient(pg_advantages))

def from_logits(behaviour_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, bootstrap_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
    target_log_probs = log_probs_from_logits_and_actions(target_policy_logits, actions)
    behaviour_log_probs = log_probs_from_logits_and_actions(behaviour_policy_logits, actions)
    log_rhos = target_log_probs - behaviour_log_probs     # log π(a) − log μ(a)
    return from_importance_weights(log_rhos, discounts, rewards, values,
                                   bootstrap_value, clip_rho_threshold, clip_pg_rho_threshold)

def off_policy_targets(behaviour_policy_logits, target_policy_logits, actions,
                       discounts, rewards, values, bootstrap_value):
    returns = from_logits(behaviour_policy_logits, target_policy_logits, actions,
                          discounts, rewards, values, bootstrap_value)
    return returns.vs, returns.pg_advantages
```

And the learner loss — the three weighted terms. The cross-entropy carries the `−log π`, so multiplying it by the corrected advantage and summing gives the policy-gradient ascent on `ρ_s ∇log π · (q_s − V)`; the baseline term regresses V toward v_s; the entropy term keeps the policy from collapsing:

```python
def baseline_loss(vs, values):
    return 0.5 * tf.reduce_sum(tf.square(vs - values))    # (v_s − V(x_s))^2

def entropy_loss(logits):
    policy, log_policy = tf.nn.softmax(logits), tf.nn.log_softmax(logits)
    return -tf.reduce_sum(tf.reduce_sum(-policy * log_policy, axis=-1))

def policy_gradient_loss(logits, actions, advantages):
    neg_log_pi = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
    return tf.reduce_sum(neg_log_pi * tf.stop_gradient(advantages))  # advantages = ρ_s (r_s + γ v_{s+1} − V)

def build_learner(agent, trajectories, baseline_cost=0.5, entropy_cost=0.01):
    out = agent.unroll(trajectories.actions, trajectories.env_outputs,
                       trajectories.initial_state)        # recompute target π's logits & V
    bootstrap_value = out.value[-1]                       # V(x_T) bootstrap for the tail
    value_targets, pg_advantages = off_policy_targets(
        behaviour_policy_logits=trajectories.behaviour_policy_logits,
        target_policy_logits=out.policy_logits, actions=trajectories.actions,
        discounts=trajectories.discounts, rewards=trajectories.rewards,
        values=out.value, bootstrap_value=bootstrap_value)
    loss  = policy_gradient_loss(out.policy_logits, trajectories.actions, pg_advantages)
    loss += baseline_cost * baseline_loss(value_targets, out.value)
    loss += entropy_cost * entropy_loss(out.policy_logits)
    optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.99, momentum=0.0, epsilon=0.01)
    return optimizer.minimize(loss)
```

So the causal chain: to get throughput I decoupled many cheap asynchronous actors from one GPU learner and had actors ship trajectories instead of gradients — which made the learner's data off-policy by a growing policy-lag. Plain importance sampling on n-step returns would correct it but its trace product explodes with the lag, and Retrace targets Q not the V my actor-critic learns and doesn't collapse to the on-policy update. So I built a state-value off-policy target `v_s = V(x_s) + Σ γ^{t-s}(∏ c_i) δ_t V` with `δ_t V = ρ_t(r_t + γV(x_{t+1}) − V(x_t))`, splitting the correction into a per-step residual ratio ρ_t (clipped at ρ̄) and a back-propagation trace ∏ c_i (clipped at c̄): the operator it defines is a contraction whose fixed point `V^{π_ρ̄}` is set by ρ̄ alone — sliding the evaluated policy from μ to π and thus controlling bias — while c̄ truncates the variance-exploding trace product and only sets the contraction speed, leaving the fixed point untouched; the policy gradient reuses ρ_s on the advantage `r_s + γ v_{s+1} − V(x_s)` (with `q_s = r_s + γ v_{s+1}` the one Q-estimate that stays unbiased under a perfect value); and with no lag the whole thing collapses to the ordinary on-policy n-step actor-critic.
