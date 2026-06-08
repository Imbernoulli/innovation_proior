Let me start from where it actually hurts. I have an episodic MDP — H steps per episode, K episodes, an agent that sees a state, picks an action, gets a reward in [0,1], and moves to a next state drawn from an unknown kernel P_h(·|x,a). I want low regret: over K episodes, the cumulative gap between the optimal value V*₁(x₁) and what my policy actually earns. And the whole reason I'm stuck is the state space. In the problems I care about — board positions, game frames, robot configurations — the number of states S is astronomically large or just infinite. I will never see the same state twice; the neighborhood of most states is never visited even once during all my episodes.

So the tabular toolbox, which is otherwise excellent, is useless to me here, and I should be precise about *why* before I reach for anything else. The best tabular algorithms — optimistic model-based ones that build confidence sets on the empirical transition matrix and plan optimistically, and the model-free ones that bonus the Q-table — get regret like √(H²SAT) or √(H³SAT), essentially matching the minimax lower bound Ω(√(H²SAT)). That lower bound is the thing I keep circling back to. It says: information-theoretically, you cannot do better than √S in the tabular world. There is no clever algorithm that escapes it. If S is exponential, every one of these bounds is vacuous. The √S isn't a weakness of a particular algorithm; it's a wall that the *generic* problem puts up. The only way through a lower bound is to change the problem — to assume structure that ties the states together, so that experience at one state teaches me about others.

What structure? I want a guarantee that depends not on S but on some intrinsic complexity d of a function class — the dimension of a feature map φ(x,a) ∈ ℝ^d that I'm given. If I had that, then the √S in the regret should become something like √d, and d can be tiny even when S is infinite. So the target is: regret polynomial in d and H and √T, and *completely independent of S and A*. Whether such a thing exists is genuinely open, even in the most basic linear case. Let me see if I can build it.

Where do I already know how to get an S-free, structure-aware bound? Linear bandits. There, the reward of pulling arm x is ⟨x, θ*⟩ + noise, θ* unknown in ℝ^d, and the optimism-in-the-face-of-uncertainty algorithm — OFUL — gets regret Õ(d√T) with *no dependence on the number of arms*, however many there are, even infinitely many. That's exactly the shape I want. The mechanism is beautiful and I should hold it in my hands: maintain a ridge estimate θ̂ from past (x,y) pairs with Gram matrix Λ = Σ xx⊤ + λI; then θ* lives, with high probability, in a confidence ellipsoid around θ̂ whose radius in the Λ-norm is about √(d log(...)); to act, pick the arm whose *optimistic* reward — best over the ellipsoid — is largest. Concretely that's ridge prediction ⟨x, θ̂⟩ plus a bonus β·√(x⊤Λ⁻¹x). The quantity √(x⊤Λ⁻¹x) = ‖x‖_{Λ⁻¹} is the right notion of uncertainty: it's large precisely in directions where I've collected few samples, and (x⊤Λ⁻¹x)⁻¹ acts like the effective number of samples seen along x. Add that bonus and the agent is automatically pulled toward under-explored directions. The reason the total regret is only d√T and not worse is a clean potential argument I'll come back to: each informative pull grows det(Λ), det can only grow by about d log T total, so the cumulative uncertainty Σ x⊤Λ⁻¹x is bounded by ≈ 2d log T. The learner runs out of directions.

So I have a tool that does exactly the S-free thing I want — but for bandits. A bandit is an MDP with H=1: one decision, then the world resets, no state transition, no future. My problem has a horizon and dynamics. Can I just run OFUL at each step? Treat each h as its own linear bandit on the immediate reward?

Let me try it and watch it break. The trouble is credit assignment across the horizon. In a bandit, the value of an action is its immediate reward — fully observed, fully local. In an MDP, the value of an action at step h is the immediate reward *plus* the expected value of where it lands me, V_{h+1}, which itself depends on actions h+1, …, H. If I'm optimistic at every step independently, the optimism doesn't just add up — it compounds through the recursion. An optimistic estimate at step H feeds into the target at step H−1, which inflates the optimism there, which feeds into H−2… A back-of-the-envelope says a naive per-step bandit optimism gives regret that blows up *exponentially in H*. That's the real difference between linear bandits and linear RL, and it's the difference I have to handle: the temporal structure is exactly what makes exploration in RL hard, and it's exactly what a flat bandit reduction ignores.

So I can't just bolt OFUL onto each step. I need optimism that propagates *correctly* through the Bellman recursion. And for OFUL's machinery to even apply, I need two things to be true that aren't true for a generic MDP: (1) the action-value function has to be linear in φ, so that the ridge/ellipsoid apparatus has something to estimate; and (2) the Bellman backup has to *be* a linear regression, so that one step of value iteration is one ridge fit. Let me figure out what assumption on the MDP buys me both.

For the Bellman expectation to stay linear after I feed in an arbitrary next-step value function, the transition itself has to be linear in φ, and the reward has to share the same coordinates. So I posit an unknown vector θ_h with r_h(x,a) = ⟨φ(x,a), θ_h⟩, and — the bold part — d unknown signed measures μ_h = (μ_h^{(1)}, …, μ_h^{(d)}) over the state space with

  P_h(·|x,a) = ⟨φ(x,a), μ_h(·)⟩.

I want to be careful what I'm assuming. I am *not* assuming the policy is linear — that's a different and much-studied assumption. I'm assuming the dynamics-generating *model* is linear in the features: a statistical modeling assumption about how the data is generated, the way one posits a linear regression model and then studies estimators of it. The tabular case is a special instance — take d = SA, let φ(x,a) = e_{(x,a)} the canonical basis vector, and set e_{(x,a)}⊤μ_h = P_h(·|x,a), e_{(x,a)}⊤θ_h = r_h(x,a). So I'm not throwing away the tabular world; I'm embedding it.

And here is the subtle, crucial thing about this μ_h being a *measure*, not a matrix. Even though P_h is "linear in φ," the model is enormous: μ_h is an unknown signed measure over a possibly-infinite state space, so it has infinitely many degrees of freedom. This is *not* the same as the linear-quadratic regulator or a low-rank model where the dynamics are pinned down by a finite matrix that I just have to estimate. Here I have a finite-dimensional handle φ but an infinite-dimensional unknown μ_h behind it. That tension — finite handle, infinite unknown — is going to matter enormously later, both for what's hard and for what saves me.

Now check claim (1): does this make Q linear? Take any policy π and write out the Bellman equation Q^π_h = r_h + P_h V^π_{h+1}. Plug in the linear forms:

  Q^π_h(x,a) = ⟨φ(x,a), θ_h⟩ + (P_h V^π_{h+1})(x,a)
            = ⟨φ(x,a), θ_h⟩ + ∫ V^π_{h+1}(x') ⟨φ(x,a), dμ_h(x')⟩
            = ⟨φ(x,a), θ_h + ∫ V^π_{h+1}(x') dμ_h(x')⟩.

So Q^π_h(x,a) = ⟨φ(x,a), w^π_h⟩ with w^π_h = θ_h + ∫ V^π_{h+1}(x') dμ_h(x'). The expectation under the linear kernel is itself linear in φ, because the inner product pulls straight through the integral. And this holds for *every* policy π, not just the optimal one — every action-value function in this model is linear in φ. That's the payoff of assuming the *kernel* is linear rather than just the optimal value: linearity is closed under the Bellman backup for all π. So when I design an algorithm, I only ever need to maintain a linear Q. Good. Claim (1) holds.

Let me also bound that weight, because I'll need it. With rewards in [0,1], every value function is in [0,H], so V^π_{h+1}(x') ≤ H. Under the normalization ‖θ_h‖ ≤ √d and ‖μ_h(S)‖ ≤ √d, I get ‖θ_h‖ ≤ √d and ‖∫ V^π_{h+1} dμ_h‖ ≤ H·√d, so ‖w^π_h‖ ≤ 2H√d. The true weights live in a ball of radius 2H√d.

The same structure has to turn the Bellman backup into a regression. Value iteration wants Q_h ← r_h + P_h max_{a'} Q_{h+1}(·,a'). The target r_h(x,a) + (P_h V_{h+1})(x,a) is, by the computation above, exactly ⟨φ(x,a), θ_h + ∫V_{h+1}dμ_h⟩ — linear in φ with unknown coefficient. And I have data: from past episodes, pairs (φ(x_h^τ, a_h^τ), y_h^τ) where the response is y_h^τ = r_h(x_h^τ,a_h^τ) + max_a Q_{h+1}(x_{h+1}^τ, a) = r_h^τ + V_{h+1}(x_{h+1}^τ). The next-state value V_{h+1}(x_{h+1}^τ) is a noisy, single-sample realization of the expectation (P_h V_{h+1})(x_h^τ,a_h^τ). So fitting w_h to these targets by least squares is precisely an empirical version of the Bellman backup. This is least-squares value iteration — LSVI, the classical algorithm for the linear setting — and now I see *why* it's the natural object: the linear-MDP assumption is exactly the condition under which one step of value iteration equals one linear regression.

Plain least squares is too brittle here: in early episodes I have fewer than d samples and Σφφ⊤ is singular, and even later I need the fitted weights to stay in a bounded set so the value functions have finite capacity. The small λI term buys both invertibility and a radius I can cover, so the fit is ridge:

  w_h ← argmin_w Σ_{τ=1}^{k−1} [ r_h^τ + max_a Q_{h+1}(x_{h+1}^τ, a) − w⊤φ(x_h^τ,a_h^τ) ]² + λ‖w‖²,

with closed form w_h = Λ_h⁻¹ Σ_τ φ_h^τ [r_h^τ + max_a Q_{h+1}(x_{h+1}^τ,a)], where Λ_h = Σ_τ φ_h^τ (φ_h^τ)⊤ + λI. Let me bound ‖w_h‖ now: for any unit v, |v⊤w_h| = |v⊤Λ_h⁻¹ Σ_τ φ_h^τ y_h^τ| ≤ Σ_τ |v⊤Λ_h⁻¹φ_h^τ|·2H, since each target r+V is in [0, 2H]. Cauchy-Schwarz gives ≤ 2H √(Σ_τ v⊤Λ_h⁻¹v) √(Σ_τ (φ_h^τ)⊤Λ_h⁻¹φ_h^τ). The second sum is ≤ d (trace identity: Σ_τ φ⊤Λ⁻¹φ = tr(Λ⁻¹ Σφφ⊤) = Σ_j λ_j/(λ_j+λ) ≤ d), and the first is at most k‖v‖²/λ. So ‖w_h‖ ≤ 2H√(dk/λ). Now I have a radius for the *estimated* weights too: order H√(dk).

But LSVI by itself is inert. Run pure least-squares value iteration and the agent never seeks out the directions of φ-space it knows nothing about; it just exploits its current fit, and it can get stuck having never explored. I need to inject exploration — and I know the shape it should take, because the bandit case told me. The uncertainty of my ridge estimate at a query (x,a) is ‖φ(x,a)‖_{Λ_h⁻¹} = √(φ⊤Λ_h⁻¹φ). So add the bonus, and clip at H since values can't exceed H:

  Q_h(x,a) ← min{ w_h⊤φ(x,a) + β·√(φ(x,a)⊤Λ_h⁻¹φ(x,a)), H }.

Act greedily w.r.t. this. The bonus β√(φ⊤Λ⁻¹φ) is the elliptical confidence width, the same self-normalized form OFUL uses; β is a scalar I'll have to set. Intuitively, with m := (φ⊤Λ⁻¹φ)⁻¹ the effective sample count along φ, the bonus is β/√m — uncertainty shrinking as I gather data along φ. I want to choose β so that this is a genuine *upper* confidence bound: with high probability Q_h(x,a) ≥ Q*_h(x,a) everywhere. If I can guarantee that, optimism does the exploration for me.

So the whole algorithm, which I'll call least-squares value iteration with UCB, is: each episode, sweep h from H down to 1 building (Λ_h, w_h, Q_h); then sweep h from 1 to H executing the greedy policy and collecting the new trajectory. For k=1 the sums are empty, Λ_h = λI, w_h = 0.

Choosing β so Q_h ≥ Q*_h, and then bounding the regret, both come down to controlling how well the ridge fit ⟨φ, w_h⟩ approximates the Bellman target. Let me put one fixed policy π in play and expand w_h − w^π_h. Using w^π_h = θ_h + ∫V^π_{h+1}dμ_h and the fact that the target's expectation is r_h + P_h V^π_{h+1},

  w_h − w^π_h = Λ_h⁻¹ Σ_τ φ_h^τ [ r_h^τ + V_{h+1}(x_{h+1}^τ) ] − w^π_h.

Write the target as the fixed-policy Bellman part plus the learned-value noise and recursion:

  r_h^τ + V_{h+1}(x_{h+1}^τ)
    = [r_h^τ + (P_h V^π_{h+1})(x_h^τ,a_h^τ)]
      + [V_{h+1}(x_{h+1}^τ) − (P_h V_{h+1})(x_h^τ,a_h^τ)]
      + (P_h(V_{h+1} − V^π_{h+1}))(x_h^τ,a_h^τ).

Now Λ_h⁻¹(Σφφ⊤) = I − λΛ_h⁻¹, so the difference splits into three pieces:

  w_h − w^π_h = −λ Λ_h⁻¹ w^π_h  (call it q₁, the regularization bias)
             + Λ_h⁻¹ Σ_τ φ_h^τ [ V_{h+1}(x_{h+1}^τ) − (P_h V_{h+1})(x_h^τ,a_h^τ) ]  (q₂, the stochastic noise)
             + Λ_h⁻¹ Σ_τ φ_h^τ (P_h (V_{h+1} − V^π_{h+1}))(x_h^τ,a_h^τ).  (q₃, the recursion)

Take inner products with φ(x,a) and bound each. For q₁: |⟨φ, q₁⟩| = λ|⟨φ, Λ_h⁻¹w^π_h⟩|. Cauchy-Schwarz gives λ√(φ⊤Λ_h⁻¹φ)√((w^π_h)⊤Λ_h⁻¹w^π_h), and Λ_h ⪰ λI turns the second square root into at most ‖w^π_h‖/√λ. With ‖w^π_h‖ ≤ 2H√d and λ=1, that's ≤ 2H√d · √(φ⊤Λ_h⁻¹φ). A clean O(H√d)·‖φ‖_{Λ⁻¹} term.

For q₃, here's where the linear-transition assumption pays off again. Each P_h(V_{h+1}−V^π_{h+1})(x_h^τ,a_h^τ) = (φ_h^τ)⊤ ∫(V_{h+1}−V^π_{h+1})dμ_h. So

  ⟨φ, q₃⟩ = ⟨φ, Λ_h⁻¹ Σ_τ φ_h^τ (φ_h^τ)⊤ ∫(V_{h+1}−V^π_{h+1})dμ_h⟩.

But Λ_h⁻¹ Σ_τ φφ⊤ = I − λΛ_h⁻¹, so this equals ⟨φ, ∫(V_{h+1}−V^π_{h+1})dμ_h⟩ − λ⟨φ, Λ_h⁻¹ ∫(...)dμ_h⟩. The first term is exactly (P_h(V_{h+1}−V^π_{h+1}))(x,a) — the recursion term I *want* — and the second is another O(H√d)·‖φ‖_{Λ⁻¹} bias (same Cauchy-Schwarz, ‖∫(V−V^π)dμ_h‖ ≤ 2H√d). So q₁ and q₃ together give me the Bellman recursion P_h(V_{h+1}−V^π_{h+1}) plus a controlled error of size O(H√d)·‖φ‖_{Λ⁻¹}.

Everything now hinges on q₂, the stochastic term, the difference between the single-sample next-state value and its expectation, projected through Λ_h⁻¹. This is where the self-normalized concentration lives. Σ_τ φ_h^τ [V_{h+1}(x_{h+1}^τ) − P_h V_{h+1}(x_h^τ,a_h^τ)] — if V_{h+1} were a *fixed* function, independent of the samples, then each bracket would be a zero-mean random variable in [−H,H] (x_{h+1}^τ is a genuine sample from P_h(·|x_h^τ,a_h^τ)), and I'd be in exactly the setting of the self-normalized tail inequality for vector-valued martingales: ‖Σ_τ φ_h^τ ε_τ‖²_{Λ_h⁻¹} ≤ 2H² log(det(Λ_h)^{1/2}det(λI)^{−1/2}/δ), giving a Λ_h⁻¹-norm bound on the q₂ numerator of order √(d log(T/δ))·H. Multiply by ‖φ‖_{Λ⁻¹} and I'd be done, with a width like H√(d·log).

But — stop. V_{h+1} is *not* fixed and independent of the samples. It's the value function my own algorithm built, by running LSVI at steps h+1, …, H on the *same* data {x_{h+1}^τ}. The function I'm taking expectations of is statistically entangled with the very samples I'm concentrating. The brackets are not martingale differences with respect to a filtration that V_{h+1} is measurable against, so the self-normalized bound does not apply. This is the wall, and it's a real one — it's the heart of why RL with function approximation is harder than bandits. In a bandit there's no V_{h+1} to learn; here the regression target is itself a learned, data-dependent object.

How do I get around a data-dependent V? I make the concentration *uniform* over a whole class of value functions, one large enough to contain whatever V^k_{h+1} my algorithm could produce, but small enough that a union bound (a covering argument) is cheap. What do my value functions actually look like? Every one is of the form

  V(·) = min{ max_a [ φ(·,a)⊤w + β√(φ(·,a)⊤Λ⁻¹φ(·,a)) ], H },

for some w, β, Λ. So define 𝒱 to be exactly this parametric class, with the parameters ranging over bounded sets (‖w‖ ≤ L, β bounded, λ_min(Λ) ≥ λ — all of which I've already established hold for the algorithm's actual iterates: ‖w_h^k‖ ≤ 2H√(dk/λ), Λ_h^k ⪰ λI). I guarantee my algorithm only ever uses V ∈ 𝒱, and I concentrate uniformly over 𝒱. The cost is log N_ε, the log covering number of 𝒱 — and *now* I see the second reason the ridge regularization mattered: it's what bounds L and keeps λ_min(Λ) ≥ λ, which is what makes 𝒱 a bounded class with a finite, small covering number. Without the λ‖w‖², w could be unbounded and 𝒱 would have unbounded capacity.

Let me compute that covering number, because its size determines my final d-power and I have to get it right. A function in 𝒱 is determined by (w, β, Λ). Reparametrize: let A = β²Λ⁻¹, so the bonus β√(φ⊤Λ⁻¹φ) = √(φ⊤Aφ), and V(·) = min{max_a [w⊤φ(·,a) + √(φ⊤Aφ)], H}, with ‖w‖ ≤ L and ‖A‖ ≤ B²λ⁻¹. For two such functions with parameters (w₁,A₁) and (w₂,A₂), since min{·,H} and max_a are contractions (1-Lipschitz), the sup-distance is at most

  sup_{‖φ‖≤1} |(w₁−w₂)⊤φ| + sup_{‖φ‖≤1} √(|φ⊤(A₁−A₂)φ|) = ‖w₁−w₂‖ + √(‖A₁−A₂‖) ≤ ‖w₁−w₂‖ + √(‖A₁−A₂‖_F),

using |√x − √y| ≤ √(|x−y|). So an ε-cover of 𝒱 comes from an (ε/2)-cover of the w-ball (in ℝ^d) and an (ε²/4)-cover of the A-ball (in ℝ^{d×d}, Frobenius). The covering number of a Euclidean ball of radius R is (1 + 2R/ε)^{dim}. The w-ball has dimension d → log cover ≤ d log(1 + 4L/ε). The A-ball has dimension d² → log cover ≤ d² log(1 + 8√d B²/(λε²)). Hence

  log N_ε ≤ d log(1 + 4L/ε) + d² log(1 + 8√d B²/(λε²)).

There it is — the dominant term is the d², from the bonus matrix A ∈ ℝ^{d×d}. The covering number of the *quadratic* part of the value function (the elliptical bonus, which is a quadratic form in φ) carries d² of capacity, and that's what I pay for handling the data-dependence of V.

Feed this into the self-normalized-plus-covering bound: with V ∈ 𝒱, ‖V‖_∞ ≤ H, the uniform version of the concentration gives

  ‖Σ_τ φ_h^τ [V_{h+1}(x_{h+1}^τ) − P_h V_{h+1}(x_h^τ,a_h^τ)]‖²_{Λ_h⁻¹}
     ≤ 4H² [ (d/2)log((k+λ)/λ) + log(N_ε/δ) ] + (8k²ε²/λ),

and with log N_ε ~ d² and the choice ε = dH/k (so the discretization slack 8k²ε²/λ ~ d²H² is absorbed), the right side is O(d²H² · log(dT/δ)). Take the square root: ‖q₂-numerator‖_{Λ_h⁻¹} ≤ O(dH√(log(dT/δ))). So ⟨φ, q₂⟩ ≤ O(dH√ι)·√(φ⊤Λ_h⁻¹φ), with ι = log(2dT/p). The d here, the full d (not √d), comes from the d² inside the log — √(d²H²) = dH. That's strictly worse than the √d I'd have gotten from a fixed-V bandit-style argument, and the gap is exactly the price of uniform concentration over the value class.

Combine the three terms. q₁ and q₃ each contributed O(H√d)·‖φ‖_{Λ⁻¹}, which is dominated by q₂'s O(dH√ι)·‖φ‖_{Λ⁻¹}. So, on the high-probability event,

  | ⟨φ(x,a), w_h^k⟩ − Q^π_h(x,a) − (P_h(V^k_{h+1} − V^π_{h+1}))(x,a) | ≤ c'·dH√ι · √(φ(x,a)⊤(Λ_h^k)⁻¹φ(x,a))

for an absolute constant c'. This is the key relation. It says my ridge estimate ⟨φ,w_h^k⟩ equals the Bellman recursion term P_h(V^k_{h+1}−V^π_{h+1}) plus the true reward/value, up to a slack of size dH√ι·‖φ‖_{Λ⁻¹}. So I should set β = c·dH√ι with c ≥ c' — and there's a small fixed-point to check, because the failure-probability bookkeeping makes the constant inside the log depend on c itself: I need c'√(ι + log(c+1)) ≤ c√ι for all ι ≥ log 2. Since c' is absolute and ι ≥ log 2, I can pick c large enough that c'√(log 2 + log(c+1)) ≤ c√(log 2), and then it holds for all larger ι. Fine — β = Θ(dH√ι).

With β set this way, optimism falls out by backward induction. Claim: Q_h^k(x,a) ≥ Q*_h(x,a) for all (x,a,h,k). Base case h = H: V_{H+1} ≡ 0, so the key relation with π = π* and the empty recursion term gives |⟨φ,w_H^k⟩ − Q*_H(x,a)| ≤ β‖φ‖_{Λ⁻¹}, hence Q*_H ≤ min{⟨φ,w_H^k⟩ + β‖φ‖_{Λ⁻¹}, H} = Q_H^k. Inductive step: suppose Q*_{h+1} ≤ Q^k_{h+1}, so V*_{h+1} ≤ V^k_{h+1} pointwise, hence P_h(V^k_{h+1} − V*_{h+1}) ≥ 0. The key relation gives |⟨φ,w_h^k⟩ − Q*_h − P_h(V^k_{h+1}−V*_{h+1})| ≤ β‖φ‖_{Λ⁻¹}; since the recursion term is ≥ 0, Q*_h ≤ ⟨φ,w_h^k⟩ + β‖φ‖_{Λ⁻¹}, and capping at H is harmless because Q*_h ≤ H. So Q*_h ≤ Q_h^k. The optimism propagates *down* the horizon, correctly — the recursion term being nonnegative is what lets the induction go through, and it's exactly the thing a naive per-step bandit reduction would have mishandled.

Now the regret. Let δ_h^k = V_h^k(x_h^k) − V^{π_k}_h(x_h^k), the gap between my optimistic value and the value my greedy policy actually realizes, evaluated at the state I visit. The key relation, applied with π = π_k and turned into a statement about Q^k − Q^{π_k}, gives

  Q_h^k(x,a) − Q^{π_k}_h(x,a) ≤ (P_h(V^k_{h+1} − V^{π_k}_{h+1}))(x,a) + 2β√(φ⊤Λ_h⁻¹φ).

Evaluate at the visited (x_h^k, a_h^k); since π_k is greedy, δ_h^k = Q_h^k(x_h^k,a_h^k) − Q^{π_k}_h(x_h^k,a_h^k). The expectation P_h(V^k_{h+1}−V^{π_k}_{h+1})(x_h^k,a_h^k) equals E[δ_{h+1}^k | x_h^k,a_h^k] — the gap at the next step, in expectation. Write the actual next-step gap as δ_{h+1}^k and the deviation of the conditional expectation from the realized value as ζ_{h+1}^k := E[δ_{h+1}^k|x_h^k,a_h^k] − δ_{h+1}^k. Then

  δ_h^k ≤ δ_{h+1}^k + ζ_{h+1}^k + 2β√((φ_h^k)⊤Λ_h⁻¹φ_h^k).

Unrolling over h from 1 to H, and using optimism (V*₁ ≤ V₁^k, so V*₁ − V^{π_k}₁ ≤ δ₁^k),

  Regret(K) = Σ_k [V*₁(x₁^k) − V^{π_k}₁(x₁^k)] ≤ Σ_k δ₁^k ≤ Σ_{k,h} ζ_h^k + 2β Σ_{k,h} √((φ_h^k)⊤(Λ_h^k)⁻¹φ_h^k).

Two terms. The first, Σ ζ_h^k, is a sum of martingale differences: V_h^k is computed before the new observation x_h^k, so each ζ_h^k has conditional mean zero, and |ζ_h^k| ≤ 2H. By Azuma-Hoeffding, with probability 1−p/2, Σ_{k,h} ζ_h^k ≤ √(2TH²·log(2/p)) ≤ 2H√(Tι). That's the lower-order √T term, no d in it.

The second term is the cumulative exploration bonus, and *this* is where the elliptical potential lemma earns its keep — it's the same det-growth argument that gave OFUL its d√T, lifted into the MDP. Fix h. The sequence ‖φ_h^k‖_{(Λ_h^k)⁻¹}² is summed over k, and with λ_min(Λ_h^k) ≥ λ = 1 and ‖φ_h^k‖ ≤ 1 the potential lemma gives

  Σ_{k=1}^K (φ_h^k)⊤(Λ_h^k)⁻¹ φ_h^k ≤ 2 log[ det(Λ_h^{K+1})/det(Λ_h^1) ].

Now det(Λ_h^{K+1}) is at most (λ+K)^d (operator norm ‖Λ_h^{K+1}‖ ≤ λ + K since each rank-one update adds ‖φφ⊤‖ ≤ 1, and det ≤ ‖·‖^d), while det(Λ_h^1) = λ^d, so the log is ≤ 2d log((λ+K)/λ) ≤ 2dι. The learner runs out of directions: only d eigen-directions, each contributing log K to the determinant. Then by Cauchy-Schwarz over k,

  Σ_{k=1}^K √((φ_h^k)⊤(Λ_h^k)⁻¹φ_h^k) ≤ √K · √(Σ_k (φ_h^k)⊤(Λ_h^k)⁻¹φ_h^k) ≤ √K · √(2dι),

and summing over the H steps, Σ_{k,h} √(·) ≤ H√(2dKι). Multiply by 2β:

  Regret(K) ≤ 2H√(Tι) + 2β·H√(2dKι).

Plug in β = c·dH√ι:

  2β·H√(2dKι) = 2c·dH√ι · H√(2dKι) = O( dH · H · √(dK) · ι ) = O( d^{3/2} H² √K · ι ) = O( √(d³ H⁴ K · ι²) ).

And H⁴K = H³·(HK) = H³T, so this is O(√(d³ H³ T)·ι). Let me triple-check the powers, because this is the headline. β carries dH (the d from the d² covering, the H from the value scale). The bonus sum carries H√(dK) (one H from summing over steps, √d from the potential lemma, √K from Cauchy-Schwarz over episodes). Product: dH · H√(dK) = d^{3/2}H²√K. Square it: d³H⁴K = d³H³T. Square root: √(d³H³T). So

  Regret(K) = Õ(√(d³ H³ T)),

with probability 1−p, no dependence on S or A anywhere. The first term 2H√(Tι) is dominated. Done — and it's the shape I wanted at the very start: the √S of the tabular world is gone, replaced by d^{3/2}, the intrinsic dimension of the feature space.

Let me situate the powers against what I know. The tabular minimax rate is √(H²SAT); here SA → d (the intrinsic complexity), and there's an extra √(dH) relative to that minimax form. Part of that — the extra √H — is the standard loss from using a Hoeffding-type bonus instead of a sharper Bernstein-type one; the same √H gap shows up in the tabular Q-learning UCB analysis (Hoeffding gives √(H⁴SAT), Bernstein √(H³SAT)). And critically, the dependence on H is only polynomial — H^{3/2} — *not* exponential. Avoiding the exponential-in-H blowup of the naive bandit reduction was the whole game, and the recursion-aware optimism is what bought it.

One more thing I want to be honest with myself about, because it tells me what I've really proven. The transition model P_h(·|x,a) = ⟨φ, μ_h⟩, despite being "linear," has infinite degrees of freedom — μ_h is an unknown measure. With finitely many samples, *no* algorithm can establish that an estimated P̂_h is close to P_h in total variation; the model is just too big to learn. So I have *not* learned the model. What did I actually use? Only that my empirical Bellman operator P̂_h V ≈ P_h V for the specific value functions V in the small class 𝒱. Let me make that bridge explicit. Define P̂_h(·|x,a) = φ(x,a)⊤Λ_h⁻¹ Σ_τ φ_h^τ δ(·, x_{h+1}^τ), the empirical next-state measure, and an intermediate P̄_h(·|x,a) = φ(x,a)⊤Λ_h⁻¹ Σ_τ φ_h^τ P_h(·|x_h^τ,a_h^τ). Then P̂_h V ≈ P̄_h V is exactly the value-aware uniform concentration (q₂, the self-normalized bound over 𝒱), and the linear-transition step turns P̄_h into P_h plus the same ridge bias I already bounded: substituting P_h(·|x_h^τ,a_h^τ) = (φ_h^τ)⊤μ_h(·) gives φ(x,a)⊤[Λ_h⁻¹ Σ_τ φ_h^τ(φ_h^τ)⊤]μ_h = φ(x,a)⊤(I − λΛ_h⁻¹)μ_h as a signed measure, so after applying a bounded value function V the residual is λφ(x,a)⊤Λ_h⁻¹∫V dμ_h. That last term is controlled by the O(H√d)·‖φ‖_{Λ⁻¹} regularization-bias calculation. The linear structure is what lets least-squares from *other* state-action pairs infer the value at a never-before-seen one — the answer to "how do I learn anything about a state I've visited once" is "I don't learn the state, I learn the d-dimensional coefficient, and the linear model ties them together." So I only ever needed P̂_h V ≈ P_h V on a small function class, never the full model — which is why this algorithm, though it does a regression, is really *model-free* in the way that matters.

Now I should stress-test the linearity assumption, because it's strong and I don't want a result that shatters the instant the MDP is slightly non-linear. Suppose the model is only ζ-approximately linear: ‖P_h(·|x,a) − ⟨φ,μ_h⟩‖_TV ≤ ζ and |r_h − ⟨φ,θ_h⟩| ≤ ζ for all (x,a), with ζ ≤ 1. Trace ζ through. First, the value-linearity becomes approximate: with w^π_h = θ_h + ∫V^π_{h+1}dμ_h, |Q^π_h(x,a) − ⟨φ,w^π_h⟩| ≤ |r_h − ⟨φ,θ_h⟩| + |P_h V^π_{h+1} − ⟨φ,∫V^π_{h+1}dμ_h⟩| ≤ ζ + Hζ ≤ 2Hζ (the second piece because ‖V‖_∞ ≤ H and the kernels differ by ζ in TV). The weight bound ‖w^π_h‖ ≤ 2H√d survives unchanged. The decomposition of w_h − w^π_h now gains a fourth term q₄ from the model error: a term Λ_h⁻¹Σ_τ φ_h^τ ε_τ with |ε_τ| ≤ O(Hζ), where the noise ε_τ can be *adversarial*, not stochastic — it's a fixed bias, not a mean-zero fluctuation, so the self-normalized bound doesn't apply to it. For that I use a deterministic Cauchy-Schwarz bound: |φ⊤Λ_h⁻¹Σ_τ φ_h^τ ε_τ| ≤ B√(dk)·√(φ⊤Λ_h⁻¹φ) when |ε_τ| ≤ B (same two-sided Cauchy-Schwarz as the ‖w_h‖ bound, plus Σφ⊤Λ⁻¹φ ≤ d). With B ~ Hζ this contributes 2Hζ√(dk)·‖φ‖_{Λ⁻¹}, which grows with √k — so I let the bonus grow with the episode: β_k = c·(d√ι + ζ√(kd))H. The optimism becomes Q_h^k ≥ Q*_h − 4H(H+1−h)ζ, an accumulating slack of order H²ζ over the horizon. And the regret picks up two effects: a per-step bias of 4Hζ at each of T steps → an additive 4HTζ, and the √k-growing bonus inflating the elliptical-potential sum. Carrying the Σ_k β_k² through (Σ_k β_k² ≤ 2c²(d²HTι + ζ²T²d), so √(Σβ_k²) ≤ √2 c(d√(HTι) + ζT√d)) and combining with the √(dι) potential bound gives the bonus contribution O(√(d³H³Tι²) + ζdHT√ι). So

  Regret(K) = Õ( √(d³ H³ T) + ζ·dHT ).

The misspecification costs an extra term *linear* in T, ζdHT — which is exactly right and unavoidable: a wrong linear model suffers an O(ζ) bias *every* step, no matter how much data, so it must accumulate O(ζT) regret. When ζ is small the algorithm degrades gracefully rather than breaking; the √T term is untouched. Good — the result is robust to the assumption, not brittle.

So here is the causal chain, start to finish. The pain was √S, a wall the tabular lower bound makes unavoidable, so I had to import structure. Linear bandits showed me how structure kills the dependence on the number of options — ridge estimate plus elliptical bonus, optimism, the potential lemma — but only for H=1. To carry it into the horizon I defined the *linear MDP* (kernel and reward linear in φ), which simultaneously makes every Q linear in φ (so there's a d-vector to estimate) and makes the Bellman backup a ridge regression (so value iteration is a sequence of fits) — that's LSVI. Pure LSVI doesn't explore, so I added the elliptical/self-normalized confidence width β√(φ⊤Λ⁻¹φ) as a UCB bonus, the same uncertainty notion as the bandit. The catch — the value function in my own regression target is data-dependent — broke the off-the-shelf concentration, and I fixed it with uniform concentration over the bounded value class 𝒱 (bounded thanks to ridge), paying d² in the covering number, which is what makes β ~ dH and pushes the d-power to d^{3/2}. With β chosen so the deviation never exceeds β‖φ‖_{Λ⁻¹}, optimism Q^k ≥ Q* propagates down the horizon by induction; the regret telescopes into a martingale part (Azuma, the small term) and a cumulative-bonus part (elliptical potential, the d^{3/2}H²√K term), summing to Õ(√(d³H³T)) with no S and no A, polynomial in H, and degrading by only an inevitable ζdHT under model misspecification.

```python
import numpy as np

# LSVI-UCB: Least-Squares Value Iteration with Upper-Confidence Bound.
# Episodic linear MDP with feature map phi(x,a) -> R^d, horizon H, A actions.
# Theory: regret  Õ(sqrt(d^3 H^3 T)),  independent of |S| and |A|.

class LSVI_UCB:
    def __init__(self, phi, d, H, A, beta, lam=1.0):
        # beta = c * d * H * sqrt(log(2 d T / p))  -- makes the bonus a true UCB.
        self.phi, self.d, self.H, self.A = phi, d, H, A
        self.beta, self.lam = beta, lam

    def plan(self, history_by_h):
        # Backward sweep h = H..1, building (Lambda_h, w_h) and the Q_h closures.
        # history_by_h[h] = list of (phi_ha, r, x_next) collected at step h.
        d, H = self.d, self.H
        Lam_inv = [None] * (H + 1)
        w       = [None] * (H + 1)
        Q       = [ (lambda x, a: 0.0) ] * (H + 2)          # Q_{H+1} == 0
        for h in range(H, 0, -1):
            data = history_by_h[h]
            Lam = self.lam * np.eye(d)                      # ridge: lambda*I keeps Lam invertible,
            for (phi_ha, r, x_next) in data:                #        bounds ||w||, controls capacity
                Lam += np.outer(phi_ha, phi_ha)             # Gram matrix  sum phi phi^T + lambda I
            Lam_inv[h] = np.linalg.inv(Lam)                 # (Sherman-Morrison in practice: O(d^2)/step)
            # ridge target y = r + max_a Q_{h+1}(x', a)  -- the empirical Bellman backup
            target = np.zeros(d)
            for (phi_ha, r, x_next) in data:
                v_next = max(Q[h + 1](x_next, a) for a in range(self.A))   # = V_{h+1}(x')
                target += phi_ha * (r + v_next)
            w[h] = Lam_inv[h] @ target                      # w_h = Lam^{-1} sum phi (r + V_{h+1})
            Li = Lam_inv[h]; wh = w[h]; beta = self.beta
            def make_Q(wh, Li, beta):
                def Q_h(x, a):
                    f = self.phi(x, a)
                    bonus = beta * np.sqrt(max(f @ Li @ f, 0.0))   # elliptical confidence width
                    return min(wh @ f + bonus, self.H)             # optimism, clipped at H
                return Q_h
            Q[h] = make_Q(wh, Li, beta)
        return Q

    def run(self, env, K):
        H = self.H
        history_by_h = {h: [] for h in range(1, H + 1)}
        for k in range(K):
            Q = self.plan(history_by_h)                     # re-plan from all data so far
            x = env.reset()                                 # adversary's x_1^k
            for h in range(1, H + 1):
                a = max(range(self.A), key=lambda a: Q[h](x, a))   # greedy w.r.t. optimistic Q
                r, x_next, done = env.step(a)
                history_by_h[h].append((self.phi(x, a), r, x_next))
                x = x_next
        return history_by_h
```
