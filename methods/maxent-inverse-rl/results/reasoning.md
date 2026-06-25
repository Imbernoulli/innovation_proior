I have a pile of GPS traces — taxi drivers threading the same road network day after day — and I want to learn what they're actually trying to do, so I can predict where a driver will turn next, or guess their destination from half a trip. The obvious thing is to clone the behavior: at each road segment, copy the turn the driver tended to take. But that memorizes the segments I happened to see and says nothing about a segment I never observed, or a slightly different goal. What carries over is the *reason* — the trade-off between distance, speed, road type, number of turns — the thing the driver is optimizing. So I'll model the network as an MDP: states are road segments, actions are the transitions at intersections, the destination is an absorbing state, and the driver is assumed to optimize some reward. Reinforcement learning, run forward, takes a reward and gives behavior. I have behavior and want the reward. Inverse.

Let me set up the optimization the driver is supposedly doing. Each segment `s_j` has a feature vector `f_{s_j}` — its road type, speed, lanes, the kind of turn taken to enter it. The driver values a state by some weights `θ` on those features, and values a whole path `ζ` by the sum along it. So `reward(f_ζ) = θ·f_ζ` where `f_ζ = Σ_{s_j ∈ ζ} f_{s_j}` is the path's feature counts. From the demonstrations I can compute the empirical expected feature counts, `f̄ = (1/m) Σ_i f_{ζ̃_i}` over my `m` observed trips. That much is clean: a low-dimensional, transferable summary of "what these drivers do."

Now, what do I actually want from `θ`? The first instinct is: find `θ` that makes the demonstrated paths optimal. Make the drivers' choices the best choices under the recovered reward. That's the natural reading of the inverse problem, and it's exactly how it's been posed — given an MDP and a demonstrated optimal policy, find rewards under which that policy is optimal (Ng & Russell 2000). Let me actually look at the solution set, because that's where the trouble is. Their characterization says the policy that always picks action `a_1` is optimal iff, for every other action `a`, `(P_{a_1} − P_a)(I − γP_{a_1})^{-1} R ⪰ 0`. Stare at this. It's a system of inequalities on `R`. And the all-zero reward satisfies it trivially — `R = 0` makes the left side zero, `⪰ 0` holds, and indeed with zero reward *every* policy is optimal because nothing distinguishes any behavior. So `R = 0` always "explains" my drivers. So does any constant reward. And beyond those degeneracies there's a whole cone of reward vectors that make the demonstration optimal. The constraint "make the demonstration optimal" carries almost no information; it picks out a huge set, most of whose members are useless.

Ng & Russell saw this and patched it with a tie-breaker: among all the rewards that make the demonstration optimal, prefer the one that maximizes the margin by which the demonstrated action beats the next-best action, `Σ_s (Q*(s,a_1) − max_{a≠a_1} Q*(s,a))`, with an `ℓ_1` penalty, solved as a linear program. It works, but the margin objective is arbitrary — there's no reason the *true* driver behaves to maximize that particular gap, it's a heuristic to make the LP pick something. And it wants the full optimal policy handed to me, which I don't have; I have noisy traces. And if the drivers are imperfect — and GPS-tracked humans absolutely are imperfect, they take the scenic route, they miss a turn — then no single reward makes their behavior strictly optimal at all, and the whole "make it optimal, then break ties by margin" frame has nothing to stand on.

So maybe I shouldn't insist the demonstration be *optimal*. Maybe I should only insist that whatever behavior I model has the same *feature counts* as the demonstrations — drives the same total distance on highways, makes the same number of hard lefts, and so on. There's a reason to think this is a good target rather than an arbitrary one. Define the expected feature counts of a behavior, `μ = E[Σ_t γ^t f(s_t)]`. Because reward is linear in features, the *value* of that behavior is `θ·μ`. So if I produce a behavior whose feature counts equal the demonstrated `f̄`, then under *any* linear reward whatsoever — including the true unknown one — my behavior should have the same value as the demonstrations. Let me sanity-check that I'm not fooling myself with the word "value": if two behaviors have feature counts `μ₁` and `μ₂`, the gap in their values is `θ·μ₁ − θ·μ₂ = θ·(μ₁ − μ₂)`, and when `μ₁ = μ₂` that is `θ·0 = 0` for *every* `θ`. So yes — matching feature counts forces equal value under all linear rewards, with no dependence on knowing `θ`. That is the property I want, and crucially it doesn't need the demonstration to be optimal, just matched (Abbeel & Ng 2004).

Abbeel & Ng build an algorithm on this: alternate an IRL step that finds a weight vector on which the expert currently out-scores my candidate policies by a margin, and an RL step that solves the MDP under that weight; iterate until the expert's feature counts are matched, and return the resulting collection of policies. And here's where I hit the wall again. What they return is a *mixture* of policies — flip a coin at the start of the trip to decide which policy to follow — because a single deterministic policy generally can't hit an arbitrary target feature-count vector, but a convex combination of several can. A mixture isn't a coherent model of one driver; it's "be driver A on Mondays, driver B on Tuesdays." And worse, the deeper problem from before hasn't gone away, it's just moved. Feature matching is satisfied by *many* different behaviors. Lots of distinct policies, and lots of distinct mixtures of policies, all hit the same `f̄`. The constraint `Σ_ζ P(ζ) f_ζ = f̄` is a handful of linear equations on a distribution over astronomically many paths — wildly underdetermined. Which of the matching behaviors is *the* model of my drivers? Max-margin picks one by another arbitrary tie-break. There's still no principled answer to "given that many distributions match the feature counts, which one do I commit to?"

Let me stop and name exactly what's wrong, because I think the wrongness is the clue. I have a set of constraints — match these feature expectations, be a normalized distribution over paths — and infinitely many distributions satisfy them, and I have no principled rule for choosing one. This is not a new situation. This is the oldest problem in statistical inference: assign a probability distribution when your knowledge fixes only some expectation values and leaves the rest open. Laplace's principle of insufficient reason was the first attempt — make things equal when you've no reason to prefer otherwise — but it's ad hoc and breaks under reparameterization. The cleaner resolution is Jaynes's: among all distributions consistent with what you know, choose the one of maximum entropy, `H(P) = −Σ P log P`. The claim is that this is the unique distribution that is maximally noncommittal about everything the constraints *don't* pin down — it injects no preference, no spurious structure, beyond what the feature-matching forces; any other choice would smuggle in information I don't have. I should hold that "unique / maximally noncommittal" billing as a hope rather than a proven fact for now, and let the derivation either earn it or expose it. Either way, it matches my situation exactly: many path-distributions hit `f̄`, and I want the one that hits `f̄` and otherwise assumes as little as possible about which paths the driver prefers.

So let me stop reasoning about policies and mixtures, and reason directly about a distribution `P(ζ)` over entire paths. I want the distribution that maximizes entropy subject to feature-matching and normalization:

maximize `H(P) = −Σ_ζ P(ζ) log P(ζ)`
subject to `Σ_ζ P(ζ) f_ζ = f̄` (match the demonstrated feature counts, componentwise)
and `Σ_ζ P(ζ) = 1`.

Now just turn the crank Jaynes turned. Form the Lagrangian, with multipliers `θ` (a vector, one per feature) for the matching constraints and `μ` for normalization. I need the sign convention to make a positive reward increase probability, so I attach `θ` to model feature counts minus demonstrated feature counts:

`J = −Σ_ζ P(ζ) log P(ζ) + Σ_k θ_k (Σ_ζ P(ζ) f_{ζ,k} − f̄_k) + μ(Σ_ζ P(ζ) − 1)`.

Differentiate with respect to a single `P(ζ)` and set to zero:

`∂J/∂P(ζ) = −log P(ζ) − 1 + Σ_k θ_k f_{ζ,k} + μ = −log P(ζ) − 1 + θ·f_ζ + μ = 0`.

So `log P(ζ) = θ·f_ζ + μ − 1`, i.e.

`P(ζ) = exp(θ·f_ζ) · exp(μ − 1)`.

The second factor is a constant that doesn't depend on `ζ`; normalization fixes it. Summing over all paths, `Σ_ζ P(ζ) = exp(μ−1) Σ_ζ exp(θ·f_ζ) = 1`, so `exp(μ−1) = 1/Z(θ)` with the partition function `Z(θ) = Σ_ζ exp(θ·f_ζ)`. Therefore

`P(ζ | θ) = (1/Z(θ)) exp(θ·f_ζ) = (1/Z(θ)) exp(Σ_{s_j ∈ ζ} θ·f_{s_j})`.

The probability of a path comes out exponential in its reward. I should check that this is the right kind of object before reading more into it. The stationarity condition was linear in `log P(ζ)` and the only nonlinearity in `H` is the `−P log P`, whose second derivative is `−1/P(ζ) < 0` — so `J` is strictly concave in `P` and this stationary point is the *unique* maximizer, not a saddle or one of many. That retroactively earns the "unique distribution" billing I was holding on credit: the constraints really do single out exactly one max-entropy distribution, and it's this one. So I didn't pick the exponential form by taste — it's what falls out of "match the counts, assume nothing else." Reading it off: higher-reward paths are exponentially more likely; two paths with *equal* reward get *equal* probability; and the Lagrange multipliers `θ` that enforce feature-matching are exactly the reward weights. The reward is the dual variable of "match the demonstrated feature counts while assuming nothing else." This is a Boltzmann distribution over behaviors, an exponential-family model, and it's a single, globally normalized object: one distribution over all paths, not a mixture, not a per-state patchwork.

I want to check that this global normalization is doing real work, not just looking elegant. The other obvious probabilistic move is to normalize *locally*: set the probability of each action at a state proportional to `exp(Q*(s,a))`, a softmax over action values at that branch. The worry I've heard about such locally-normalized sequence models is "label bias," but I don't want to take that on faith — let me build the smallest case where it should bite and actually compute both models. Start state `S` with two choices. Action `L` runs `S → u → goal`, where `u` has a single onward action, no branch. Action `R` runs `S → v → goal`, where `v` has four onward actions, exactly one of which reaches the goal (the other three hit dead states). Put zero reward everywhere, so the two complete routes to the goal, path A (`L`) and path B (`R`), have *identical* total reward 0. A model that says "high reward ⟹ high probability" must call these two equally likely. Global model: `P ∝ exp(0)` for each valid path, so after normalizing over the two they're `0.5` and `0.5` — equal, as required. Local model: at `S`, equal `Q`'s give `P(L)=P(R)=1/2`; at `u` the single action takes probability `1`; at `v` four equal actions split the mass, so the goal-reaching one gets `1/4`. Multiplying along the routes, `P(path A) = 1/2·1 = 1/2` while `P(path B) = 1/2·1/4 = 1/8`. Their ratio is `1/4`, not `1`. So the local model declares path B four times less likely than the equally-rewarded path A, for no reason but that `v` had more outgoing edges — the missing `3/8` of mass leaked into dead-end paths and never came back. That's label bias, made concrete: per-branch conservation forces structure that the reward never asked for, and you can get a higher-reward path that is *less* probable than a lower-reward one just because it passed through busier intersections. My globally normalized `P(ζ) ∝ exp(θ·f_ζ)` compares whole paths in one shared pool with no per-branch conservation, so the equal-reward paths tie exactly (I just saw it: `0.5`/`0.5`). The global normalization isn't decoration; it's what buys that coherence.

I've been assuming the path is determined by the driver's choices alone. On a real road the dynamics can be stochastic — you intend a turn and the world sometimes lands you elsewhere. The driver shouldn't be "rewarded" or "blamed" for the dice the environment rolls, so the distribution over paths has to factor in the transition randomness. Let `T` be the transition distribution and let an *outcome* `o` fix the next state for every action — so given the action choices, `o` pins down the realized path. The MDP is deterministic once `o` is fixed. The exact max-entropy distribution constrained to match feature counts, now conditioned on `T`, is

`P(ζ | θ, T) = Σ_{o ∈ T} P_T(o) · (exp(θ·f_ζ) / Z(θ,o)) · I_{ζ ∈ o}`,

summing over outcome samples, with the indicator picking out outcomes compatible with `ζ`. Exact, but that sum over all outcomes is intractable. If I assume the transition randomness has only a limited effect on which behavior the driver favors — so the partition function is roughly constant across the outcomes `o` — I can pull it out and approximate

`P(ζ | θ, T) ≈ (exp(θ·f_ζ) / Z(θ,T)) · Π_{(s_{t+1}, a_t, s_t) ∈ ζ} P_T(s_{t+1} | a_t, s_t)`.

The reward still drives an exponential preference over paths; the transition product just folds in the probability of the environment actually realizing that path given the actions. And this immediately gives me a stochastic *policy*: the probability of taking action `a` at the start is the total probability of all paths that begin with `a`, `P(action a | θ,T) ∝ Σ_{ζ: a ∈ ζ_{t=0}} P(ζ | θ,T)` — one coherent distribution over each state's actions, no mixture, no coin flip.

Now, how do I fit `θ` to the data? I have a parametric family `P(ζ|θ,T)` and demonstrated paths. The natural thing is maximum likelihood: choose `θ` to maximize the probability of the demonstrations,

`θ* = argmax_θ L(θ) = argmax_θ (1/m) Σ_{examples} log P(ζ̃ | θ, T)`.

There's a consistency worth noticing — maximizing the likelihood of the data under this exponential-family distribution turns out to be the *same* optimization as feature-matching, which is the same as the max-entropy problem I started from; that the three coincide is a standard exponential-family duality, and I'll let the gradient below confirm it concretely rather than just asserting it. Let me write the log-likelihood with the transition product separated out as a base measure. Call it `b_T(ζ) = Π_{(s_{t+1}, a_t, s_t) ∈ ζ} P_T(s_{t+1} | a_t, s_t)`, with `b_T(ζ)=1` in the deterministic case. Then the normalizer is not just a decorative denominator; it has to sum the exponentiated reward *with* that base measure:

`Z(θ,T) = Σ_ζ b_T(ζ) exp(θ·f_ζ)`,

and each demonstrated trajectory contributes `θ·f_{ζ̃_i} + log b_T(ζ̃_i) − log Z(θ,T)`. The `log b_T(ζ̃_i)` term is fixed by the dynamics, so it disappears from the gradient. The first θ-dependent term is linear, gradient `f_{ζ̃_i}`. For the partition function,

`∇_θ log Z(θ,T) = (1/Z) Σ_ζ b_T(ζ) exp(θ·f_ζ) f_ζ = Σ_ζ P(ζ|θ,T) f_ζ`.

That's the *expected* feature counts under the current model, including the transition probabilities in the path mass. So summing over the `m` demonstrations and writing `f̄` for their average,

`∇_θ L(θ) = f̄ − Σ_ζ P(ζ|θ,T) f_ζ`.

The gradient is the empirical feature counts minus the model's expected feature counts. Demonstrated minus expected. And it says exactly what learning *is*: when the model under-visits highway segments relative to the drivers, the highway weight goes up; when it over-visits, down; and you stop precisely when the model's expected feature counts equal the demonstrated ones. At a fixed point `∇L = 0` reads `Σ_ζ P(ζ|θ,T) f_ζ = f̄` — feature expectations match — which is also exactly the feature-matching condition, so the "three faces are one" claim above isn't a slogan, the same equation is the stationarity condition of all three. If it holds, I inherit Abbeel & Ng's performance property (same feature counts ⟹ same value under the true reward, which I checked above is a `θ·0 = 0` identity), but now realized by a *single* stochastic policy chosen by a principle rather than a tie-break — provided the gradient ascent actually reaches that fixed point. I shouldn't take that on faith either; I'll come back and test it once I have the machinery to evaluate the gradient.

Is this optimization well-behaved? Differentiate once more — the Hessian. `∇²_θ log Z = ∇_θ Σ_ζ P(ζ|θ,T) f_ζ`. Differentiating the model expectation, `∂/∂θ [Σ_ζ P(ζ|θ,T) f_{ζ}]`: each `P(ζ|θ,T)` carries its own `∂ log P/∂θ = f_ζ − Σ_{ζ'} P(ζ') f_{ζ'}`, so the derivative is `Σ_ζ P(ζ|θ,T) f_ζ (f_ζ − E[f])^T = Σ_ζ P(ζ) f_ζ f_ζ^T − E[f] E[f]^T`, which is exactly the covariance `Cov_{P(·|θ,T)}[f_ζ]` — and a covariance matrix is positive semidefinite. The empirical term `f̄·θ` is linear in `θ`, so it contributes nothing to the second derivative, and the average over demonstrations leaves the same `log Z`. Hence `∇²_θ L = −Cov_{P(ζ|θ,T)}[f_ζ] ⪯ 0`. The log-likelihood is concave; maximizing it is a convex optimization problem; there are no local maxima below the global one. (This is the exponential-family identity that the second derivative of a log-partition is a variance, surfacing here as curvature — the same identity, incidentally, that made `H` strictly concave above.) Convex maximum likelihood, gradient = demonstrated − expected feature counts. That's the whole learning story, and it's exactly what the ambiguity-ridden margin heuristics were missing — *if* I can evaluate that gradient.

Except I've buried the hard part in one symbol. To take a gradient step I need `Σ_ζ P(ζ|θ,T) f_ζ`, the expected feature counts, and the obvious way to get it is to enumerate paths, weight each by `P(ζ|θ,T)`, and sum. But the number of paths grows exponentially with the time horizon — on a 300,000-state road network that's hopeless. I need the expected feature counts without ever enumerating a path. Notice the expected feature counts are just `Σ_{s_i} D_{s_i} f_{s_i}`, where `D_{s_i}` is the expected number of times the behavior visits state `s_i` — the expected state-visitation frequency. So the real computational object I need is `D_{s_i}` for every state. And that I *can* get by dynamic programming, the same way value iteration and the forward-backward algorithm for sequence models avoid enumerating sequences — back the probabilities up through the planning horizon instead.

Two passes. First a backward pass that computes, for each state and each remaining horizon, a partition function `Z^{(h)}_{s_i}` — the total exponentiated reward of all paths from that state with `h` counted state positions left. Initialize the zero-step boundary with `Z^{(0)}_{s_i} = 1` (an empty remaining path contributes a factor of one). Then recurse for `h = 1, …, N`: the value of taking action `a_{i,j}` at state `s_i` aggregates, over possible next states `s_k`, the transition probability times this state's exponentiated reward times the downstream partition function,

`Z^{(h)}_{a_{i,j}} = Σ_k P(s_k | s_i, a_{i,j}) · exp(reward(s_i | θ)) · Z^{(h-1)}_{s_k}`,

and the state's partition function sums over its actions, `Z^{(h)}_{s_i} = Σ_{a_{i,j}} Z^{(h)}_{a_{i,j}}`. This is exactly the kind of backup value iteration does, but with a sum (a "soft" max) over actions instead of a hard max — which is what the exponential-family / max-entropy model asks for: actions are weighted by their exponentiated downstream reward, not winner-take-all. For a finite horizon I have to keep the remaining-horizon index: at time `t`, with `N−t` state positions left to count, the local action probability is

`P_t(a_{i,j} | s_i) = Z^{(N-t)}_{a_{i,j}} / Z^{(N-t)}_{s_i}`.

I want to make sure I have this recursion and its boundary right before I trust it on a 300k-state graph, so let me run it by hand on the smallest case that exercises the soft action-split. Three states: `s0` branches to `s1` under action 0 and to `s2` under action 1; `s1` and `s2` are absorbing. Give `s1` reward `log 3` and `s0, s2` reward 0, so the only thing distinguishing the two actions at `s0` is that one leads to a state worth `exp(log 3) = 3` and the other to a state worth `1`. Count `N = 2` state positions. By hand: `Z^{(0)}_s = 1` for all. For `h = 1`, `Z^{(1)}_{a}(s) = exp(reward(s))·Σ_k P(k|s,a)·1 = exp(reward(s))`, so `Z^{(1)}_{s1} = exp(log 3) = 3`, `Z^{(1)}_{s2} = 1`, and at `s0` both actions give `exp(0) = 1`, so `Z^{(1)}_{s1}=3, Z^{(1)}_{s2}=1, Z^{(1)}_{s0}=2`. For `h = 2` at `s0`: action 0 leads to `s1`, `Z^{(2)}_{a0}(s0) = exp(0)·Z^{(1)}_{s1} = 3`; action 1 leads to `s2`, `Z^{(2)}_{a1}(s0) = exp(0)·Z^{(1)}_{s2} = 1`. So the policy at `s0` with two positions left should be `P(a0) = 3/(3+1) = 0.75`, `P(a1) = 0.25` — the action toward the higher-reward state gets exactly its exponentiated-reward share. I coded the recursion and ran it: the returned policy at two-states-left is `[0.75, 0.25]` at `s0` and `[0.5, 0.5]` at the absorbing states, and at one-state-left it is `[0.5, 0.5]` everywhere — matching the hand computation to the digit. The boundary `Z^{(0)}=1`, the `exp(reward)` weighting, and the action ratio are all behaving as derived; the `0.75` is the reward, not an artifact. If I am using a large fixed horizon only to approximate an infinite-horizon absorbed problem, those backups can be iterated until this ratio stabilizes and the policy becomes stationary. But for a true finite horizon the time index matters: `P_t` is the fraction of a state's total exponentiated path-mass that flows through each action at that remaining horizon, higher-reward continuations get exponentially more, equal-reward continuations split evenly — the path distribution, expressed one decision at a time.

Then a forward pass that pushes visitation mass through that policy. Seed each state with the probability it's where a trip starts, `D_{s_i,0} = P(s_i = s_{initial})` (the empirical start-state distribution). Then propagate: the mass at `s_k` at time `t+1` is everything that flows into it — sum over predecessor states `s_i` and actions `a_{i,j}` of the mass that was at `s_i` at time `t`, times the chance the policy picks `a_{i,j}` there, times the chance that action lands in `s_k`,

`D_{s_k, t+1} = Σ_{s_i, a_{i,j}} D_{s_i, t} · P_t(a_{i,j} | s_i) · P(s_k | s_i, a_{i,j})`,

run for `t = 0, …, N−2` when `N` is the number of state positions being counted. Finally sum visitation over the counted time slices, `D_{s_i} = Σ_{t=0}^{N-1} D_{s_i, t}`. Now `Σ_{s_i} D_{s_i} f_{s_i}` is the model's expected feature counts — computed in time polynomial in states and horizon, never touching an individual path — and the gradient `f̄ − Σ_{s_i} D_{s_i} f_{s_i}` is in hand. Step `θ` along it and repeat.

Now I can finally cash the promise I made twice above — that the gradient actually drives the model's expected counts to the demonstrated ones — instead of asserting it. On that same three-state MDP, give `s0` feature `[1,0]`, `s1` feature `[0,1]`, `s2` feature `[0,0]`, so the second feature simply counts visits to `s1`. Feed in four demonstrated trips, three taking the `s0→s1` branch and one taking `s0→s2`; the demonstrated counts are `f̄ = [1, 0.75]` (every trip visits `s0`, three of four visit `s1`). Start `θ = 0` and run the full loop — backward pass, forward pass, gradient `f̄ − Σ_s D_s f_s`, ascent step — for a few hundred epochs. The model's expected counts converge to `[1, 0.75]` and the final gradient is `[0, 2.2·10⁻¹⁶]`, i.e. zero to machine precision. So the fixed point really is where expected feature counts equal demonstrated ones, gradient ascent really reaches it, and the recovered `θ` produces a `0.75` probability on the `s1` branch — reproducing the empirical frequency, not by a tie-break but as the optimum of one convex objective. That closes the loop: concave likelihood, exact gradient, and a fixed point that delivers the feature-matching (hence performance-matching) property with a single stochastic policy.

One caution on the partition function itself: does `Z(θ)` even converge? For a finite horizon, or an infinite horizon with discounting, yes. For an undiscounted infinite horizon with zero-reward absorbing states it can diverge even when all rewards are negative — paths can dawdle forever. But the demonstrations are trips that *do* reach a destination in finitely many steps, and constrained to reproduce those, the entropy-maximizing weights have to produce paths that get absorbed in finite time, so the relevant `Z` converges. Good enough to proceed.

A last refinement on the step itself. Plain additive gradient ascent already has the right target because `L` is concave and the gradient is exact once the expected counts are computed — which is what the run above demonstrated. If I want the finite-sample regularized version, I can swap the optimizer for exponentiated-gradient updates. That keeps the optimization efficient and, as a side effect, induces an `ℓ_1`-flavored regularization on the coefficients, which is well-motivated here: matching the *true* feature expectations only up to sampling error from finitely many demonstrations is exactly a max-entropy problem with bounded uncertainty in the feature expectations, whose solution is the same exponential-family model but with an `ℓ_1` regularizer whose strength scales with that uncertainty (Dudík & Schapire 2006), giving an error that depends only logarithmically on the number of features. So the regularization isn't bolted on; it's what "match the demonstrated counts, but only to within the noise" turns into.

So the whole chain, end to end: I wanted the driver's reward, not a clone of the driver. Recovering the reward is ill-posed — zero reward and a whole cone of others make any behavior "optimal," and matching feature counts alone is satisfied by countless behaviors. Instead of breaking the tie with an arbitrary margin, I committed to the one principle that *says* what to do with leftover ambiguity: match the demonstrated feature expectations and otherwise maximize entropy. Lagrange multipliers turn that into `P(ζ) ∝ exp(θ·f_ζ)` — and the strict concavity of the entropy made that distribution the unique maximizer, a single, globally normalized Boltzmann distribution over paths whose reward weights are the dual variables, free of the label bias that I watched cost a local model a factor of four on equal-reward paths. Maximizing the concave log-likelihood is a convex optimization problem, and its gradient is demonstrated-minus-expected feature counts, with the partition function contributing exactly the model expectation; I checked on a small MDP that ascent drives that gradient to zero and the expected counts onto the demonstrated `[1, 0.75]`, so the gradient vanishes when feature expectations match and the performance guarantee comes with it, via one coherent stochastic policy. The only hard piece, the expected feature counts, reduces to expected state-visitation frequencies, computed by a backward pass that builds remaining-horizon partition functions — which I hand-checked reproduces the exact `0.75/0.25` reward-weighted split — and a forward pass that flows visitation through the induced time-indexed policy — polynomial, no path enumeration. The reward drops out of the dual; the behavior model drops out of the principle; the algorithm drops out of the gradient.

```python
import numpy as np

def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = np.exp(x - m)
    shifted[~np.isfinite(shifted)] = 0.0
    out = m + np.log(np.sum(shifted, axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def _log_transition(transition):
    log_t = np.full_like(transition, -np.inf, dtype=float)
    positive = transition > 0
    log_t[positive] = np.log(transition[positive])
    return log_t

def find_feature_expectations(feature_matrix, trajectories):
    # f̄ : average demonstrated feature counts (the constraint target)
    fe = np.zeros(feature_matrix.shape[1])
    for traj in trajectories:
        for s in traj:
            fe += feature_matrix[s]
    return fe / len(trajectories)

def find_time_indexed_policy(n_states, reward, n_actions, transition, horizon):
    # Backward pass: Z_s^(0)=1, store P_t(a|s) for each remaining horizon.
    log_t = _log_transition(transition)
    log_z_s = np.zeros(n_states)                    # log Z_{s,0}
    policies_by_remaining = []
    for _ in range(1, horizon + 1):
        log_z_a = np.empty((n_states, n_actions))
        for a in range(n_actions):
            downstream = _logsumexp(log_t[:, a, :] + log_z_s[None, :], axis=1)
            log_z_a[:, a] = reward + downstream
        log_z_s = _logsumexp(log_z_a, axis=1)       # log Z_s = log Σ_a Z_a
        policy = np.exp(log_z_a - log_z_s[:, None]) # P(a|s,h)=Z_a^h/Z_s^h
        policies_by_remaining.append(policy / policy.sum(axis=1, keepdims=True))
    return np.asarray(policies_by_remaining)

def find_expected_svf(n_states, reward, n_actions, transition,
                      trajectories, horizon):
    # Forward pass: expected state-visitation frequencies D_{s_i} = Σ_t D_{s_i,t}.
    policies = find_time_indexed_policy(n_states, reward, n_actions,
                                         transition, horizon)
    n_traj = len(trajectories)
    # D_{s,0} = empirical start-state distribution p_0(s)
    start = np.zeros(n_states)
    for traj in trajectories:
        start[traj[0]] += 1.0
    start /= n_traj
    D = np.zeros((n_states, horizon))
    D[:, 0] = start
    for t in range(horizon - 1):
        policy = policies[horizon - t - 1]          # remaining counted states
        for i in range(n_states):
            for a in range(n_actions):
                # D_{s_k,t+1} += D_{s_i,t} P_t(a|s_i) P(s_k|s_i,a)
                D[:, t + 1] += D[i, t] * policy[i, a] * transition[i, a, :]
    return D.sum(axis=1)                          # Σ_t D_{s_i,t}

def irl(feature_matrix, n_actions, transition, trajectories,
        epochs, lr, horizon=None):
    # Convex maximum-likelihood fit; gradient = demonstrated − expected counts.
    n_states, d = feature_matrix.shape
    horizon = horizon or max(len(traj) for traj in trajectories)
    theta = np.random.uniform(size=(d,))
    f_bar = find_feature_expectations(feature_matrix, trajectories)
    for _ in range(epochs):
        reward = feature_matrix.dot(theta)                  # r(s) = θ·f_s
        D = find_expected_svf(n_states, reward, n_actions,
                              transition, trajectories, horizon)
        expected = feature_matrix.T.dot(D)                  # Σ_s D_s f_s
        grad = f_bar - expected                             # ∇L(θ)
        theta += lr * grad                                  # gradient ascent
    return feature_matrix.dot(theta)                        # recovered reward
```
