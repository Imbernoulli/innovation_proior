Let me start where it actually hurts. I want to learn to predict — given experience with some system I don't fully understand, forecast what it will do. Will this chess position lead to a win, will these clouds bring rain, how far will the market move. And the thing about all of these is that the answer doesn't arrive in a tidy pair (question, answer). It dribbles in. A game unfolds move by move, the weather develops across the week, the walk steps from state to state, and at every step I learn a little more about an outcome that is only fully revealed at the very end.

So how does everybody do prediction right now? You force it into supervised learning. You make pairs. To predict Saturday's weather, you pair Monday's measurements with the actual Saturday outcome, Tuesday's measurements with the same Saturday outcome, and so on. Then you run the delta rule on those pairs. For a linear predictor P_t = wᵀx_t the Widrow-Hoff update is Δw_t = α(z − wᵀx_t)x_t — the error between the prediction and what it should have been, z, pushed back onto the weights through x_t. It's robust, it's well understood, it's the LMS filter, it's everywhere. Fine.

But stare at that error term, z − wᵀx_t. It depends on z. The actual outcome. And z isn't known until the sequence ends. So every increment Δw_t for every t in the sequence has to *wait* until the end, when z finally lands. Which means I have to hold on to every x_t (or at least every gradient) for the whole sequence, and then at the last moment dump all the updates at once. For a sequence of length M that's O(M) memory sitting around all week and a burst of O(M) computation on Saturday. That's ugly. Learning ought to be something I can do *as I go*, a little on each step, with bounded memory. The supervised framing structurally can't do that, because its signal is anchored at the end of the sequence.

That's the computational pain. There's a second pain, and it's deeper. Think about that two-person game. There's a state I've learned is bad — it loses 90% of the time. Now I reach a novel position I've never seen, the play runs through the bad state, and by sheer luck the game ends in a win. What should I now think of the novel position? Supervised learning pairs the novel state with the win and concludes: novel state, looks like a winner. But that's clearly wrong. The novel state led straight into a position I *know* usually loses. What happened after that — the lucky win — is noise. The honest verdict is: the novel state is bad, because it led to a bad state. And notice how I got that verdict. I compared the novel state not to the final outcome but to the *next prediction*, the evaluation of the bad state. The final outcome z is corrupted by all the random stuff that happens *after* the state I care about. A subsequent prediction is uncorrupted by that later noise — it can be a *better* performance standard than the truth itself. That's a strange and wonderful thing to say, so let me hold onto it: the actual outcome of a sequence is often not the best estimate of its expected value.

So both pains point the same way. The end-of-sequence outcome is both too late (computationally) and too noisy (statistically). The information I actually want is sitting in how my prediction *changes* from one step to the next. When my estimate jumps from a 50% chance of rain on Monday to 75% on Tuesday, that change is itself the signal — Monday-like days should be revised upward, and I don't have to wait for Saturday to do it.

Let me try to make that precise, because right now it's just an intuition. I have a sequence of predictions P_1, P_2, …, P_m and then the outcome z. The supervised error for prediction P_t is z − P_t. Can I rewrite that gap in terms of changes in prediction? Define P_{m+1} ≝ z, just treat the outcome as one more "prediction" at the end. Then

  z − P_t = P_{m+1} − P_t = (P_{t+1} − P_t) + (P_{t+2} − P_{t+1}) + … + (P_{m+1} − P_m) = Σ_{k=t}^{m} (P_{k+1} − P_k).

It telescopes. The total error between a prediction and the final outcome is *exactly* the sum of the one-step changes in prediction from that point on. There's no approximation here — it's an identity. The supervised error literally *is* a sum of temporal differences. Huh.

That reframes everything. The supervised batch update over a whole sequence is

  w ← w + Σ_{t=1}^{m} α(z − P_t)∇P_t.

Substitute the telescoping in:

  Σ_{t=1}^{m} α(z − P_t)∇P_t = Σ_{t=1}^{m} Σ_{k=t}^{m} α(P_{k+1} − P_k)∇P_t.

Now I have a double sum over pairs (t, k) with t ≤ k. Let me swap the order — sum over k on the outside, and for each k let t run from 1 up to k:

  = Σ_{k=1}^{m} α(P_{k+1} − P_k) Σ_{t=1}^{k} ∇P_t.

Let me just rename the outer index k back to t to read it as a per-step rule:

  Δw_t = α(P_{t+1} − P_t) Σ_{k=1}^{t} ∇P_k.

And look at what this needs. To make the increment at step t I need the change in prediction (P_{t+1} − P_t) — available the moment I see step t+1 — and the *running sum of all past gradients*, e_t = Σ_{k≤t} ∇P_k. That sum I can carry along incrementally: when a new gradient comes in I just add it. I never have to store the individual ∇P_k. So this rule produces *exactly the same total weight change as Widrow-Hoff over the sequence* — it has to, it's an algebraic rearrangement of the same sum — but it can be computed one increment per step, with O(1) extra memory in M instead of O(M). The peak computation is spread evenly over the sequence instead of dumped at the end.

So I've found a temporal-difference form of the supervised rule. Let me call this one TD(1) for a reason that'll be clear in a second. It's not yet anything new in *what* it learns — it lands on the same weights as Widrow-Hoff — but it's already a strict win on *how*: same answer, computed incrementally. That alone matters: speech recognition, process monitoring, anything where data streams in over time and you'd otherwise have to store everything until the label arrives.

Now, can I get something that learns *differently*, and better? The telescoping gave equal weight to every past gradient — Σ_{k=1}^t ∇P_k, all coefficients 1. When the change (P_{t+1} − P_t) happens, it gets attributed equally to every preceding prediction back to the start. But should a prediction I made ten steps ago really be adjusted just as hard as the one I made last step, in response to this latest little wiggle? Recent predictions are more likely to be the relevant cause. Let me weight by recency: a prediction from k steps in the past gets weight λ^{t−k} for some 0 ≤ λ ≤ 1. So

  Δw_t = α(P_{t+1} − P_t) Σ_{k=1}^{t} λ^{t−k} ∇P_k.

Call it TD(λ). At λ = 1 every weight λ^{t−k} = 1 and I'm back to the telescoping, TD(1) ≡ Widrow-Hoff. Good, it's a genuine generalization, the supervised rule is one end of a family.

Is this still incremental? The only thing I need to carry is e_t = Σ_{k=1}^{t} λ^{t−k} ∇P_k. Let me check its recursion. At the next step,

  e_{t+1} = Σ_{k=1}^{t+1} λ^{t+1−k} ∇P_k = ∇P_{t+1} + Σ_{k=1}^{t} λ^{t+1−k} ∇P_k = ∇P_{t+1} + λ Σ_{k=1}^{t} λ^{t−k} ∇P_k = ∇P_{t+1} + λ e_t.

Clean: e_{t+1} = ∇P_{t+1} + λ e_t. One multiply-and-add. So the whole family is O(1)-memory. And this is *why* exponential recency and not some arbitrary decay — exponential is the weighting that folds into a single running trace. Any other shape would force me to remember the individual gradients again, and I'd have thrown away the whole computational advantage. The trace is an eligibility memory: how eligible each weight currently is to be changed by an incoming prediction-difference.

Now push λ all the way to the other extreme, λ = 0. Then λ^{t−k} is 1 only when k = t and 0 otherwise, so the sum collapses to a single term:

  Δw_t = α(P_{t+1} − P_t)∇P_t.

This is TD(0). Only the most recent prediction gets adjusted, toward the very next prediction. And now compare it to the supervised rule it grew out of, Δw_t = α(z − P_t)∇P_t. They're *identical* except for one substitution: the actual outcome z has been replaced by the next prediction P_{t+1}. Same machinery, same gradient, different target. Supervised learning aims each prediction at the final outcome; TD(0) aims each prediction at its own successor. That's the whole move, sitting right there in the algebra. For λ strictly less than 1 this is genuinely *not* any supervised rule — its weight changes can't be reproduced by aiming at z — and TD(0) is the purest version of the new idea, so it's the one to understand first.

Let me make sure I believe the substitution is principled and not just convenient. Why is aiming at P_{t+1} a *good* thing to do, and not merely a cheaper thing? Go back to what I'm really trying to minimize. My predictor P(x, w) is supposed to approximate the expected outcome given the observation, E{z | x}. The honest objective is

  J(w) = E_x[ (E{z | x} − P(x, w))² ].

Per observation, the gradient direction is α(E{z | x_t} − P(x_t, w))∇P. But I don't *know* E{z | x_t} — that's the whole point, it's what I'm trying to learn. I have to estimate it. And there are two natural estimates. I can estimate E{z | x_t} by z, the single outcome that actually followed — and that gives me back the classical supervised rule. Or I can estimate E{z | x_t} by P(x_{t+1}, w), my prediction at the very next step — and that gives me TD(0). The key recognition is in the objective itself: my real goal is for each prediction to match the *expected* subsequent outcome, not the *actual* one that happened to occur in the training set. And the actual outcome is one noisy sample, corrupted by everything random that happens after x_t. P(x_{t+1}) is an already-learned estimate of that same expectation, with the post-successor noise integrated out. So when my predictor is even roughly trustworthy, the next prediction is the *lower-variance* target. That's why the bad-state example came out right: the bad state's learned 90%-loss evaluation is a cleaner standard for the novel state than the one lucky win was. Bootstrapping isn't a hack I tolerate for the sake of incrementality — it's using a better estimate of the thing I actually want.

(I should be honest that aiming at P(x_{t+1}, w) means my target depends on w too, so TD(0) isn't strictly the gradient of a fixed J — the target moves as I learn. I'll keep that flag in mind. But the estimate-substitution argument is the right way to see *why* the next prediction belongs in the target.)

Now I'm uneasy about one thing, and I should chase it down rather than wave at it. This whole scheme rests on updating predictions toward *other predictions*, most of which are themselves wrong, especially early on. I'm pulling myself up by my own bootstraps. What stops it from spinning off into nonsense or settling on garbage? Samuel hit exactly this in his checker player — he adjusted each position's evaluation toward its successor's, and discovered that the constraint "every evaluation matches its successor" is satisfied by *useless* functions, like the constant function where every position scores the same. There's nothing pinning the values to reality. He had to bolt on a fixed piece-count term and, when self-play made things worse, literally zero out the largest weight to jar it loose. So the danger is real: bootstrapping without an anchor can converge to something self-consistent but worthless.

What's the anchor? The end of the sequence. Every sequence terminates with a definite, externally supplied outcome z, and the *last* prediction is tied to that z, not to another prediction. That single grounded constraint propagates backward and rules out the constant-function fixed points. So I do need a terminal outcome that's pinned to the world — that's the discipline Samuel's procedure was missing.

That settles whether the fixed points are sane, but not whether the iteration actually *gets* there. Nobody has ever proved a temporal-difference method stable or convergent to the right predictions — that's the unease that's kept these methods looking like black magic. Let me try to prove it, at least for linear TD(0) on the cleanest case: sequences generated by an absorbing Markov chain (the random walk is one), with a distinct observation vector per state.

Set it up. Nonterminal states N, transition matrix Q with entries Q_ij = p_ij for i, j ∈ N, and h_i = Σ_{j∈T} p_ij z̄_j collecting the expected outcomes on transitions into terminal states. What are the ideal predictions? The expected outcome starting from i is the expected outcome one step out, plus the expected continuation:

  E{z | i} = h_i + Σ_{j∈N} Q_ij E{z | j},  i.e. v = h + Qv,  so v = (I − Q)^{-1} h = Σ_{k=0}^∞ Q^k h.

That last equality needs (I − Q)^{-1} = Σ Q^k, which holds when Q^k → 0 — and for an absorbing chain it does, because Q^k holds the k-step nonterminal-to-nonterminal probabilities and everything eventually gets absorbed. (That's the accessory fact: if A^n → 0 then I − A is invertible with inverse Σ A^k.) So the target is well-defined, and notice it's the Bellman/dynamic-programming fixed point — the same consistency equation DP solves, except I'm going to learn it from sampled experience instead of sweeping a known model.

Now the dynamics. Linear TD(0), updating after each sequence, accumulates

  w_{n+1} = w_n + Σ_t α(P_{t+1} − P_t)∇P_t = w_n + Σ_t α(wᵀx_{q_{t+1}} − wᵀx_{q_t})x_{q_t},

where q_t is the state at step t (and the last "prediction" is z). Group these increments by which transition i → j produced them, with η_ij the count of that transition in the sequence; then take expectations, since transitions and outcomes are independent. Let d_i be the expected number of visits to state i in a sequence, so the expected count of i → j is d_i p_ij. Writing w̄_n for the expected weights, and letting D = diag(d_i), X the matrix of observation vectors as columns, the mean update becomes

  Xᵀw̄_{n+1} = α XᵀX D h + (I − α XᵀX D (I − Q)) Xᵀw̄_n.

That's an affine recursion Xᵀw̄_{n+1} = b + M (Xᵀw̄_n) with M = I − α XᵀX D (I − Q). It converges iff M^n → 0, and then to the fixed point (I − M)^{-1} b. Let me check the fixed point first, assuming convergence: I − M = α XᵀX D (I − Q), so

  (I − M)^{-1} b = (α XᵀX D (I − Q))^{-1} · α XᵀX D h = (I − Q)^{-1} D^{-1} (XᵀX)^{-1} (XᵀX) D h = (I − Q)^{-1} h.

Exactly the ideal predictions. (D^{-1} exists because every d_i > 0 — a never-visited state can just be dropped — and (XᵀX)^{-1} exists because the observation vectors are linearly independent.) So *if* it converges, it converges to the truth. Good.

Now the real work: M^n → 0, i.e. every eigenvalue of M = I − α XᵀX D (I − Q) has modulus below 1 for small enough α. The eigenvalues of M are 1 − αμ where μ ranges over eigenvalues of A ≝ XᵀX D (I − Q). If I can show every such μ has *positive real part*, then for small α each 1 − αμ sits just inside the unit circle and I win. So I need: A has all eigenvalues with positive real part.

The crux is D(I − Q). Claim: it's positive definite. It isn't symmetric, so I can't just look at it — but a matrix A is positive definite exactly when its symmetric part S = A + Aᵀ is, since yᵀAy = ½yᵀ(A + Aᵀ)y (the antisymmetric part contributes zero). So examine S = D(I − Q) + (D(I − Q))ᵀ and use the diagonal-dominance fact I need: a real symmetric matrix with positive diagonal, nonpositive off-diagonal, weak diagonal dominance in every row, and a strict row in each connected component is positive definite. In the irreducible case this is the strict diagonal-dominance lemma Varga gives. Check the conditions. Diagonal: S_ii = 2[D(I − Q)]_ii = 2 d_i(1 − p_ii) > 0. Off-diagonal, i ≠ j: S_ij = [D(I − Q)]_ij + [D(I − Q)]_ji = −d_i p_ij − d_j p_ji ≤ 0. Since the diagonal is positive and the off-diagonals are nonpositive, diagonal dominance is equivalent to nonnegative row sums. Compute the row sum:

  Σ_j S_ij = d_i(1 − Σ_{j∈N} p_ij) + Σ_j d_j(I − Q)_ji = d_i(1 − Σ_{j∈N} p_ij) + μ_i,

using the absorbing-chain identity dᵀ = μᵀ(I − Q)^{-1}, i.e. dᵀ(I − Q) = μᵀ, where μ_i is the probability the sequence *starts* in i. The first term is d_i times the probability of leaving N from i, which is ≥ 0, and μ_i ≥ 0, so every row is diagonally dominant. I still need strictness in each connected component of S. Any retained state has d_i > 0, so it is reachable from some starting state; the directed path that reaches it gives nonzero off-diagonal links in S along the same undirected component. Thus each component contains a start state r with μ_r > 0, and that row has Σ_j S_rj ≥ μ_r > 0. So every component has a strict row, S is positive definite by the diagonal-dominance lemma, and therefore D(I − Q) is positive definite.

Now lift that to A = XᵀX D (I − Q). Let (μ, y) be an eigenpair of A, and set z = (XᵀX)^{-1} y so that y = XᵀX z. Then

  y* D(I − Q) y = z* XᵀX D(I − Q) y = z* A y = μ z* XᵀX z = μ (Xz)* (Xz).

Take real parts: the left side, aᵀD(I−Q)a + bᵀD(I−Q)b for y = a + bi, is strictly positive because D(I − Q) is positive definite; and (Xz)*(Xz) is strictly positive. So Re μ > 0. Every eigenvalue of A has positive real part.

Finish it. For μ = a + bi with a > 0, the eigenvalue 1 − αμ of M has modulus

  |1 − αμ|² = (1 − αa)² + (αb)² = 1 − 2αa + α²(a² + b²),

which is < 1 exactly when α(a² + b²) < 2a, i.e. 0 < α < 2a/(a² + b²). Pick ε as the smallest such bound over all eigenvalues; then for every 0 < α < ε all eigenvalues of M lie strictly inside the unit circle, M^n → 0, and Xᵀw̄_n → (I − Q)^{-1} h. Linear TD(0) converges in the mean to the ideal predictions. There it is — the first proof that a temporal-difference method is stable and lands on the truth. (The predictions themselves keep jittering around their expected values, exactly as Widrow-Hoff's do; shrinking α, or scheduling α ∝ 1/n, controls that variance.)

That's asymptotic correctness. But the finite-data question is still alive: both rules can converge, yet they can aim at different fixed points on a finite set of experience. Take a finite training set and present it over and over until the weights settle. Widrow-Hoff, everyone knows, converges to the weights minimizing RMS error against the actual outcomes *in that training set*. Where does TD(0) go? Build the maximum-likelihood Markov model from the data — Q̂_ij = fraction of times state i transitioned to j, ĥ from the observed terminal outcomes — and compute that model's ideal predictions, (I − Q̂)^{-1} ĥ. I can run the convergence proof again with estimated quantities substituted for true ones throughout — the one extra thing to check is that the analog of dᵀ = μᵀ(I − Q)^{-1} still holds for the counts, which it does because every appearance of a state is either the destination of a transition or a sequence start, giving d̂ᵀ = μ̂ᵀ + d̂ᵀQ̂ ⇒ d̂ᵀ = μ̂ᵀ(I − Q̂)^{-1}. So under repeated presentation TD(0) converges to the predictions that are *optimal if the maximum-likelihood Markov model were exactly right* — the certainty-equivalence estimate.

And those are different answers. Picture the smallest case: I see episodes where A is always followed by B and B sometimes ends in 1, sometimes 0 — say A→B→0 once, and B→1 six times, B→0 once. What's V(A)? Minimizing error against the outcomes that literally followed A gives V(A) = 0, because the one and only A-sequence ended in 0. That's what Widrow-Hoff (and any aim-at-the-actual-outcome method) settles on. But if I respect the sequential structure — A always leads to B, and B is worth 6/8 = 0.75 — then V(A) should be 0.75. That's what TD(0) gives, because it builds in the transition A→B and inherits B's value. The Widrow-Hoff answer overfits the one noisy A-trajectory; the TD answer uses the chain structure to pool the evidence about B. TD is converging to the certainty-equivalence answer of the best-fit Markov model, not just the best fit to the raw outcomes. That's why it should use finite experience better. The λ tradeoff is now clear before I test anything: λ = 0 leans hardest on that bootstrapped model structure but propagates information only one step per presentation, while larger λ reaches farther back toward the exact outcome-based telescoping.

Now let me generalize past "predict the final outcome," because most real problems accumulate. Each step incurs a cost c_{t+1}, and I want to predict the remaining total. The thing to predict is z_t = Σ_{k=0}^∞ γ^k c_{t+k+1}, a discounted sum — discounted both to keep it finite over an infinite horizon and to weight near-term consequences over far-term ones (γ is exactly the "imminence" knob: 0 ≤ γ < 1, with γ = 1 the undiscounted finite-episode case). Do the same telescoping I did at the very start, but now peel off the first term:

  z_t = c_{t+1} + γ c_{t+2} + γ² c_{t+3} + … = c_{t+1} + γ(c_{t+2} + γ c_{t+3} + …) = c_{t+1} + γ z_{t+1}.

So accurate predictions must satisfy the recursive consistency relation

  P_t = c_{t+1} + γ P_{t+1}.

That's the Bellman equation again, falling straight out of the definition of the discounted sum. And the mismatch in this relation — the amount by which my current predictions violate it — is the obvious error to drive learning:

  δ_t = c_{t+1} + γ P_{t+1} − P_t.

This is the temporal-difference error in its general form. Plug it into the same trace-weighted update,

  Δw_t = α(c_{t+1} + γ P_{t+1} − P_t) Σ_{k=1}^{t} (γλ)^{t−k} ∇P_k,

and the discounted, cost-accumulating TD(λ) family appears; the three convergence/optimality results carry over with the obvious modifications. This δ is precisely the learning signal the Adaptive Heuristic Critic was using inside the pole-balancer and that Witten's controller had stumbled onto — but now it's standing on its own as a prediction method with a proof, not buried in a control loop.

Let me write the general recipe I keep re-running, because it's the same three moves every time. First, if the prediction I care about doesn't naturally sit in a sequence of related predictions, embed it in a family that does. Second, write down the recursive equation the predictions *should* satisfy — P_t = P_{t+1} in the simplest case, P_t = c_{t+1} + γ P_{t+1} for discounted costs. Third, drive the weights with the mismatch in that recursion. These are exactly the steps of setting up a dynamic-programming problem — except DP needs the full model and sweeps the state space, while this learns the same fixed point from sampled experience.

And that contrast is the real place this sits. Three ways to estimate a value. Use the actual returned outcome z as the target: that *samples* (no model needed, you just run the system) but it does *not bootstrap* (the target is the real outcome, so you wait until the end and eat the full variance). Use the Bellman expectation E[c + γP(s′)] as the target, à la dynamic programming: that *bootstraps* (the target is built from your current estimate at the successor) but it does *not sample* (it needs the whole expectation, i.e. the model). The temporal-difference target c_{t+1} + γP(x_{t+1}) does *both at once* — it samples one real successor, like working from the outcome, and it bootstraps off the current estimate there, like DP. That's the unification. It inherits DP's bootstrapping efficiency without needing a model, and it inherits sampling's model-freeness without waiting for the final outcome. No prior method occupies that corner.

One more thing makes me trust the bootstrap, and it comes from outside engineering entirely. In Pavlovian conditioning, Rescorla and Wagner modeled learning by ΔV_i = β(λ − V̄)·α_i X_i — the change in a stimulus's association is driven by λ − V̄, the discrepancy between the actual reinforcement λ on the trial and the summed prediction V̄. That is, beat for beat, the same form as the supervised error z − P: learning happens when the outcome violates the expectation. It's a beautiful model, but it's a *trial-level* model — one λ, one V̄, one update per trial — and it's blind to *when* things happen inside a trial. And it breaks on second-order conditioning: pair B with an already-trained A, no actual reward on those trials, so λ = 0 and λ − V̄ ≤ 0, so the model says B can only lose association — yet animals reliably grow a positive association to B. The fix is to stop treating λ as a single trial-level number. Let the thing being predicted be the discounted sum of *future* reinforcement, V̄_t = λ_{t+1} + γλ_{t+2} + γ²λ_{t+3} + …, and telescope it exactly as before into V̄_t = λ_{t+1} + γV̄_{t+1}. Then the discrepancy that should drive learning is λ_{t+1} + γV̄_{t+1} − V̄_t — which is just Rescorla-Wagner's λ − V̄ with λ promoted from "the reward this trial" to "the reward next step plus the discounted prediction next step." It's a discrete-time time-derivative of the prediction. And it dissolves the second-order problem for free: when B precedes a trained A, the next-step prediction γV̄_A is positive, so B sees reinforcement even with no primary reward present — the prediction itself acts as a secondary reinforcer, which is exactly the psychological mechanism by which credit travels back along a chain. So the same δ I derived from telescoping the supervised error is also the natural real-time generalization of the best model of animal conditioning. That's not a coincidence; that's the bootstrap being the *right* idea, showing up independently in the psychology and in the engineering.

Let me land it. Predictions are P(x, w); in the linear case P = wᵀx and ∇P = x. I keep a running eligibility trace e (one component per weight). On each step, after seeing the next observation and the cost/reward, I form the temporal-difference error δ = c + γP(x′) − P(x), decay-and-accumulate the trace e ← γλ e + ∇P(x), and nudge the weights w ← w + α δ e. Set γ = 1 for an episodic outcome-prediction problem and δ collapses to P(x′) − P(x) (with P at the terminal step pinned to the actual outcome z); set λ = 0 and the trace collapses to ∇P(x) and I'm doing pure TD(0), w ← w + α(c + γP(x′) − P(x))∇P(x) — the supervised rule with the next prediction in place of the outcome; in that same undiscounted outcome setting, set λ = 1 and I recover the incremental form of Widrow-Hoff. The whole family runs in memory that doesn't grow with sequence length, learns as the data streams in, targets the expected outcome through a lower-noise standard than the actual one, and provably converges — under repeated presentation — to the certainty-equivalence estimate of the underlying Markov process, which is a more structured use of finite experience than fitting the raw outcomes.

The causal chain, start to finish: the outcome-anchored supervised signal is both too late and too noisy → the supervised error telescopes exactly into a sum of successive prediction-differences → so learning can be driven incrementally, change by change, weighting recent predictions by an exponentially-decaying eligibility trace that recurs in O(1) memory → pushing the recency parameter to zero replaces the noisy outcome with the lower-variance next prediction, which the gradient-descent objective justifies because the goal was always the *expected* outcome, not the actual one → grounding the last prediction in a real terminal outcome rules out the worthless self-consistent fixed points → and on absorbing Markov chains this iteration provably converges in the mean to the true predictions, and under repeated presentation to the maximum-likelihood/certainty-equivalence predictions, explaining why it can use finite experience better → the same telescoping on a discounted sum yields δ = c + γP(x′) − P(x), the temporal-difference error, which both unifies dynamic programming (bootstrap, no model) with outcome-based learning (sample, no sweep) and is the real-time generalization of Rescorla-Wagner, with the prediction acting as its own secondary reinforcer.

```python
import numpy as np

class TDLambda:
    """Linear TD(lambda) prediction. Predictions P(x,w)=w·x are driven by the
    difference between successive predictions, not by the final outcome."""

    def __init__(self, n_features, alpha=0.1, gamma=1.0, lam=0.0):
        self.w = np.zeros(n_features)
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.e = np.zeros(n_features)      # eligibility trace e_t = sum (gamma*lam)^(t-k) grad P_k

    def predict(self, x):
        return float(self.w @ x)           # linear case: gradient of P wrt w is x

    def step(self, x, reward, x_next):
        # TD error: next prediction replaces the (as-yet-unknown) actual outcome
        v      = self.predict(x)
        v_next = 0.0 if x_next is None else self.predict(x_next)   # terminal: reward/cost carries the outcome
        delta  = reward + self.gamma * v_next - v                 # delta = r + gamma*V(s') - V(s)
        # trace recurrence e <- gamma*lam*e + grad P(x); grad is x in the linear case
        self.e = self.gamma * self.lam * self.e + x
        # one incremental weight change per step, O(1) memory in sequence length
        self.w += self.alpha * delta * self.e
        return delta

    def end_episode(self):
        self.e = np.zeros_like(self.e)     # reset eligibility at sequence boundary

# lam=0 -> TD(0): w += alpha*(r + gamma*V(s') - V(s)) * x   (bootstrap on the next prediction)
# gamma=1 and lam=1 -> incremental Widrow-Hoff (the telescoped supervised rule)
# gamma=1 -> undiscounted episodic outcome prediction; the last transition carries the real outcome
```
