Let me start from what actually goes wrong, because the problem here is a clean one and I want to feel exactly where the wall is before I reach for anything. I have an objective that is an average over a fixed, finite set of examples, P(w) = (1/n) Σ_i ψ_i(w), each ψ_i a smooth convex per-example loss and P strongly convex. Two ways to descend it, and they are infuriatingly complementary. Full gradient descent steps along ∇P(w) = (1/n) Σ_i ∇ψ_i(w); with a constant step η < 1/L it contracts the gap by a fixed factor every iteration, a linear rate O((1 − γ/L)^t), so it reaches accuracy ε in about κ ln(1/ε) iterations where κ = L/γ. Beautiful rate, but every single iteration touches all n examples, so the real cost is n κ ln(1/ε) gradient evaluations, and for large n that is a lot of work per step. The other way, stochastic gradient descent, draws one example i_t uniformly and steps w ← w − η_t ∇ψ_{i_t}(w). Cost per step is one gradient, independent of n, gorgeous — and it's an honest unbiased estimate of the descent direction, E_i[∇ψ_i(w)] = ∇P(w). But its rate is only O(1/t), sublinear, and worse, I have to shrink the step η_t toward zero or it doesn't even converge. So I'm forced to choose between cheap-but-slow and fast-but-expensive, and I refuse to accept that the average-of-finite-functions structure really demands this.

So why is SGD stuck? I want the mechanism, not the folklore. The cleanest place to look is the optimum itself. At w* we have ∇P(w*) = 0 — that's what optimum means for P. But the individual ∇ψ_i(w*) are, in general, *not* zero; only their average is. Picture standing exactly at w* and running an SGD step: I draw some i, and ∇ψ_i(w*) ≠ 0, so I take a nonzero step and walk *away* from the solution. The method does not stop at the answer even when handed the answer. Let me make sure this isn't just a story I'm telling myself — I'll build a tiny finite sum and look. Take a least-squares finite sum, ψ_i(w) = ½(a_iᵀw − b_i)² + (λ/2)‖w‖² with λ = 0.5, n = 8 examples in d = 3 drawn at random, P = (1/n) Σ ψ_i. Solve for w* in closed form: ‖∇P(w*)‖ comes out 3·10⁻¹⁷, machine zero, so that's a true optimum. Now the leftover I claimed: σ² = (1/n) Σ_i ‖∇ψ_i(w*)‖². I get σ² = 0.546 — emphatically nonzero. So the picture is literally true on this instance: sitting at the exact solution, the average per-example squared gradient is 0.55, and a single draw moves me off it.

Now trace what that does to the iterate. Take the SGD recursion and expand the squared distance to w*: E‖w_t − w*‖² = ‖w_{t-1} − w*‖² − 2η (w_{t-1} − w*)^T ∇P(w_{t-1}) + η² E‖∇ψ_{i_t}(w_{t-1})‖². Strong convexity makes the middle term a genuine contraction proportional to ‖w_{t-1} − w*‖², but the last term has a piece that doesn't vanish as w_{t-1} → w*: near the optimum E‖∇ψ_i(w_{t-1})‖² → σ² > 0. So the recursion looks like E‖w_t − w*‖² ≲ (1 − 2ηγ + …)‖w_{t-1} − w*‖² + η² σ², and its fixed point is not zero — it's a floor of order η σ² / γ. The iterate should contract down to a ball of that radius and then rattle around inside it. Let me confirm the floor is real and not an artifact of my bookkeeping: I run constant-step SGD (η = 0.1/L, here L ≈ 7.5) on the same instance and watch P(w) − P(w*). Across 12 epochs of 2n steps it goes 1.7e-2, 1.7e-2, 8.9e-3, 8.0e-3, 7.3e-3, 1.6e-3, 1.0e-3, 1.8e-3, 1.3e-3, 1.4e-3, 3.0e-4, 4.2e-4 — it drops a bit, then sticks and bounces noisily around 10⁻³, going *up* as often as down. That's the floor, observed: a constant step cannot get below the noise level set by σ². To shrink the ball I have to shrink η; but a shrinking η also weakens the contraction, and the well-known balance lands me at η_t = O(1/t) and the rate at O(1/t). So the obstruction is the variance σ², not any bias — the bouncing is symmetric, not a systematic pull. If I could make that per-step variance shrink as the iterate settles, I could keep a constant, big step and the floor would go to zero with it.

Before I get excited, I should check whether this is even possible — maybe O(1/t) is simply the law. And there *is* a law: in the oracle model where the algorithm is only ever allowed to ask for *unbiased noisy gradient measurements*, O(1/t) is optimal for strongly convex problems (Nemirovski & Yudin 1983; Nemirovski et al. 2009). So I cannot beat it with that information alone. Stare at that qualifier though — "unbiased noisy measurements," meaning the oracle hands me a fresh anonymous sample each time and I know nothing else. My problem is *not* that oracle. My functions are a fixed finite list ψ_1, …, ψ_n; the same n of them recur; sampling is from that fixed set, not a bottomless fresh stream. The lower bound assumes I can't ever reuse structure across samples, but I can: I can, now and then, look at *all* of them. SAG and SDCA already proved this loophole is real — they get a linear rate at SGD-like per-step cost on finite sums — so the question isn't *whether* but *how*, and how to do it without their cost.

What is their cost, precisely, and what does each really do? SAG (Le Roux, Schmidt & Bach) keeps a table of one stored gradient y_i per example and steps along the running *average* of the whole table, w ← w − (η/n) Σ_i y_i, refreshing only the single y_{i_t} it just recomputed. So every step carries information about all n examples — the averaged direction is close to the true ∇P — at the cost of one fresh gradient. That's the randomized version of IAG (Blatt, Hero & Gauchman 2007), which did the same with cyclic instead of random index choice and could only be analyzed for quadratics. SDCA (Shalev-Shwartz & Zhang) optimizes the dual of the regularized problem one coordinate at a time and stores the n dual variables. Both get linear convergence. But both pay an O(n)-sized memory bill: a gradient (or, only for linear models ψ_i(w) = φ_i(w^T x_i) where the gradient is a scalar times x_i, a single scalar) per example, or n duals. On a structured-prediction loss or a neural network there is no such table I can afford, and SDCA's dual machinery doesn't even make sense there. And honestly SAG's convergence proof is a tangle — a joint Lyapunov function bounding gradients and iterates simultaneously — which leaves me without a clean picture of *why* the rate is linear. I want the mechanism that SAG's averaged table is implicitly exploiting, stated plainly, so I can get the same effect without the table.

Let me name what SAG's averaged-table step is doing in one sentence: it replaces the high-variance single-sample direction with something whose variance shrinks as the iterate settles, because as the stored gradients all become recent and consistent, the averaged step approaches the true ∇P. That's variance reduction. And there's a completely standard tool for exactly "reduce the variance of an estimator of E[X] when you have side information," from Monte Carlo: if I want to estimate the mean of a random quantity X and I have another quantity Y that is correlated with X and whose mean E[Y] I happen to know exactly, then I form X − Y + E[Y]. Check its mean: E[X − Y + E[Y]] = E[X] − E[Y] + E[Y] = E[X], so it estimates the same thing as X, unbiased, for *any* such Y. Now its variance: Var(X − Y + E[Y]) = Var(X − Y) = Var(X) + Var(Y) − 2 Cov(X, Y). If X and Y are highly correlated, that 2 Cov term eats most of Var(X) + Var(Y) and the corrected estimator has *much* smaller variance than X alone. The art is choosing a Y that (a) is strongly correlated with the thing I'm sampling, and (b) has a mean I can actually compute. This is the lever. (Greensmith, Bartlett & Baxter 2004 used precisely this idea to tame gradient-estimate variance in another setting, so it's a known move for gradients, not just for scalar Monte Carlo.)

So: my X, the noisy thing I sample, is the SGD direction ∇ψ_i(w) at the current point w. I need a Y, indexed by the *same* i so it's correlated with X, whose average over i I can compute. The most natural correlated companion: the gradient of the *same* example at some *reference* point w̃ that I'm holding fixed, Y = ∇ψ_i(w̃). Why would X and Y be correlated? Because they're the same function's gradient at two nearby arguments — by smoothness, ∇ψ_i(w) − ∇ψ_i(w̃) ≈ ∇²ψ_i · (w − w̃), so when w is close to w̃ the two gradients move together example-by-example; the noisy idiosyncrasy of "example i happens to have a big gradient" is shared between X and Y and cancels in X − Y. And do I know E_i[Y]? E_i[∇ψ_i(w̃)] = (1/n) Σ_i ∇ψ_i(w̃) = ∇P(w̃). I can compute that — it costs one full pass over the data at the fixed reference w̃, but it's exact. Call it μ̃ = ∇P(w̃). Then my variance-reduced direction is

  g = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃.

Two things I have to check immediately. First, is it still an honest descent direction? E_i[g] = ∇P(w) − ∇P(w̃) + μ̃ = ∇P(w) − ∇P(w̃) + ∇P(w̃) = ∇P(w). Yes — unbiased for ∇P(w), for *any* reference w̃ I choose. So I can swap this g straight into the SGD update, w ← w − η g, and in expectation it's still gradient descent on P. That's the safety rail: whatever reference I pick, I'm never optimizing the wrong thing.

Second — and this is what matters — does the variance actually go to zero where SGD's didn't? Recall SGD's trouble was at the optimum: ∇ψ_i(w*) ≠ 0, a constant offset that never died. Look at g there. Suppose the iterate w → w* and I also arrange the reference w̃ → w*. Then ∇ψ_i(w) − ∇ψ_i(w̃) → ∇ψ_i(w*) − ∇ψ_i(w*) = 0 example-by-example, and μ̃ = ∇P(w̃) → ∇P(w*) = 0. So algebraically g → 0. The very offset ∇ψ_i(w*) — the part that's specific to example i and doesn't average away — now appears in *both* ∇ψ_i(w) and ∇ψ_i(w̃) at the same i and seems to cancel against itself, leaving only the part of the gradient that genuinely depends on how far w is from the optimum.

I should put a number on that, because "→ 0" on paper and "small in practice" are different claims, and the whole method lives or dies here. On the same tiny instance, sit at a point w near the optimum (w = w* + 10⁻³·noise) and take the snapshot at the same place, w̃ = w. Then I compute, over all i, the mean squared length of the two candidate directions. The plain SGD direction ∇ψ_i(w) averages ‖·‖² = 0.546 — exactly σ², it hasn't shrunk at all, the offset is fully present. The control-variate direction ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃ averages ‖·‖² = 5.2·10⁻⁶. Five orders of magnitude smaller, at the *same* w. So the cancellation is not a paper trick; the per-example offset that SGD carries at full strength is genuinely gone here, leaving only the O(‖w − w*‖²) residual. That's the cancellation SGD couldn't do, because SGD had nothing to subtract. As g → 0, the step η g → 0 *on its own*, with no need to shrink η. So a constant, large η should survive — I'll want to confirm that on a full run, not just at one point.

There's a second way to see that g is "the right thing," which I like because it tells me I haven't smuggled in anything ad hoc. Define an auxiliary per-example function ψ̃_i(w) = ψ_i(w) − (∇ψ_i(w̃) − μ̃)^T w — I've just subtracted a *linear* function of w from each ψ_i, with the slope being the per-example deviation of the reference gradient from its mean. A linear shift changes the gradient by a constant: ∇ψ̃_i(w) = ∇ψ_i(w) − (∇ψ_i(w̃) − μ̃), which is exactly g. And what did I do to the objective? Average the shifts: (1/n) Σ_i (∇ψ_i(w̃) − μ̃) = μ̃ − μ̃ = 0, the slopes cancel in the mean, so (1/n) Σ_i ψ̃_i(w) = (1/n) Σ_i ψ_i(w) = P(w). The averaged objective is *unchanged*. So g is nothing but the plain SGD direction on a cleverly re-centered representation of the very same P — re-centered so that each example's gradient at the reference point sits at the common mean. Same problem, lower per-sample variance. That makes me confident this is principled, not a trick.

Now the engineering question: what is w̃, and how often do I refresh it and recompute μ̃? The exact mean μ̃ = ∇P(w̃) is the expensive part — one full pass over the data — so I can't recompute it every step or I've just rebuilt full gradient descent and thrown away the cheap-step win. But I also can't fix w̃ forever, because the variance reduction is only good while w stays close to w̃ (that's where ∇ψ_i(w) and ∇ψ_i(w̃) are correlated); let w drift far from a stale w̃ and the cancellation weakens. The resolution writes itself: amortize. Work in epochs. At the start of an epoch, snapshot the current parameters as w̃, pay once for μ̃ = ∇P(w̃) in a single full pass, then run m cheap inner steps, each using a freshly drawn i_t and the *same* held (w̃, μ̃):

  w_t = w_{t-1} − η (∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃),

and at the end of the epoch refresh w̃ from the inner iterates and recompute μ̃. As the whole thing converges, each epoch's w̃ creeps toward w*, μ̃ shrinks, and the inner variance shrinks with it — a constant step the entire time. The memory cost is the punchline: I store w̃ (one parameter vector) and μ̃ (one gradient vector), both O(d). No per-example table, no duals. Exactly the bill SAG and SDCA refused to keep small, and it's small because the reference is a *single shared snapshot* rather than n separately stored gradients. That immediately frees the method from the linear-model special case: it doesn't care whether ∇ψ_i has the φ_i'(w^T x_i) x_i structure, so it works for structured losses and for neural nets, where the per-example table was the dealbreaker.

Let me count the cost honestly, because the accounting axis is gradient-evaluations-per-n. An epoch costs n for the μ̃ pass, plus, per inner step, two example-gradients — one at w_{t-1} and one at w̃ — so 2m, total n + 2m. (For a linear model I *could* cache the n scalars φ_i'(w̃^T x_i), one for each example at the snapshot, and avoid recomputing ∇ψ_i(w̃) during the inner loop; but in general I pay the two gradients.) So I want m on the order of n — large enough that the one-time n cost of μ̃ is amortized over many cheap steps, not so large that w drifts hopelessly far from w̃. m = O(n) it is; something like m = 2n for convex problems and a bit larger, m = 5n, for nonconvex, where the landscape is rougher and a longer inner run helps. Two cheap things to note about refreshing w̃: setting it to the last inner iterate w_m is the obvious practical choice, or I could set it to the average of the inner iterates. There's a subtlety the analysis will force me to confront about *which* choice is provable, but for running the algorithm, last-iterate is natural.

Time to prove the linear rate, because "the variance goes to zero" is intuition and I want a theorem. I'll prove: under each ψ_i convex and L-smooth, P γ-strongly convex with γ > 0, if m is large enough that

  α = 1/[γ η (1 − 2Lη) m] + 2Lη/(1 − 2Lη) < 1,

then E[P(w̃_s) − P(w*)] ≤ α^s [P(w̃_0) − P(w*)] — geometric decay of the suboptimality across epochs s. The whole proof has to hinge on turning "variance" into "function-value gap," because the gap is what I want to contract. So the first thing I need is a lemma that bounds a squared gradient *difference* by a function-value gap.

Fix an example i and define g_i(w) = ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*). This is ψ_i with its value and its tangent plane at w* subtracted off — the Bregman-divergence-looking remainder. Its gradient is ∇g_i(w) = ∇ψ_i(w) − ∇ψ_i(w*), so ∇g_i(w*) = 0, meaning w* is a stationary point of g_i; and g_i is convex (it's ψ_i minus an affine function) with the same smoothness constant L as ψ_i, so w* is its global minimizer and g_i(w*) = 0, with g_i ≥ 0 everywhere. Now use smoothness on g_i the standard way — take one gradient step on g_i from w with step η' and use the smoothness upper bound:

  0 = g_i(w*) = min_v g_i(v) ≤ min_{η'} g_i(w − η' ∇g_i(w)) ≤ min_{η'} [ g_i(w) − η' ‖∇g_i(w)‖² + 0.5 L η'² ‖∇g_i(w)‖² ].

The bracket is a quadratic in η'; minimizing over η' gives η' = 1/L and value g_i(w) − (1/2L)‖∇g_i(w)‖². So 0 ≤ g_i(w) − (1/2L)‖∇g_i(w)‖², i.e. ‖∇g_i(w)‖² ≤ 2L g_i(w), which spelled out is

  ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L [ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*)].

Sum over i, divide by n, and notice that the linear term averages to zero: (1/n) Σ_i ∇ψ_i(w*)^T(w − w*) = ∇P(w*)^T(w − w*) = 0 since ∇P(w*) = 0. That leaves

  (1/n) Σ_i ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L [P(w) − P(w*)].   (8)

This is the bridge: the average squared deviation of the per-example gradients from their optimum values is controlled by how suboptimal w is. As w → w*, the left side → 0. Before I lean on it, let me check (8) numerically on the tiny least-squares instance, since I derived it through a min-over-η' step that's easy to get wrong. Drawing five random w at various distances from w* and comparing both sides: lhs 0.14 vs rhs 0.93; 1.4 vs 5.0; 3.1 vs 16.5; 12.4 vs 29.1; 11.8 vs 22.3. The inequality holds every time, and not tightly — there's slack, as a 2L bound should have. So (8) is sound, and it's exactly the handle I need on the variance.

Now bound the second moment of the actual update direction. Let v_t = ∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃, the thing I step along, and condition on w_{t-1} so the only randomness is i_t. I want E‖v_t‖². The move is to insert ∇ψ_{i_t}(w*) and split v_t into two centered-ish pieces. Write

  v_t = [∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w*)] + [∇ψ_{i_t}(w*) − ∇ψ_{i_t}(w̃) + μ̃].

Call the first bracket a and the second b. Use ‖a + b‖² ≤ 2‖a‖² + 2‖b‖²:

  E‖v_t‖² ≤ 2 E‖∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w*)‖² + 2 E‖[∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)] − μ̃‖²,

where in the second term I flipped the sign inside the norm (harmless) and used μ̃ = ∇P(w̃). Now look hard at that second term. Let ξ = ∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*); its expectation over i_t is E[ξ] = ∇P(w̃) − ∇P(w*) = ∇P(w̃) = μ̃. So the second term is exactly E‖ξ − E[ξ]‖², a variance, and for any random vector E‖ξ − E[ξ]‖² = E‖ξ‖² − ‖E[ξ]‖² ≤ E‖ξ‖². That's the place where subtracting the known mean μ̃ *pays off in the algebra* — it turns a raw second moment into a variance that I can only-shrink by dropping the −‖E[ξ]‖². So

  E‖v_t‖² ≤ 2 E‖∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w*)‖² + 2 E‖∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)‖².

Both terms are now exactly the average-squared-deviation that (8) controls — the first at w_{t-1}, the second at w̃:

  E‖v_t‖² ≤ 4L [P(w_{t-1}) − P(w*)] + 4L [P(w̃) − P(w*)].   (★)

There it is in one line, and it's the heart of everything: the second moment of my update direction is bounded by L times the suboptimality at the current point *plus* the suboptimality at the reference. As both w_{t-1} and w̃ approach w*, the right side → 0, so E‖v_t‖² → 0. The direction's variance is provably squeezed to zero by progress — that is the variance reduction, now a theorem, not a hope. And note it needed no separate variance bound: smoothness-times-suboptimality *is* the variance bound here.

Now the per-step contraction in squared distance. E[v_t | w_{t-1}] = ∇P(w_{t-1}) (unbiasedness, shown earlier). Expand:

  E‖w_t − w*‖² = ‖w_{t-1} − w*‖² − 2η (w_{t-1} − w*)^T E[v_t] + η² E‖v_t‖²
             = ‖w_{t-1} − w*‖² − 2η (w_{t-1} − w*)^T ∇P(w_{t-1}) + η² E‖v_t‖².

Bound the cross term by convexity of P: −(w_{t-1} − w*)^T ∇P(w_{t-1}) ≤ P(w*) − P(w_{t-1}) (the function lies above its tangent, rearranged). And substitute (★) for the last term:

  E‖w_t − w*‖² ≤ ‖w_{t-1} − w*‖² − 2η [P(w_{t-1}) − P(w*)] + 4Lη² [P(w_{t-1}) − P(w*) + P(w̃) − P(w*)]
             = ‖w_{t-1} − w*‖² − 2η(1 − 2Lη) [P(w_{t-1}) − P(w*)] + 4Lη² [P(w̃) − P(w*)].

I'll need 1 − 2Lη > 0 for the middle coefficient to be a genuine *decrease*, i.e. η < 1/(2L). That's the step-size ceiling, and it's a *constant* ceiling — it does not shrink with t. Hold that thought.

Now telescope across the inner loop. Fix the epoch s, so w̃ = w̃_{s-1} is constant throughout, and w_0 = w̃. Sum the inequality over t = 1, …, m. The ‖·‖² terms telescope: Σ_t (‖w_{t-1} − w*‖² − ‖w_t − w*‖²) = ‖w_0 − w*‖² − ‖w_m − w*‖². Taking full expectation over all the history,

  E‖w_m − w*‖² + 2η(1 − 2Lη) Σ_{t=1}^m E[P(w_{t-1}) − P(w*)] ≤ E‖w_0 − w*‖² + 4Lmη² E[P(w̃) − P(w*)].

Here's where the choice of how to set the next reference w̃_s matters. If I pick w̃_s to be a *uniformly random* inner iterate w_t over t ∈ {0, …, m−1}, then by definition E[P(w̃_s) − P(w*)] = (1/m) Σ_{t=1}^m E[P(w_{t-1}) − P(w*)] — the average of the inner suboptimalities is exactly the expected suboptimality of the random pick, with no Jensen gap. *That's* the reason the analysis wants the random-iterate option even though last-iterate (or averaging) is what I'd run in practice: it makes the telescoped sum equal the quantity I'm trying to bound, cleanly. So Σ_t E[P(w_{t-1}) − P(w*)] = m E[P(w̃_s) − P(w*)]. Drop the nonnegative E‖w_m − w*‖² ≥ 0, and use w_0 = w̃ so E‖w_0 − w*‖² = E‖w̃ − w*‖². Strong convexity gives one more handle: P(w̃) − P(w*) ≥ (γ/2)‖w̃ − w*‖², i.e. ‖w̃ − w*‖² ≤ (2/γ)[P(w̃) − P(w*)]. Substitute:

  2η(1 − 2Lη) m E[P(w̃_s) − P(w*)] ≤ (2/γ) E[P(w̃) − P(w*)] + 4Lmη² E[P(w̃) − P(w*)]
                                  = (2/γ + 4Lmη²) E[P(w̃) − P(w*)].

Divide both sides by 2η(1 − 2Lη) m:

  E[P(w̃_s) − P(w*)] ≤ [ 1/(γ η (1 − 2Lη) m) + 2Lη/(1 − 2Lη) ] E[P(w̃_{s-1}) − P(w*)] = α E[P(w̃_{s-1}) − P(w*)].

Iterate over s and the geometric bound E[P(w̃_s) − P(w*)] ≤ α^s [P(w̃_0) − P(w*)] falls out. Done.

Now I have to make sure α can actually be made < 1, because a "linear rate" with α ≥ 1 is no rate at all. The two terms pull against each other in a way I can exploit. The second term, 2Lη/(1 − 2Lη), depends only on η: it's small when η is small (it's 0 at η = 0 and blows up as η → 1/(2L)), and it sets a hard requirement that I keep η comfortably below 1/(2L) — say 2Lη around 0.2, giving 0.2/0.8 = 0.25 for this term. The first term, 1/(γ η (1 − 2Lη) m), is then driven down by making m large: it's O(1/m) for fixed η. So the recipe is: pick η a constant fraction of 1/L to keep the second term a constant below 1, then take m a large-enough multiple of κ = L/γ to push the first term small enough that their sum is < 1. Concretely, the most telling case is when conditioning is bad, κ = L/γ = n. Take η = 0.1/L: then 2Lη = 0.2, 1 − 2Lη = 0.8, the second term is 0.25, and the first term is 1/(γ · 0.1/L · 0.8 · m) = L/(0.08 γ m) = κ/(0.08 m) = n/(0.08 m). Choosing m as a sufficiently large constant multiple of n — and I should check the arithmetic rather than eyeball it — m = 50n drives the first term to L/(0.08 γ · 50n) = κ/(4n) = n/(4n) = 0.25 at κ = n, so α = 0.25 + 0.25 = 0.5. (I plugged the numbers in to be sure: with L = 1, γ = 1/50, η = 0.1, m = 50·50, the formula α = 1/(γη(1−2Lη)m) + 2Lη/(1−2Lη) returns exactly 0.5.) What does that buy? To reach accuracy ε I need ln(1/ε)/ln(1/α) ≈ ln(1/ε)/ln 2 epochs, each costing n + 2m = O(n) gradient evaluations, so O(n ln(1/ε)) gradient computations total. Compare batch GD at this conditioning: κ ln(1/ε) = n ln(1/ε) iterations times n gradients each = n² ln(1/ε). That's a full factor of n.

The theorem says geometric decay; I haven't yet *seen* it, and a bound being < 1 on paper is not the same as the run actually contracting at a fixed ratio. So I run the full method on the tiny instance — same η = 0.1/L that floored SGD, m = 2n (I don't need 50n here; the conditioning is mild, κ ≈ 12) — and read off P(w) − P(w*) per epoch: 2.2e-2, 1.4e-2, 9.2e-3, 6.0e-3, 3.9e-3, 2.6e-3, 1.7e-3, 1.1e-3, 7.5e-4, 5.1e-4, 3.4e-4, 2.3e-4. The successive ratios are 0.65, 0.64, 0.65, 0.65, 0.66, 0.66, 0.66, 0.67, 0.67, 0.67, 0.67 — flat. That flatness is the signature of a *linear* rate: a fixed per-epoch contraction factor, here ≈ 0.66, with no sign of a floor. Side by side with the SGD run on the identical problem and step (which stalled at ~10⁻³ and bounced), the difference is unmistakable — same step size, one floors and one keeps falling geometrically. That's the claim made good, on a run rather than in a margin. This matches the rate SAG and SDCA achieve, but with O(d) memory instead of an O(n)-sized table, and the proof is short: smoothness turns the gradient deviation into a value gap (eq 8), the value gaps bound the direction's second moment (★), the second moment feeds a per-step distance contraction, telescoping plus strong convexity closes the epoch.

Two extensions I can read straight off the same machinery without re-deriving from scratch. If P is smooth and convex but *not* strongly convex (γ → 0), the strong-convexity substitution ‖w̃ − w*‖² ≤ (2/γ)[…] is unavailable, but the rest of the convex analysis can be adapted to yield an O(1/T) rate — better than SGD's O(1/√T) on smooth convex problems. And for a nonconvex objective like a neural network, the global theorem doesn't apply, because the lemma and the contraction proof used convexity; still, the estimator itself remains unbiased, E_i[g] = ∇P(w), and the matched-index cancellation that drives the update toward zero near a shared limiting point does not require a convex formula. The practical move is to warm-start with a few SGD steps to get near a good basin and then run the variance-reduced method to accelerate the local convergence; if that basin is locally strongly convex, the theorem applies locally and gives local geometric convergence with a constant step. So the same idea reaches exactly the problems — structured prediction, neural nets — where the stored-gradient methods couldn't go.

One more thing nags at me. SDCA gets a linear rate too, by completely different-looking bookkeeping — dual coordinates, convex conjugates. If my "vanishing variance" reading of the mechanism is right, SDCA ought to be doing the same thing under the hood; if it isn't, then I've found one gadget among several, not the underlying reason. Let me push on it. Take the regularized form P(w) = (1/n) Σ_i φ_i(w) + (λ/2)‖w‖². At the optimum, differentiating, the dual variables satisfy α_i* = −(1/λn) ∇φ_i(w*), and the primal is reconstructed as w = Σ_i α_i. SDCA's effective update direction on coordinate i, when you push it through, is proportional to ∇φ_i(w) + λn α_i. Now watch that quantity as (w, α) → (w*, α*): ∇φ_i(w) + λn α_i → ∇φ_i(w*) + λn α_i* = ∇φ_i(w*) − ∇φ_i(w*) = 0, for every i. So (1/n) Σ_i (∇φ_i(w) + λn α_i)² → 0 — the variance of SDCA's update direction goes to zero at the optimum, the same property that let me keep a constant step. So SDCA is a variance-reduction method too; its stored dual α_i plays the role my snapshot ∇ψ_i(w̃) plays — a per-example reference that cancels the offending offset as things converge. SAG is the same story with the stored per-example gradients as the (biased, table-based) reference. So the three are one mechanism — drive the update variance to zero so a constant step survives — wearing three different costumes, and mine is the one that keeps the reference as a single shared snapshot plus its mean, which is why it costs O(d) and not O(n), and why it ports to nonconvex models. The reading survived the check rather than just sounding good, which is what I wanted from it.

So let me write it as the algorithm I'd run, then as code. Parameters: inner-loop length m and a constant step η < 1/(2L). Initialize w̃_0. For each epoch s: set w̃ = w̃_{s-1}; compute μ̃ = (1/n) Σ_i ∇ψ_i(w̃) in one full pass; set w_0 = w̃; for t = 1 … m draw i_t uniformly and update w_t = w_{t-1} − η(∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃); then set w̃_s to w_m (last iterate, the practical choice) — or, the version the proof uses, a uniformly random inner iterate. The state held between steps is just w̃ and μ̃, both the size of the parameters.

```python
import torch


class FiniteSumOptimizer:
    """Variance-reduced stochastic optimizer for P(w) = (1/n) Σ_i ψ_i(w).

    Each epoch: snapshot w̃, pay one full pass for μ̃ = ∇P(w̃), then run m cheap inner
    steps along the control-variate direction
        v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃     (E_i[v] = ∇P(w), variance → 0 as w, w̃ → w*),
    so the step size η stays CONSTANT (no decay) and the rate is linear. Only w̃ and the
    single vector μ̃ are stored — O(d) memory, no per-example gradient table.
    """

    def __init__(self, problem, lr, inner_steps):
        self.problem = problem            # exposes grad_batch(w_state, idx), full_grad(), loss_batch(idx)
        self.params = problem.params      # the live parameters w
        self.lr = lr                      # constant η < 1/(2L)
        self.inner_steps = inner_steps    # m random inner updates per epoch
        self.snapshot = None              # w̃  (one parameter-sized copy)
        self.mu = None                    # μ̃ = ∇P(w̃)  (one parameter-sized vector)

    def _clone_params(self):
        return [p.data.clone() for p in self.params]

    def _set_params(self, state):
        for p, s in zip(self.params, state):
            p.data.copy_(s)

    def train_one_epoch(self):
        # --- snapshot: hold w̃ and pay once for the exact mean μ̃ = ∇P(w̃) (one full pass) ---
        self.snapshot = self._clone_params()
        self.mu = self.problem.full_grad()

        n, b = self.problem.n, self.problem.batch_size
        total_loss, n_batches = 0.0, 0

        for _ in range(self.inner_steps):
            idx = torch.randint(n, (b,))

            # ∇ψ_i at the current point w_{t-1}
            grad_cur = self.problem.grad_batch(self.params, idx)
            # ∇ψ_i at the SAME indices but at the snapshot w̃  (the correlated control variate)
            grad_snap = self.problem.grad_batch(self.snapshot, idx)

            # control-variate direction  v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃   (unbiased for ∇P(w))
            with torch.no_grad():
                for p, gc, gs, mu in zip(self.params, grad_cur, grad_snap, self.mu):
                    v = gc - gs + mu
                    p.data.add_(v, alpha=-self.lr)        # constant-step in-place update

            total_loss += float(self.problem.loss_batch(idx))
            n_batches += 1

        # refresh the reference from the last inner iterate (practical "option I")
        return {"avg_loss": total_loss / max(n_batches, 1), "full_grad_count": 1}
```

Tracing the causal chain end to end: I started boxed in by SGD's variance — the per-example gradients don't vanish at the optimum, so a constant step leaves a noise floor of order η σ²/γ and the rate collapses to O(1/t) unless η → 0, which is itself the disease. The O(1/t) lower bound only binds an *unbiased-measurement* oracle, and a finite fixed sum gives me more than that, so a faster rate is not forbidden. SAG and SDCA already cross the line but pay an O(n)-sized per-example table I can't afford on structured or neural objectives. Reading SAG's averaged step as "reduce the update variance" pointed me at the classical control variate X − Y + E[Y]: take X the current single-example gradient, Y the same example's gradient at a fixed reference w̃, and E[Y] = ∇P(w̃) = μ̃ computed exactly in one pass. The resulting g = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃ is unbiased for ∇P(w) for any w̃, and as both w and w̃ approach w* the matched-index difference cancels the per-example offset that floored SGD, so g → 0 and a *constant* step survives. Recomputing μ̃ exactly is the only expensive piece, so I amortize it in epochs — snapshot, one full pass for μ̃, m cheap inner steps, refresh — keeping only w̃ and μ̃, which is O(d), not O(n). The proof is short and mechanical: smoothness turns each squared gradient deviation into a value gap (eq 8); those gaps bound the update's second moment (★), which is the variance reduction made rigorous; the second moment drives a per-step contraction in ‖w − w*‖²; telescoping over the inner loop with the random-iterate reference plus strong convexity yields α-geometric decay per epoch, and α is pushed below 1 by η a constant fraction of 1/L and m = O(κ), which is O(n) in the κ = n comparison, giving O(n ln(1/ε)) gradient evaluations there — a factor of n past batch GD, matching the stored-table methods at O(d) memory. And the same vanishing-variance condition is exactly what SDCA's dual offset and SAG's stored gradients satisfy, so all three are one mechanism, with this one being the memory-light, model-agnostic member of the family.
