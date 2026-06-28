I keep coming back to the same picture and it keeps bothering me. I have a finite training set, n examples, and I want to minimize the average loss P(w) = (1/n) Σ_i ψ_i(w), with each ψ_i a smooth convex per-example loss and the whole average strongly convex — say from an L2 regularizer. There are exactly two things people reach for, and they sit at opposite corners of a trade-off that feels like it shouldn't have to exist.

If I run full gradient descent, w ← w − η∇P(w), and each ψ_i is L-smooth while P is γ-strongly convex, then a *constant* step size η < 1/L already gives me geometric convergence: P(w_t) − P(w*) shrinks like (1 − γ/L)^t. That's the Nesterov result, the clean linear rate. The cost is cut by a fixed fraction every step. Beautiful — except each of those steps is a full pass over all n examples. To get to accuracy ε I need about κ ln(1/ε) iterations, κ = L/γ the condition number, and each iteration is n gradients, so n·κ·ln(1/ε) gradient evaluations. For large n that's brutal.

If instead I run SGD, w ← w − η_t ∇ψ_{i_t}(w), each step is a single gradient — cost independent of n, which is the whole reason it dominates large-scale learning. And it's unbiased: averaging over the random index, E_i[∇ψ_i(w)] = ∇P(w), so on average I'm going the right way. But the rate collapses to O(1/k), sublinear, and I'm forced onto a decaying step size η_t = O(1/t) to get even that. So: linear but n-per-step, versus 1-per-step but sublinear. I want the bottom-left corner — cheap steps *and* a linear rate — and the folklore is that you can't have it.

Let me not accept the folklore. Why exactly is SGD stuck at O(1/k)? It's not the bias; it's unbiased. So it has to be the variance. Let me actually watch what happens at the optimum. At w = w*, the full gradient is zero, ∇P(w*) = 0. That's the whole point of being at the minimum. But the *individual* gradients are not zero there — in general ∇ψ_i(w*) ≠ 0. Each example still pulls in its own direction; it's only their average that cancels. So if I'm sitting exactly at w* and I take an SGD step, I pick a random i and move along −η∇ψ_i(w*), which is some nonzero vector. The method doesn't stop at the solution. It can't. Gradient descent stops at w* because ∇P(w*) = 0; SGD walks right out of it.

So let me put a number on that. Define σ² = E_i‖∇ψ_i(w*)‖², the average squared per-example gradient at the optimum — generically positive. Now look at how ‖w_t − w*‖² evolves under a constant-step SGD. Expanding,

E‖w_t − w*‖² = ‖w_{t−1} − w*‖² − 2η(w_{t−1} − w*)^T∇P(w_{t−1}) + η² E‖∇ψ_{i_t}(w_{t−1})‖².

Strong convexity makes the middle term a genuine contraction, roughly −2ηγ‖w_{t−1} − w*‖², and near the optimum the last term is about η²σ². So schematically E‖w_t − w*‖² ≲ (1 − 2ηγ)‖w_{t−1} − w*‖² + η²σ². Iterate that and it converges not to zero but to a fixed point of the recursion: ‖w − w*‖² ≈ η²σ² / (2ηγ) = ησ²/(2γ). A noise ball sits around the optimum and never shrinks, with squared radius on the order of ησ²/γ. The only way to send the ball to zero is η → 0, and η → 0 destroys the rate.

I should pin σ² down on something concrete before I trust this story, because everything downstream hinges on σ² being genuinely nonzero. Take a tiny one-dimensional finite sum I can compute by hand: four quadratics ψ_i(w) = 0.5a_i(w − c_i)² + 0.15w² with a = (1, 2, 1.5, 0.5), c = (3, −1, 2, 0.5). The minimizer solves (1/4)Σa_i(w − c_i) + 0.3w = 0, which gives w* = 0.6855, and full_grad(w*) comes out to 0 as it must. But the four per-example gradients at w* are ∇ψ_i(w*) = (−2.109, 3.577, −1.766, 0.298): nonzero, summing to zero, and σ² = E_i‖∇ψ_i(w*)‖² = 5.11. So at the exact optimum, plain SGD picks one of those four vectors at random and takes a step of size η×(something with squared magnitude ~5) — it cannot sit still. The noise floor is real and I've now watched it. The thing that caps SGD is the *variance* η²σ², not any bias; the bias is zero throughout.

And I should be honest that this isn't a tuning failure I can out-clever within the same setting. If all I'm allowed is unbiased measurements of the gradient, O(1/k) is the best possible rate for strongly convex problems — that's the Nemirovski–Yudin lower bound, sharpened by Nemirovski, Juditsky, Lan and Shapiro. With unbiased gradient samples alone you cannot beat 1/k. So I can't just be cleverer about step sizes. I need to change the information I'm using.

Where's the slack? The lower bound assumes an endless stream of fresh, independent unbiased samples. But my problem isn't that. I have *n* examples, a fixed finite set, and I revisit them. That's extra structure the lower bound doesn't get to assume — n is known, the ψ_i are the same functions every epoch, I can in principle touch every one of them. SAG and SDCA already exploited exactly this and broke the 1/k barrier on finite sums: both get a genuine linear rate. So linear *is* achievable here; the question is at what cost.

Let me look hard at *how* they do it, because the mechanism is the thing I want. Take SAG, the stochastic average gradient of Le Roux, Schmidt and Bach. It keeps a table: y_i = the most recent gradient it ever computed for example i. Each step it picks a random i_k, refreshes only that one entry, y_{i_k} ← ∇ψ_{i_k}(x_k), leaves the other n−1 stale, and steps along the average of the whole table, x ← x − (α/n) Σ_i y_i. Per step it computes a single new gradient — cheap, like SGD. But the *direction* is the average of n stored gradients, like full GD. And here's the mechanism: as x → x*, every freshly computed entry y_i → ∇ψ_i(x*), the table average → ∇P(x*) = 0, so the update shrinks to zero on its own. No noise ball. That's why a constant-order step works and the rate is linear.

SDCA does the same thing in dual clothes. For P(w) = (1/n)Σφ_i(w) + 0.5λ‖w‖² it maintains dual variables α_i with w = Σ_i α_i, and at the optimum α_i* = −(1/λn)∇φ_i(w*). Its effective per-step direction is ∇φ_i(w) + λnα_i, and as (w, α) → (w*, α*) that goes to zero. So (1/n)Σ_i‖∇φ_i(w) + λnα_i‖² → 0 — again the per-step quantity vanishes at the optimum. Different bookkeeping, identical effect: a stored per-example quantity that, near w*, exactly cancels the offset that was σ².

Lining the two up, the difference between them starts to look cosmetic. SAG stores a per-example gradient and steps along the table average; SDCA stores a per-example dual and its effective direction is ∇φ_i(w) + λnα_i. In both, near the optimum the per-step quantity goes to zero because a stored per-example value cancels the part of ∇ψ_i that *doesn't* vanish there. So what they actually have in common isn't "average gradients" or "duals" — it's that each carries a per-example memory whose job is to subtract off the persistent offset, leaving a direction that dies at w*. Reduce the variance of the stochastic direction so it goes to zero at the optimum, and a constant step size converges. That's the property I want; the storage is the part I want to get rid of.

But both pay the same toll, and it's a heavy one for me. SAG stores n gradients — an n×d table. SDCA stores n duals. For least squares or plain logistic regression that compresses (the gradient is a scalar times x_i, so you can keep one number per example), and it's tolerable. But I care about structured-prediction models and neural networks, where ∇ψ_i is a full dense parameter-sized vector with no such compression, and an n×d table is simply impractical. So I can't just use SAG. I need its vanishing-variance effect *without* the per-example table.

Let me also note something that nags me about SAG specifically: its update direction is biased. The stored average is not, in expectation, ∇P(x_k) — most of the entries are stale, evaluated at old iterates. It still converges, but the bias is presumably why its proof is so intricate and its constants are loose. If I'm going to redesign this, I'd rather keep the direction *unbiased*. Unbiased is what makes everything provable cleanly.

So let me state precisely what I want from a stochastic direction g, built from a single sampled gradient per step:

  (i) unbiased: E_i[g] = ∇P(w);
  (ii) vanishing variance: E‖g‖² → 0 as w → w*;
  (iii) cheap: O(1) gradients per step, O(d) memory — no n-sized table.

If I can build that, the noise-ball argument above runs in reverse: the η²·(variance) term dies on its own as I approach w*, so I can hold η constant and still contract all the way to zero. Linear rate, cheap steps.

How do I drive the variance of a single sampled gradient to zero while keeping it unbiased? This is exactly the shape of a problem the Monte Carlo people solved long ago — control variates. If I want to estimate E[X] and I happen to have another random variable Y, correlated with X, *whose mean E[Y] I can compute*, then

  g = X − Y + E[Y]

is unbiased for *any* such Y, because E[g] = E[X] − E[Y] + E[Y] = E[X]. And its variance is Var(g) = Var(X) + Var(Y) − 2Cov(X, Y), which is small exactly when X and Y are strongly correlated. The whole game is to find a Y that tracks X closely and has a known mean. That's the lever.

Now map it onto my gradient. X should be the SGD direction itself, X = ∇ψ_i(w), with E_i[X] = ∇P(w) — that's the thing I want an unbiased estimate of. What's a Y that's correlated with ∇ψ_i(w) and whose mean over i I can actually compute? The same per-example gradient ψ_i, but evaluated at some *fixed reference point* w̃ that I'll keep close to w: Y = ∇ψ_i(w̃). Why is that a good Y? Because for a fixed i, ∇ψ_i(w) and ∇ψ_i(w̃) are the *same smooth function* evaluated at two nearby points — when w and w̃ are close they're nearly equal, so X and Y are tightly correlated, which is precisely the regime where the control variate kills variance. And its mean is computable: E_i[∇ψ_i(w̃)] = (1/n) Σ_i ∇ψ_i(w̃) = ∇P(w̃). I'll call that μ̃ = ∇P(w̃). One full pass over the data computes it exactly.

Drop it into the control-variate form:

  g = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃,    μ̃ = ∇P(w̃).

Let me check the two properties I demanded. Unbiased: take E over i, E_i[g] = ∇P(w) − ∇P(w̃) + μ̃ = ∇P(w) − μ̃ + μ̃ = ∇P(w). Exactly the full gradient, no bias — and notice that's *why* I add μ̃ back rather than just using ∇ψ_i(w) − ∇ψ_i(w̃): the correction term restores the mean. Without it I'd be subtracting something with nonzero mean and biasing the whole thing toward the wrong point.

Vanishing variance: as the iterate w → w* and the reference w̃ → w* as well, μ̃ = ∇P(w̃) → ∇P(w*) = 0, and ∇ψ_i(w) − ∇ψ_i(w̃) → ∇ψ_i(w*) − ∇ψ_i(w*) = 0 term by term, so g → 0. The mechanism is the same-index pairing: because I subtract ∇ψ_i at the same i, the persistent per-sample offset ∇ψ_i(w*) — the very thing that was σ² and kept SGD jittering — cancels against itself.

I want to actually watch this happen, and I want to test the claim I'd otherwise just assert: that pairing the *same* index is what reduces variance, and that a different index would make it worse. Back to the four-quadratic toy, where I already have σ² = 5.11. Put both w and w̃ near w* at distance δ — say w = w* + δ, w̃ = w* + 0.5δ — and tabulate the variance of three directions over the four possible draws: plain SGD X = ∇ψ_i(w); the matched control variate g = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃; and a *mismatched* one g' = ∇ψ_i(w) − ∇ψ_j(w̃) + μ̃ where j is drawn independently of i (averaged over all i, j pairs).

  δ = 1.0:   Var(X) = 6.70   Var(g_match) = 0.0781   E‖g_mismatch‖² = 14.9
  δ = 0.1:   Var(X) = 5.24   Var(g_match) = 0.00078  E‖g_mismatch‖² = 10.4
  δ = 0.01:  Var(X) = 5.12   Var(g_match) = 7.8e-6   E‖g_mismatch‖² = 10.2

Three things land at once. Plain SGD's variance does not move as δ shrinks — it sits at σ² ≈ 5.1, the floor I derived. The matched control variate's variance falls like δ²: 0.078 → 0.00078 → 7.8e-6, exactly a factor of 100 each time I shrink δ by 10. And the mismatched version is *larger* than plain SGD (≈10–15) and, tellingly, does not shrink with δ at all — which is the Var(X) + Var(Y) outcome I'd expect when Cov(X, Y) ≈ 0. So the matched index isn't a detail; swapping it for an independent index destroys the whole effect and actually hurts. I also confirmed the mean: E[g_match] equals full_grad(w) to machine precision at every δ, so the variance reduction costs nothing in bias.

And this needs no per-example table. To form g I only need w̃ and the single vector μ̃. That's O(d) memory, not O(nd) — the vanishing-variance effect SAG and SDCA got with their tables, but with O(d) storage.

There's a price, of course, and I should name it. Y's mean E[Y] = μ̃ = ∇P(w̃) is an *exact* full gradient at w̃; computing it is one full pass over the n examples. If I recomputed μ̃ every step, that's just full gradient descent again — I've defeated the purpose. So I must *amortize*: fix w̃ for a while, pay the one full pass to get μ̃, then take many cheap inner steps that all reference the same (w̃, μ̃), and only occasionally refresh w̃ to a newer, better point. An epoch structure. Snapshot w̃, one full pass for μ̃, then m cheap stochastic steps, then re-snapshot. As long as the inner steps keep w close to w̃, the control variate stays effective and the variance stays low.

Let me write the algorithm down concretely. Parameters: an inner length m and a constant step η. Initialize w̃_0. Then for each outer stage s = 1, 2, …: set w̃ = w̃_{s−1}; compute μ̃ = (1/n) Σ_i ∇ψ_i(w̃) in one pass; set w_0 = w̃; and for t = 1, …, m pick a random index i_t and update

  w_t = w_{t−1} − η(∇ψ_{i_t}(w_{t−1}) − ∇ψ_{i_t}(w̃) + μ̃).

At the end of the stage I need to pick the next reference w̃_s from the inner trajectory. The natural practical choice is w̃_s = w_m, the last inner iterate — or an average of the inner iterates. The cost of a stage is n for μ̃ plus 2m for the inner steps (two gradients each, one at w_{t−1} and one at w̃), so I'll want m on the order of n — large enough that the m cheap steps earn back the full pass, not so large that w drifts far from w̃. Something like m = O(n). (For these convex linear models there's a nice shortcut: ∇ψ_i(w) = φ'_i(w^T x_i) x_i, so ∇ψ_i(w̃) is just a stored scalar φ'_i(w̃^T x_i) times x_i — the same per-example memory footprint as SAG, and I needn't recompute it each step. For a general nonconvex net I do have to recompute it, paying the second gradient.)

I notice I've also just re-derived something. If I define an auxiliary function ψ̃_i(w) = ψ_i(w) − (∇ψ_i(w̃) − μ̃)^T w, then since Σ_i (∇ψ_i(w̃) − μ̃) = 0, the average is unchanged: (1/n) Σ_i ψ̃_i(w) = P(w). And plain SGD on this *reparameterized* finite sum gives exactly my update, because ∇ψ̃_i(w) = ∇ψ_i(w) − (∇ψ_i(w̃) − μ̃). So what I'm doing is plain SGD on a problem I've reshaped to have the same minimizer but per-example gradients that all agree near w̃. Nice sanity check — it's not a hack, it's SGD on a better-conditioned representation.

Now the real question: does the variance actually decay fast enough to give me a *linear* rate, and with what constants? I need to prove it, and the proof has to convert "variance small" into "geometric contraction."

The bridge I need is to bound a per-example gradient *difference* by a function-value *suboptimality*, because suboptimality is what I'm trying to shrink. Fix i and define g_i(w) = ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*). Then ∇g_i(w) = ∇ψ_i(w) − ∇ψ_i(w*), and ∇g_i(w*) = 0, so w* minimizes g_i and g_i(w*) = 0. Since ψ_i is convex, g_i ≥ 0 everywhere, and since ψ_i is L-smooth, so is g_i. For any L-smooth function, taking one gradient step from w shrinks the value by at least (1/2L)‖∇g_i(w)‖²:

  0 = g_i(w*) ≤ min_η g_i(w − η∇g_i(w)) ≤ min_η [ g_i(w) − η‖∇g_i(w)‖² + 0.5Lη²‖∇g_i(w)‖² ],

and minimizing the quadratic in η at η = 1/L gives ≤ g_i(w) − (1/2L)‖∇g_i(w)‖². Rearranging,

  ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L[ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*)].

Sum over i, divide by n, and use ∇P(w*) = 0 so that the (1/n)Σ_i ∇ψ_i(w*)^T(w − w*) term is ∇P(w*)^T(w − w*) = 0:

  (1/n) Σ_i ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L[P(w) − P(w*)].    (★)

This is the lever: the average squared gradient *gap* is controlled by how suboptimal the value is. As P(w) → P(w*), all those gaps go to zero. Exactly the quantitative version of "the variance dies at the optimum."

Now bound the second moment of the actual update direction. Write v_t = ∇ψ_{i_t}(w_{t−1}) − ∇ψ_{i_t}(w̃) + μ̃, conditioned on w_{t−1}, expectation over i_t. I want E‖v_t‖² in terms of suboptimalities, so I'll insert ∇ψ_{i_t}(w*) and split into a current-point piece and a snapshot piece. Since μ̃ = ∇P(w̃) and ∇P(w*) = 0, I can write μ̃ = ∇P(w̃) − ∇P(w*), which lets the snapshot piece appear as a centered random vector:

  v_t = [∇ψ_{i_t}(w_{t−1}) − ∇ψ_{i_t}(w*)] − [ (∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)) − (∇P(w̃) − ∇P(w*)) ].

Call the first bracket a and the second bracket b, so v_t = a − b. Using ‖a − b‖² ≤ 2‖a‖² + 2‖b‖²,

  E‖v_t‖² ≤ 2 E‖∇ψ_{i_t}(w_{t−1}) − ∇ψ_{i_t}(w*)‖² + 2 E‖ (∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)) − ∇P(w̃) ‖².

The second term is exactly a centered random vector: let ξ = ∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*); then E_{i_t}[ξ] = ∇P(w̃) − ∇P(w*) = ∇P(w̃), so the bracket is ξ − E[ξ]. For any random vector, E‖ξ − E ξ‖² = E‖ξ‖² − ‖E ξ‖² ≤ E‖ξ‖². That variance-vs-second-moment inequality is the place I cash in being centered — dropping the −‖Eξ‖² only loosens the bound. So

  E‖v_t‖² ≤ 2 E‖∇ψ_{i_t}(w_{t−1}) − ∇ψ_{i_t}(w*)‖² + 2 E‖∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)‖².

Both terms are averages over i of squared gradient gaps, so (★) applies to each — the first at w_{t−1}, the second at w̃:

  E‖v_t‖² ≤ 4L[P(w_{t−1}) − P(w*)] + 4L[P(w̃) − P(w*)].    (♦)

This is the quantitative heart of it: the update's second moment is bounded by smoothness times the suboptimality at the current point *plus* at the snapshot. When both w_{t−1} and w̃ approach w*, the right side → 0, so E‖v_t‖² → 0 — the variance reduction is bounded by how close I am to the optimum, not merely hoped for.

Before I lean on (♦), let me sanity-check it numerically on the same four quadratics, where each ψ_i has smoothness a_i + reg, so L = max_i(a_i + 0.3) = 2.3. At w − w* = 1.0, w̃ − w* = 0.5 I get E‖v‖² = 2.48 against the bound 4L[P(w) − P*] + 4L[P(w̃) − P*] = 8.91 — holds, with the constant 4L being loose as expected from the two ‖a − b‖² ≤ 2‖a‖² + 2‖b‖² steps. At w − w* = 0.1, w̃ − w* = 0.07 it's E‖v‖² = 0.024 against 0.106 — still holds, and both sides have dropped by roughly the square of the distance, which is the decay the bound is supposed to certify. Good; (♦) is not just formally derived, it actually brackets the real second moment here.

Now turn that into a contraction. The direction is unbiased, E_{i_t}[v_t] = ∇P(w_{t−1}). Expand the squared distance:

  E‖w_t − w*‖² = ‖w_{t−1} − w*‖² − 2η(w_{t−1} − w*)^T E[v_t] + η² E‖v_t‖²
              = ‖w_{t−1} − w*‖² − 2η(w_{t−1} − w*)^T∇P(w_{t−1}) + η² E‖v_t‖².

For the inner product, convexity of P gives −(w_{t−1} − w*)^T∇P(w_{t−1}) ≤ P(w*) − P(w_{t−1}). And substitute (♦) for the last term:

  E‖w_t − w*‖² ≤ ‖w_{t−1} − w*‖² − 2η[P(w_{t−1}) − P(w*)] + 4Lη²[P(w_{t−1}) − P(w*) + P(w̃) − P(w*)]
              = ‖w_{t−1} − w*‖² − 2η(1 − 2Lη)[P(w_{t−1}) − P(w*)] + 4Lη²[P(w̃) − P(w*)].

The structure is exactly what I wanted. There's a genuine *negative* term proportional to the current suboptimality, with coefficient 2η(1 − 2Lη), which is positive as long as η < 1/(2L) — and that's the only step-size constraint, a constant one, no decay. The only thing standing in its way is the +4Lη²[P(w̃) − P(w*)] term, the residual from the snapshot. So the inner loop reduces the current suboptimality at the cost of a term tied to the *snapshot's* suboptimality, which is smaller and which the outer loop will keep shrinking. That coupling is the engine.

Hold the snapshot fixed for the whole stage, w̃ = w̃_{s−1}, and sum the inequality over the inner steps t = 1, …, m, taking full expectation. The ‖·−w*‖² terms telescope — each step's right-hand ‖w_{t−1} − w*‖² is the previous step's left-hand value — leaving only the endpoints:

  E‖w_m − w*‖² + 2η(1 − 2Lη) Σ_{t=1}^m E[P(w_{t−1}) − P(w*)] ≤ E‖w_0 − w*‖² + 4Lmη² E[P(w̃) − P(w*)].

Now I see exactly why the analysis wants the snapshot chosen as a *random* inner iterate rather than the last one. If w̃_s = w_t for a t drawn uniformly from {0, …, m−1}, then E[P(w̃_s) − P(w*)] = (1/m) Σ_{t=1}^m E[P(w_{t−1}) − P(w*)], so that whole sum collapses into a single clean term m·E[P(w̃_s) − P(w*)]. (In practice I'd take w̃_s = w_m or an average and it works fine, but this random-iterate option is what makes the telescoped sum turn into the next snapshot's suboptimality without any Jensen slack.) Using also w_0 = w̃ and dropping the nonnegative E‖w_m − w*‖²:

  2η(1 − 2Lη)m · E[P(w̃_s) − P(w*)] ≤ E‖w̃ − w*‖² + 4Lmη² E[P(w̃) − P(w*)].

Last ingredient: strong convexity converts the iterate distance of the snapshot into its suboptimality. From P(w) − P(w*) ≥ 0.5γ‖w − w*‖² at w = w̃, I get ‖w̃ − w*‖² ≤ (2/γ)[P(w̃) − P(w*)]. Substitute:

  2η(1 − 2Lη)m · E[P(w̃_s) − P(w*)] ≤ (2/γ + 4Lmη²) · E[P(w̃) − P(w*)].

Divide through by 2η(1 − 2Lη)m:

  E[P(w̃_s) − P(w*)] ≤ [ 1/(γη(1 − 2Lη)m) + 2Lη/(1 − 2Lη) ] · E[P(w̃_{s−1}) − P(w*)].

So with

  α = 1/(γη(1 − 2Lη)m) + 2Lη/(1 − 2Lη),

I have E[P(w̃_s) − P(w*)] ≤ α · E[P(w̃_{s−1}) − P(w*)], and iterating over stages,

  E[P(w̃_s) − P(w*)] ≤ α^s [P(w̃_0) − P(w*)].

If α < 1, that's geometric — *linear convergence with a constant step size*, exactly the corner I was told I couldn't reach. Let me make sure α < 1 is genuinely attainable and isn't quietly demanding η or m I can't have. Two terms. The second, 2Lη/(1 − 2Lη), is the floor from the snapshot residual; it's small when η is small (e.g. η = 0.1/L makes 2Lη = 0.2, so this term is 0.2/0.8 = 0.25). The first, 1/(γη(1 − 2Lη)m), is the term I beat down by making the inner loop long enough; it → 0 as m → ∞. So I pick η < 1/(2L) small enough that the second term is comfortably below 1, then take m large enough that the first term closes the gap. They trade off — too-small η helps the floor but slows the first term's decay in m, so there's a balance — but α < 1 is clearly reachable.

Let me put numbers on the indicative case where the conditioning is bad, κ = L/γ = n. Take η = 0.1/L: then 2Lη = 0.2, 1 − 2Lη = 0.8, the second term is 0.25. The first term is 1/(γ · 0.1/L · 0.8 · m) = L/(0.08 γ m) = κ/(0.08 m) = n/(0.08 m); if I want α = 1/2 with these displayed constants, I can take m = 50n so that the first term is also 0.25. So α = 0.25 + 0.25 = 0.5 < 1 — and I checked that arithmetic with κ = n = 100 plugged in directly, getting exactly 0.5. The important scaling is m = O(n): a fixed constant number of passes inside each stage makes the contraction a fixed constant. To reach accuracy ε I need O(ln(1/ε)) stages, each costing n + 2m = O(n) gradient evaluations, for a total of O(n ln(1/ε)). Compare that to batch gradient descent at this conditioning: n·κ·ln(1/ε) = n²ln(1/ε). I've turned n² into n. And it matches what SAG and SDCA achieve — but with a simpler argument and O(d) memory instead of an n×d table.

The α bound is conservative — those displayed constants come from a chain of loose inequalities — so before I believe the *rate* I should just run the thing and watch the suboptimality per stage. I take a small L2-regularized logistic regression, n = 200, d = 10, find w* to machine precision with a long GD run, and run the algorithm exactly as I wrote it: η = 0.3/L (comfortably below 1/(2L), here 1/(2L) ≈ 0.070 so η ≈ 0.042), m = 2n, refreshing w̃ to the last inner iterate. The per-stage residual P(w̃_s) − P(w*) reads:

  stage 1: 5.8e-3   stage 2: 4.7e-5   stage 3: 2.0e-6   stage 4: 2.1e-8   stage 5: 2.3e-10   ...   stage 8: 7.9e-15

Each stage knocks the residual down by a roughly constant factor (here a brutal ~0.01–0.04 per stage, far better than my conservative α = 0.5 — the bound is loose, as expected), and it's a *constant* ratio, not a slowing one: that's the geometric signature, on a log plot a straight line down to machine precision by stage 8. And the control: plain SGD with the *same constant η* and the same total number of gradient evaluations stalls at P − P* ≈ 2.7e-3 — it never gets below where SVRG already was after one stage. So the noise ball I derived at the start is exactly what SVRG escapes and SGD does not. The linear rate isn't only on paper; I watched the residual fall geometrically.

I want to step back and double-check the surprise I just stepped over, because it's the whole point. I argued earlier that with unbiased gradient samples alone, O(1/k) is optimal and can't be beaten. And every one of my v_t *is* an unbiased sample of ∇P(w_{t−1}). So how did I get linear? Because the lower bound is about a fixed *quality* of unbiased oracle — a fixed noise level σ². My oracle's noise isn't fixed; the variance bound (♦) says it *shrinks* as I approach the optimum, down to zero, because I'm reusing the finite set of functions through the snapshot. That's the structure the lower bound doesn't assume, and it's precisely the finite-sum structure SAG and SDCA were exploiting too. I'm not violating the bound; I'm leaving its model.

Looking back at SDCA through this lens, its per-step direction ∇φ_i(w) + λnα_i is, near the optimum, a quantity going to zero — and the stored dual α_i is playing the same role as my snapshot gradient: a per-example value that cancels the persistent offset so the update's variance vanishes. So SDCA was a variance-reduction method too; it hid the control variate inside the dual variables, and SAG hid a biased cancellation inside the stored gradient average. Once all three look like the same vanishing-variance idea, the only real design choice is *which* control variate, and the one I picked — a periodically refreshed snapshot gradient — is the one that needs no per-example storage. That's also why I keep the full +μ̃ term: unbiasedness is what gave the clean proof I just ran, with no bias terms to chase, and the numeric checks above (mean exact, variance ∝ δ², residual geometric) are what convince me it's right rather than merely plausible.

The constraints to remember are just: each ψ_i convex and L-smooth, P γ-strongly convex with γ > 0, and η < 1/(2L). For a smooth convex problem without strong convexity, the same machinery degrades gracefully to an O(1/T) rate, still better than SGD's O(1/√T). And for a nonconvex model like a neural net, the variance-reduction identity E_i[g] = ∇P(w) and the second-moment shrinkage still hold structurally; if I warm-start with a bit of SGD to get near a local minimum where the landscape is locally strongly convex, the same theorem applies locally and I get local geometric convergence with a constant step — which is exactly the regime where plain SGD crawls.

So the whole chain, start to finish: SGD is unbiased but its per-sample gradients don't vanish at the optimum, so a constant step leaves a squared-distance floor of order ησ²/γ and forces η → 0 and an O(1/k) rate — a variance floor, not a bias problem, and one the unbiased-oracle lower bound says is unbeatable *with that information*. Finite sums carry extra structure the lower bound ignores; the way to use it is a control variate: subtract the same example's gradient at a snapshot point w̃ and add back the snapshot's full gradient μ̃ = ∇P(w̃), giving g = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃, which is unbiased for any w̃ and whose variance is bounded by 4L times the suboptimalities at w and w̃, hence vanishes as both → w*. Amortize the cost of μ̃ over an epoch of m cheap inner steps, refresh the snapshot, and the telescoped one-step contraction plus strong convexity gives E[P(w̃_s) − P(w*)] ≤ α^s[P(w̃_0) − P(w*)] with α < 1 for a constant η < 1/(2L) and m = O(n) — linear convergence, O(n ln(1/ε)) gradient evaluations, O(d) memory, no per-example table.

```python
import numpy as np

def svrg(problem, w0, eta, m, n_outer):
    """Stochastic Variance Reduced Gradient.
    g = grad psi_i(w) - grad psi_i(w_tilde) + mu_tilde is an unbiased estimate of
    grad P(w) whose variance -> 0 as w, w_tilde -> w*, so a CONSTANT eta < 1/(2L)
    still gives geometric convergence."""
    w_tilde = w0.copy()
    for s in range(n_outer):
        # snapshot: one full pass for the exact mean of the control variate
        mu_tilde = problem.full_grad(w_tilde)          # mu_tilde = grad P(w_tilde)
        w = w_tilde.copy()                             # inner loop starts at the snapshot
        for t in range(m):
            i = np.random.randint(problem.n)
            # control-variate direction: matched index i makes grad psi_i(w) and
            # grad psi_i(w_tilde) cancel near w*, so the variance vanishes; + mu_tilde
            # keeps E_i[g] = grad P(w) (unbiased)
            g = problem.grad_i(w, i) - problem.grad_i(w_tilde, i) + mu_tilde
            w = w - eta * g                            # constant step size, no decay
        # refresh the reference point from the inner trajectory
        w_tilde = w                                    # option: last iterate (or average)
    return w_tilde
```
