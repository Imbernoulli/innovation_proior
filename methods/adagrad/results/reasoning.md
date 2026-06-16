Let me start from what actually hurts. I am running a subgradient method on text — bag-of-words, hundreds of thousands or millions of features, and within any single example only a handful are non-zero. I predict a weight vector x_t, suffer a convex loss f_t, read off a subgradient g_t, and I want my total regret R(T) = Σ_t f_t(x_t) − inf_x Σ_t f_t(x) to grow sublinearly so that, averaged over the stream, I do as well as the single best fixed predictor in hindsight. The standard thing is Zinkevich's projected gradient descent: x_{t+1} = Π_X(x_t − η g_t), with the decaying schedule η_t = η/√t, and that gives me √2 D_2 (Σ_t ||g_t||²_2)^{1/2}, on the order of √(dT). Fine in the worst case — Abernethy showed that √T rate is minimax tight, so I cannot beat it by being cleverer about the same isotropic algorithm. But it grates on me because of what it does to my data. Most of my coordinates are zero almost all the time; the gradient mass lives on a few features. And the features that fire rarely are exactly the discriminative ones — that is the whole TF-IDF intuition, rare terms carry the information. Yet one global η_t scales every coordinate identically: a feature I have seen ten thousand times gets the same step as one I have seen twice. When a rare feature finally fires, I want the learner to take notice, to move a lot on that coordinate; a single global rate physically cannot do that. So I do not want a better global rate. I want a per-coordinate rate, and I want it to come out of the data, not out of my hand-tuning.

How do the existing methods even let me influence the geometry? The cleanest handle is mirror descent. Think of plain gradient descent as linearize-and-stay-close: I replace f_t by its tangent and add a penalty that keeps me near x_t, x_{t+1} = argmin_x { η⟨g_t, x⟩ + ½||x − x_t||²_2 }, and setting the gradient to zero gives η g_t + (x_{t+1} − x_t) = 0, i.e. x_{t+1} = x_t − η g_t, exactly gradient descent. The squared Euclidean penalty is just one choice of "stay close." Replace it with a Bregman divergence B_ψ(x, y) = ψ(x) − ψ(y) − ⟨∇ψ(y), x − y⟩ of any strongly convex ψ, and I get the general mirror-descent step x_{t+1} = argmin_x { η⟨g_t, x⟩ + η φ(x) + B_ψ(x, x_t) }, with φ an optional fixed regularizer like an ℓ1 penalty. ψ = ½||·||²_2 gives projected gradient back; negative entropy gives the multiplicative-weights / exponentiated-gradient update. And the regret for any such fixed ψ has a known shape,

  R_φ(T) ≤ (1/η) B_ψ(x*, x_1) + (η/2) Σ_{t=1}^{T} ||g_t||²_{ψ*},

where ||·||_{ψ*} is the dual norm of the one ψ induces. The dual-averaging cousins — Nesterov's primal-dual method, Xiao's RDA — predict from the running average gradient instead and get the same shape, √T ψ(x*) + (1/(2√T)) Σ_t ||g_t||²_*. Different bookkeeping, same lesson.

Now stare at that bound for a second, because it is telling me where my leverage is. The data-dependent term is Σ_t ||g_t||²_{ψ*}, a sum of dual norms of the gradients, and that dual norm is fixed entirely by the single choice of ψ. Everyone picks ψ once — Euclidean for a box, entropy for the simplex — before a single gradient has arrived, and the same ψ governs every coordinate. So on my sparse heavy-tailed data the Euclidean ψ pays Σ_t ||g_t||²_2 which carries the full ambient dimension d, even though the gradient mass is concentrated on a few coordinates and most of the d dimensions contribute essentially nothing to the loss. The looseness is not in the analysis; it is in ψ being chosen blind. So here is the thing I keep circling back to: ψ is a free knob, and the bound is a function of ψ. Why am I fixing it in advance? Why not pick ψ to minimize the very bound it controls — and let it depend on the gradients I have actually seen, and even let it change over time as more gradients arrive? That reframes the whole problem. Instead of "run mirror descent with a chosen ψ," it becomes a meta-problem: choose the proximal function ψ_t itself, adaptively, to make Σ_t ||g_t||²_{ψ_t*} as small as the best ψ could in hindsight. I will let ψ_t grow monotonically, ψ_{t+1} ⪰ ψ_t, which is what the regret algebra will want for the divergence terms to telescope, and I will choose its shape from the data.

To make "choose ψ_t" concrete I need a parameterized family I can actually optimize over. Take ψ_t to be a squared Mahalanobis form, ψ_t(x) = ½⟨x, H_t x⟩ with H_t ⪰ 0 symmetric. Two reasons: the dual norm is then a clean quadratic, ||g||²_{ψ_t*} = ⟨g, H_t^{-1} g⟩, which I can differentiate; and the resulting update is just a rescaled-and-projected gradient step, no harder than what I already run. So the data-dependent regret term becomes Σ_t ⟨g_t, H_t^{-1} g_t⟩, and my meta-question sharpens to: which sequence of preconditioners H_t makes this sum small, and how small can it possibly be?

Let me first answer the hindsight question — forget online for a moment and ask, if I had all T gradients in front of me and had to commit to one fixed H = diag(s) (start diagonal, it is the cheap case and I want intuition), what s minimizes the gradient term? The problem is

  min_s  Σ_{t=1}^{T} Σ_{i=1}^{d} g_{t,i}² / s_i   s.t.  s ⪰ 0,  ⟨1, s⟩ ≤ c,

a budget c on the total "size" of the metric so the trivial s → ∞ is ruled out. Lagrangian, multipliers λ ⪰ 0 for s ⪰ 0 and θ ≥ 0 for the budget:

  L(s, λ, θ) = Σ_i ( Σ_t g_{t,i}² ) / s_i − ⟨λ, s⟩ + θ(⟨1, s⟩ − c).

Write Σ_t g_{t,i}² = ||g_{1:T,i}||²_2, the squared ℓ2 norm of the whole history of coordinate i (g_{1:T,i} is the i-th row of the concatenated gradient matrix). Take ∂L/∂s_i: −||g_{1:T,i}||²_2 / s_i² − λ_i + θ = 0. Complementary slackness forces λ_i s_i = 0; at any coordinate that carries gradient mass s_i > 0 so λ_i = 0, leaving −||g_{1:T,i}||²_2 / s_i² + θ = 0, i.e. s_i = θ^{−1/2} ||g_{1:T,i}||_2. The budget normalizes θ: s_i = c ||g_{1:T,i}||_2 / Σ_j ||g_{1:T,j}||_2. So the optimal denominator for coordinate i is proportional to ||g_{1:T,i}||_2 — the accumulated ℓ2 norm of that coordinate's gradients. That is a real surprise worth pausing on: the best fixed per-coordinate scale is not the count of times the feature fired, not the average gradient, but the square root of the sum of its squared gradients. And plug s back in to see how small the term gets:

  inf_s { Σ_t Σ_i g_{t,i}²/s_i : s ⪰ 0, ⟨1, s⟩ ≤ c } = (1/c) ( Σ_{i=1}^{d} ||g_{1:T,i}||_2 )².

The hindsight-optimal value is governed by Σ_i ||g_{1:T,i}||_2, the sum over coordinates of each coordinate's accumulated gradient norm. This is the quantity I want my online method to chase, because on sparse data it is tiny: a coordinate that rarely fires has a small ||g_{1:T,i}||_2, contributing almost nothing, whereas the isotropic bound charged me the full d.

So the target is set: I want a preconditioner whose denominator on coordinate i is ||g_{1:T,i}||_2. The trouble is I cannot use the full history — online, at round t I have only seen g_1,…,g_t. The honest thing is to set it incrementally: at round t use the running accumulated norm

  s_{t,i} = ||g_{1:t,i}||_2 = sqrt( Σ_{τ=1}^{t} g_{τ,i}² ),

and take H_t = δI + diag(s_t), where δ ≥ 0 is a small floor I will justify in a moment. Equivalently, writing G_t = Σ_{τ≤t} g_τ g_τ^T for the running outer-product matrix, the diagonal denominator is diag(G_t)^{1/2}, and the mirror-descent step with this changing metric is

  x_{t+1} = Π_X^{H_t} ( x_t − η H_t^{-1} g_t ),

projection in the metric induced by the current diagonal. Coordinate by coordinate, when X = R^d and I drop the projection, this is simply

  x_{t+1,i} = x_{t,i} − η g_{t,i} / ( δ + sqrt( Σ_{τ≤t} g_{τ,i}² ) ).

Look at what this does. A coordinate that fires constantly builds up a large Σ g², so its denominator is large and its effective step is small. A coordinate that has barely fired has a tiny denominator, so the first time it does fire it gets a big step. That is exactly the "take notice of a rare feature" behavior I wanted, and it dropped out of minimizing the regret bound, not out of a heuristic. The δ is there for one concrete reason: before a coordinate has seen any gradient mass its accumulated norm is sqrt(0) = 0, and I would be dividing by zero; δ (or, in a finite-precision implementation, a small additive constant) floors the denominator so the metric is always invertible. It needs to be small enough not to perturb the active coordinates, where Σ g² is well above it.

But I have to earn this — the incremental accumulated-norm denominator is not the hindsight optimum, it is a causal approximation to it, and I need to show I do not pay much for being causal. The thing I must control is Σ_t ⟨g_t, diag(s_t)^{−1} g_t⟩ = Σ_t Σ_i g_{t,i}² / ||g_{1:t,i}||_2, where each term uses only the norm up to t, not up to T. I want to bound this by something like 2 Σ_i ||g_{1:T,i}||_2 — twice the hindsight quantity. Reduce to one coordinate, a scalar sequence a_1,…,a_T with partial vectors a_{1:t} = [a_1 … a_t]; I claim

  Σ_{t=1}^{T} a_t² / ||a_{1:t}||_2 ≤ 2 ||a_{1:T}||_2,

with the convention 0/0 = 0. Induct on T. T = 1 gives a_1²/||a_1||_2 = a_1²/|a_1| = |a_1| ≤ 2|a_1|. Assume it for T−1. Split off the last term and use the hypothesis on the first T−1:

  Σ_{t=1}^{T} a_t²/||a_{1:t}||_2 = Σ_{t=1}^{T−1} a_t²/||a_{1:t}||_2 + a_T²/||a_{1:T}||_2
    ≤ 2 ||a_{1:T−1}||_2 + a_T²/||a_{1:T}||_2.

Write b_T = Σ_{τ=1}^{T} a_τ² = ||a_{1:T}||²_2, so ||a_{1:T−1}||_2 = sqrt(b_T − a_T²). I need to turn that sqrt(b_T − a_T²) into something I can cancel against sqrt(b_T). Concavity of the square root: for a concave function its graph lies below any tangent, and the tangent of sqrt at b_T is sqrt(b_T) + (1/(2 sqrt(b_T)))·(· − b_T), so evaluating at b_T − a_T² gives sqrt(b_T − a_T²) ≤ sqrt(b_T) − a_T²/(2 sqrt(b_T)) (valid as long as b_T − a_T² ≥ 0, which it is). Therefore

  2 ||a_{1:T−1}||_2 + a_T²/||a_{1:T}||_2 = 2 sqrt(b_T − a_T²) + a_T²/sqrt(b_T)
    ≤ 2( sqrt(b_T) − a_T²/(2 sqrt(b_T)) ) + a_T²/sqrt(b_T)
    = 2 sqrt(b_T) − a_T²/sqrt(b_T) + a_T²/sqrt(b_T) = 2 sqrt(b_T) = 2 ||a_{1:T}||_2.

The a_T² terms cancel exactly, and the induction closes. Summing the scalar bound over coordinates, Σ_t ⟨g_t, diag(s_t)^{−1} g_t⟩ ≤ 2 Σ_i ||g_{1:T,i}||_2. So the price for using the causal running norm instead of the unknown full-history norm is a factor of two. The whole adaptive scheme is provably within a constant of the best diagonal preconditioner I could have chosen knowing all the gradients in advance.

Now assemble the regret. I start, as always, from convexity: f_t(x_t) − f_t(x*) ≤ ⟨g_t, x_t − x*⟩, so R_φ(T) ≤ Σ_t ⟨g_t, x_t − x*⟩ (folding in the composite φ where present). With the changing Mahalanobis metric ψ_t(x) = ½⟨x, H_t x⟩, the mirror-descent machinery gives a bound of the form I had before but now with the time-varying ψ_t: the divergence head term plus the gradient term. Because I forced ψ_{t+1} ⪰ ψ_t — and H_t = δI + diag(s_t) is monotone since s_{t,i} = ||g_{1:t,i}||_2 only grows as more gradients accumulate — the successive Bregman-divergence terms telescope. Carrying that through, the composite mirror-descent version lands at

  R_φ(T) ≤ (1/(2η)) max_{t≤T} ||x* − x_t||²_∞ Σ_{i=1}^{d} ||g_{1:T,i}||_2 + η Σ_{i=1}^{d} ||g_{1:T,i}||_2,

where the second term is exactly the η/2 times the gradient sum after I apply my doubling lemma (the factor 2 from the lemma meeting the 1/2). Both terms are proportional to the same Σ_i ||g_{1:T,i}||_2. With X compact, set D_∞ = sup_x ||x − x*||_∞ and balance the two terms by choosing η = D_∞/√2:

  R_φ(T) ≤ √2 D_∞ Σ_{i=1}^{d} ||g_{1:T,i}||_2 = √2 D_∞ γ_T,

with γ_T = Σ_i ||g_{1:T,i}||_2 the hindsight-optimal quantity from the Lagrangian. That is the payoff: the regret is, up to the dimension-free constant √2 D_∞, the best the optimal fixed diagonal preconditioner could have done. Sublinear in the worst case — ||g_{1:T,i}||_2 ≤ G_∞ √T so R = O(√T) — but the real point is what happens on sparse data.

Let me actually compute γ_T on a sparse feature model rather than wave at it, because this is where the method either wins big or it does not. Take 0/1 features where feature i appears with probability p_i = min{1, c i^{−α}} for some power-law exponent α > 1 (rare features in the tail), with hinge loss so a gradient term is ±1 when the feature is active. Then ||g_{1:T,i}||_2 = sqrt(|{t : |g_{t,i}| = 1}|), the square root of the number of times coordinate i was active. Take expectations and push the sqrt inside by Jensen (sqrt is concave):

  E Σ_i ||g_{1:T,i}||_2 = Σ_i E sqrt(|{t : |g_{t,i}| = 1}|) ≤ Σ_i sqrt( E|{t : |g_{t,i}| = 1}| ) = Σ_i sqrt(p_i T) = √T Σ_i sqrt(p_i).

And Σ_i sqrt(p_i) = sqrt(c) Σ_i i^{−α/2}. For α ≥ 2 that sum is Σ_i i^{−α/2} = O(log d); for α ∈ (1, 2) it is O(d^{1−α/2}), still strictly sub-d. So on a box X = {||x||_∞ ≤ 1}, where D_∞ = 2, the adaptive regret is O((log d) √T) in the heavy-tailed α ≥ 2 case. Compare the isotropic bound: Zinkevich has D_2 = 2√d and ||g_t||²_2 ≥ 1, giving best-case regret O(√(dT)). The adaptive method is smaller by a factor that grows like √d / log d — exponentially better in the dimension. That is the concrete reason to adapt: on the sparse heavy-tailed data I actually have, the per-coordinate scaling turns a √(dT) regret into a (log d)√T regret. (And I should be careful: this advantage is real but it depends on the geometry of X. If the optimal predictor is dense over a box, the relevant comparison shifts and the gain can shrink; the win is precisely in the sparse-features regime.)

I went diagonal for intuition and for cost, but let me ask whether the diagonal restriction is leaving something on the table — what is the true optimal preconditioner if I allow a full matrix? Repeat the hindsight argument without the diagonal constraint: minimize the gradient term over all PSD S with a trace budget,

  min_S  Σ_{t=1}^{T} ⟨g_t, S^{−1} g_t⟩  s.t.  S ⪰ 0,  tr(S) ≤ c.

Let G_T = Σ_{τ≤T} g_τ g_τ^T. I can rewrite the objective as tr(S^{−1} G_T). The minimizer of tr(S^{−1} A) over the trace-budgeted PSD cone is S = c A^{1/2}/tr(A^{1/2}) — to see it, build the Lagrangian L(S, θ, Z) = tr(S^{−1} A) + θ(tr(S) − c) − tr(SZ), set ∂L/∂S = −S^{−1} A S^{−1} + θI − Z = 0; for S full rank complementarity gives Z = 0, so S^{−1} A S^{−1} = θI, i.e. A = θ S², hence S ∝ A^{1/2}, and the trace budget fixes the constant, S = c A^{1/2}/tr(A^{1/2}). The optimal value is tr(A^{1/2})²/c. So with A = G_T the best full preconditioner is the (scaled) matrix square root of the outer-product matrix, S = c G_T^{1/2}/tr(G_T^{1/2}), and the regret target becomes tr(G_T^{1/2}) — a genuine generalization of Σ_i ||g_{1:T,i}||_2, which is what you get when you keep only the diagonal of G_T. The online version sets S_t = G_t^{1/2}, H_t = δI + S_t, and steps with H_t^{-1}; when δ = 0 and the projection is inactive, that is x_{t+1} = x_t − η G_t^{-1/2} g_t, interpreted on the range of G_t.

I have to redo the doubling argument for matrices, and this is where I need to be careful because the scalar concavity step becomes a trace concavity step. The matrix statement I want is Σ_t ⟨g_t, S_t^{†} g_t⟩ ≤ 2 tr(G_T^{1/2}), with S_t = G_t^{1/2} and the pseudo-inverse used on the range when G_t is singular. The engine is the concavity of A ↦ tr(A^{1/2}): at B ≻ 0 its gradient is ½B^{-1/2}, so concavity gives tr(A^{1/2}) ≤ tr(B^{1/2}) + ½ tr(B^{-1/2}(A − B)). Put A = B − νgg^T, then let the usual δI regularization go to zero when B is singular; because B − νgg^T ⪰ 0 forces g into the range of B, the limiting pseudo-inverse expression is legitimate. The result is

  2 tr((B − νgg^T)^{1/2}) ≤ 2 tr(B^{1/2}) − ν tr(B^{-1/2} gg^T).

Now the induction mirrors the scalar one. The base case is ⟨g_1, S_1^{†} g_1⟩ = ||g_1||_2 = tr(G_1^{1/2}) ≤ 2 tr(G_1^{1/2}). For the step, assume the bound through T−1. Then

  Σ_{t≤T} ⟨g_t, S_t^{†} g_t⟩
    ≤ 2 tr(G_{T−1}^{1/2}) + ⟨g_T, S_T^{†} g_T⟩
    = 2 tr((G_T − g_Tg_T^T)^{1/2}) + tr(G_T^{-1/2} g_Tg_T^T)
    ≤ 2 tr(G_T^{1/2}).

The last line is exactly the trace-concavity inequality with B = G_T and ν = 1. So the same doubling shape survives: the causal matrix square root costs only a factor of two in the gradient term. The composite mirror-descent regret has the same two pieces as the diagonal case,

  R_φ(T) ≤ (1/(2η)) max_{t≤T} ||x* − x_t||_2² tr(G_T^{1/2}) + η tr(G_T^{1/2}),

and with D = sup_x ||x − x*||_2 and η = D/√2 it becomes R_φ(T) ≤ √2 D tr(G_T^{1/2}), the full-matrix analogue of the diagonal bound.

So the full-matrix version is the genuinely optimal object — it can exploit correlations between coordinates, capturing a rotation of the space in which the gradients are sparse. But now I hit the wall I anticipated: maintaining G_t, forming its square root, and inverting it is O(d²) memory and roughly O(d³) (or O(d²) with care) per step. In a problem with d in the millions that is simply impossible — I cannot even store the d×d matrix. The diagonal restriction H_t = δI + diag(G_t)^{1/2} is linear time and space, because a diagonal matrix is just a vector; the memory overhead over vanilla subgradient descent is a single d-dimensional accumulator, and the per-step work is a handful of elementwise operations. And critically, the diagonal version keeps the part that delivered the sparse-feature win — the per-coordinate accumulated-norm denominator. The off-diagonal correlations that the full matrix captures are a refinement; the per-coordinate adaptation is the bulk of the gain. So the diagonal version is the one that actually ships, and the full-matrix result stands as the ideal it approximates.

Can I recover the methods I came from as corners of this family? That is the test of whether I have found the right generalization rather than a third option. If I ignore the accumulated geometry and choose the time-scaled isotropic metric H_t = √t I, the update degenerates to x_{t+1,i} = x_{t,i} − (η/√t) g_{t,i}, which is exactly Zinkevich's projected gradient with its √t schedule. So the non-adaptive method is the corner of my family where the preconditioner is isotropic and only its global scale changes. And the dual-averaging form recovers RDA: the dual-averaging update with my diagonal proximal term ψ_t(x) = ½⟨x, H_t x⟩, for the ℓ1 case φ(x) = λ||x||_1, solves min_x { η⟨ḡ_t, x⟩ + (1/t)ψ_t(x) + ηλ||x||_1 }, which is separable across coordinates; standard subgradient calculus on each coordinate gives the soft-threshold

  x_{t+1,i} = sign(−ḡ_{t,i}) · (ηt/H_{t,ii}) · [ |ḡ_{t,i}| − λ ]_+,

where the three cases (the solution is 0 when |ḡ_{t,i}| ≤ λ, and the differentiable branches give x̂_i = (ηt/H_{t,ii})(−ḡ_{t,i} ∓ λ) when ḡ_{t,i} ≷ ±λ) combine into that one expression. Put it next to vanilla RDA, x_{t+1,i} = sign(−ḡ_{t,i}) · η√t · [|ḡ_{t,i}| − λ]_+. The only difference is the per-coordinate step ηt/H_{t,ii} in place of the global η√t — the adaptive method has handed each coordinate its own learning rate, inversely proportional to that coordinate's accumulated gradient norm, while leaving the soft-thresholding structure that produces sparse iterates untouched. That is the cleanest statement of what I have done: I took the single √t-scaled proximal term and replaced its one global scale by a per-coordinate scale read off the data. And because H_{t,ii} stays intact between updates of a coordinate, when the gradient stream is sparse I can do all of this lazily — only touch coordinate i on the rounds it is non-zero, applying the accumulated shrinkage in one shot — which is what keeps the per-step cost proportional to the number of non-zeros, not to d.

Now let me write the thing I actually ship. The setting in front of me is unconstrained, X = R^d, no composite φ, so the mirror-descent step collapses to the plain per-coordinate rule: keep a running sum of squared gradients per coordinate, and step by the gradient divided by the square root of that sum plus a small floor. Concretely, for a parameter block (I have two, the diagonal-net's u and v, each with its own independent accumulator), the per-coordinate update is x ← x − lr · g / (sqrt(Σ g²) + eps). Two state objects: the squared-gradient accumulator and a step counter. Here is the loop, filling the single empty slot in the online-learner harness:

```python
import torch


def get_hyperparameters(dim, sparsity, delta):
    # Single global trust knob lr (the per-coordinate scaling lives in the
    # accumulated-norm denominator, so lr stays one scalar). eps floors the
    # denominator so a coordinate with zero accumulated gradient mass is still
    # invertible (sqrt(0) = 0 would divide by zero).
    return {"lr": 0.01, "eps": 1e-6}


def init_state(u, v, hyperparameters):
    # Per-coordinate accumulator of sum-of-squared-gradients: this IS diag(G_t),
    # and its square root is the optimal-in-hindsight diagonal denominator
    # ||g_{1:t,i}||_2 derived from the regret bound. Initialised to 0 = no history.
    d = u.shape[0]
    return {
        "t": 0,
        "state_sum_u": torch.zeros(d, dtype=torch.float64),
        "state_sum_v": torch.zeros(d, dtype=torch.float64),
    }


def step(u, v, grad_u, grad_v, state, hyperparameters):
    lr = float(hyperparameters["lr"])
    eps = float(hyperparameters["eps"])
    # Accumulate the running sum of squared gradients per coordinate:
    #   state_sum_{t,i} = sum_{tau<=t} g_{tau,i}^2 = ||g_{1:t,i}||_2^2
    state_sum_u = state["state_sum_u"] + grad_u * grad_u
    state_sum_v = state["state_sum_v"] + grad_v * grad_v
    # Per-coordinate step = lr / (sqrt(state_sum) + eps):
    # frequent coords -> large denominator -> small step; rare coords -> large step.
    #   x_{t+1,i} = x_{t,i} - lr * g_{t,i} / ( sqrt(sum_{tau<=t} g_{tau,i}^2) + eps )
    std_u = torch.sqrt(state_sum_u) + eps
    std_v = torch.sqrt(state_sum_v) + eps
    u_new = u - lr * grad_u / std_u
    v_new = v - lr * grad_v / std_v
    return u_new, v_new, {
        "t": state["t"] + 1,
        "state_sum_u": state_sum_u,
        "state_sum_v": state_sum_v,
    }
```

The causal chain, end to end. I started with a subgradient method whose single global step size is structurally wrong for sparse heavy-tailed data, where rare features carry the information yet get the same rate as constant ones, and where the isotropic regret √(dT) is provably the best I can do without exploiting the sparsity. The leverage was that the mirror-descent regret bound is governed entirely by the dual norms of the gradients, which are set by the proximal function ψ — a knob everyone fixes a priori. Treating ψ as something to learn from the data, I restricted it to a squared Mahalanobis form ⟨x, H_t x⟩ so the regret term became Σ_t ⟨g_t, H_t^{−1} g_t⟩, and solving the hindsight problem under a trace budget showed the optimal diagonal denominator for coordinate i is its accumulated gradient norm ||g_{1:T,i}||_2 — the square root of the running sum of squared gradients, not a count or an average. Using the causal running norm online costs only a factor of two, which the doubling lemma proved through the concavity of the square root, yielding regret √2 D_∞ Σ_i ||g_{1:T,i}||_2 — competitive with the best fixed diagonal preconditioner in hindsight, and exponentially smaller in the dimension on sparse heavy-tailed features, O((log d)√T) instead of O(√(dT)). The full-matrix version with the matrix square root tr(G_T^{1/2}) is the genuinely optimal object but its O(d²) memory and matrix roots are unaffordable, so the diagonal restriction — linear time and space, keeping the per-coordinate adaptation that drives the win — is what ships. The frozen-metric corner recovers Zinkevich's gradient descent, and the dual-averaging form recovers RDA with one global rate replaced by a per-coordinate rate, leaving the sparse soft-thresholding intact. And the whole thing drops into the online subgradient loop as a single per-coordinate accumulator, a square root, and a divide.
