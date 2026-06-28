I have a nested problem and I want its gradient. There is an inner training procedure that, given hyperparameters lambda, produces weights w by minimizing a training loss J(w, lambda); and there is an outer validation error E that I want small at those weights. So the thing I actually want to minimize is f(lambda) = E(w_T(lambda)), where w_T is what the inner optimizer spits out after T steps. If lambda were two or three numbers I would just grid-search it. But the cases I care about have lambda enormous — one weight per training example, so I can down-weight the mislabeled ones; or a whole dense task-interaction matrix; or a per-parameter regularizer. Random search and Bayesian optimization both die here, because each completed training run tells me essentially one scalar, the validation loss, and I would need a number of runs that explodes with the dimension of lambda. The only way out is the gradient df/dlambda: with it I can move all million coordinates of lambda at once. So the entire problem reduces to one question — how do I compute the gradient of the validation error with respect to lambda when w_T(lambda) is not a formula but the output of an iterative optimizer?

Let me look at what the inner optimizer actually is, because that is the object I have to differentiate. It is a sequence of update steps. Whether it is plain SGD, momentum, RMSProp, Adam, each step takes the current state and produces the next, and lambda is baked into it. Let me write the state as s_t — for plain gradient descent that is just the weights w_t, for momentum it also carries the velocity, but call it s_t in general — and write the step as a map

  s_t = Phi_t(s_{t-1}, lambda),   t = 1, ..., T,

with s_0 the initialization. The t subscript on Phi_t lets it depend on the minibatch at step t. For gradient descent with momentum, concretely, s_t = (v_t, w_t) with v_t = mu·v_{t-1} + ∇J_t(w_{t-1}) and w_t = w_{t-1} − eta·(...), so lambda = (mu, eta) lives inside Phi_t. The point is: f(lambda) = E(s_T(lambda)) is a composition of T known, smooth maps, ending in a known scalar function E. That is exactly the kind of object that the chain rule chews through. There is nothing mysterious about whether the gradient exists; the only question is how to compute it without drowning in memory or time, because s_T depends on lambda through every one of the T steps, and the weights are millions of numbers.

So just apply the chain rule and see what falls out, then worry about cost. lambda enters Phi_t in two ways: directly, through Phi_t's own dependence on lambda, and indirectly, through s_{t-1}, which already depends on lambda. Differentiate the recursion totally with respect to lambda:

  ds_t/dlambda = (∂Phi_t/∂s_{t-1})·(ds_{t-1}/dlambda) + ∂Phi_t/∂lambda.

Let me name the two partials, because they will be everywhere: A_t = ∂Phi_t/∂s_{t-1}, which is d×d (how this step's output moves when its input state moves), and B_t = ∂Phi_t/∂lambda, which is d×m (how this step's output moves when lambda moves directly). And let me name the total derivative Z_t = ds_t/dlambda, a d×m matrix. Then the recursion is

  Z_t = A_t Z_{t-1} + B_t,   Z_0 = 0.

And the hypergradient I want, by the chain rule on f = E(s_T), is

  ∇f(lambda) = ∇E(s_T)·Z_T,

with ∇E(s_T) a row vector of length d. Unroll the recursion to see the whole thing in one line:

  Z_T = A_T Z_{T-1} + B_T = A_T(A_{T-1}Z_{T-2} + B_{T-1}) + B_T = ... = Σ_{t=1}^T (Π_{s=t+1}^T A_s) B_t,

so

  ∇f(lambda) = ∇E(s_T) · Σ_{t=1}^T (Π_{s=t+1}^T A_s) B_t.

That is a closed form for the hypergradient, but I do not yet trust it — I unrolled the recursion by eye, and it is easy to be off by an index in the product or the sum. Let me pin it down on a problem small enough to differentiate by brute force. Take the inner objective J(w, lambda) = ½·a·w² − lambda·w with a > 0, one scalar weight w and one scalar hyperparameter lambda; gradient descent with step gamma gives Phi(w, lambda) = w − gamma·(a·w − lambda) = (1 − gamma·a)·w + gamma·lambda, so here A = 1 − gamma·a and B = gamma are just numbers, constant across steps. Let the outer objective be E(w) = ½·(w − c)², so ∇E = w − c and E does not depend on lambda directly. Pick a = 2, gamma = 0.3 (so A = 0.4, B = 0.3), c = 1, w_0 = 0, lambda = 0.5, and run T = 8 inner steps; that lands at w_8 = 0.249836. The formula above then predicts ∇f = (w_8 − c)·Σ_{t=1}^8 A^{8−t}·B = (−0.750164)·Σ_{t=1}^8 0.4^{8−t}·0.3 = −0.374836. Now check it against a derivative that has nothing to do with my algebra: perturb lambda by ±1e−6, rerun the whole 8-step procedure, and take the central finite difference of f. That gives −0.374836 as well, agreeing to six digits. So the unrolled expression is the right object, indices and all. But staring at it, it is a computational nightmare if taken literally. A_t is d×d with d in the millions — I can never form it, let alone multiply chains of them. B_t is d×m. The sum runs over all T steps. If I were foolish enough to evaluate this as written I would be doing dense d×d matrix products T times over. So the real work is not deriving this formula — it is finding an arithmetic order that never materializes A_t or B_t and never stores anything d×d.

I can evaluate that sum-of-products from the right or from the left, and the two cost wildly different things — the same fork that shows up when you differentiate a recurrent net.

Take the recursion Z_t = A_t Z_{t-1} + B_t at face value and just carry Z_t forward alongside the training. That works: I start with Z_0 = 0, and at each step, as I compute s_t from s_{t-1}, I also compute Z_t from Z_{t-1}. At the end I have Z_T and dot it with ∇E(s_T). The appeal is that I never look back: Z_t marches forward in lockstep with the optimizer, I can throw away s_{t-1} the moment I have s_t, and I even get a partial hypergradient ∇E(s_t)·Z_t at every step if I want it. But what is Z_t? It is d×m. If lambda has m coordinates, I am carrying a d×m matrix and, at each step, computing A_t Z_{t-1}, which is m Jacobian-vector products of the map Phi_t (one per column of Z), plus the Jacobian B_t. By the basic algorithmic-differentiation fact, a single Jacobian-vector product of a map that costs O(g) to evaluate also costs O(g) in forward mode. So this costs O(T·m·g) time and O(g) space — the space is gorgeous, constant in T, because Z overwrites itself, but the time carries a factor of m, the number of hyperparameters. For m a handful — a learning rate, a momentum — that is wonderful and I would happily run it while training, even in real time on a data stream. But for m in the thousands or millions, the per-step factor of m is fatal. I cannot afford to carry a d×m matrix and update it when m is the dimension of my whole problem. Wall — for high-dimensional lambda the forward way is too slow.

So flip it. The reason forward mode pays a factor of m is that it propagates a d×m object, and m is large. But the thing I ultimately want, ∇E(s_T)·Z_T, is a 1×m object — a single gradient. There is a complementary algorithmic-differentiation fact: a vector-times-Jacobian product q^T J also costs O(g), in reverse mode. So if I could push the *row vector* ∇E(s_T) leftward through the chain of A_t's, contracting from the left, I would only ever carry a 1×d row, never a d×m matrix, and the factor of m would vanish. Let me see if the algebra cooperates. Group the sum

  ∇E(s_T)·Σ_t (Π_{s=t+1}^T A_s) B_t = Σ_t [∇E(s_T)·A_T·A_{T-1}···A_{t+1}]·B_t.

Define alpha_t = ∇E(s_T)·A_T·A_{T-1}···A_{t+1}, the row vector you get by starting at ∇E(s_T) and multiplying by the A's down to t+1. Then ∇f = Σ_t alpha_t·B_t, and the alpha's satisfy a backward recursion: alpha_T = ∇E(s_T), and alpha_t = alpha_{t+1}·A_{t+1}. Each alpha_{t+1}·A_{t+1} is a row-times-Jacobian — a transposed-Jacobian-vector product — costing O(g), and each alpha_t·B_t is likewise a row-times-Jacobian wrt lambda, costing O(g). So the whole hypergradient comes out in O(T·g) time, with no factor of m at all. That is the direction I want for big lambda.

But I should make sure I am deriving this and not just asserting it, because the backward recursion has a structure I want to trust and reuse — it is the adjoint of a constrained problem, and that view will also tell me how to handle constraints on lambda and how to be sure I have not dropped a term. Let me re-derive it cleanly as a Lagrangian, the way back-propagation itself is derived. The thing I want is min_lambda E(s_T) where the s_t are *forced* by the dynamics. So write it as a constrained problem in all the variables at once:

  min over lambda, s_1, ..., s_T of E(s_T)   subject to   s_t = Phi_t(s_{t-1}, lambda),  t = 1, ..., T.

Attach a row vector of multipliers alpha_t to each constraint and form the Lagrangian

  L(s, lambda, alpha) = E(s_T) + Σ_{t=1}^T alpha_t·(Phi_t(s_{t-1}, lambda) − s_t).

Now take partials and set them to zero, which is just the optimality conditions. Differentiating in alpha_t recovers the constraint, fine. The interesting ones are the s_t. For the last state,

  ∂L/∂s_T = ∇E(s_T) − alpha_T,

because s_T appears in E(s_T) and in the −alpha_T·s_T term of the t=T constraint, and nowhere else (there is no Phi_{T+1}). Setting it to zero: alpha_T = ∇E(s_T). For an interior t in 1..T−1, s_t appears in two places: as the −alpha_t·s_t in the t-th constraint, and inside Phi_{t+1}(s_t, lambda) in the (t+1)-th constraint. So

  ∂L/∂s_t = alpha_{t+1}·(∂Phi_{t+1}/∂s_t) − alpha_t = alpha_{t+1}·A_{t+1} − alpha_t.

Setting it to zero gives the backward recursion alpha_t = alpha_{t+1}·A_{t+1} — the exact same one I guessed, now with the multipliers identified as the adjoint state. Solving it: alpha_t = ∇E(s_T)·A_T·A_{T-1}···A_{t+1} for t < T, and alpha_T = ∇E(s_T). And finally, at a feasible point (constraints satisfied) the gradient of the response in lambda is

  ∂L/∂lambda = Σ_{t=1}^T alpha_t·(∂Phi_t/∂lambda) = Σ_{t=1}^T alpha_t·B_t.

Combine with the solved alphas and I get ∇f(lambda) = ∇E(s_T)·Σ_t (Π_{s=t+1}^T A_s) B_t — symbolically the same expression the forward derivation produced. But I want to be sure the *backward* arithmetic, not just the algebra, lands on the same number, because the whole point is to run it from the right end. Let me trace the recursion on the same toy instance. Start alpha_8 = ∇E(w_8) = w_8 − c = −0.750164 and g = 0. Step j = 8: g += alpha_8·B = −0.750164·0.3 = −0.225049; then alpha_7 = alpha_8·A = −0.750164·0.4 = −0.300066. Step j = 7: g += alpha_7·B = −0.090020, so g = −0.315069; alpha_6 = −0.120026. Step j = 6: g += −0.036008, g = −0.351077; alpha_5 = −0.048010. The alpha is collapsing by a factor of 0.4 each step, so the early steps barely move g, and carrying the loop down to j = 1 lands at g = −0.374836 — the same value the finite difference and the forward unroll gave, to six digits. So the two ways of organizing the arithmetic do compute the identical gradient; they just contract the matrix product from opposite ends, and I have now watched the left-contraction produce the right answer rather than assumed it. The Lagrangian view is doing real work here, not decoration. It tells me alpha_t is the adjoint, the sensitivity of the final error to a perturbation of the state at time t; it hands me the recursion without my having to guess the grouping; and because it is a constrained formulation, the moment I want to put constraints on lambda — a box, an L1 budget, a symmetry — they slot in as constraints on the same problem rather than forcing me to rederive anything. This is structurally back-propagation through time: the forward sweep computes the states, the backward sweep carries the adjoint from the end to the beginning, accumulating the gradient. I have effectively recognized that differentiating a training run is BPTT over the optimizer's own steps.

Let me write the procedure out concretely, now that the recursion is pinned down and checked. Forward: run s_t = Phi_t(s_{t-1}, lambda) for t = 1..T, keeping the states. Initialize alpha_T = ∇E(s_T) and g = 0. Backward, for j = T down to 1: accumulate g += alpha_j·B_j, and, if j > 1, propagate alpha_{j-1} = alpha_j·A_j. Return g. This indexing harvests every transition's direct contribution, including the first step Phi_1(s_0, lambda); the propagation stops after alpha_1 because there is no alpha_0 term in the Lagrangian sum. Every operation in the backward pass is a transposed-Jacobian-vector product: alpha·A is the gradient of "alpha dotted with the next state" with respect to the current state, and alpha·B is the same with respect to lambda. Reverse-mode automatic differentiation computes both in O(g) without ever building A or B. So I never touch a d×d matrix. Good.

Now the cost, honestly. Time is O(T·g): one forward step and one backward step per t, each O(g), no m factor. That is the win over forward mode for high-dimensional lambda. But the space — here is the price. To compute A_{t+1} and B_{t+1} in the backward pass I need s_t, the state going *into* step t+1. In the forward pass I overwrote nothing only if I planned for the backward pass; but the backward pass walks t from T down to 1 and needs s_{T-1}, s_{T-2}, ..., s_0 in reverse order, each to recompute the local Jacobians. So I must keep the whole trajectory s_0, s_1, ..., s_T in memory. That is O(T·d) space. For a small model fine; for a network with millions of weights run for thousands of steps, the trajectory is the weight vector — on the order of a gigabyte — times T, and that is the memory wall everyone hits. Wall: reverse mode is fast but its space grows linearly with the length of training times the size of the model.

How have people dodged this wall? The sharpest prior attempt reverses the dynamics instead of storing them. The observation is that if Phi_t is *invertible*, I do not need to store s_{t-1}; I can recover it from s_t by running the step backward, regenerating the trajectory on the fly during the reverse pass at O(1) extra space. For SGD with momentum this almost works: given (v_t, w_t) and the gradient, you can step back to (v_{t-1}, w_{t-1}). The trouble is finite precision. The momentum decay multiplies the velocity by gamma < 1 every step, which shifts bits off the bottom of the float; reversing requires repeatedly multiplying by 1/gamma, and the lost low-order bits accumulate into error that, over thousands of steps, sends the reversed trajectory far from the true one and usually overflows. And you cannot just set gamma = 1 to avoid the lossy multiply — gamma = 1 is the leapfrog limit, perfectly reversible but non-converging; gamma > 1 is unstable. The fix prior work found is ingenious but fragile: store the exact bits that fall off the bottom in an "information buffer," using an integer divide-by/multiply-by trick so that the reversal is bit-exact, which still costs only a fraction of a bit per step on average — a ~200× memory saving at gamma = 0.9 over storing the trajectory. But it is bought with brittleness. It is welded to one specific dynamics (momentum with 0 < gamma < 1), it needs the exact-reversal bit machinery, and it does not generalize to an arbitrary optimizer step. I do not want my method to require Phi_t to be invertible. I would rather keep the clean, exact, optimizer-agnostic reverse-mode computation and pay the memory — or, better, find a way to pay *less* of it without giving up generality.

Before that, let me make sure I am not missing a fundamentally cheaper route — implicit differentiation. If the inner optimizer actually reaches the minimizer w*(lambda), defined by ∇_w J(w*, lambda) = 0, then I never need the trajectory at all. Differentiate the stationarity condition totally in lambda: ∇_{w,w}J·(dw*/dlambda) + ∇_{lambda,w}J = 0, so dw*/dlambda = −(∇_{w,w}J)^{-1}·∇_{lambda,w}J, and then ∇f = ∇_lambda E − ∇_{lambda,w}J·(∇_{w,w}J)^{-1}·∇_w E. No T, no trajectory, O(d) memory if I solve the linear system with the Hessian by an iterative solver using only Hessian-vector products. Beautiful — when it applies. But look at what it assumes. It holds at an *exact* minimizer, so I have to actually solve the inner problem (or run it to a tolerance epsilon and design a schedule that drives epsilon down). It needs ∇_{w,w}J invertible and well-conditioned for the linear solve to converge, which in practice means I have to add a strong-convexity-inducing regularizer to the inner objective. And — this is the decisive limitation — because it differentiates the *optimum* w*, it is blind to anything that does not show up in ∇_w J(w*, lambda) = 0. A hyperparameter that controls the inner *optimizer* — a learning rate, a momentum factor, the inner step count — leaves no fingerprint on the stationarity condition, so implicit differentiation simply cannot produce its gradient. My trajectory-differentiation route, by contrast, differentiates the *procedure I actually ran*: it needs no optimum, no strong convexity, no Hessian solve, and it naturally yields gradients for the optimizer's own hyperparameters, because those literally appear inside Phi_t. So the two are not competitors so much as opposite trade-offs, and for the problems I care about — non-convex inner objectives, hyperparameters that include the optimizer's settings, no guarantee of reaching the optimum — differentiating the trajectory is the right tool, memory cost and all. I will hold onto implicit differentiation as a sanity reference, though, because I suspect the two should agree in the limit.

Now, can I shave the memory of the reverse-mode trajectory differentiation without going back to the fragile reversibility trick? The backward pass walks from t = T down toward 1, carrying the adjoint alpha. Most of the gradient signal in well-behaved inner problems lives near the *end* of training, where the iterates are close to the inner solution — the early steps, while the optimizer is still far from convergence, contribute little to how the final error responds to lambda. What if I just do not back-propagate all the way? Run the full forward inner optimization for all T steps, so the final weights w_T are properly converged, but in the backward pass only walk back K transitions, from T down to T−K+1, and stop. Then I only need to store the K+1 suffix iterates w_{T−K}, ..., w_T, not all T+1 iterates. The approximate hypergradient is

  h_{T−K} = ∇_lambda E + Σ_{t=T−K+1}^T alpha_t·B_t,

i.e. the same backward accumulation, truncated to the last K steps, plus the direct ∂E/∂lambda term. The question is whether this truncation costs me anything real, and I want to *derive* the answer, not assume it.

Suppose the inner objective J(·, lambda) is, near its minimizer w*, twice differentiable and strongly convex — say alpha-strongly-convex and beta-smooth — and the inner step is gradient descent, Phi(w, lambda) = w − gamma·∇_w J(w, lambda), with step gamma ≤ 1/beta. Then the per-step state Jacobian is A_t = I − gamma·∇_{w,w}J(w_t, lambda), the Hessian shifted. Strong convexity and smoothness pin its spectrum: every eigenvalue of gamma·∇_{w,w}J lies in [gamma·alpha, gamma·beta] ⊆ [gamma·alpha, 1], so every eigenvalue of A_t lies in [0, 1 − gamma·alpha], and hence ‖A_t‖ ≤ 1 − gamma·alpha < 1. The map is a contraction. Now look at what truncation drops. The full backward sum's t-th term carries the factor Π_{s=t+1}^T A_s; for a term at depth T−t from the end, that is a product of (T−t) contraction matrices, so its size is bounded by (1 − gamma·alpha)^{T−t}. Truncating at K throws away exactly the terms with T−t ≥ K, the deepest ones, whose combined contribution is therefore bounded by something on the order of (1 − gamma·alpha)^K — the bias decays *geometrically* in K. Writing the constant out, with M_B = max_t ‖B_t‖, the error of the truncated hypergradient is

  ‖h_{T−K} − ∇f‖ ≤ ((1 − gamma·alpha)^K / (gamma·alpha))·‖∇_w E(s_T)‖·M_B

in the globally strongly convex case (and, when only the stored suffix is in the strongly convex neighborhood, the local bound carries the extra factor 2^{T−K+1}). The norm is at the approximate final lower solution, not at the limiting minimizer; the limiting minimizer only enters once I pass to the fixed-point picture. The base 1 − gamma·alpha is strictly below 1, so to get the bias down to epsilon I would need only K on the order of log(1/epsilon) — a *constant* number of stored iterates, independent of how long T is.

I derived that bound, but a geometric decay claim is exactly the kind of thing that looks clean and is wrong by a factor that grows with T, so let me put a number on it with the same toy instance, where I can compute the full hypergradient (K = T) and every truncation exactly. Here gamma·alpha = gamma·a = 0.6, so 1 − gamma·alpha = A = 0.4, and the predicted error at truncation K is (0.4^K / 0.6)·|w_8 − c|·B = (0.4^K / 0.6)·0.750164·0.3. Truncating the backward sum to its last K terms and comparing to the full g = −0.374836:

  K = 1:  h = −0.225049,  |h − g| = 1.498e−1,   bound = 1.500e−1
  K = 2:  h = −0.315069,  |h − g| = 5.977e−2,   bound = 6.001e−2
  K = 3:  h = −0.351077,  |h − g| = 2.376e−2,   bound = 2.401e−2
  K = 4:  h = −0.365480,  |h − g| = 9.356e−3,   bound = 9.602e−3

The actual error sits just under the bound at every K and shrinks by very nearly the factor A = 0.4 each time I keep one more step — geometric, with no creeping T-dependence. So the bound is not just an upper estimate that happens to hold; it is tight here, and the decay is real. That settles the worry: I can run the inner optimizer as long as I like to get good weights, but pay only K-worth of memory for the gradient, with K small. This is the lever I was looking for against the space wall, and it does not require reversibility — it requires only that the inner map contracts, which strong convexity already gives me.

There is a subtlety I should not gloss over. The numbers above show truncation has real bias — at K = 1 the estimate is off by 1.5e−1, about 40% of the true hypergradient's magnitude. So a short K is decidedly *not* a good estimate of ∇f as a vector. What I actually need for the outer optimization to work is weaker than a good estimate: I need the truncated direction to still point downhill, i.e. to have positive inner product with the true hypergradient. In the toy instance every truncation h_K has the same sign as g (all negative) and lies between 0 and g, so −h_K is a descent direction for f even at K = 1, even though its length is wrong. That matches what I would expect when the outer objective depends on lambda only through the weights (∇_lambda E = 0, the non-interference case — true for data hyper-cleaning): every retained term of the backward sum has the form (positive geometric weight)·(common B), so they cannot cancel, and the truncated sum stays aligned with the full one rather than rotating away from it. I would conjecture, then, that for this problem class even K = 1 can drive the outer problem toward a stationary point — the *direction* survives truncation even when the *magnitude* does not. I have not proven that the alignment stays bounded below by a positive multiple of ‖∇f‖² in general (the toy case is one scalar, where alignment is trivial), so I will hold it as an expectation to test on the actual MNIST hyper-cleaning runs rather than a settled fact. What the computation does settle is the weaker and more useful claim: truncation shrinks the magnitude geometrically without flipping the sign, so the truncated method is not a crude approximation to be tolerated but a reweighting of a descent direction, at a fraction of the memory.

Now I want to come back to the implicit-differentiation route I set aside, because the contraction I just used has a suggestive consequence. Take the backward sum in the limit where the inner iterates have converged, so A_t → A_infty = I − gamma·∇_{w,w}J(w*) and B_t → B_infty = −gamma·∇_{w,lambda}J(w*) in row-gradient orientation. The row-form full backward accumulation becomes ∇_{w*}E · Σ_{k=0}^∞ A_infty^k · B_infty. The sum Σ_{k=0}^∞ A_infty^k is a geometric series of matrices, and because ‖A_infty‖ < 1 it converges to (I − A_infty)^{-1} = (gamma·∇_{w,w}J)^{-1} — the Neumann series of the inverse. So the infinite backward sum equals ∇_{w*}E · (gamma·∇_{w,w}J)^{-1} · (−gamma·∇_{w,lambda}J) = −∇_{w*}E · (∇_{w,w}J)^{-1} · ∇_{w,lambda}J, with the gammas cancelling. In the usual column-gradient notation that is the transpose-equivalent formula −∇_{lambda,w}J · (∇_{w,w}J)^{-1} · ∇_{w*}E — exactly the implicit-function hypergradient I wrote down at the start.

That equality is worth checking rather than admiring. On the toy instance, ∇_{w,w}J = a = 2 and ∇_{w,lambda}J = −1, so the implicit formula gives ∇f = ∇E(w*)·(1/a)·(−(−1))·... let me just compute it: dw*/dlambda = −(1/a)·(∇_{w,lambda}J) = −(1/2)·(−1) = 1/2, and ∇f = (w* − c)·(dw*/dlambda) = (0.25 − 1)·0.5 = −0.375. And the Neumann side: Σ_k A^k · B = (1/(1 − A))·B = (1/0.6)·0.3 = 0.5 = dw*/dlambda, so it reproduces the same 1/2 and hence the same −0.375. Meanwhile the T = 8 unroll gave −0.374836, off from the converged value by 1.6e−4 ≈ A^8 = 0.4^8 = 6.5e−5·(constant) — the trajectory hypergradient is converging to the implicit one geometrically in T, exactly as the fixed-point picture predicts. So differentiating the converged trajectory in full *is* implicit differentiation, just computed by summing a Neumann series instead of solving a linear system; and the K-step truncation is keeping the first K terms of that series — which is why K = 1 above matched the leading geometric term of the bias. The two pictures are the same object viewed from two ends — the unroll from the trajectory side, implicit differentiation from the fixed-point side — and the truncation parameter K is the order of the Neumann approximation to the inverse Hessian.

One more thing I should check before I commit, since I keep invoking "the trajectory converges, so its derivative converges": is that actually true, that the hypergradient of the t-step unroll converges to the true hypergradient as t grows? It is not automatic — a function converging does not force its derivative to. But under the same contraction, with q = ‖A_t‖-bound < 1, I can bound the difference between the t-step unrolled hypergradient ∇f_t and the true one ∇f by something of the form (c_1 + c_2·t/q + c_3)·q^t: the q^t contraction eventually beats the polynomial-in-t growth from having more steps to back-propagate through, so the unrolled hypergradient converges to the true hypergradient *geometrically* in t. So the whole approach is sound from both ends — longer inner solving gives a better hypergradient, and deeper truncation gives a lower-bias one, both at geometric rates governed by the inner contraction.

Let me also make the complexity trade-off concrete so I know which mode to reach for, since both forward and reverse computations are now in hand. Take a network with d weights, m hyperparameters. Forward mode carries the d×m sensitivity Z_t alongside training: time O(T·m·g), space O(g) (constant in T). Reverse mode carries the 1×d adjoint backward: time O(T·g) (no m factor), space O(T·d) (the trajectory) — or O(K·d) if I truncate. So: when m is tiny (a learning rate, a momentum) and d huge, forward mode wins — constant memory, and only m times the training cost, and it even gives partial hypergradients while training, suitable for real-time updates on a stream. When m is huge (one weight per example, a dense interaction matrix) and I want a single gradient cheaply, reverse mode wins — time independent of m — and truncation tames its only weakness, the memory. The data-hyper-cleaning problem has m in the thousands, so it is squarely reverse-mode territory, full unroll when memory allows, truncated when it does not.

Now let me write it as code I would actually run, because the abstract A_t, B_t must become autograd operations. The trick that makes the inner step differentiable is to build it with create_graph=True, so the gradient-descent update is itself a node in the graph and autograd can take alpha·A and alpha·B as vector-Jacobian products through it. The inner update map for the data-hyper-cleaning problem is one gradient-descent step on the lambda-weighted training loss — each training example i is weighted by sigmoid(x_i), x being the hyperparameter vector — and the outer loss is the cross-entropy on the clean validation set:

```python
import torch
import torch.nn.functional as F


def inner_update(params, hparams, lr_inner, data, dirty_target, reg=0.0):
    """One DIFFERENTIABLE gradient-descent step on the lambda-weighted training loss.
    Phi_t(s_{t-1}, lambda): returns the new params as a graph node in `params` and `hparams`,
    so autograd can later form A = d(new)/d(params) and B = d(new)/d(hparams) as vjps."""
    x = hparams[0]                                           # per-example weights (the lambda)
    logits = forward(params, data)
    per_example = F.cross_entropy(logits, dirty_target, reduction='none')
    loss = (torch.sigmoid(x) * per_example).mean() + reg * sum((p * p).sum() for p in params)
    grads = torch.autograd.grad(loss, params, create_graph=True)   # create_graph => step is differentiable
    return [p - lr_inner * g for p, g in zip(params, grads)]        # the gradient-descent map Phi_t


def outer_loss(params, hparams, val_data, val_target):
    """E(s_T): validation cross-entropy on cleanly-labeled data."""
    return F.cross_entropy(forward(params, val_data), val_target)
```

And the hypergradient itself, the reverse-mode backward sweep, walking the stored iterates from last to first, carrying the adjoint `alpha` and accumulating the gradient `grads`. Truncation is built in for free: I pass only the suffix of K transitions, meaning K+1 consecutive iterates, so the loop only ever sees K steps and only those need to have been stored.

```python
def reverse(params_history, hparams, update_map_history, outer_loss, set_grad=True):
    """Reverse-mode hypergradient by back-propagating through the stored suffix of the inner
    trajectory and the update maps that produced it.

    params_history     : inner iterates [w_{T-K}, ..., w_T] (first to last)
    hparams            : the outer variables lambda, each requires_grad=True
    update_map_history : the inner update maps Phi applied along that suffix
    outer_loss         : (params, hparams) -> validation scalar
    """
    # treat each stored iterate as a fresh leaf so we can re-apply the map and read off A, B
    params_history = [[w.detach().requires_grad_(True) for w in ws] for ws in params_history]

    o_loss = outer_loss(params_history[-1], hparams)
    # alpha_T = dE/dw at the last iterate ; also the direct dE/dlambda term (often zero here)
    grad_outer_w = torch.autograd.grad(o_loss, params_history[-1], retain_graph=True, allow_unused=True)
    grad_outer_hparams = torch.autograd.grad(o_loss, hparams, retain_graph=True, allow_unused=True)
    grad_outer_w = [torch.zeros_like(w) if g is None else g for g, w in zip(grad_outer_w, params_history[-1])]

    alphas = grad_outer_w                                   # alpha <- nabla E(s_T)
    grads = [torch.zeros_like(h) for h in hparams]          # accumulator for the hypergradient
    K = len(params_history) - 1
    for k in range(-2, -(K + 2), -1):                       # walk backward over the stored steps
        w_mapped = update_map_history[k + 1](params_history[k], hparams)   # re-apply Phi_{k+1}
        # g += alpha B : transposed-Jacobian-vector product of the map wrt lambda
        bs = torch.autograd.grad(w_mapped, hparams, grad_outputs=alphas,
                                 retain_graph=True, allow_unused=True)
        grads = [g + (torch.zeros_like(h) if b is None else b)
                 for g, b, h in zip(grads, bs, hparams)]
        # alpha <- alpha A : transposed-Jacobian-vector product of the map wrt the input state
        alphas = torch.autograd.grad(w_mapped, params_history[k], grad_outputs=alphas)

    grads = [g + (torch.zeros_like(h) if go is None else go)
             for g, go, h in zip(grads, grad_outer_hparams, hparams)]   # add direct dE/dlambda
    if set_grad:
        for h, g in zip(hparams, grads):
            h.grad = (torch.zeros_like(h) if h.grad is None else h.grad) + g
    return grads
```

The two `torch.autograd.grad(w_mapped, ·, grad_outputs=alphas)` calls are where the recursion actually happens: the first, against `hparams`, is alpha·B_{k+1} — this step's direct contribution to the gradient through lambda; the second, against the input iterate, is alpha·A_{k+1} — pushing the adjoint one step further back. Neither forms a matrix; both are vector-Jacobian products costing one step's worth of autograd. After the loop I add the direct ∂E/∂lambda term (zero in data hyper-cleaning, where the validation loss does not depend on the per-example weights, nonzero in general) and write the result into `hparams.grad` so an outer optimizer can step on it. And the outer loop that ties it together: re-initialize the inner model, run the inner optimizer forward for T steps keeping (the last K+1 of) the iterates, call `reverse`, and take a projected outer step.

```python
def hyperopt_loop(hparams, T, K, lr_inner, hyper_lr, num_outer_steps, project,
                  train_data, dirty_target, val_data, val_target):
    hyper_opt = torch.optim.Adam(hparams, lr=hyper_lr)
    phi = lambda ws, hp: inner_update(ws, hp, lr_inner, train_data, dirty_target)
    e   = lambda ws, hp: outer_loss(ws, hp, val_data, val_target)
    K = min(K, T)
    for _ in range(num_outer_steps):
        params = fresh_params()
        history = [params]
        for t in range(T):                                 # full forward inner solve
            params = phi(params, hparams)
            history.append(params)
            if len(history) > K + 1:                        # keep K transitions = K+1 iterates
                history.pop(0)
        hyper_opt.zero_grad()
        reverse(history, hparams, [phi] * (len(history) - 1), e, set_grad=True)
        hyper_opt.step()
        with torch.no_grad():
            for h in hparams:
                h.copy_(project(h))                        # enforce constraints on lambda (box, L1)
    return hparams
```

With K = T this is the full reverse-mode unroll — the exact hypergradient of the T-step procedure. With K < T it is the truncated version, storing K+1 suffix iterates and back-propagating through only the last K transitions, with bias geometrically small in K. So the causal chain is: I want the gradient of a validation error through an iterative training procedure with high-dimensional hyperparameters; the chain rule gives that gradient as a sum of products of per-step Jacobians, which I must never materialize; contracting that product from the right (forward) carries a d×m sensitivity matrix at cost O(T·m) — fine for few hyperparameters, hopeless for many; contracting it from the left (reverse) carries a 1×d adjoint at cost O(T) with no hyperparameter-count penalty, which the Lagrangian/adjoint derivation pins down exactly and which is back-propagation through time over the optimizer's steps; its one cost, storing the trajectory, is the memory wall, dodged not by the fragile reverse-the-dynamics trick (which needs invertible momentum dynamics and bit bookkeeping) but by truncating the backward pass to the last K transitions, whose bias decays geometrically because the inner gradient-descent map contracts; and that truncated backward sum is exactly the K-term Neumann series of the inverse-Hessian implicit-differentiation formula, so the trajectory view and the fixed-point view coincide, with K the order of the approximation. The whole thing lands as a forward inner loop that keeps the K+1 suffix iterates and a backward loop that pushes one adjoint through them with vector-Jacobian products, computing the hypergradient that an outer optimizer descends.
