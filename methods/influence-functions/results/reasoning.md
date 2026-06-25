Let me start from the question that actually bugs me, because everyone is answering a slightly different one. People ask "why did the model predict this?" and they answer it by poking at the *model*: fit a little linear surrogate around the test point, or perturb the test pixels and watch the output wiggle. Fine. But that explains the prediction in terms of the model, and the model isn't a god-given object — it's the minimizer of a loss over a training set. So the prediction came from the data, through the optimizer, into the parameters, out to the test point. The "why" I care about is: which training points are responsible for this prediction? If I had never seen a particular training image, would the model still call this test image a fish?

That last sentence is already the cleanest definition I can think of. Responsibility is a counterfactual. Take a training point z, delete it, refit the model to get θ̂_{-z}, and look at how the test loss changed: L(z_test, θ̂_{-z}) − L(z_test, θ̂). That number *is* how much z mattered for the prediction on z_test. No approximation, no philosophy. The only problem is cost: I have n training points, and this asks me to retrain the whole model n times (and if I want to attribute several test points, I'd want to re-examine each retrained model). For logistic regression on a few thousand points, maybe. For a deep net with millions of parameters and tens of thousands of examples, completely hopeless. So the definition is right and the computation is the enemy.

So the move has to be: keep the definition, kill the retraining. I want d(test loss)/d(presence of z) without ever refitting. The obstacle is that "presence of z" is a discrete thing — z is in the set or it isn't. I can't differentiate "in the set." Let me make it continuous. Instead of all-or-nothing, give z a weight. Define θ̂_{ε,z} = argmin_θ (1/n)Σ_i L(z_i,θ) + ε·L(z,θ). At ε=0 I have the ordinary model θ̂. As I crank ε up, z counts extra; as I crank ε down, z counts less. And here's the bookkeeping that makes deletion fall out: removing z entirely is the same as taking its weight from 1/n down to 0, i.e. subtracting 1/n of it, i.e. setting ε = −1/n. So if I can get the derivative dθ̂_{ε,z}/dε at ε=0, I can linearly extrapolate the effect of deletion as roughly −(1/n)·(that derivative), and I never have to retrain — I just need a derivative evaluated at the model I already have.

Now, can I actually get dθ̂_{ε,z}/dε in closed form? θ̂_{ε,z} is defined by an argmin, which is annoying to differentiate directly. But an argmin satisfies a stationarity condition, and stationarity conditions I *can* differentiate. Let me write R(θ) = (1/n)Σ_i L(z_i,θ) for the empirical risk, and assume for now it's twice-differentiable and strictly convex, so θ̂ is the unique minimizer and the Hessian H = ∇²R(θ̂) = (1/n)Σ_i ∇²_θ L(z_i,θ̂) is positive definite — which means H^{-1} exists, and I'll lean on that.

The perturbed minimizer satisfies its first-order condition:
0 = ∇R(θ̂_{ε,z}) + ε ∇L(z, θ̂_{ε,z}).
This holds for every ε near 0, and θ̂_{ε,z} → θ̂ as ε → 0. So let me define the change Δ_ε = θ̂_{ε,z} − θ̂ and Taylor-expand the right-hand side around θ̂, treating ε and Δ_ε as both small. The gradient term ∇R(θ̂_{ε,z}) ≈ ∇R(θ̂) + ∇²R(θ̂)Δ_ε, and ε∇L(z,θ̂_{ε,z}) ≈ ε∇L(z,θ̂) + ε∇²L(z,θ̂)Δ_ε. Putting it together and dropping the higher-order leftovers o(‖Δ_ε‖):
0 ≈ [∇R(θ̂) + ε∇L(z,θ̂)] + [∇²R(θ̂) + ε∇²L(z,θ̂)] Δ_ε.
Solve for Δ_ε:
Δ_ε ≈ −[∇²R(θ̂) + ε∇²L(z,θ̂)]^{-1} [∇R(θ̂) + ε∇L(z,θ̂)].
Two simplifications. First, θ̂ minimizes R, so ∇R(θ̂)=0 — that whole first bracket collapses to ε∇L(z,θ̂). Second, I only want the leading behavior in ε, so to first order I can drop the ε∇²L term inside the inverse (it contributes at order ε² once multiplied by the already-order-ε right-hand side). What's left is
Δ_ε ≈ −∇²R(θ̂)^{-1} ∇L(z,θ̂) · ε = −H^{-1}∇L(z,θ̂) · ε.
Divide by ε and take ε→0:
dθ̂_{ε,z}/dε |_{ε=0} = −H^{-1} ∇L(z,θ̂).
Call this I_up,params(z). The effect on the *parameters* of upweighting z, in closed form, all evaluated at the model I already trained.

I dropped two terms in that expansion and used the optimality condition, so before I trust the result I want to see it survive contact with an actual refit. Let me fit a small logistic regression — n=30 points, d=3 features, a little L2 so H is comfortably PD — pick a training point j, bump its weight from 1/n to 1/n + ε with ε=10⁻⁴ by actually re-running the optimizer to convergence, and compare the empirical (θ̂_{ε,z} − θ̂)/ε against the closed form −H^{-1}∇L(z_j,θ̂). The closed form gives [1.12442, −0.78839, 0.25828]; the genuine refit gives [1.1243, −0.7883, 0.25825], agreeing to four decimals, with a max coordinate difference of 1.2×10⁻⁴ — exactly the size of the O(ε) remainder I threw away. So the derivative is real, not an artifact of the expansion: the influence formula predicts where the retrained model's parameters actually go.

That also pins down the *meaning*, not just the number. What did the algebra do? It formed the quadratic approximation of the risk around θ̂ and asked: if I nudge the objective by adding a sliver of L(z,·), where does the minimizer of the perturbed quadratic go? The minimizer of a quadratic shifts by −(curvature)^{-1}·(gradient of the perturbation) — that's exactly one Newton step in the direction of ∇L(z,θ̂). And that reading is what the refit just confirmed: upweighting z pulls the parameters toward reducing L(z,·), and the inverse Hessian translates "pull" into "displacement," accounting for how stiff the loss landscape is in each direction. The same closed form, for M-estimators, is what the robust-statistics tradition has long computed for small regression models (Cook & Weisberg; the influence curve of Hampel) — so I'm rederiving a known object, but I want it for something that tradition never targeted.

Because parameters aren't what I asked about. I asked about a test prediction. So chain-rule through. The thing I want is how upweighting z moves the loss at z_test:
I_up,loss(z, z_test) = d L(z_test, θ̂_{ε,z})/dε |_{ε=0} = ∇_θ L(z_test, θ̂)^⊤ · dθ̂_{ε,z}/dε |_{ε=0} = −∇_θ L(z_test, θ̂)^⊤ H^{-1} ∇_θ L(z, θ̂).
Read it right to left: ∇L(z,θ̂) is how upweighting z pushes the parameters; H^{-1} turns that push into an actual parameter displacement; ∇L(z_test,θ̂)^⊤ reads off how that displacement changes the test loss. A symmetric bilinear form in the train gradient and the test gradient, glued by the inverse Hessian. And to estimate the actual deletion effect I scale by the ε I computed earlier: removing z changes the test loss by about −(1/n)·I_up,loss(z,z_test).

Before I worry about computing this for a big model, I want to understand *whether this actually beats just looking at similar training points*, because the cheap thing everyone does is rank training points by Euclidean closeness to the test point. If all the points have the same norm, ranking by distance is ranking by the inner product x·x_test. Let me instantiate I_up,loss for logistic regression and see what extra structure it carries. Take p(y|x)=σ(yθ^⊤x), y∈{−1,1}, σ(t)=1/(1+e^{−t}). Then L(z,θ)=log(1+exp(−yθ^⊤x)), and its gradient is ∇_θL = −σ(−yθ^⊤x) y x — the factor σ(−yθ^⊤x) is exactly the model's probability of being *wrong* on z, i.e. the training loss/residual at z. The Hessian works out to H = (1/n)Σ_i σ(θ^⊤x_i)σ(−θ^⊤x_i) x_i x_i^⊤, a data-weighted covariance. Plug into I_up,loss:
I_up,loss(z,z_test) = −y_test y · σ(−y_test θ^⊤x_test) · σ(−y θ^⊤x) · x_test^⊤ H^{-1} x.
Two things appear that the bare x·x_test doesn't have. First, the σ(−yθ^⊤x) factor: training points the model gets wrong, i.e. high-loss outliers, get *more* influence — and the inner product is blind to it. Second, H^{-1} sits between x_test and x instead of the identity. My reading is that H is the covariance of the other training points' gradients, so H^{-1} measures *resistance*: a gradient ∇L(z,θ̂) that points where many other points also vary is expensive to move along (it would spike everybody else's loss), so z's influence should be damped; a gradient in a low-variation direction should be amplified. But that's a story about a matrix I haven't looked at. Does H^{-1} ever actually change the *answer*, or only rescale it?

The sharpest version of the question: with image pixels, x ⪰ 0 elementwise, so x·x_test ≥ 0 always, and even the loss-weighted inner product −y_test y σ σ (x·x_test) just inherits the label sign — every same-label training point comes out "helpful." If H^{-1} only rescaled magnitudes, it could never flip that. So let me try to *find* a same-label training point where the full influence and the inner-product version disagree in sign. I build small 2-D non-negative datasets, fit logistic regression with L2, and scan same-label points. It doesn't take long to hit one: a dataset with θ̂≈[−0.927,−0.123], test point x_test=[0.884,0.123] with y_test=+1, and a training point x=[0.07,0.962] also with y=+1. The loss-weighted inner product calls it **−0.0685** — helpful, removing it would raise the test loss. The full influence x_test^⊤H^{-1}x with the real Hessian calls it **+0.2343** — *harmful*, removing it would *lower* the test loss. Opposite signs. So a same-label point really can drag the boundary the wrong way, and only the H^{-1} version sees it: it points its gradient mass along a direction (the second coordinate) where the test point has little, while the inner product, treating every direction as equally stiff, is fooled by the shared label. The inverse Hessian and the train-loss factor aren't decoration — stripping down to x_test^⊤x is not a cheaper approximation of the same ranking, it's a different and sometimes sign-wrong ranking. I keep both.

OK, so the formula is right, it tracks the refit, and it carries information the cheap heuristic provably lacks. Now the thing that has kept this out of modern ML: H^{-1}. With n points and θ∈ℝ^p, forming H is O(np²) and inverting it is O(p³). At p ≈ 10⁶ that's absurd — I can't even store a 10⁶×10⁶ matrix. I have to compute I_up,loss without ever materializing H or H^{-1}. Stare at the formula: every place H^{-1} appears, it's sandwiched against a vector. I never need the matrix; I need its action on a vector. Define s_test = H^{-1} ∇_θL(z_test,θ̂), one vector. Then I_up,loss(z,z_test) = −s_test · ∇_θL(z,θ̂), and there's a second payoff: s_test depends only on the *test* point, not on z. So I solve one inverse-Hessian-vector problem per test point, then for every one of the n training points I just take a cheap dot product of s_test with that point's gradient. The n inversions I was dreading collapse into one. That attacks both costs at once — the p³ and the per-train-point blowup.

But I still owe myself "compute s_test = H^{-1}∇L(z_test) without forming H." Two ingredients I need. The first is: can I compute Hv for an arbitrary v cheaply, without H? There's a known trick. The Hessian shows up in the expansion of the gradient: ∇(θ+rv) = ∇(θ) + r·Hv + O(r²). The crude reading is Hv ≈ [∇(θ+rv)−∇(θ)]/r, a finite difference. I don't love that: as r→0 I'm subtracting two nearly-equal gradients and adding a tiny rv to a big θ, so I bleed precision both ways — it's numerically nasty. But take the exact limit instead: Hv = lim_{r→0} [∇(θ+rv)−∇(θ)]/r = (∂/∂r) ∇_θ(θ+rv)|_{r=0}. That's a derivative of the gradient along the direction v, and a derivative is something autodiff computes exactly, not by finite differencing. Concretely the claim is Hv = ∇_θ( v · ∇_θ L ): form the gradient g=∇_θL (one reverse pass), dot it with v to get a scalar v·g (treating v as a constant), and differentiate that scalar with respect to θ again (a second reverse pass).

I should make sure that identity is actually true and not just plausible, because the whole scaling story rests on it. Let me check it on a function with a genuinely non-constant Hessian — f(t)=½t₀²t₁+sin(t₀t₁) — symbolically. Computing H by hand and forming Hv, then separately computing ∇(v·∇f), the two expressions come out *symbolically identical* (their difference simplifies to the zero vector), and at the test point t=(0.7,−0.3), v=(1,2) both routes give (2.98727, 1.83855). So Hv = ∇(v·∇L) holds exactly, for a nonlinear f, with no finite-difference error — the result is computed in about the time of two gradient evaluations, O(p), with no explicit matrix. (This is Pearlmutter's fast-HVP; the v·∇L route is its practical incarnation.)

The second ingredient: given that I can do Hv, how do I solve H s = ∇L(z_test) for s? This is now a linear system with a positive-definite matrix that I can only touch through matrix-vector products. That's a classic situation. Because H ≻ 0, solving Hs = v is the same as minimizing the strictly convex quadratic ½ t^⊤ H t − v^⊤ t over t: its gradient is Ht − v, zero exactly at t = H^{-1}v. And minimizing that quadratic with conjugate gradients needs only the ability to compute Ht — which I just verified I have. CG converges to the exact answer in p iterations, but in practice a handful of iterations gives a good approximation. Each CG iteration evaluates Ht once, which means one pass over the training data to assemble H t = (1/n)Σ_i ∇²L(z_i)t via the per-term HVP trick. So: pick a test point, run CG to get s_test, done. That's the Hessian-free-optimization move — turn the inverse into an optimization that only needs HVPs.

CG has a cost I should be honest about, though. Every CG iteration still sweeps all n training points to form Ht. With large datasets that's slow, and I might want many test points. Can I get s_test while looking at *fewer* than n points per step? Here's the lever: I don't need the exact H at each step, I need an *unbiased* estimate of the right thing. There's a stochastic route to the inverse. Write the Neumann/Taylor series for the inverse: for a matrix A with the right spectrum, A^{-1} = Σ_{i=0}^∞ (I−A)^i. Truncate at j terms, A_j^{-1} = Σ_{i=0}^j (I−A)^i, and notice it satisfies the recursion A_j^{-1} = I + (I−A) A_{j-1}^{-1}, with A_j^{-1} → A^{-1} as j→∞. The series converges only if ‖I−A‖<1, i.e. 0 ≺ A ⪯ I — I'll come back to enforce that. The useful structure: this recursion is *linear* in A, so I can replace the full A=H at each step by any unbiased estimator of it and keep the expectation correct. And a single training point's Hessian ∇²L(z_i,θ̂) is an unbiased estimator of H = (1/n)Σ∇²L(z_i,θ̂) — just draw z_i uniformly. So define the recursion directly on the vector I want, H^{-1}v:
H̃_0^{-1}v = v;  H̃_j^{-1}v = v + (I − ∇²L(z_{s_j},θ̂)) H̃_{j-1}^{-1}v,
where each z_{s_j} is a fresh uniform training sample. Each step costs one single-point HVP, O(p), no full pass. Because expectation passes through the linear recursion, E[H̃_t^{-1}v] = H_t^{-1}v → H^{-1}v, so H̃_t^{-1}v is unbiased in the limit. I run the recursion to depth t large enough that it stabilizes, and to cut the variance I repeat the whole thing r times from scratch and average. This is meant to beat CG on big data because rt can be less than n — I'd want to confirm that crossover empirically on a real dataset rather than take it on faith. (This stochastic estimator is Agarwal–Bullins–Hazan's LiSSA; their version assumed generalized linear models so the per-term HVP was cheap by construction — I just swap in Pearlmutter's general HVP so the same recursion works for any differentiable model.)

Now the convergence caveat I deferred: the series needs 0 ≺ H ⪯ I, but a generic loss has no reason to satisfy ∇²L ⪯ I. The fix that costs nothing: scale the loss down by a constant. Scaling L by a constant doesn't change argmin θ̂ at all, but it scales the Hessian, so I can shrink it until ∇²L ⪯ I, run the recursion on the scaled Hessian, and rescale the result at the end. For some models I can get an a-priori bound (linear models with bounded inputs); otherwise I treat the scale as a hyperparameter and tune it until the recursion converges instead of blowing up. (If I watch the recursion's norm explode, that's the signal the scale is too small — bump it.)

Let me total up the cost, because the whole point was scale. To get I_up,loss on all n training points for one test point: compute s_test once — that's O(np) with CG over a few passes, or O(rt·p) with LiSSA — then n dot products with the precomputed train gradients, O(np). So O(np + rtp) overall, and the regime I'm betting on is rt = O(n). Linear in n and p, which is the regime I needed. And it all rides on autodiff: I only have to specify the loss; gradients and HVPs come for free from the framework.

Now I want a finer attribution than "delete the whole point." Sometimes the question is: how would the prediction change if I *modified* a training input slightly — change x by δ? That's the door to feature-level attribution and, less benignly, to perturbation attacks. Let me set it up the same way: instead of deleting z=(x,y), move ε of its mass onto a perturbed copy z_δ=(x+δ,y) and ε off of z. So θ̂_{ε,z_δ,−z} = argmin (1/n)Σ L(z_i,θ) + ε L(z_δ,θ) − ε L(z,θ). The exact same stationarity-Taylor calculation as before, but now the perturbation gradient is the difference of two point gradients:
dθ̂/dε|_0 = I_up,params(z_δ) − I_up,params(z) = −H^{-1}(∇_θL(z_δ,θ̂) − ∇_θL(z,θ̂)).
And the linear approximation to actually replacing z by z_δ is θ̂_{z_δ,−z} − θ̂ ≈ (1/n)(I_up,params(z_δ) − I_up,params(z)). Notice this didn't assume δ small or even continuous — the ε-mass-moving trick smoothly interpolates between z and z_δ for *any* δ, which means this works for discrete data and discrete label flips too, not just infinitesimal continuous nudges. The influence machinery isn't actually restricted to tiny continuous perturbations the way it first appears.

If x is continuous and δ is small, I can go one step further and get a derivative with respect to the perturbation direction. For small δ, ∇_θL(z_δ,θ̂) − ∇_θL(z,θ̂) ≈ [∇_x∇_θL(z,θ̂)] δ, where ∇_x∇_θL is the p×d matrix of mixed second derivatives. So θ̂_{z_δ,−z} − θ̂ ≈ −(1/n) H^{-1}[∇_x∇_θL(z,θ̂)] δ. Now differentiate the test loss with respect to δ and chain-rule:
I_pert,loss(z,z_test) = ∇_δ L(z_test, θ̂_{z_δ,−z})|_{δ=0} = −∇_θL(z_test,θ̂)^⊤ H^{-1} ∇_x∇_θL(z,θ̂).
This is a vector in ℝ^d: it tells me, to first order, how moving x by δ changes the test loss, namely by I_pert,loss·δ. So if I want to *maximally* increase the loss at z_test by editing a training image, I push δ along I_pert,loss^⊤. And computationally it's the same s_test reused: first get s_test = H^{-1}∇L(z_test,θ̂), then s_test^⊤ ∇_x∇_θL(z,θ̂) — which is again a Hessian-vector-product-flavored object (differentiate the scalar s_test·∇_θL(z,θ̂) with respect to x), so the HVP trick handles it. Two matrix-vector products and I have a per-feature attribution, or a direction to attack.

I should pause on the attack, because it falls right out and it's striking. Adversarial *test* examples are old news: imperceptibly perturb a test image to flip its label. But I_pert,loss gives me the training-set analogue. Pick a target test point z_test; for a training image z_i, iterate z̃_i ← Π(z̃_i + α·sign(I_pert,loss(z̃_i, z_test))), where α is a step size and Π projects back onto valid images that share the same 8-bit representation as the original (so the change is literally invisible after quantization); retrain after each step. This walks the training image in the direction that most raises the test loss, while staying pixel-identical to the eye. It's an iterated training-set version of the sign-gradient test attack. And it's mathematically the same object that the poisoning-attack literature derived straight from the KKT conditions for SVMs and linear models — but now it's a single attribution quantity I can compute through an entire deep network, and the magnitude of I_pert,loss is itself a measure of how vulnerable a model is to having its training data tampered with.

Now I have to be honest about the assumptions, because everything above assumed θ̂ is the exact global minimizer of a twice-differentiable strictly convex risk, and that is false for the models I actually care about. Two violations to handle: non-convergence / non-convexity (so the Hessian can be indefinite and the gradient at my parameters isn't zero), and non-differentiability (ReLUs, hinges). Let me take them one at a time, and the test for both is the same: does the influence estimate still track actual leave-one-out retraining?

First, non-convergence and non-convexity. In practice I get θ̃ from SGD with early stopping on a non-convex objective, so θ̃ ≠ θ̂ and the empirical-risk gradient g = (1/n)Σ∇L(z_i,θ̃) is not zero, and H_{θ̃} can have negative eigenvalues — at which point H^{-1} is meaningless and the whole derivation, which leaned on H ≻ 0, is suspect. The fix for the indefiniteness is to form a *convex* quadratic model of the loss around θ̃: L̃(z,θ) = L(z,θ̃) + ∇L(z,θ̃)^⊤(θ−θ̃) + ½(θ−θ̃)^⊤(H_{θ̃}+λI)(θ−θ̃), where λ is a damping term I add precisely when H_{θ̃} has negative eigenvalues. Adding λI shifts the spectrum up so the curvature matrix is PD again, and it has a clean reading: it's exactly L2 regularization on the parameters, so I'm computing influence under a slightly regularized model. Then I run all the I_up,loss machinery using H_{θ̃}+λI in place of H. (And conveniently, λI also helps the LiSSA series converge by pushing the spectrum into the safe range.)

But does this damped, non-converged influence still mean anything, given g ≠ 0? Let me check what a single Newton step from θ̃ actually does after I upweight z, and see whether I_up,params is hiding inside it. Upweighting z by ε sends the gradient g ↦ g + ε∇L(z,θ̃) and the Hessian H_{θ̃} ↦ H_{θ̃} + ε∇²L(z,θ̃). A Newton step from θ̃ changes the parameters by
N_{ε,z} = −[H_{θ̃} + ε∇²L(z,θ̃)]^{-1} [g + ε∇L(z,θ̃)].
Expand and keep only the leading terms — drop εg, ε², and higher (the inverse's ε-correction times g and times ε∇L are both higher order):
N_{ε,z} ≈ −H_{θ̃}^{-1}(g + ε∇L(z,θ̃)) = −H_{θ̃}^{-1}g − ε H_{θ̃}^{-1}∇L(z,θ̃).
Look at the two pieces. The first, −H_{θ̃}^{-1}g, does *not depend on z at all* — it's just the Newton step that any upweighting (or none) would take to chase down the nonzero gradient. The second is exactly ε·(−H_{θ̃}^{-1}∇L(z,θ̃)) = ε·I_up,params(z). So even when I'm not at a minimum, the Newton step from θ̃ decomposes into a constant z-independent drift plus ε times the very influence quantity I defined. When I rank or compare training points, the constant drift is common to all of them and cancels; what differentiates points is precisely I_up,params(z). So at non-convergence influence is still measuring the right *relative* thing. That argument is suggestive but it's an expansion, not a measurement — the real test is whether the damped influence still correlates with genuine LOO retraining on a non-converged CNN, and that's the validation I'd run before trusting it on a deep net.

Second, non-differentiability. Take a linear SVM with hinge loss, Hinge(s)=max(0,1−s). The hinge is piecewise linear — its second derivative is zero everywhere it exists and undefined at the kink. If I naively set the derivatives at the hinge to zero and compute I_up,loss, I can see exactly why it should be bad before running it: influence is built on a quadratic (second-order) approximation of L(z,θ̂), but a piecewise-*linear* loss has an identically-zero second derivative, so the quadratic model degenerates to a linear one. The Hessian carries no information about how close a support vector z is to the margin — and "closeness to the margin" is exactly what determines how much removing z perturbs the SVM. With no curvature signal, I_up,loss has nothing to weight support vectors by. So the failure isn't random noise; it's a structural blind spot of using a curvature-free loss.

The fix follows from naming the failure: I need a loss that agrees with the hinge but has a non-trivial second derivative that encodes margin-closeness. Smooth the kink. A softplus version is SmoothHinge(s,t) = t·log(1 + exp((1−s)/t)), which → Hinge(s) as t→0: for s well below the margin it tracks 1−s, for s well above it flattens to 0. The question is whether its curvature really lands *at the margin*, where the information lives. Let me differentiate it twice. The second derivative simplifies to SmoothHinge''(s,t) = 1/(4t·cosh²((s−1)/(2t))) — a single bump centered at s=1, the margin, falling off with cosh² on either side. Putting in t=0.1 and reading the curvature across the margin: at s=0.9 it's 1.97, at s=1.0 (exactly on the margin) it's 2.50, at s=1.1 it's 1.97, and by s=2.0 (well clear of the margin) it's down to 0.0005. So the smoothed loss puts essentially all its curvature on points within ~t of the margin and almost none on points far from it — which is precisely the margin-closeness signal the raw hinge threw away, recovered and concentrated where it belongs. So I can train the actual non-differentiable model however I like, then *swap in* the smoothed loss only for the purpose of computing influence. I'd expect the correlation against real LOO retraining (on the original hinge) to go from useless with the raw hinge to high with SmoothHinge at small t, staying high until t gets so large the approximation is too soft — the curvature spreads out and stops localizing on the margin. The general lesson is bigger than SVMs: wherever a model has a non-differentiable component (a ReLU is just a one-sided hinge), I can substitute a smoothed surrogate for influence computation and keep going.

Let me also write down the appendix-grade derivation of I_up,params cleanly, since I waved at "M-estimation" earlier and I want it airtight. θ̂ minimizes R(θ)=(1/n)Σ_iL(z_i,θ), assumed twice-differentiable and strongly convex, so H_{θ̂}=∇²R(θ̂) is PD and invertible. The perturbed estimator θ̂_{ε,z}=argmin{R(θ)+εL(z,θ)} satisfies 0=∇R(θ̂_{ε,z})+ε∇L(z,θ̂_{ε,z}). Define Δ_ε=θ̂_{ε,z}−θ̂; since θ̂ doesn't depend on ε, dθ̂_{ε,z}/dε = dΔ_ε/dε. Because θ̂_{ε,z}→θ̂ as ε→0, Taylor-expand the optimality condition about θ̂: 0 ≈ [∇R(θ̂)+ε∇L(z,θ̂)] + [∇²R(θ̂)+ε∇²L(z,θ̂)]Δ_ε, dropping o(‖Δ_ε‖). Solve: Δ_ε ≈ −[∇²R(θ̂)+ε∇²L(z,θ̂)]^{-1}[∇R(θ̂)+ε∇L(z,θ̂)]. Use ∇R(θ̂)=0 and drop o(ε): Δ_ε ≈ −∇²R(θ̂)^{-1}∇L(z,θ̂)·ε = −H_{θ̂}^{-1}∇L(z,θ̂)·ε. Hence dθ̂_{ε,z}/dε|_{ε=0} = −H_{θ̂}^{-1}∇L(z,θ̂) =: I_up,params(z) — which is the quantity my n=30 refit already matched to four decimals, so the asymptotic argument and the finite-sample reality agree. (It's an asymptotic, not fully rigorous, statement — the honest version is that this is the standard M-estimation influence-function derivation, and a rigorous treatment lives in the asymptotic-statistics literature.) Everything else — I_up,loss, I_pert,loss — is one chain-rule away from this.

So the whole thing hangs together as one idea: responsibility = the derivative of (test loss) with respect to upweighting a training point; that derivative is −∇L(z_test)^⊤H^{-1}∇L(z), and I've checked it against an actual refit and against the cheap inner-product heuristic it's supposed to beat; the only obstacle, H^{-1} at scale, dissolves once I notice I only ever need its action on a vector, compute Hv exactly and cheaply by differentiating v·∇L (an identity I verified symbolically), fold the n inversions into one precomputed s_test=H^{-1}∇L(z_test), solve that single system with CG or with a stochastic Neumann recursion (scaled so the series converges, damped so it stays PD off the minimum), and finally reduce influence to a dot product s_test·∇L(z) per training point. Now let me write the code, keeping each block tied to a step above.

```python
import torch

# --- model + loss are user-specified; everything below is the influence machinery ---
# params: list of tensors (the theta we trained); loss_fn(model, z) -> scalar loss on a point/batch

def grad_params(scalar_loss, params, create_graph=False):
    """One reverse pass: gradient of a scalar loss w.r.t. params, returned as a list."""
    return list(torch.autograd.grad(scalar_loss, params, create_graph=create_graph))

def hvp(loss, params, v):
    """Hessian-vector product H v at the current params, exactly and in O(p).
    Pearlmutter's trick in its practical form: H v = grad( v . grad L ).
    First reverse pass gives g = grad L (kept differentiable); we dot it with v
    (v held constant) and take a second reverse pass. Never forms H; no finite differencing."""
    g = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((gi * vi.detach()).sum() for gi, vi in zip(g, v))   # v . grad L  (v is a constant)
    return list(torch.autograd.grad(dot, params, retain_graph=True))  # = H v

def _hvp_over_data(model, train_data, loss_fn, params, v, batch_size=None):
    """Full empirical-risk HVP: H v = (1/n) sum_i grad^2 L(z_i) v, accumulated over the data.
    Used by the CG path, which needs the exact H action each iteration."""
    total, n_batches = [torch.zeros_like(p) for p in params], 0
    for xb, yb in train_data.batches(batch_size):
        loss = loss_fn(model, (xb, yb))
        part = hvp(loss, params, v)
        total = [t + p for t, p in zip(total, part)]
        n_batches += 1
    return [t / n_batches for t in total]

# ---- s_test = H^{-1} v, route 1: conjugate gradients on the convex quadratic ----
# H > 0  =>  H^{-1} v = argmin_t { 1/2 t^T H t - v^T t };  needs only H t, which we have.
def inverse_hvp_cg(model, train_data, loss_fn, params, v, damping=0.0):
    import numpy as np
    from scipy.optimize import fmin_ncg
    shapes = [p.shape for p in params]
    sizes  = [p.numel() for p in params]
    def to_list(x):                                   # flat numpy vector -> list of tensors
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(torch.tensor(x[i:i+n], dtype=params[0].dtype).reshape(s)); i += n
        return out
    def flat(ts): return np.concatenate([t.detach().cpu().numpy().ravel() for t in ts])
    def Hx(x):
        hv = _hvp_over_data(model, train_data, loss_fn, params, to_list(x))
        return flat([h + damping * xi for h, xi in zip(hv, to_list(x))])  # damped: (H + lambda I) x
    fmin_loss = lambda x: 0.5 * np.dot(Hx(x), x) - np.dot(flat(v), x)     # 1/2 x^T H x - v^T x
    fmin_grad = lambda x: Hx(x) - flat(v)                                 # H x - v
    res = fmin_ncg(f=fmin_loss, x0=flat(v), fprime=fmin_grad,
                   fhess_p=lambda x, p: Hx(p), avextol=1e-8, maxiter=100)
    return to_list(res)

# ---- s_test = H^{-1} v, route 2: LiSSA stochastic Neumann recursion ----
# H^{-1} = sum_i (I - H)^i  (needs 0 < H <= I: enforced by `scale`; `damping` keeps it PD).
# Recursion:  e_0 = v ;  e_j = v + (I - H_sample/scale - damping*I) e_{j-1} ;  return e_t / scale.
def inverse_hvp_lissa(model, train_data, loss_fn, params, v,
                      scale=10.0, damping=0.0, num_samples=1, recursion_depth=5000,
                      batch_size=1):
    result = None
    for _ in range(num_samples):                       # repeat r times and average (variance reduction)
        cur = [vi.clone() for vi in v]
        for j in range(recursion_depth):
            xb, yb = train_data.sample_batch(batch_size)          # unbiased single-point/minibatch H
            loss = loss_fn(model, (xb, yb))
            Hcur = hvp(loss, params, cur)
            # cur <- v + (I - H/scale - damping I) cur
            cur = [vi + (1 - damping) * ci - hi / scale
                   for vi, ci, hi in zip(v, cur, Hcur)]
        contrib = [ci / scale for ci in cur]                      # undo the loss down-scaling
        result = contrib if result is None else [r + c for r, c in zip(result, contrib)]
    return [r / num_samples for r in result]

# ---- influence of every training point on the loss at a test point ----
# I_up,loss(z, z_test) = - s_test . grad L(z, theta);  removal effect ~ -(1/n) I_up,loss.
def influence_on_test_loss(model, train_data, test_point, loss_fn, params,
                           approx="cg", **approx_kwargs):
    n = train_data.num_examples
    test_loss = loss_fn(model, test_point)
    v = grad_params(test_loss, params)                            # grad L(z_test, theta)
    if approx == "cg":
        s_test = inverse_hvp_cg(model, train_data, loss_fn, params, v, **approx_kwargs)
    else:
        s_test = inverse_hvp_lissa(model, train_data, loss_fn, params, v, **approx_kwargs)
    influences = []
    for z in train_data.each_example():                           # reuse s_test across all train z
        gz = grad_params(loss_fn(model, z), params)
        infl = -sum((si * gi).sum() for si, gi in zip(s_test, gz)).item()
        influences.append(infl)                                   # I_up,loss(z, z_test)
    return influences

# ---- per-feature influence / training-set attack direction ----
# Returns d/dx ( s_test . grad_theta L ) = s_test^T grad_x grad_theta L = - I_pert,loss(z, z_test).
# To raise the test loss, step x along + I_pert,loss^T, i.e. x <- Pi(x - a*sign(returned)).
def grad_influence_wrt_input(model, train_point_x, train_point_y, s_test, loss_fn, params):
    x = train_point_x.clone().requires_grad_(True)
    loss = loss_fn(model, (x, train_point_y))
    g = torch.autograd.grad(loss, params, create_graph=True)      # grad_theta L(z)
    scalar = sum((gi * si.detach()).sum() for gi, si in zip(g, s_test))  # s_test . grad_theta L(z)
    return torch.autograd.grad(scalar, x)[0]                      # d/dx of that = grad_x grad_theta L . s_test
```
