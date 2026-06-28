Let me start from what actually goes wrong when I train a model across many clients without moving their data. The thing I'm minimizing is an average of per-client losses, f(x) = (1/N) Σ_i f_i(x), and each f_i is itself a stochastic objective — I only ever see g_i(x), a noisy unbiased gradient of f_i, computed on a minibatch sitting on client i. The brutal constraint is the network: talking to clients is slow and flaky, so the only resource I really care about is the number of *communication rounds*. That forces a pattern where each round a client copies the current server model, runs many local steps on its own data, and ships back only the change. And on top of that, in any round I can only reach a small subset S of the N clients, and — the part that bites — the clients' data distributions are arbitrarily different, so the f_i are arbitrarily different functions, with their individual minimizers x_i* scattered far apart and far from the true x*.

So what do I have today? The obvious recipe: each sampled client sets y_i ← x, takes K SGD steps y_i ← y_i − η_l g_i(y_i), and the server averages the results, x ← x + (η_g/|S|) Σ_{i∈S}(y_i − x). For identical clients this is just parallel SGD and it's lovely. But picture two clients with very different data. Client 1's local steps walk y_1 toward x_1*, client 2's walk y_2 toward x_2*. Average the two endpoints and the server lands near (x_1* + x_2*)/2 = (1/N)Σ_i x_i*. And that point is *not* x*. There's no reason the average of the per-client optima should be the optimum of the average loss — unless the clients are identical, in which case all the x_i* coincide. So the aggregated step is systematically biased toward (1/N)Σ_i x_i*, and the size of that bias is exactly the dissimilarity of the clients. Let me name it: client-drift. And the nasty thing is it doesn't come from stochastic noise — set σ = 0, give everyone full-batch gradients, let all clients participate, and the drift is still there, because it's a property of where the *deterministic* local trajectories go. The more local steps K I take to save communication, the further each y_i drifts toward its own x_i*, so the very lever I pull to be communication-efficient makes the bias worse. To stay stable I'm forced to shrink η_l, which directly slows me down. That's the wall: heterogeneity turns my best efficiency lever into a liability.

Let me quantify the heterogeneity so I can reason about it precisely. The clean way is gradient dissimilarity: there exist G, B with (1/N)Σ_i ||∇f_i(x)||² ≤ G² + B²||∇f(x)||² for all x. G measures how much the per-client gradients disagree even where the global gradient is small — at x*, ∇f(x*) = 0 but the individual ∇f_i(x*) need not vanish, and their spread is G. I'd like to know whether that G is something I can tune away or whether it's structural, so let me build the smallest instance that isolates it and actually run the plain method on it. Take two one-dimensional clients, f_1(x) = (μ/2)x² + Gx with ∇f_1 = μx + G, and f_2(x) = −Gx with ∇f_2 = −G. The average is f = (μ/4)x², so ∇f = (μx)/2, and the global optimum is x* = 0. The constant G has cancelled in f — it's pure heterogeneity, invisible to the global objective — yet each client's own gradient carries it. With full batch (σ = 0) and both clients participating every round, the round map is exactly affine in x, so I can solve for its fixed point in closed form rather than guess. Client 1 runs y ← (1 − η_l μ)y − η_l G for K steps; client 2 runs y ← y + η_l G; the server averages the two displacements. Solving the resulting affine map for its fixed point and sweeping η_l (μ = 1, G = 10, K = 10, η_g = 1) I get x_∞ = 0.933 at η_l = 0.02, 0.458 at 0.01, 0.227 at 0.005, 0.0903 at 0.002, 0.0451 at 0.001 — a clean factor-of-2 each time I halve η_l. So the fixed point is nonzero for *every* positive η_l, with the bias scaling like η_l·G·(something); it only reaches x* = 0 in the limit η_l → 0, where progress also stops. That settles it numerically: under heterogeneity the plain method has a residual bias I cannot tune to zero at any usable step size, and its size is governed by G. Anything I build has to make the dependence on G *vanish*, not merely shrink with the step — otherwise I haven't solved heterogeneity, I've only rescaled it.

What's been tried? The first instinct is to stop the clients from wandering: add a penalty (μ/2)||w − x||² to each client's local objective, so the local gradient becomes g_i(w) + μ(w − x), anchoring w to the shared model. Let me think about whether that fixes the drift. It shrinks how *far* w can move in K steps, yes. But stare at the corrected direction: g_i(w) + μ(w − x) still has g_i(w) in it, and g_i(w) points toward x_i*. The penalty pulls back toward x, it doesn't rotate the direction toward x*. So at any fixed distance from x the client is still heading toward its own optimum, just on a shorter leash. The aggregate is still biased toward (1/N)Σ x_i* — the bias is throttled, not removed — and I pay for the throttling with slower progress. If I work out its rate it comes out the same as plain averaging up to the heterogeneity term; there's no actual escape from G. Wall again. Damping the *magnitude* of the local move is the wrong knob. I need to fix the *direction*.

So let me ask the right question: what direction *should* the client take? If communication were free, the ideal local update on client i would use the global gradient, y_i ← y_i − η_l ∇f(y_i) = y_i − η_l (1/N) Σ_j ∇f_j(y_i). That's unbiased toward x* by construction — it's gradient descent on f itself — and it would converge beautifully even for one client at a time, with zero client-drift, because every client would be descending the *same* function f. The whole problem is that computing (1/N)Σ_j ∇f_j(y_i) needs every other client's gradient at y_i, which is exactly the communication I can't afford every step. So the real task is sharp: approximate the global direction (1/N)Σ_j ∇f_j(y_i) using only client i's own gradient g_i(y_i) plus some cheap state, and do it well enough that the approximation error doesn't depend on how heterogeneous the clients are.

Now I recognize this shape. This is the variance-reduction / control-variate situation. The classical move: to estimate E[X] when I have a correlated Y whose mean E[Y] I know, use X − Y + E[Y]; it's unbiased, E[X − Y + E[Y]] = E[X], and its variance is Var(X) + Var(Y) − 2Cov(X,Y), which is small when X and Y are tightly correlated. In finite-sum optimization that's exactly how SAGA and SVRG kill the variance of the sampled gradient: the SGD direction X = g_j(x) gets corrected to g_j(x) − g_j(φ_j) + (1/n)Σ_i g_i(φ_i), where g_j(φ_j) is a stale stored gradient for the sampled component j and (1/n)Σ_i g_i(φ_i) is the table average. The correction subtracts the part of g_j(x) that is *specific to component j* and adds back the average — it turns a one-component sample into something whose mean is the full-sum gradient. The analogy to my setting writes itself: my "components" are the clients, the "sampled component" is client i, and the thing I want is for the corrected local gradient on client i to have mean (1/N)Σ_j ∇f_j — the full-average gradient — instead of ∇f_i.

So let me posit two pieces of state. Each client keeps a control variate c_i that is meant to track its own gradient, c_i ≈ g_i(y_i); the server keeps a control variate c that is meant to track the average gradient, c ≈ (1/N)Σ_j g_j(y_i). Then the corrected local direction is g_i(y_i) − c_i + c. Check what it estimates: if c_i ≈ g_i(y_i) and c ≈ (1/N)Σ_j g_j(y_i), then g_i(y_i) − c_i + c ≈ g_i(y_i) − g_i(y_i) + (1/N)Σ_j g_j(y_i) = (1/N)Σ_j g_j(y_i), which is exactly the ideal global direction. The local update becomes

  y_i ← y_i − η_l (g_i(y_i) − c_i + c).

The correction term (c − c_i) is, intuitively, an estimate of the drift: c_i is where *my* gradient points, c is where the *average* gradient points, and (c − c_i) is the rotation that bends my local step away from x_i* and back toward x*. That's the structural fix the penalty couldn't do — it operates on the direction, not the leash length. And there's a consistency condition I should impose for c to mean "the average": keep the invariant c = (1/N) Σ_i c_i. If c is literally the average of the client control variates, then c being a good estimate of the average gradient follows from each c_i being a good estimate of its own gradient. Initialize everything to zero, c = c_i = 0, which trivially satisfies the invariant; and notice that if I freeze every c_i ≡ 0 forever, the correction (c − c_i) vanishes and I recover plain averaging exactly. So this is a strict generalization of local averaging, with the extra state used only to change the local direction.

I want to know whether this actually removes the G dependence or just relocates it, because that's the whole point and I don't trust the substitution-by-eye above. Let me compare the corrected update against the ideal one and bound the residual

  Σ_i ||(∇f_i(y) − c_i + c) − ∇f(y)||²

by hand. Write e_i := c_i − ∇f_i(y), the per-client tracking error, and use the invariant c = (1/N)Σ_j c_j together with ∇f(y) = (1/N)Σ_j ∇f_j(y), so that c − ∇f(y) = (1/N)Σ_j (c_j − ∇f_j(y)) = ē, the *mean* tracking error. Then a single client's residual is

  (∇f_i(y) − c_i + c) − ∇f(y) = −e_i + ē.

That's a clean cancellation: the corrected direction lands wrong by exactly (mean tracking error − this client's tracking error). Summing the squares, Σ_i ||−e_i + ē||² = Σ_i ||e_i||² − N||ē||² — the standard variance decomposition (the cross term −2⟨e_i, ē⟩ sums to −2N||ē||², leaving Σ||e_i||² − N||ē||²). Since N||ē||² ≥ 0,

  Σ_i ||(∇f_i(y) − c_i + c) − ∇f(y)||² = Σ_i ||c_i − ∇f_i(y)||² − N||ē||² ≤ Σ_i ||c_i − ∇f_i(y)||².

So it's an inequality, not an identity, and the slack is exactly N times the squared mean tracking error. Let me sanity-check the algebra on a tiny instance before I lean on it: N = 2, one dimension, gradients (g_1, g_2) = (3, −1) and controls (c_1, c_2) = (2, 1), so ḡ = 1 and c̄ = 3/2. Corrected residuals are g_1 − c_1 + c̄ − ḡ = 3 − 2 + 1.5 − 1 = 1.5 and g_2 − c_2 + c̄ − ḡ = −1 − 1 + 1.5 − 1 = −1.5, giving LHS = 1.5² + 1.5² = 4.5. The right side Σ||c_i − g_i||² = (2−3)² + (1−(−1))² = 1 + 4 = 5, and the predicted gap N||ē||² with ē = mean(c_i − g_i) = mean(−1, 2) = 0.5 is 2·0.25 = 0.5. Indeed 4.5 = 5 − 0.5. The decomposition holds. What matters is the upshot: the residual depends *only* on how well each c_i tracks ∇f_i(y) — and *not at all* on G, not on how dissimilar the clients are. That's the escape. Plain averaging's residual is ||∇f_i(y) − ∇f(y)||, the raw gradient dissimilarity bounded by G; mine is the tracking error ||c_i − ∇f_i(y)||, which I get to control. And here's why I can control it: f_i is β-smooth, so ∇f_i(y) doesn't move fast as y moves; if c_i is a recently-computed gradient of f_i it stays close to ∇f_i(y). So I should keep the c_i *stateful across rounds* — a client holds onto its c_i and refreshes it when it participates — because smoothness makes a slightly stale gradient still a good approximation. Statefulness isn't an accident; it's what makes the cheap approximation valid.

Now, how exactly do I refresh c_i? The honest version: after the local work, recompute the client's gradient and set c_i⁺ = g_i(x), the gradient at the server model x. That needs one extra pass over the local data beyond the K training steps — call it Option I, clean and stable but costing an extra gradient computation. Can I avoid the extra pass and reuse the gradients I already computed during the K local steps? Let me telescope the local trajectory. Over the round, y_{i,0} = x and each step is y_{i,k} = y_{i,k−1} − η_l (g_i(y_{i,k−1}) + c − c_i). Sum the increments from k = 1 to K:

  y_{i,K} − x = −η_l Σ_{k=1}^K (g_i(y_{i,k−1}) + c − c_i) = −η_l [ Σ_{k=1}^K g_i(y_{i,k−1}) + K(c − c_i) ].

Rearrange to isolate the average local gradient:

  (1/(K η_l)) (x − y_{i,K}) = (1/K) Σ_{k=1}^K g_i(y_{i,k−1}) + (c − c_i).

The left side is just the total displacement divided by Kη_l — quantities the server already has, since it sees x and y_{i,K}. So if I define the refresh as

  c_i⁺ = c_i − c + (1/(K η_l)) (x − y_{i,K}),

then substituting the identity gives c_i⁺ = c_i − c + (1/K)Σ_k g_i(y_{i,k−1}) + (c − c_i) = (1/K) Σ_{k=1}^K g_i(y_{i,k−1}) — the average of the K minibatch gradients I *already computed* during local training, for *free*, no extra pass. That's Option II. Before I commit to it I should verify the telescoping really holds numerically, because the whole "free" claim rests on it: take K = 10, η_l = 0.01, c = 0.3, c_i = −0.2 on the f_1 client above, run the K corrected steps accumulating Σ_k g_1(y_{k−1}), and compare. I get (x − y_K)/(Kη_l) = 10.9961 and (1/K)Σ_k g_1 + (c − c_i) = 10.9961 to machine precision, so c_i⁺ = c_i − c + (x − y_K)/(Kη_l) = 10.4961 equals the direct mean Σ_k g_1/K = 10.4961. The identity is exact, as it must be — it's just the sum of the update increments.

Option II tracks the gradient at the *visited* iterates rather than at x, so the two refreshes are not interchangeable, and I want to know how much that matters before defaulting to it. Let me run both to the fixed point on the heterogeneous two-client instance (μ = 1, G = 10, K = 10, η_g = 1), where I know x* = 0. With Option I (c_i⁺ = g_i(x)) the model goes to |x − x*| = 4.5e−5 by R = 200 and 8.9e−16 by R = 1000 — exact convergence — and the controls settle at c_1 → 10 = ∇f_1(0), c_2 → −10 = ∇f_2(0), the true gradients at the optimum. So Option I genuinely kills the G-floor that defeated plain averaging: same instance, same G, bias gone. I expected Option II to do the same, but running it to R = 50000 it parks at |x − x*| = 0.229 and *stays* there, with c_1 → 5, c_2 → −5 — not the true gradients. That stopped me. It's not a transient; it's a biased fixed point. Hunting for the cause: at K = 1 Option II converges to 0 exactly (there's no within-round drift to corrupt the gradient average), but the residual grows with K (0.025 at K = 2, 0.101 at K = 5, 0.229 at K = 10, 0.492 at K = 20), and — the tell — it scales linearly with η_l: halving η_l from 0.01 to 0.005 to 0.0025 to 0.00125 takes |x − x*| from 0.2291 to 0.1135 to 0.0565 to 0.0282, a clean factor of ≈0.5 each time. So Option II carries an O(η_l) bias for K > 1 because it averages g_i over the *drifted* iterates y_{i,k} instead of at x; that error vanishes as η_l → 0, exactly the small-step regime the convergence analysis will need anyway, whereas plain averaging's bias also scaled like η_l but never vanished relative to the target because it was an order-G term, not a tracking error. That reconciles it: Option II is sound but only in the small-η_l regime; Option I is fixed-point-exact for any η_l. Option II costs nothing and is what the telescoping identity hands me for free, so I'll keep it as the default while remembering this O(η_l) caveat, and keep Option I as the more stable fallback.

Crucially, the local update step uses plain SGD — just g_i plus the fixed correction (c − c_i) added to the gradient, then a vanilla step. I should resist the urge to bolt on momentum here, because the whole Option-II refresh rests on the exact telescoping identity Σ_k (y_{i,k−1} − y_{i,k}) = η_l Σ_k v_{i,k} with v_{i,k} = g_i + c − c_i; a momentum buffer would break that clean relationship between the net displacement and the sum of corrected gradients, and I'd lose the free control-variate update. Plain SGD it is.

Now the server side — aggregation. The model aggregates the usual way: x ← x + (η_g/|S|) Σ_{i∈S}(y_i − x), the average client displacement scaled by a global step η_g. But the control variate c needs care because of sampling. Only the S participating clients refreshed their c_i this round; the other N − S kept their old c_i. I want to maintain c = (1/N)Σ_i c_i, the average over *all* clients. The change in the global average is

  Δc = (1/N) Σ_i (c_i^{new} − c_i^{old}) = (1/N) Σ_{i∈S} (c_i⁺ − c_i),

since only the sampled clients changed. So the server update is

  c ← c + (1/N) Σ_{i∈S} (c_i⁺ − c_i).

Note the divisor is N, not |S| — that's the subtle but essential point. Each participating client sends back its delta Δc_i = c_i⁺ − c_i, the server sums those S deltas and divides by N. Equivalently, c ← c + (|S|/N) · (1/|S|) Σ_{i∈S} Δc_i: take the *mean* of the sampled clients' control deltas and scale it down by the participation fraction |S|/N. If I'd divided by |S| instead, c would no longer equal the true average (1/N)Σ_i c_i — it would over-weight the few clients I happened to sample, the invariant the whole correction relies on would break, and the drift estimate (c − c_i) would be wrong. The |S|/N factor is exactly what keeps c honest as the all-client average despite seeing only S of them.

Why two separate step sizes, η_l local and η_g global? Because they trade off against the drift differently. The effective per-round step is η̃ = K η_l η_g, and that's what sets the descent speed. But the drift accumulated within a round scales with η_l (how big each *local* step is), while the global progress scales with η̃. So I can make η_g large and η_l correspondingly small at fixed η̃: same net progress per round, but smaller local steps mean less within-round drift. Decoupling the two gives me a knob to suppress drift without sacrificing speed — which is also why a careful analysis of even the plain method improves once you allow η_g ≠ 1. In practice I'll often just set η_g = 1, but the freedom matters for the theory.

Let me now convince myself this converges, and at the right rate, because I'm claiming I've removed the heterogeneity dependence and that's a strong claim. I'll track three error quantities round to round. First the client-drift E_r = (1/(KN)) Σ_{k,i} E||y_{i,k}^r − x^{r−1}||², how far clients wander. Second the control-lag C_r = (1/N) Σ_i ||E[c_i^r] − ∇f_i(x*)||², how stale the control variates are relative to the gradients at the optimum. Third the distance to optimum ||x − x*||². The plan is the standard one: bound the second moment of the server update, bound how the drift and the lag evolve, and assemble a one-round contraction.

Start with the variance of the server update. Δx = −(η̃/(KS)) Σ_{k,i∈S} (g_i(y_{i,k−1}) + c − c_i). I want E||Δx||². Split it with the relaxed triangle inequality ||Σ v_j||² ≤ τ Σ||v_j||² into four groups: the gradient-vs-its-mean part (which contributes the within-client variance σ²/(KS) after separating mean from noise), the c term, the (∇f_i(x*) − c_i) term, and the (∇f_i(x) − ∇f_i(x*)) term. The last one is ≤ 2β(f(x) − f*) by smoothness-convexity. Using c = (1/N)Σ c_i to fold the c and (∇f_i(x*) − c_i) pieces together into the control-lag, and Lipschitzness to turn Σ||∇f_i(y_{i,k−1}) − ∇f_i(x)||² into β²·(drift), I get

  E||x^r − x^{r−1}||² ≤ 8β η̃² (f(x) − f*) + 8 η̃² C_{r−1} + 4 η̃² β² E_r + 12 η̃² σ²/(KS).

Good — the server-update size is controlled by suboptimality, lag, drift, and noise, with no bare G anywhere.

Next, how does the control-lag evolve? After round r, c_i⁺ = (1/K)Σ_k g_i(y_{i,k−1}) for the sampled clients (probability S/N each) and c_i⁺ = c_i otherwise. Taking expectations, E[c_i^r] = (1 − S/N) E[c_i^{r−1}] + (S/N)·(1/K)Σ_k E[∇f_i(y_{i,k−1})]. Plug into the definition of C_r, use Jensen to pull the average inside the norm, then the relaxed triangle inequality plus Lipschitzness (∇f_i(y_{i,k−1}) − ∇f_i(x*) split through ∇f_i(x^{r−1})) to land

  C_r ≤ (1 − S/N) C_{r−1} + (S/N)( 4β(f(x) − f*) + 2β² E_r ).

So the lag *contracts* by (1 − S/N) each round — it forgets stale information at the participation rate — and is refreshed by terms proportional to current suboptimality and drift. The more clients I sample, the faster the lag heals.

Now the drift itself. Expand E||y_{i,k} − x||² one local step at a time. The update inside is y_i − η_l(g_i(y_i) + c − c_i); separate the noise (η_l²σ²), then use the relaxed triangle inequality with parameter a = 1/(K−1) to split into a contractive piece and the correction piece. The contractive piece is ||y_i − η_l∇f_i(y_i) + η_l∇f_i(x) − x||²: this is exactly the contractive-mapping lemma — a gradient step on a β-smooth μ-convex function is a contraction for η_l ≤ 1/β — so it's ≤ ||y_i − x||². The correction piece T_3 = (1/N)Σ_j||c − c_j + ∇f_j(x)||² expands, using c = (1/N)Σc_j and smoothness, into ≤ (6/N)Σ_j||c_j − ∇f_j(x*)||² + 6β(f(x) − f*). Assemble into a within-round recursion

  (1/N)Σ_i E||y_{i,k} − x||² ≤ (1 + 1/(K−1)) (1/N)Σ_i E||y_{i,k−1} − x||² + 7η_l²σ² + 6η_l²Kβ(f(x)−f*) + 6Kη_l² C_{r−1},

and unroll over k. The growth factor (1 + 1/(K−1))^k is the only delicate bit; I claim (K−1)((1+1/(K−1))^K − 1) ≤ 3K, and I should actually check it rather than wave at it. K = 2: (1)((1+1)² − 1) = (4 − 1) = 3 ≤ 6. K = 3: (2)((1.5)³ − 1) = 2(3.375 − 1) = 4.75 ≤ 9. For K ≥ 4 the slick bound is (1+1/(K−1))^K < exp(K/(K−1)) ≤ exp(4/3) ≈ 3.79, so the whole expression is < (K−1)(exp(4/3) − 1) < (K−1)(2.79) < 3K. To be sure the constant 3 isn't optimistic I tabulated the ratio (K−1)((1+1/(K−1))^K − 1)/(3K) up to K = 10⁵: it rises monotonically from 0.50 (K=2) toward a limit of (e−1)/3 ≈ 0.573, never exceeding 0.573, so 3K is a valid bound with room to spare. So the accumulated drift is bounded by 3K times the per-step terms, and after averaging and multiplying by 3β,

  3β η̃ E_r ≤ (2η̃²/3) C_{r−1} + (η̃/(25η_g²))(f(x) − f*) + (η̃²/(Kη_g²)) σ²,

once η_l ≤ 1/(81βKη_g) so that η̃ ≤ 1/(81β) makes the messy constants (54β²η̃² etc.) collapse below 1/25 and 2/3. The drift is small precisely when the local step is small — which is exactly the regime the decoupled η_g lets me sit in without losing round-level speed.

Assemble. I want a Lyapunov function that contracts. The model distance alone won't close because the server-variance bound feeds on C_{r−1}, and C evolves on its own. So combine them: Φ_r = ||x^r − x*||² + (9N η̃²/S) C_r. The weight 9Nη̃²/S on the lag is chosen so the lag terms telescope cleanly across the three inequalities. Starting from ||x + Δx − x*||² = ||x − x*||² − (2η̃/(KS)) Σ_{k,i∈S}⟨∇f_i(y_{i,k−1}), x − x*⟩ + ||Δx||², bound the cross term T_4 by perturbed strong convexity (with h = f_i, the gradient read at the drifted point y_{i,k−1}, progress measured from x toward x*): T_4 ≤ −2η̃(f(x) − f* + (μ/4)||x − x*||²) + 2βη̃ E_r. Plug in the server-variance bound for ||Δx||², add (9Nη̃²/S) times the control-lag recursion and the drift bound, and watch the bookkeeping. The (f − f*) coefficients sum to (44βη̃² − (49/25)η̃), which is ≤ −η̃ once η̃ ≤ 1/(81β); the drift coefficients sum to (22β²η̃² − βη̃) ≤ 0; the leftover lag coefficient (9μη̃N/(2S) − 1/3)η̃² C_{r−1} is ≤ 0 once η̃ ≤ S/(15μN). What survives is

  E[Φ_r] ≤ (1 − μη̃/2) E[Φ_{r−1}] − η̃ (f(x^{r−1}) − f*) + (12η̃²/(KS))(1 + S/η_g²) σ².

That's a clean contraction: Φ shrinks by (1 − μη̃/2) per round, pays down suboptimality at rate η̃, and accumulates only the genuine within-client noise σ², scaled by 1/(KS) — and *no G term anywhere*. Heterogeneity has dropped out of the rate entirely, which is what I set out to prove.

Unroll it. With weights w_r = (1 − μη̃/2)^{1−r} the suboptimality terms telescope, and tuning η̃ (small enough for the contraction, large enough to make progress) gives, in the strongly convex case, a number of rounds

  R = Õ( σ²/(μ K S ε) + β/μ + N/S ).

Read the terms. σ²/(μKSε) is the statistical term: it's exactly the rate of SGD with a batch K times larger, since each round does KS stochastic gradients — so SCAFFOLD is *at least as fast as SGD*, and unlike plain averaging this holds for *arbitrarily heterogeneous* clients. β/μ is the optimization condition number, unavoidable. N/S is the price of partial participation: it's how long it takes the control-lag to heal when I only see a fraction S/N of clients per round, and it shows up additively, not multiplicatively, so sampling slows me by a bounded amount rather than crippling me. Set S = N (full participation) and σ = 0 and the strongly convex rate is just the condition-number term plus the one-round participation term — linear convergence with no heterogeneity penalty.

There's a sanity check sitting right here that I can actually run, not just assert. Take σ = 0, K = 1, S = 1, η_g = 1, and use Option I (c_i⁺ = g_i(x)). Then each round samples one client i, takes one corrected step x ← x − η(∇f_i(x) − c_i + c) with c = (1/N)Σ_j c_j, and overwrites c_i ← ∇f_i(x). Line that up against SAGA on the finite sum (1/N)Σ_j f_j: sample j, step x ← x − γ(∇f_j(x) − φ_j + (1/N)Σ_l φ_l), then set φ_j ← ∇f_j(x). With c_i ↔ φ_i and c = mean φ these are character-for-character the same map. Rather than trust my eye on that, I coded both on a random N = 4, d = 3 strongly-convex quadratic sum, started them from identical state, drove them with the same sampled index sequence for 500 steps, and measured max ||x_SCAFFOLD − x_SAGA||: it was 0.0 — bit-for-bit identical, not merely close — and the common iterate converged to x* (residual 3.6e−16). So in this corner the construction *is* SAGA, with clients as the finite-sum components and the stored c_i as the gradient table; the whole thing is the lift of SAGA's variance reduction from "sample one component, one step" to "sample S clients, K local steps each, intermittent availability." Because the iterations coincide exactly, the σ = 0 rate must inherit SAGA's condition-number-plus-table form, β/μ + N/S in the strongly convex case (N/S being the table-refresh time when only 1 of N components is touched per step) — that's not a separate claim to prove but the same algorithm read in two notations.

Let me push on one more question, because it's been nagging: when are local steps actually *worth it*? Plain averaging gets *worse* with more local steps under heterogeneity (more drift), but intuitively, if the clients are similar, taking many local steps should let each client make real progress on a shared landscape and save communication. Which notion of similarity governs this? Not gradient dissimilarity G — that can be huge even when local steps help. Let me look at the corrected update with a first-order expansion around x, taking quadratics for clarity (so the Hessians A_i are constant). The corrected direction is ∇f_i(y) − ∇f_i(x) + ∇f(x). Taylor-expand each piece about x:

  ∇f_i(y) − ∇f_i(x) ≈ ∇²f_i(x)(y − x),   and   ∇f(x) ≈ ∇f(y) + ∇²f(x)(x − y).

Add them:

  ∇f_i(y) − ∇f_i(x) + ∇f(x) ≈ ∇f(y) + (∇²f_i(x) − ∇²f(x))(y − x).

So the corrected local direction equals the *ideal* global direction ∇f(y) plus an error (∇²f_i(x) − ∇²f(x))(y − x). For quadratics the Hessians are constant, so this expansion isn't an approximation — it should be an exact equality, and that's worth confirming because the whole "δ not G" conclusion hangs on it. I took N = 3 random SPD quadratics f_i(x) = ½xᵀA_i x − b_iᵀx in d = 4, a random x and a random y = x + perturbation, and compared the exact corrected direction ∇f_i(y) − ∇f_i(x) + ∇f̄(x) against ∇f̄(y) + (A_i − Ā)(y − x) for each client. The two matched to machine precision for all three clients (e.g. ||corrected − ideal|| = 8.124 = ||(A_i − Ā)(y − x)|| exactly), confirming the corrected direction is the ideal one plus precisely (A_i − Ā)(y − x). That error is bounded by δ·||y − x|| where δ is the *Hessian* dissimilarity ||∇²f_i − ∇²f|| ≤ δ. Not G — δ. The drift therefore contracts through ||I − η(A_i − A)||² ≤ (1 + ηδ)² ≤ 1 + 3ηδ, so it stays controlled exactly when the Hessians are close. Working the quadratic analysis through, the σ = 0 communication rate becomes (β + δK)/(μK) = β/(μK) + δ/μ. The β/(μK) term falls with local work until around K = β/δ, where it is comparable to the δ/μ floor; beyond that, extra local steps mainly spend computation because the communication bound has saturated at the Hessian-mismatch term. If the clients share a common curvature (δ = 0) the rate improves *linearly* with K — local steps are pure profit — even if their optima x_i* are arbitrarily far apart and G is unbounded. That's the surprise the gradient-dissimilarity lens completely misses: what makes local steps pay off is similarity of the *Hessians*, not closeness of the optima. And it explains the contrast with the exact proximal-point ancestor that also takes "large K": solving the subproblem exactly buys a δ²/μ² dependence, while taking K stochastic gradient steps and correcting them with control variates buys δ/μ — a quadratic improvement — because the control variate fixes the direction directly instead of relying on a proximal anchor.

Let me also confirm the partial-participation story holds together, since sampling is where naive variance reduction usually falls apart. The danger is that with only S of N clients refreshing their c_i, the global c drifts away from the true average gradient. The defense is the invariant c = (1/N)Σ_i c_i, which the |S|/N-scaled server update c ← c + (1/N)Σ_{i∈S}(c_i⁺ − c_i) is supposed to preserve — but "supposed to" is exactly the kind of claim I keep finding to be subtler than it looks, so let me drive it. With N = 6, S = 3, feeding arbitrary refresh targets to the sampled clients for 300 rounds and tracking ||c − mean_i c_i|| at each step: with the divide-by-N update the gap stays at 1.2e−15 (machine zero) for all 300 rounds, so c tracks the true all-client mean exactly despite only ever touching half the clients. To be sure the N is load-bearing and not incidental I reran the same sampled stream with a divide-by-|S| update, and the gap grew to ≈1.9 — c no longer equals mean_i c_i, the invariant the whole correction depends on is broken, and the over-weighting of the sampled clients is precisely the failure I worried about. So the |S|/N factor is doing real work, not cosmetic bookkeeping. Granting the invariant, the control-lag recursion C_r ≤ (1 − S/N)C_{r−1} + … shows the rest: the lag introduced by *not* updating N − S clients decays geometrically at rate (1 − S/N) and only adds the N/S term to the round count. The method is "client-variance-reduced": it's relatively unaffected by which clients I happen to sample, which is exactly what I need when availability is out of my control.

So let me write the whole thing as code that drops into the federated harness — filling the slots the strategy left open: the state I keep (the global control variate, and one per client), the correction added to the local gradient before each plain-SGD step, the Option-II control refresh from the free telescoped gradient average, and the server aggregation with the |S|/N-scaled control update.

```python
from collections import OrderedDict
import random

import torch
from torch import optim
from torch.utils.data import DataLoader


class Strategy:
    """SCAFFOLD: stochastic controlled averaging. Keeps a server control variate c
    and one client control variate c_i per client; corrects each local gradient by
    (c - c_i), refreshes c_i from the free telescoped gradient average (Option II),
    and aggregates with the |S|/N-scaled control update so c stays = mean_i c_i."""

    def __init__(self, global_model, args):
        self.args = args
        self.num_clients = args.num_clients
        # server control variate c, initialized to 0 (so c = mean_i c_i = 0 holds)
        self.global_control = OrderedDict(
            (k, torch.zeros_like(v)) for k, v in global_model.state_dict().items()
        )
        self.client_controls = {}          # client_idx -> OrderedDict (its c_i), stateful
        self._pending_delta_c = {}          # client_idx -> Delta c_i to aggregate this round
        self.global_lr = getattr(args, "global_lr", 1.0)   # eta_g

    def _zero_like(self, state_dict):
        return OrderedDict((k, torch.zeros_like(v)) for k, v in state_dict.items())

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()

        # c_i for this client (lazily created, zero on first sight); c is the server control.
        c = OrderedDict((k, v.to(device)) for k, v in self.global_control.items())
        if client_idx not in self.client_controls:
            self.client_controls[client_idx] = self._zero_like(model.state_dict())
        c_i = OrderedDict((k, v.to(device))
                          for k, v in self.client_controls[client_idx].items())

        # snapshot the server model x (= y_{i,0}) for the Option-II telescoping refresh
        x = OrderedDict((n, p.detach().clone()) for n, p in model.named_parameters())
        # correction (c - c_i) per trainable parameter, fixed across the K local steps
        correction = {}
        for n, p in model.named_parameters():
            if n in c:
                correction[id(p)] = c[n] - c_i[n]

        optimizer = optim.SGD(model.parameters(), lr=local_lr)   # plain SGD: keeps telescoping exact
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)

        total_loss, total_samples, local_steps = 0.0, 0, 0
        for _ in range(local_epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                # corrected local gradient: g_i + (c - c_i)  -> direction tracks the global gradient
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    corr = correction.get(id(p))
                    if corr is not None:
                        p.grad.add_(corr)
                optimizer.step()                                  # y_i <- y_i - eta_l (g_i + c - c_i)
                local_steps += 1
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        # Option II refresh: c_i^+ = c_i - c + (x - y_{i,K}) / (K * eta_l)  =  mean_k g_i(y_{i,k-1})
        if local_steps > 0 and local_lr > 0.0:
            denom = local_steps * local_lr
            delta_c = OrderedDict()
            new_ci = OrderedDict()
            for n, p in model.named_parameters():
                if n not in c:
                    continue
                new = c_i[n] - c[n] + (x[n] - p.detach()) / denom
                delta_c[n] = new - c_i[n]                          # Delta c_i to send to the server
                new_ci[n] = new
            self._pending_delta_c[client_idx] = delta_c
            self.client_controls[client_idx] = OrderedDict(
                (k, v.cpu()) for k, v in new_ci.items())            # keep c_i stateful across rounds

        final_state = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())
        avg_loss = total_loss / max(total_samples, 1)
        return final_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # model update: x <- x + eta_g * mean_i(y_i - x).
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not torch.is_floating_point(ref):
                new_state[key] = ref.clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for st, n, _ in client_updates:
                acc += st[key].float() - ref.float()
            acc = ref.float() + self.global_lr * (acc / max(len(client_updates), 1))
            new_state[key] = acc.to(ref.dtype)

        # server control update: c <- c + (|S|/N) * mean_i Delta c_i  (divide by N, not |S|)
        deltas = self._pending_delta_c
        if deltas:
            n_updates = len(deltas)
            weight = n_updates / max(self.num_clients, 1)             # |S| / N
            for key in self.global_control:
                acc = None
                for dc in deltas.values():
                    if key in dc:
                        contrib = dc[key].to(self.global_control[key].device)
                        acc = contrib.clone() if acc is None else acc + contrib
                if acc is not None:
                    self.global_control[key] = (
                        self.global_control[key] + (weight / n_updates) * acc)
            self._pending_delta_c = {}
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
```

So the causal chain is whole. I started stuck with local-averaging under heterogeneous clients, where each client descends its own loss and drags the aggregate toward the average of the client optima rather than the true optimum — a client-drift that gets worse the more local steps I take and whose size is exactly the gradient dissimilarity G, provably impossible to remove by tuning the step size. Penalizing distance to the shared model only shortens the leash; it leaves the local direction pointing the wrong way, so the bias survives. The fix had to act on the direction: approximate the unaffordable ideal global gradient (1/N)Σ_j ∇f_j(y_i) using only the local gradient plus cheap state. Recognizing this as a control-variate problem — the same variance-reduction template that turns SAGA's sampled component-gradient into the full-sum gradient — I keep a per-client c_i tracking that client's gradient and a server c tracking the average, with the invariant c = (1/N)Σ c_i, and correct the local step to g_i − c_i + c, whose residual against the ideal depends only on the tracking error ||c_i − ∇f_i||, not on G; smoothness keeps that tracking error small, so the c_i are kept stateful. Telescoping the local trajectory hands me a free refresh c_i⁺ = mean of the already-computed local gradients (Option II), the server keeps c as the true all-client average by scaling the control update by |S|/N, and decoupled local/global step sizes let me suppress within-round drift without losing per-round speed. The Lyapunov analysis — server-variance, control-lag contraction at rate (1 − S/N), drift bound, assembled into Φ = ||x − x*||² + (9Nη̃²/S)C — gives a contraction with no G term, hence a rate Õ(σ²/(μKSε) + β/μ + N/S) that matches SGD with K× the batch under arbitrary heterogeneity and reduces to SAGA when σ = 0, K = 1, S = 1. And a first-order expansion of the correction shows the leftover error is the *Hessian* dissimilarity δ, not G, so local steps pay off — linearly in K when the curvatures match — exactly when the Hessians are close, regardless of how far apart the client optima sit.
