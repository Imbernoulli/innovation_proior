OK, let me think about memory from scratch. I have a signal arriving one value at a time — f(t) for t ≥ 0 — and at every instant I want to carry forward a summary of everything seen so far, f(x) for x ≤ t, in a fixed amount of storage that I update incrementally as each new value lands. That's the whole game. The trouble with the obvious tool, a recurrent hidden state, is well known: I update some vector h ← φ(h, f(t)) and hope it remembers the distant past, but when I backpropagate, the sensitivity of a late state to an early input is a long product of Jacobians, and that product almost always decays (or explodes) geometrically in the time gap. So the *effective* horizon is short no matter how big I make h. Gates in LSTMs and GRUs help — they make the update a convex blend, h ← (1−g)h + g·(something), which can hold a value still — and orthogonal RNNs constrain the recurrent matrix to have unit-modulus spectrum so the product of Jacobians doesn't shrink. But all of that is patching the *dynamics*. None of it says what the state is *supposed to be*. I keep feeling like I'm choosing an update rule and then hoping it remembers, instead of saying what "remember" means and letting the rule fall out.

So let me try to say what it means. "Compress the history" — the history is a function f(x) on x ≤ t, which lives in an infinite-dimensional space, and I have N numbers. The cleanest way to compress a function into N numbers is to approximate it inside an N-dimensional subspace and store the coordinates. But "approximate" needs a notion of distance, and distance between functions needs a weighting — which parts of the past do I care about getting right? That's not a nuisance, that's the entire content of "memory": a weighting of the past *is* a memory policy. So let me make it explicit. Put a measure μ on the time axis, giving the inner product ⟨f,g⟩_μ = ∫ f(x) g(x) dμ(x) and the norm ‖f‖_μ = ⟨f,f⟩_μ^{1/2}. Now "best summary in an N-dim subspace 𝒢" has a precise meaning: the g ∈ 𝒢 minimizing ‖f_{≤t} − g‖_μ. And because the past I'm allowed to see only runs up to t, and because what I care about changes as t grows, the measure should itself depend on t: μ^(t) supported on (−∞, t]. So at each t I want the projection of f_{≤t} onto 𝒢 under μ^(t), and I want to maintain its coordinates online.

What subspace 𝒢? I want something where the optimal projection has a *closed form*, not an optimization I have to run at every t. That's exactly what orthogonal polynomials give me. For a measure μ there's a unique family P_0, P_1, … with deg P_n = n and ⟨P_i, P_j⟩_μ = 0 for i ≠ j — just Gram–Schmidt on the monomials. Take 𝒢 = polynomials of degree < N. Then if {p_n} are the orthonormal versions, the best approximation is literally g = Σ_{n<N} c_n p_n with c_n = ⟨f, p_n⟩_μ. No optimization — projection onto an orthonormal basis is just reading off inner products. So the summary I store *is* the coefficient vector c(t) = (⟨f_{≤t}, p_n^{(t)}⟩_{μ^{(t)}})_{n<N}. Polynomials are also more general than they look — sin and cos are polynomials e^{2πix·n} in disguise on the unit circle, so this even contains the Fourier story.

Now the real question: I have c_n(t) = ⟨f_{≤t}, p_n^{(t)}⟩_{μ^{(t)}} as an integral, and I need to update it as t advances without redoing the integral over the whole history every step. So differentiate it in t and see what comes out. Writing the inner product as an integral against the density ω(t,x) = dμ^{(t)}/dx (assume the measure is smooth and normalized; arbitrary scaling doesn't change the optimal projection), and for now no fancy reweighting, so p_n is orthonormal under ω itself:

c_n(t) = ∫ f(x) p_n(t,x) ω(t,x) dx.

The integrand depends on t in two places — the basis function p_n(t,x) and the weight ω(t,x) — so the product rule gives

d/dt c_n(t) = ∫ f(x) (∂_t p_n(t,x)) ω(t,x) dx + ∫ f(x) p_n(t,x) (∂_t ω(t,x)) dx.

Stare at this. The first term: ∂_t p_n is, for the kind of bases I'm using, again a polynomial — and a polynomial of degree no higher than n. So I can expand it back in the same basis p_0, …, and the integral against f turns into a *linear combination of the c_k themselves*. The second term: if ∂_t ω can be written in terms of ω again (plus boundary pieces where the window edge moves), then that integral is also a linear combination of c_k, plus a contribution from the new data f(t) at the moving edge. If both of those happen, then

d/dt c(t) = A(t) c(t) + B(t) f(t)

for some matrices A(t), B(t). That's it — the summary obeys a *linear ODE driven by the input*. The state I was groping for in the RNN is exactly the coefficient vector of the optimal polynomial approximation, and its update law isn't a heuristic, it's forced by differentiating the projection. I just have to pick a measure and turn the crank.

One technical thing about that second term, because the window has a hard edge that moves. When I differentiate an integral whose limits depend on t I get boundary terms — the Leibniz rule. The slick way to keep it automatic is to write the limits as an indicator and differentiate it as a distribution: ∂_t 𝟙_{[α(t), β(t)]} = β'(t) δ_{β(t)} − α'(t) δ_{α(t)}. So ∂_t ω will generically produce a bulk piece (the weight reshaping) plus Dirac spikes at the moving endpoints, and a spike at endpoint x = β just evaluates the rest of the integrand there — i.e. injects the current value f(β). Good, that's where new information enters.

Let me make this concrete with the most natural memory I can think of: remember the most recent stretch of fixed length θ, uniformly. So μ^(t) = (1/θ) 𝟙_{[t−θ, t]} — a window of width θ sliding along with t. Legendre polynomials are the orthogonal family for a uniform measure, so I translate and scale them from [−1,1] onto [t−θ, t]. The change of variable that maps x ∈ [t−θ, t] to [−1,1] is x ↦ 2(x−t)/θ + 1, and the normalized basis (so that ‖p_n‖ = 1 under the uniform probability measure) is

p_n(t,x) = (2n+1)^{1/2} P_n( 2(x−t)/θ + 1 ).

The (2n+1)^{1/2} is just the Legendre normalization: (2n+1)/2 ∫_{−1}^1 P_n² = 1. I'll carry an extra free scale λ_n on each basis vector — λ_n = ±1 keeps it orthonormal, but leaving it loose costs nothing (orthogonality is unchanged by per-vector scaling) and might simplify things later. So g_n = λ_n p_n.

Two endpoint values I'll need, using P_n(1) = 1 and P_n(−1) = (−1)^n:
g_n(t, t) = λ_n (2n+1)^{1/2}, and g_n(t, t−θ) = λ_n (−1)^n (2n+1)^{1/2}.

Now the two derivatives. The weight: ω = (1/θ) 𝟙_{[t−θ, t]}, so by the indicator trick ∂_t ω = (1/θ)(δ_t − δ_{t−θ}) — a spike at the leading edge t and a negative spike at the trailing edge t−θ, no bulk reshaping because the window just translates.

The basis derivative: differentiating P_n(2(x−t)/θ + 1) in t pulls down a factor −2/θ and a P_n'. And here's where the polynomial structure pays off — the derivative of a Legendre polynomial is a finite sum of lower-degree ones: P_n' = (2n−1)P_{n−1} + (2n−5)P_{n−3} + …. So

∂_t g_n = λ_n (2n+1)^{1/2} (−2/θ) [ (2n−1) P_{n−1}(·) + (2n−5) P_{n−3}(·) + … ]

and re-expressing each P_k in terms of the basis vectors g_k = λ_k (2k+1)^{1/2} P_k, i.e. P_k = g_k / (λ_k (2k+1)^{1/2}),

∂_t g_n = −λ_n (2n+1)^{1/2} (2/θ) [ λ_{n−1}^{−1} (2n−1)^{1/2} g_{n−1} + λ_{n−3}^{−1} (2n−5)^{1/2} g_{n−3} + … ].

Now here's a wall. When I assemble d/dt c_n, the second term has those endpoint spikes, and the spike at the *trailing* edge needs f(t−θ) — the value θ ago. But I don't have it anymore; it scrolled out of the window and I refused to store the raw signal, that was the whole point. So the recurrence as written isn't closed. What can I do? I do have a *running approximation* of the history — that's literally what c is for. If the compression is doing its job, then f near t−θ is well modeled by my reconstruction g^{(t)}(x) = Σ_k λ_k^{−1} c_k (2k+1)^{1/2} P_k(2(x−t)/θ + 1), and evaluating that at x = t−θ (where the Legendre argument is −1, so P_k = (−1)^k) gives

f(t−θ) ≈ Σ_k λ_k^{−1} c_k (2k+1)^{1/2} (−1)^k.

So I close the loop by *feeding the state's own estimate of the dropped value back in*. It's an approximation on top of the projection — a real one, worth being honest about — but it makes the dynamics self-contained in c.

Assemble d/dt c_n = ∫ f (∂_t g_n) ω + ∫ f g_n (∂_t ω). The first integral, using the expansion above, gives the lower-triangular-ish coupling to c_{n−1}, c_{n−3}, …; the second gives (1/θ) f(t) g_n(t,t) − (1/θ) f(t−θ) g_n(t,t−θ), and I substitute the reconstruction for f(t−θ). Grinding it through, every term carries λ_n (2n+1)^{1/2} / θ out front and a 1/λ_k on the c_k inside, and the structure collapses to

d/dt c_n = −(λ_n/θ)(2n+1)^{1/2} Σ_k M_{nk} (2k+1)^{1/2} c_k/λ_k + (2n+1)^{1/2} (λ_n/θ) f(t),

where M_{nk} = 1 for k ≤ n (the lower part, from the basis-derivative coupling plus the leading-edge piece) and M_{nk} = (−1)^{n−k} for k ≥ n (from the trailing-edge reconstruction term). Now I cash in the free λ_n. Take the natural λ_n = 1 (genuinely orthonormal). Then

d/dt c = −(1/θ) A c + (1/θ) B f, with A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2} · [1 if k ≤ n, (−1)^{n−k} if k ≥ n], B_n = (2n+1)^{1/2}.

There it is — a fixed, time-invariant linear recurrence (after dividing by θ) that maintains the optimal length-N Legendre summary of a sliding window. And the free λ wasn't idle: if instead I set λ_n = (2n+1)^{1/2} (−1)^n, the same expression rescales to A_{nk} = (2n+1)·[(−1)^{n−k} if k ≤ n, 1 if k ≥ n] and B_n = (2n+1)(−1)^n — which is exactly the Legendre Memory Unit's update matrix. So the LMU, which was reverse-engineered from spiking-neuron delay lines and Padé approximants in the frequency domain with no clean derivation, drops out of one line of this framework as: the optimal sliding-window Legendre projection, in a particular basis scaling. That's the unification I wanted — it's not a clever cell, it's a projection.

Quick sanity detour on how far down this generality goes. Take N = 1 with an exponentially-decaying measure instead (Laguerre family, weight e^{x−t} on x ≤ t); the same machinery gives A = B = 1, and the Euler-discretized update is c(t+Δt) = (1−Δt) c(t) + Δt f(t). If Δt is allowed to depend on the input and current state — chosen adaptively, as a learned gate — that is *exactly* a gated update c ← (1−g)c + g f. So the ubiquitous gate is the order-1, single-polynomial corner of this same picture: gated RNNs use many features each projected to degree 1; this uses one feature projected to high degree. Same principle, opposite extreme. Good. The framework explains the heuristics.

But the sliding window still bugs me. It has a hyperparameter θ, and worse, its behavior is qualitatively wrong at the two ends: when t < θ the window pokes into x < 0 where f isn't even defined, and when t is large the window has long since slid off the beginning, so the model *provably forgets* the start of the sequence by construction. If I genuinely want "remember all of it," a fixed-width window is the wrong object. What I actually want is a window that *grows* with t, covering all of [0, t], so nothing ever scrolls out. So let me set μ^(t) = (1/t) 𝟙_{[0,t]} — uniform over the entire history, normalized. The width now scales with t. Let me just turn the same crank and see what the dynamics look like.

Basis: Legendre on [0, t], normalized, g_n(t,x) = (2n+1)^{1/2} P_n(2x/t − 1) (no reweighting, λ = 1). Now the derivatives, and this is the part to be careful with because *both* the limits and the scaling depend on t.

The weight ω = (1/t) 𝟙_{[0,t]}. Two effects: the 1/t prefactor shrinks, and the upper edge at t moves. So ∂_t ω = −t^{−2} 𝟙_{[0,t]} + t^{−1} δ_t = t^{−1}(−ω + δ_t). Unlike the sliding case there *is* a bulk reshaping term −ω/t now, because the normalization is time-dependent — that's the price of a growing window, and it's going to matter.

The basis: g_n = (2n+1)^{1/2} P_n(z) with z = 2x/t − 1, and ∂_t z = −2x/t². So
∂_t g_n = (2n+1)^{1/2} (−2x/t²) P_n'(z) = −(2n+1)^{1/2} t^{−1} (2x/t) P_n'(z).
Now 2x/t = z + 1, so this is −(2n+1)^{1/2} t^{−1} (z+1) P_n'(z). And there's a Legendre identity tailored to exactly this combination: (x+1)P_n'(x) = nP_n + (2n−1)P_{n−1} + (2n−3)P_{n−2} + … . (It comes from combining P_{n+1}' = (n+1)P_n + xP_n' with the telescoped (2n+1)P_n = P_{n+1}' − P_{n−1}'.) Plugging in,
∂_t g_n = −(2n+1)^{1/2} t^{−1} [ n P_n(z) + (2n−1) P_{n−1}(z) + (2n−3) P_{n−2}(z) + … ].
Note the difference from the sliding case: there's a *diagonal* n P_n term now, not just strictly-lower terms — because the (z+1) factor (a relic of the scaling) bumps the degree back up by one. Re-expressing P_k = (2k+1)^{−1/2} g_k,
∂_t g_n = −t^{−1} (2n+1)^{1/2} [ n (2n+1)^{−1/2} g_n + (2n−1)^{1/2} g_{n−1} + (2n−3)^{1/2} g_{n−2} + … ].

Now assemble d/dt c_n = ∫ f (∂_t g_n) ω + ∫ f g_n (∂_t ω). The first integral turns the g-expansion into the same combination of coefficients:
−t^{−1}(2n+1)^{1/2}[ n(2n+1)^{−1/2} c_n + (2n−1)^{1/2} c_{n−1} + (2n−3)^{1/2} c_{n−2} + … ].
The second integral, with ∂_t ω = t^{−1}(−ω + δ_t): the −ω piece gives −t^{−1} ∫ f g_n ω = −t^{−1} c_n, and the δ_t piece gives t^{−1} f(t) g_n(t,t) = t^{−1} f(t) (2n+1)^{1/2}, since g_n(t,t) = (2n+1)^{1/2} P_n(1) = (2n+1)^{1/2}. So the new value f(t) enters through the leading edge exactly as expected — and crucially there is *no trailing edge*, no f(t−θ) I can't access; the growing window never drops anything, so the recurrence is closed with no reconstruction hack. That alone is a reason to prefer this measure.

Collect the c_n terms. From the basis derivative I get a diagonal contribution −t^{−1}(2n+1)^{1/2} · n(2n+1)^{−1/2} c_n = −t^{−1} n c_n, and from the −ω term another −t^{−1} c_n. Together −t^{−1}(n+1) c_n. Write that diagonal as −t^{−1}(2n+1)^{1/2} · (n+1)(2n+1)^{−1/2} c_n to match the off-diagonal pattern. So

d/dt c_n = −t^{−1}(2n+1)^{1/2}[ (n+1)(2n+1)^{−1/2} c_n + (2n−1)^{1/2} c_{n−1} + (2n−3)^{1/2} c_{n−2} + … ] + t^{−1}(2n+1)^{1/2} f(t).

Vectorize. d/dt c = −(1/t) A c + (1/t) B f, with

A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2} for n > k, n+1 for n = k, 0 for n < k; B_n = (2n+1)^{1/2}.

Let me double-check the off-diagonal entry. For k < n the coupling came from the term (2n − 2(n−k) + 1)^{1/2} c_{n−... } — concretely the c_k term carried coefficient −t^{−1}(2n+1)^{1/2}(2k+1)^{1/2}, so A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2}. And A is strictly lower-triangular off the diagonal (zero for k > n) because the derivative of P_n only reaches *down* to lower degrees. Diagonal n+1, off-diagonal-below the geometric-mean-of-odd-numbers structure. That matches what I derived. This is a *time-invariant* matrix A — the only time dependence is the scalar 1/t out front, which is the footprint of the scaling. Pull out the structure: A = T M T^{−1} with T = diag((2n+1)^{1/2}) and M lower-triangular, M_{nk} = 2k+1 for k < n, k+1 for k = n. So A is a fixed lower-triangular matrix conjugated by a diagonal — that's going to be the key to fast updates.

Now discretize. The basic recipe writes c(t+Δt) − c(t) = ∫_t^{t+Δt} (RHS) and approximates the integral — forward Euler keeps the left endpoint, bilinear averages the two endpoints (more stable), ZOH integrates exactly under piecewise-constant input. For this scaled ODE d/dt c = (1/t)(−A c + B f), forward Euler gives the clean recurrence

c_{k+1} = (1 − A/k) c_k + (1/k) B f_k.

Look at what just happened to the step size. In the sliding-window case I had a genuine Δt hyperparameter sitting in the discretization. Here, because *both* the A term and the B term carry the same 1/t, if I run the generalized bilinear transform on dc/dt = (1/t)(−Ac + Bf) the Δt cancels: the update at step k depends only on the ratios Δt/t = 1/k, not on Δt itself. So this thing has *no timescale hyperparameter at all* — the discrete recurrence is invariant to the step size. That's not a tuning convenience; it's the formal statement of timescale robustness I was after in goal (ii).

Let me actually prove the timescale invariance, because it's the whole selling point. Dilate the input: h(t) = f(αt). Compute its coefficients under this scaled measure,
c̃_n(t) = ⟨h, g_n^{(t)}⟩ = ∫ f(αt') (2n+1)^{1/2} P_n(2x/t − 1) (1/t) 𝟙_{[0,t]}(x) dx.
Substitute x ↦ x/α (so the dummy x rescales): the (1/t) and the P_n(2x/t − 1) and the indicator all reorganize into the same expressions evaluated at αt, and out drops c̃_n(t) = c_n(αt). In words: hippo(f∘γ) = hippo(f)∘γ for any dilation γ(t) = αt. Stretch or compress the time axis and the coefficient trajectory just gets stretched or compressed identically — the operator is equivariant to timescale, because the measure's width is *tied* to t rather than to an external θ or Δt. The sliding-window methods can't do this; their θ and Δt fix an absolute scale.

Second property, gradients — the original motivation. Take the Euler form c_{k+1} = (1 − A/k) c_k + (1/k) B f_k and unroll from step k to ℓ. The sensitivity of a later state to an earlier input is
∂c_{ℓ+1}/∂f_k = (I − A/ℓ)(I − A/(ℓ−1)) ⋯ (I − A/(k+1)) · B/k.
Now A is triangular with diagonal 1, 2, …, N, so its eigenvalues are exactly 1, …, N, and all the factors (I − A/i) are simultaneously diagonalizable in one fixed basis. So this product, in that basis, is diagonal, and its norm is governed by the largest surviving eigenvalue-mode,
ρ = (1 − 1/ℓ)(1 − 1/(ℓ−1)) ⋯ (1 − 1/(k+1)) · (1/k),
coming from the smallest eigenvalue (= 1) of A. And that product telescopes beautifully: ∏_{i=k+1}^{ℓ} (1 − 1/i) = ∏ (i−1)/i = k/ℓ. So ρ = (k/ℓ)(1/k) = 1/ℓ. Hence ‖∂c_{ℓ+1}/∂f_k‖ = Θ(1/ℓ) — and the continuous statement is ‖∂c(t_1)/∂f(t_0)‖ = Θ(1/t_1). The dependence on the gap decays only *polynomially* (in fact only through the endpoint t_1, essentially flat in t_0), not exponentially. The vanishing-gradient problem that started this whole inquiry is gone — not by constraining the recurrent matrix to be orthogonal, but as a free consequence of having derived the *right* state. (If I want the careful version of the telescoping bound: log of the product is Σ_{i=k+1}^ℓ log(1 − 1/i); since log(1 − 1/x) is increasing, the sum is bounded by ∫_k^ℓ log(1 − 1/x) dx, whose antiderivative is x log(1 − 1/x) − log(x−1), and since x log(1 − 1/x) → −1 is Θ(1), the integral is log(k/ℓ) up to Θ(1); exponentiating gives ρ = Θ(1/ℓ), with the inequality asymptotically tight.)

Third, speed. Naively each step is a matrix–vector product, O(N²). But A = T M T^{−1} = D_1 (L + D_0) D_2 where L is the all-ones lower-triangular matrix and the D's are diagonal. Multiplying by I + δA only needs diagonal scalings plus one multiply by L — and multiplying by L is just a cumulative sum, O(N). The harder operation under implicit discretizations (backward Euler, bilinear) is the inverse (I − δA)^{−1}. Factor out the diagonals; the crux is solving (L + D)x = y, i.e. x_0 + … + x_{k−1} = y_k − (1 + d_k)x_k. Let s_k = x_0 + … + x_k be the running sum; then s_k = s_{k−1} + (y_k − s_{k−1})/(1 + d_k) = d_k/(1+d_k) · s_{k−1} + y_k/(1+d_k), a scalar first-order recurrence s_k = a_k s_{k−1} + b_k. And that whole family unrolls in closed vectorized form: dividing through by the cumulative product, s = cumsum(b / cumprod(a)) · cumprod(a). All O(N). So each HiPPO-LegS step costs O(N) under any of these discretizations, versus O(N²) for a generic dense recurrence — the orthogonal-polynomial structure isn't just for the derivation, it makes the operator cheap.

Fourth, how good is the approximation as a function of N? By Parseval, the squared error of the order-N projection is just the tail of the coefficients, ‖f_{≤t} − g^{(t)}‖² = Σ_{n≥N} c_n². So I need the high-order coefficients to decay. Write c_n(t) = (2n+1)^{1/2}/2 ∫_{−1}^1 f((1+x)t/2) P_n(x) dx (change of variables onto [−1,1]) and integrate by parts using P_n = (1/(2n+1))(P_{n+1} − P_{n−1})'. The boundary term vanishes because P_{n+1} and P_{n−1} agree at both ±1 (both equal 1 at x=1, both equal (−1)^{n+1} at x=−1). What's left is c_n = −(1/4)(2n+1)^{−1/2} t ∫_{−1}^1 f'((1+x)t/2)(P_{n+1} − P_{n−1}) dx. If f is L-Lipschitz, |f'| ≤ L, and by Cauchy–Schwarz plus orthogonality (∫P_m² = 2/(2m+1)), c_n² ≤ O(1) t²L²/(2n+1) · [1/(2n+3) + 1/(2n−1)] = O(1) t²L²/n². Summing the tail Σ_{n≥N} 1/n² = O(1/N) gives ‖f_{≤t} − g^{(t)}‖ = O(tL/√N). Vanishes as N grows — and integrating by parts k times for an order-k-smooth f sharpens it to O(t^k N^{−k+1/2}). So smoother inputs are compressed dramatically better, exactly the Fourier-coefficient-decay phenomenon but for Legendre. The summary is provably faithful, not just empirically nice.

So everything I wanted falls out of one move: declare memory to be online optimal projection onto orthogonal polynomials under a chosen measure, differentiate the coefficients, and read off the linear ODE. Pick the sliding uniform measure → recover the LMU and, at order 1, the gate. Pick the *scaled* uniform measure that grows with t → a single fixed matrix A (modulated by 1/t) that needs no window length, no step size, is equivariant to timescale, has Θ(1/t) gradients, O(N) updates, and O(tN^{−1/2}) error. Now let me write the operator. I need the A,B matrices from the derivation, a discretization of the 1/t-scaled ODE into the step recurrence c_k = A_k c_{k−1} + B_k f_k, and a reconstruction that maps coefficients back through the Legendre basis.

```python
import numpy as np
from scipy import linalg as la, special as ss
import torch, torch.nn as nn, torch.nn.functional as F


def transition(measure, N):
    # The (A, B) of the continuous update dc/dt = A c + B f, derived by
    # differentiating the optimal projection coefficients in t. (Here A is the
    # machine form, with negative diagonal -1,-2,...; the prose wrote it as -A
    # with positive diagonal n+1, so this A is that matrix already negated.)
    if measure == 'legt':
        # Translated Legendre: sliding uniform window. Recovers the LMU.
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        # A_{nk} = sqrt(2n+1)sqrt(2k+1) * {1 if k<=n ; (-1)^{n-k} if k>=n}
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    elif measure == 'legs':
        # Scaled Legendre: window [0, t] grows with t. No timescale prior.
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        # M_{nk} = 2k+1 (k<n), k+1 (k=n), 0 (k>n); A = T M T^{-1}, T = diag((2n+1)^{1/2})
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)        # A_{nk}= sqrt(2n+1)sqrt(2k+1) (n>k), n+1 (n=k), 0 (n<k)
        B = np.diag(T)[:, None]             # B_n = (2n+1)^{1/2}
        B = B.copy()
    return A, B


class HiPPO_LegS(nn.Module):
    """Online optimal Legendre projection over the whole history [0, t]."""
    def __init__(self, N, max_length=1024, discretization='bilinear'):
        super().__init__()
        self.N = N
        A, B = transition('legs', N)
        B = B.squeeze(-1)
        # Discretize the 1/t-scaled ODE at each step: A_t = A/t, B_t = B/t,
        # then c_{k} = A_k c_{k-1} + B_k f_k. The step size cancels (no Delta t).
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At, Bt = A / t, B / t
            if discretization == 'forward':                 # c_{k+1}=(I+A/k)c_k + (1/k)B f_k
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else:  # zero-order hold
                A_stacked[t - 1] = la.expm(A * (np.log(t + 1) - np.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked))
        self.register_buffer('B_stacked', torch.Tensor(B_stacked))
        # Reconstruction: g^{(t)}(x) = sum_n c_n (2n+1)^{1/2} P_n(2x/t - 1).
        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor(
            (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)

    def forward(self, inputs):                 # inputs: (L, ...)
        L = inputs.shape[0]
        u = (inputs.unsqueeze(-1).transpose(0, -2) * self.B_stacked[:L]).transpose(0, -2)
        c = torch.zeros(u.shape[1:])
        cs = []
        for k in range(L):                     # c_k = A_k c_{k-1} + (B_k f_k)
            c = F.linear(c, self.A_stacked[k]) + u[k]
            cs.append(c)
        return torch.stack(cs, dim=0)          # (L, ..., N)

    def reconstruct(self, c):                  # coefficients -> approximation of f
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
```

The causal chain, end to end: I refused to pick a memory update by hand and instead asked what the state *should* be — the optimal degree-(<N) polynomial approximation of the history under a measure that encodes which past matters. Orthogonal polynomials make that projection a closed-form coefficient vector c(t). Differentiating c(t) in time turns the projection into a linear ODE dc/dt = A(t)c + B(t)f, because the derivative of a Legendre polynomial is a lower-degree Legendre polynomial and the derivative of the (indicator) measure injects the new value at the moving edge. A fixed-width sliding measure yields the LMU (and, at order 1, the gate) but forgets the start and carries a window hyperparameter. Letting the window instead *scale* with t gives a single fixed matrix A modulated by 1/t — HiPPO-LegS — which forgets nothing, is equivariant to time dilation (no timescale hyperparameter), decays gradients only as Θ(1/t), updates in O(N) via the cumsum structure of A, and approximates an L-Lipschitz signal to O(tL/√N). Discretize the 1/t-scaled ODE and the step size cancels, leaving c_k = A_k c_{k−1} + B_k f_k as the memory cell.
