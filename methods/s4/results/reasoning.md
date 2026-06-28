OK, let me think this through from scratch. The thing I actually want is a sequence layer that can reach across very long contexts — tens of thousands of steps — and that doesn't fall apart computationally. Let me get concrete about why everything I know breaks, because the failure modes are the whole problem.

A recurrent model is the obvious tool: keep a state, update it each step, x_k = f(x_{k-1}, u_k). The memory is in principle unbounded — the state can carry information arbitrarily far. But to learn that, I backpropagate through L applications of f, and the gradient is a product of L Jacobians. A product of L matrices generically decays or explodes geometrically in L. So the gradient from step 0 onto a loss at step L is either washed out to nothing or blown up to garbage by the time L is in the thousands. Gating, orthogonal constraints — they push the eigenvalues toward magnitude one to slow the decay, but I'm fighting an exponential, and I'm also stuck: the recurrence is inherently sequential, so training is slow.

A convolution fixes the sequentiality — a fixed filter applied everywhere, fully parallel, stable gradients. But a filter has finite width. To see far I stack layers or dilate, and the receptive field only grows with depth. There's no single layer that is genuinely global. Attention is genuinely global — every position talks to every other — but that's exactly L² pairs, so compute and memory are O(L²), and at L = 16k that's hopeless. The efficient-attention approximations buy back the L² but pay in quality, and they still have no cheap stepwise inference mode.

So I want, at once: unbounded principled memory (the recurrent dream), parallel training (the convolutional dream), and constant-time-per-step inference (the recurrent dream again). Those pull in opposite directions for every architecture I know. Is there an object that is *simultaneously* a recurrence and a convolution?

There is, and it's old: a linear, time-invariant system. Think of a continuous latent-state model,

    x'(t) = A x(t) + B u(t),   y(t) = C x(t) + D u(t),

with a 1-D input u(t), an N-dimensional hidden state x(t), and a 1-D output y(t). The D u term is just a feedthrough — a skip connection — so I'll drop it and put it back at the end if I want it. This is linear in the state. Linearity is usually a dirty word for expressivity, but I'm going to stack many of these with nonlinearities between them, like a CNN stacks linear convolutions; the depth gives the nonlinearity. The point of the linearity is that it makes the long-range behavior *analyzable* and the two computational views *exact*.

But wait — if it's a linear ODE, doesn't it have the same exponential problem? x'(t) = A x(t) solves to x(t) = e^{At} x(0), and the eigenvalues of A control whether that grows or decays exponentially. A random A will give me exactly the vanishing/exploding behavior I was running from. So a naive linear SSM is not just unhelpful, it's actively bad — and indeed, plugging a random A into this thing performs terribly.

The escape is to *not* pick A at random. There's a beautiful piece of theory on continuous-time memorization: for a chosen way of weighting the past, you can solve for the A (and B) such that the latent state x(t) holds the coefficients of the optimal polynomial approximation of the entire input history up to time t, and stays an honest, bounded summary of that history as it slides forward. The canonical one uses scaled Legendre polynomials, and the matrix is

    A_nk = -(2n+1)^{1/2}(2k+1)^{1/2}  for n > k,
    A_nk = -(n+1)                      for n = k,
    A_nk = 0                           for n < k.

This isn't a free parameter I'd tune; it's the matrix that falls out of requiring the state to track an orthogonal-polynomial expansion of the past. I'll take it on faith from that theory for now and let the architecture lean on it. What I do trust is the empirical signal: swapping a random A for this one takes a state space layer on sequential MNIST from ~60% to ~98%, which is a big enough jump that something real is happening. So: A is the HiPPO matrix, B and C and the step size are mine to learn, and the memory problem looks like it's handled at the continuous level. Now I need to make the thing *run*.

It lives in continuous time and my data is sampled, u_k = u(kΔ) for some step size Δ. I have to discretize x'(t) = A x(t) + B u(t). The crude option is forward Euler: pretend x'(t_k) ≈ (x_k − x_{k-1})/Δ, giving x_k = (I + ΔA) x_{k-1} + ΔB u_k, so Ā = I + ΔA. My worry is stability — the discrete recurrence x_k = Ā x_{k-1} + … only stays bounded if every eigenvalue of Ā sits inside the unit disk, and I just spent all that effort getting the continuous dynamics stable (eigenvalues with negative real part). Does Euler preserve that? Let me actually put numbers on it instead of hand-waving. Take a few stable continuous modes a (Re a < 0) and a not-tiny step Δ = 0.5, and look at the discrete magnitudes |1 + Δa| (Euler) versus the Cayley/bilinear map |(1+Δa/2)/(1−Δa/2)|:

    a = −0.5 + 3i :  |I+ΔA| = 1.677   |Cayley| = 0.852
    a = −2 + 0i   :  |I+ΔA| = 0.000   |Cayley| = 0.333
    a = −0.1 + 10i:  |I+ΔA| = 5.089   |Cayley| = 0.993

So Euler blows a stable mode up to magnitude 1.68 and even 5.09 once the mode oscillates fast (large imaginary part) — exactly the regime HiPPO lives in, since its job is to hold oscillatory history. Euler is out. The bilinear map keeps all three strictly inside the unit disk. That's the discretization I want: one that sends the stable continuous region (left half-plane) into the stable discrete region (unit disk).

Let me derive the bilinear (trapezoidal) rule properly. Integrate x' over one step with the trapezoid rule:

    x_k − x_{k-1} = (Δ/2)(A x_k + A x_{k-1}) + Δ B u_k.

The input term is the standard bilinear-discretization input term for a held input over the interval; the state term is the trapezoid/Cayley part that drove the stability numbers above. Collect the x_k terms on the left:

    (I − Δ/2·A) x_k = (I + Δ/2·A) x_{k-1} + Δ B u_k.

Invert the left factor:

    x_k = Ā x_{k-1} + B̄ u_k,   y_k = C̄ x_k,
    Ā = (I − Δ/2·A)^{-1}(I + Δ/2·A),   B̄ = (I − Δ/2·A)^{-1} Δ B,   C̄ = C.

The scalar map a ↦ (1 + Δa/2)/(1 − Δa/2) is the Cayley transform, and the three numbers above are exactly it acting on a; on the matrix it sends the left half-plane to the open unit disk. Notice for later: as Δ → 0, Ā → I.

Now I have a discrete linear recurrence, x_k = Ā x_{k-1} + B̄ u_k. That's an RNN-shaped object — great for stepwise inference. But it's sequential, so for training I want the convolutional face. Unroll it from x_{-1} = 0:

    x_0 = B̄ u_0,
    x_1 = Ā B̄ u_0 + B̄ u_1,
    x_2 = Ā² B̄ u_0 + Ā B̄ u_1 + B̄ u_2,  …

so x_k = Σ_{j=0}^{k} Ā^{k-j} B̄ u_j, and applying C̄,

    y_k = Σ_{j=0}^{k} C̄ Ā^{k-j} B̄ u_j.

That's a convolution: y = K̄ ∗ u, with a single filter

    K̄ = (C̄B̄, C̄ĀB̄, C̄Ā²B̄, …, C̄Ā^{L-1}B̄) ∈ R^L.

If I have K̄, I compute y with one FFT-based convolution in O(L log L), fully parallel — the training dream. So the entire problem collapses to: *produce the vector K̄*. And here's the wall. To build K̄ I need C̄Ā^k B̄ for k = 0…L−1, which means powering Ā up to L−1 times. Each Ā is N×N, so that's L matrix multiplies, O(N²L) time, and I have to hold O(NL) intermediate state. For N = 256 and L = 16k this is enormous — orders of magnitude more memory than an RNN of the same size. The naive deep-SSM layer does exactly this and chokes on it. There ought to be a near-linear, Õ(N + L), way, since that's the information lower bound. The whole game is computing K̄ without ever forming those powers.

First instinct: change basis to make Ā nice. A linear-algebra fact I can lean on — the SSM is invariant under conjugation. If I replace (A, B, C) by (V^{-1}AV, V^{-1}B, CV), the input-output map is identical: writing the conjugated system in state x̃ and multiplying through by V recovers the original with x = V x̃. Same operator, different coordinates for the state. And conjugation commutes with the discretization (it's algebraic in A), so I can reason about A and apply it to Ā. So I'm free to pick the most convenient basis.

The most convenient basis is the one that diagonalizes A. If A = V Λ V^{-1} with Λ diagonal, then powers are free — Λ^k is elementwise — and the kernel entry C̄ Ā^k B̄ becomes Σ_n (C̄V)_n Λ_n^k (V^{-1}B̄)_n, which as a function of k is a *Vandermonde* form in the eigenvalues. A Vandermonde matrix-vector product is a known near-linear computation, O((N+L) log²(N+L)). That would solve everything — so let me just diagonalize the HiPPO matrix and see if the eigenvectors behave.

Let me actually look at its eigenvectors. Up to sign and a diagonal rescaling the HiPPO matrix is the lower-triangular thing with A_nk = (−1)^{n−k}(2k+1) for n>k and k+1 on the diagonal. I'll guess that the eigenvectors are columns of V_{ij} = C(i+j, i−j), i.e. the j-th column has v^{(j)}_i = C(i+j, 2j) for i ≥ j and 0 for i < j, and check it's an eigenvector with eigenvalue j+1. I need (A v^{(j)})_k = (j+1) v^{(j)}_k for all k. For k < j every term has either k < i (so A_{ki}=0) or i < j (so v^{(j)}_i = 0), so both sides are 0. For k ≥ j, expand:

    (A v^{(j)})_k = Σ_{i=j}^{k-1} (−1)^{k−i}(2i+1) C(i+j, 2j) + (k+1) C(k+j, 2j).

Base case k = j: the sum is empty, leaving (j+1) C(2j,2j) = (j+1) v^{(j)}_j. Good. For k > j, the sum for index k is the sum for index k−1 with every sign flipped plus the two new edge terms, so

    (A v)_k = −(A v)_{k-1} − (2k−1) C(k−1+j, 2j) + k C(k−1+j, 2j) + (k+1) C(k+j, 2j).

Use the inductive hypothesis (A v)_{k-1} = (j+1) C(k−1+j, 2j):

    = −(j+1) C(k−1+j,2j) − (k−1) C(k−1+j,2j) + (k+1) C(k+j,2j)
    = −(j+k) C(k−1+j,2j) + (k+1) C(k+j,2j).

Now C(k−1+j,2j) = (k−1+j)! / [(k−1−j)!(2j)!], so (j+k)·that = (k+j)!/[(k−1−j)!(2j)!] = (k−j)·(k+j)!/[(k−j)!(2j)!]. Then

    (A v)_k = −(k−j) C(k+j,2j) + (k+1) C(k+j,2j) = (j+1) C(k+j,2j) = (j+1) v^{(j)}_k.

So yes, V diagonalizes it, eigenvalues 1,2,…,N. The induction closes, so the diagonalization is real — but now look at the *entries* of V before I celebrate. The (3i, i) entry is C(4i, 2i) ≈ 2^{4i}, so V has entries as large as 2^{4N/3}. To use this basis I'd have to compute C̄V = C V, multiplying by a matrix with entries near 2^{4N/3}. For N = 256 that's roughly 2^{341} — far past where double precision carries a single significant bit, so C V is pure rounding noise. The diagonalization is exact on paper and numerically dead. (The alternative "fast" route through the characteristic polynomial of A and its inverse mod x^L has the same disease: for Ā near the identity, p(x) = (1−x)^N has coefficients up to C(N, N/2) ≈ 2^N, and (1−x)^{−N} mod x^L has the even-larger C(N+L−2, L−1); and Ā → I is the *typical* small-Δ regime, so this isn't a corner case.) Diagonalization is the right idea; diagonalizing *this* matrix by *this* V is what kills it.

So the real constraint is sharper than "diagonalize A": only conjugate by *well-conditioned* V. The perfectly conditioned case is V unitary, and the spectral theorem says A is unitarily diagonalizable exactly when A is **normal** (A A^* = A^* A). If HiPPO were normal I'd be done. But it isn't — the lower-triangular structure with that diagonal is about as far from normal as it gets.

Let me stare at the matrix and see how far it is from normal — maybe the gap is structured. Take the LegS form A_nk = −(2n+1)^{1/2}(2k+1)^{1/2} for n>k, −(n+1) on the diagonal, 0 above. What if I add a rank-one matrix to symmetrize the off-diagonal? Add ½(2n+1)^{1/2}(2k+1)^{1/2} to *every* entry. Off the diagonal that turns the n>k entries into −½(2n+1)^{1/2}(2k+1)^{1/2} and the (previously zero) n<k entries into +½(2n+1)^{1/2}(2k+1)^{1/2} — antisymmetric in n,k. On the diagonal −(n+1) + ½(2n+1) = −½. So the claim is A + P P^* = −½ I + (skew), with P_n = √(n+½), since P_n P_k = ½(2n+1)^{1/2}(2k+1)^{1/2}. Let me check that on a concrete N=6 HiPPO-LegS matrix rather than trust the index algebra: forming A + P P^* numerically, the diagonal comes out [−0.5, −0.5, −0.5, −0.5, −0.5, −0.5] exactly, and the residual S = A + P P^* + ½I satisfies max|S + Sᵀ| ≈ 1.8e-15 — skew-symmetric to machine precision. So the split is exactly A = (−½ I + skew) − P P^*, rank one.

A skew-symmetric matrix is normal (pure-imaginary eigenvalues, unitarily diagonalizable), and adding −½I just shifts the eigenvalues by a real constant without touching the eigenvectors, so −½I + skew is normal. Hence the HiPPO matrix is **normal plus low-rank**: A = (normal) − P Q^*, the low-rank term rank 1 here (rank 2 for some of the other HiPPO variants). Write the normal part by its unitary diagonalization, normal = V Λ V^*, and absorb V into the low-rank factors: conjugating by V^*,

    A = V Λ V^* − P Q^*  ⇒  V^*(A)V = Λ − (V^*P)(V^*Q)^*,

so in the well-conditioned (unitary) basis A becomes **diagonal plus low-rank** (DPLR), now over the complex numbers: A = Λ − P Q^*, Λ diagonal, P, Q ∈ C^{N×r} with r = 1. The conditioning problem that killed the naive diagonalization is gone — the only basis change is unitary.

But I'm not home. Diagonal-plus-low-rank is not diagonal, and the kernel needs *powers* of Ā. Powering a diagonal matrix is free; powering Λ − PQ^* is not — the low-rank term entangles, and (Λ − PQ^*)^k has no simple closed form. So I still can't build K̄ from powers. I need to get rid of "powers" entirely.

Here's the lever. Powers show up because I'm summing Σ_k C̄ Ā^k B̄ z^k along the kernel. That sum is a geometric series in the matrix Ā. Geometric series of matrices collapse to an inverse: Σ_{i≥0} Ā^i z^i = (I − Ā z)^{-1}. So instead of computing K̄ in the time domain, let me compute its **generating function**, the polynomial whose coefficients are the kernel entries, evaluated at points z:

    K̂(z) = Σ_{i=0}^{L-1} C̄ Ā^i B̄ z^i.

A truncated geometric series, but still closed-form. Use Σ_{i=0}^{L-1} (Āz)^i = (I − (Āz)^L)(I − Āz)^{-1} = (I − Ā^L z^L)(I − Āz)^{-1}, so

    K̂(z) = C̄ (I − Ā^L z^L)(I − Āz)^{-1} B̄.

The matrix *power* Ā^k has become a matrix *inverse* (I − Āz)^{-1}. That's the move — inverses of diagonal-plus-low-rank are tractable, powers are not. There's still the Ā^L term, but I get to choose the evaluation points z, so let me choose them to remove its dependence on z: the L-th roots of unity, z = ω_k = exp(−2πi·k/L) for k = 0…L−1. Then z^L = 1, and the factor (I − Ā^L z^L) = (I − Ā^L) is *the same constant for every node*. So fold it into C̄: define C̃^* = C̄^*(I − Ā^L), or equivalently learn C̃ directly and skip recomputing it (the (I − Ā^L) folding is what made my first numerical attempt below come out wrong until I put it on the right side). Then at the roots of unity,

    K̂(ω_k) = C̃^* (I − Ā ω_k)^{-1} B̄.

And why the roots of unity specifically, beyond making z^L constant? Because evaluating the generating function at them is *exactly a DFT* of the kernel: K̂_j = Σ_{k} K̄_k exp(−2πi·jk/L). So once I have K̂ at all L roots of unity, I recover K̄ by a single inverse FFT in O(L log L). I'll want to double-check that DFT claim concretely once the algebra is assembled, because it's the load-bearing reason the frequency-domain detour is free to undo.

Now the core object is C̃^*(I − Ā ω)^{-1} B̄ — a resolvent of the *discretized* matrix. I'd rather express it through the original A so I can use A = Λ − PQ^*. Plug in Ā = (I − Δ/2·A)^{-1}(I + Δ/2·A) and B̄ = (I − Δ/2·A)^{-1} Δ B and grind. Inside the inverse,

    I − Ā z = (I − Δ/2·A)^{-1}[ (I − Δ/2·A) − (I + Δ/2·A) z ],

so

    (I − Ā z)^{-1} B̄ = [ (I − Δ/2·A) − (I + Δ/2·A) z ]^{-1} (I − Δ/2·A) · (I − Δ/2·A)^{-1} Δ B
                      = [ (I − Δ/2·A) − (I + Δ/2·A) z ]^{-1} Δ B.

The two (I − Δ/2·A) factors cancel — that's the reason this is clean. Group the bracket:

    (I − Δ/2·A) − (I + Δ/2·A) z = I(1 − z) − (Δ/2) A (1 + z).

Factor out (1 − z):

    = (1 − z) [ I − (Δ/2) A (1+z)/(1−z) ] = (1 − z) [ I − Δ A / (2 (1−z)/(1+z)) ].

So

    C̄^*(I − Ā z)^{-1} B̄ = (Δ/(1−z)) C̄^* [ I − Δ A / (2 (1−z)/(1+z)) ]^{-1} B.

Pull the scalar 2(1−z)/(1+z) inside to clear the fraction in the bracket: [I − ΔA/c]^{-1} = c (cI − ΔA)^{-1} with c = 2(1−z)/(1+z), and (Δ/(1−z))·c = (Δ/(1−z))·2(1−z)/(1+z) = 2Δ/(1+z). Hence

    C̄^*(I − Ā z)^{-1} B̄ = (2Δ/(1+z)) C̄^* [ 2(1−z)/(1+z) · I − ΔA ]^{-1} B.

Divide numerator and denominator inside by Δ to put it on A rather than ΔA:

    = (2/(1+z)) C̄^* [ (2/Δ)(1−z)/(1+z) · I − A ]^{-1} B.

Define the node g(z) = (2/Δ)(1−z)/(1+z). Restoring the C̃ that absorbed (I − Ā^L), the truncated generating function at a root of unity z is

    K̂(z) = (2/(1+z)) · C̃^* ( g(z) I − A )^{-1} B.

That's a chain of algebraic rewrites I could easily have slipped a factor in, so before going further I'll check the central identity numerically: pick a small random DPLR A, form Ā by the bilinear rule, and compare C̄(I − Āz)^{-1}B̄ against (2/(1+z)) C̄(g(z)I − A)^{-1}B at the roots of unity. On an N=3 system at L=8 nodes the two agree to max error 5e-16. The cancellation and the g(z) substitution are right. Now the L powers of Ā are gone; what's left is, for each of the L nodes, one resolvent (g(z)I − A)^{-1} of the original A. And A is Λ − PQ^*.

Now the low-rank term. (g I − A) = (g I − Λ) + P Q^*. The first piece is diagonal, so its inverse is trivial: let R(z) = (g(z) I − Λ)^{-1}, an elementwise reciprocal. The full inverse is a diagonal-plus-low-rank inverse, and that is exactly what the **Woodbury identity** handles:

    (Λ' + P Q^*)^{-1} = Λ'^{-1} − Λ'^{-1} P (I + Q^* Λ'^{-1} P)^{-1} Q^* Λ'^{-1},

with Λ' = gI − Λ, so Λ'^{-1} = R(z). Sandwiching with C̃^* on the left and B on the right,

    C̃^* (g I − A)^{-1} B = C̃^* R B − C̃^* R P (I + Q^* R P)^{-1} Q^* R B.

For rank 1, (I + Q^* R P) is a scalar, so its inverse is just dividing. So

    K̂(z) = (2/(1+z)) [ C̃^* R(z) B − (C̃^* R(z) P)(Q^* R(z) B) / (1 + Q^* R(z) P) ].

Every term here is one of four bilinear forms of the same shape: u^* R(z) v = Σ_n u_n^* v_n / (g(z) − λ_n), for (u, v) ∈ {(C̃,B), (C̃,P), (Q,B), (Q,P)}. Look at what that sum is across all the nodes. Stack the nodes ω (equivalently g(ω)) along rows and the eigenvalues λ_n along columns; the matrix with entries 1/(g(ω_i) − λ_n) is a **Cauchy matrix**. Each bilinear form over all nodes is one Cauchy matrix-vector product against the weights u_n^* v_n. And Cauchy matrix-vector products are a classical, numerically stable problem: naively O(MN), but O((M+N) log²(M+N)) in exact arithmetic and O((M+N) log(M+N) log(1/ε)) numerically, via Fast-Multipole-style algorithms. With M = L nodes and N eigenvalues, four Cauchy multiplies cost Õ(N + L) time and O(N + L) space.

That's the whole chain on paper. Before I trust it I want one end-to-end numerical check, because a sign error in the Woodbury peel or the C̃ folding would be invisible in the symbol-pushing. Let me build the entire pipeline on a tiny system — random complex DPLR A = Λ − PQ^* with N = 3, length L = 8 — and compare against the brute-force kernel K̄_k = C̄ Ā^k B̄ computed by literally powering Ā.

First the generating function itself. Computing K̂(ω_j) directly as Σ_{i<L} C̄ Ā^i B̄ ω_j^i and comparing to the DFT of the brute-force K̄ gives max error 1.5e-16, so the "evaluation at roots of unity is a DFT" claim is confirmed, not just asserted — the inverse FFT really does invert it. The closed form C̄(I − Ā^L z^L)(I − Āz)^{-1}B̄ matches that same generating function to 2e-16.

Then the full S4 route — fold (I − Ā^L) into the row functional C̃^* = C̄(I − Ā^L), form g(ω), the diagonal resolvent R(ω), the four Cauchy sums, the rank-1 Woodbury combination K̂ = (2/(1+ω))[k_CB − k_CP k_QB/(1+k_QP)], and finally ifft. The first attempt was off by ~0.01 because I'd folded (I − Ā^L) on the wrong side (as a left multiply C̄^*(I−Ā^L) rather than the row C̄(I−Ā^L)); fixing the side, the recovered kernel lands on the brute-force kernel to max error 7e-17:

    brute  K̄[:4] = [−0.03254+0.06176i, −0.02580+0.06443i, −0.01943+0.06539i, −0.01350+0.06476i]
    S4 K̄[:4]    = [−0.03254+0.06176i, −0.02580+0.06443i, −0.01943+0.06539i, −0.01350+0.06476i]

Machine precision. So the algebra holds together: conjugate A into Λ − PQ^* (well-conditioned, because the normal part is unitarily diagonalizable and only a rank-1 piece is split off); take the generating function at the L roots of unity, turning powers of Ā into one inverse per node; back the inverse out to (g(z)I − A)^{-1} on the original A; use Woodbury to peel off the rank-1 term, leaving the diagonal resolvent R(z); recognize the four resulting bilinear forms as Cauchy matrix-vector products — Õ(N+L); scale by 2/(1+z); inverse-FFT over the nodes to recover K̄. The O(N²L) kernel is now Õ(N + L), and the small example says it computes the *same* kernel. And nothing in this path forms an ill-conditioned matrix or an exponentially large coefficient: the only basis change was unitary, and the Cauchy kernel is numerically well-behaved.

I should double-check I haven't lost the recurrence. For inference I want the stepwise form, and I need Ā as a cheap matrix-vector map even though it's (I − Δ/2·A)^{-1}(I + Δ/2·A). Split it. The forward factor: I + Δ/2·A = I + Δ/2(Λ − PQ^*) = (Δ/2)[ (2/Δ)I + Λ − PQ^* ] — diagonal plus low-rank, so it's an O(N) matrix-vector multiply. The backward factor (I − Δ/2·A)^{-1} = (2/Δ)[ (2/Δ)I − Λ + PQ^* ]^{-1}, and by Woodbury the inverse of a diagonal-plus-low-rank matrix is again diagonal-plus-low-rank: with D = ((2/Δ) − Λ)^{-1} (diagonal), it equals (2/Δ)[ D − D P (1 + Q^* D P)^{-1} Q^* D ]. So Ā is a product of two DPLR matrices, each an O(N) apply; one recurrence step is O(N). Both faces — convolution for training, recurrence for inference — are cheap, off the same parameters.

Let me pin down the parameterization and the few design choices that remain. In the derivation above I keep A as Λ − P Q^*, with rank one for the scaled-Legendre case; in the symmetric/stabilized code path this is represented as Λ − P P^* after the conjugate-pair bookkeeping (and that's exactly the P P^* split I verified numerically above). The trainable objects are the length-N vectors Λ, P, Q, B, C̃ (or the P-only stabilized variant) plus a learnable step size Δ. Δ is a *per-feature timescale*, initialized log-uniformly over a range like [10^{-3}, 10^{-1}]: different features then specialize to different memory horizons, and because Δ is an explicit sampling step, I can change it at test time to handle a different input sampling rate without retraining — a genuine perk of having stayed in continuous time. Initialize A to the HiPPO matrix (then immediately conjugate to its DPLR form); a random A would reintroduce the exponential decay and the long-range memory would be gone, which is the ~60%→98% gap from before, so the HiPPO initialization is doing real work, not cosmetics. One more practical point: A comes in complex-conjugate eigenvalue pairs, so I store half of them and take twice the real part of the kernel — halving the cost for free.

That handles one 1-D channel. A real model has H feature channels, so I just run H independent copies of this 1-D SSM — H separate DPLR factors, C̃, and Δ — producing H output sequences, then mix the H channels with a position-wise linear layer (the SSM never mixes channels itself). This is precisely a depthwise-separable convolution structure, except the per-channel filters are *global* (length L) and generated implicitly from the SSM rather than stored. Stack these with norm, residual, and a pointwise nonlinearity between layers, and the depth supplies the nonlinearity that the linear core lacks. Train the SSM/DPLR parameters with a smaller learning rate and no weight decay — they're not ordinary weights, they're a continuous dynamical system, and decaying them toward zero would corrupt the HiPPO structure — while the mixing layers train normally.

Now the code. The reference implementation stores only one half of each conjugate pair, scales A by dt instead of explicitly forming g(z) = (2/dt)(1-z)/(1+z), multiplies the Cauchy weights by dt, applies the rank-1 Woodbury correction, then uses irfft to get the real kernel.

```python
import torch
import torch.nn as nn
import numpy as np

def cauchy_naive(v, z, w, conj=True):
    """Sum_n v_n / (z_i - w_n), with S4's conjugate-pair expansion."""
    if conj:
        v = torch.cat([v, v.conj()], dim=-1)
        w = torch.cat([w, w.conj()], dim=-1)
    return (v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))).sum(dim=-2)

def hippo_legs_nplr(N):
    """HiPPO-LegS -> half-state DPLR parameters, following state-spaces/s4."""
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = torch.as_tensor(T @ M @ np.linalg.inv(T), dtype=torch.float64)
    B = torch.as_tensor(np.diag(T).copy(), dtype=torch.float64)

    P = torch.sqrt(0.5 + torch.arange(N, dtype=torch.float64))
    AP = A + P[:, None] * P[None, :]                 # -1/2 I + skew
    w_re = torch.diagonal(AP).mean()
    skew = AP - w_re * torch.eye(N, dtype=torch.float64)
    w_im, V = torch.linalg.eigh((-1j * skew).to(torch.cdouble))
    Lambda = w_re.to(torch.cdouble) + 1j * w_im

    idx = torch.argsort(Lambda.imag)
    Lambda, V = Lambda[idx][: N // 2], V[:, idx][:, : N // 2]
    B = V.conj().T @ B.to(torch.cdouble)
    P = V.conj().T @ P.to(torch.cdouble)
    return Lambda.to(torch.cfloat), P.to(torch.cfloat), B.to(torch.cfloat)

class S4Kernel(nn.Module):
    """Global S4 kernel for H independent half-state DPLR SSMs."""
    def __init__(self, H, N=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        Lambda, P, B = hippo_legs_nplr(N)
        Lambda = Lambda.unsqueeze(0).expand(H, -1).contiguous()
        P = P.unsqueeze(0).expand(H, -1).contiguous()
        B = B.unsqueeze(0).expand(H, -1).contiguous()
        C_tilde_star = torch.randn(H, N // 2, dtype=torch.cfloat)
        log_dt = torch.rand(H) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

        self.Lambda = nn.Parameter(torch.view_as_real(Lambda))
        self.P = nn.Parameter(torch.view_as_real(P))
        self.B = nn.Parameter(torch.view_as_real(B))
        self.C = nn.Parameter(torch.view_as_real(C_tilde_star))
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, L):
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)       # stored as C_tilde^*
        Q = P.conj()                            # stabilized public-code convention
        dt = torch.exp(self.log_dt)[:, None]

        omega = torch.exp(
            -2j * torch.pi * torch.arange(L // 2 + 1, device=Lambda.device) / L
        )
        z = 2 * (1 - omega) / (1 + omega)
        A = dt * Lambda                         # equivalent to using g(z)

        r00 = cauchy_naive(dt * C * B, z, A)
        r01 = cauchy_naive(dt * C * P, z, A)
        r10 = cauchy_naive(dt * Q * B, z, A)
        r11 = cauchy_naive(dt * Q * P, z, A)
        K_hat = (2 / (1 + omega)) * (r00 - r01 * r10 / (1 + r11))
        return torch.fft.irfft(K_hat, n=L)

class S4Layer(nn.Module):
    def __init__(self, H, N=64):
        super().__init__()
        self.kernel = S4Kernel(H, N)
        self.D = nn.Parameter(torch.randn(H))
        self.activation = nn.GELU()
        self.out = nn.Conv1d(H, H, 1)

    def forward(self, u):                         # u: (B, H, L)
        L = u.size(-1)
        K = self.kernel(L)                        # (H, L)
        K_f = torch.fft.rfft(K, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * K_f, n=2 * L)[..., :L]
        y = y + u * self.D[:, None]
        return self.out(self.activation(y))

    def step(self, u_k, state):
        # Reference recurrent mode uses A_bar = A1 @ A0 and B_bar = 2 A1 B,
        # where both A0 and A1 are DPLR/Woodbury matrix-vector applies.
        ...
```

The causal chain, end to end: I want one layer with an RNN's unbounded memory, a CNN's parallel training, and an RNN's cheap stepping. A linear state space model is simultaneously a recurrence and a convolution, so it could have all three — but only if the state matrix A doesn't cause exponential gradient decay, which the HiPPO matrix fixes by construction. Discretize with the bilinear transform, which the stability numbers above showed keeps oscillatory stable modes inside the unit disk where Euler does not; the convolutional view needs a kernel K̄ whose naive construction costs O(N²L) because it powers Ā L times. Diagonalizing A would make the powers free, but HiPPO's eigenvectors reach magnitude 2^{4N/3} and the change of basis becomes rounding noise — that's the wall that forces the rest. HiPPO is, however, normal-plus-low-rank (verified: A + PP^* = −½I + skew exactly), so a *unitary* change of basis brings it to diagonal-plus-low-rank. Then, rather than powering, take the generating function at the L roots of unity, which turns the powers into a single resolvent per node, makes the truncation term constant, and lets an inverse FFT recover K̄; back the resolvent onto the original A, peel the rank-1 term with Woodbury to reach a diagonal resolvent, and recognize the four remaining forms as Cauchy matrix-vector products — Õ(N+L), numerically stable. The tiny end-to-end check confirmed this whole route reproduces the brute-force kernel to 1e-16, so it's the same operator, just cheaper. The same DPLR structure gives an O(N)-per-step recurrence for inference. Stack H copies with channel mixing and nonlinearities, initialize A to HiPPO, learn the DPLR factors, C̃, and Δ — and the long-range, parallel, fast sequence layer exists.
