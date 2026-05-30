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

This is not a guess; it falls out of requiring the state to track an orthogonal-polynomial expansion of the past. And empirically the difference is enormous — swapping a random A for this one takes a state space layer on sequential MNIST from ~60% to ~98%. So: A is the HiPPO matrix, B and C and the step size are mine to learn, and the memory problem is, in principle, solved. Now I need to make the thing *run*.

It lives in continuous time and my data is sampled, u_k = u(kΔ) for some step size Δ. I have to discretize x'(t) = A x(t) + B u(t). The crude option is forward Euler: pretend x'(t_k) ≈ (x_k − x_{k-1})/Δ, giving x_k = (I + ΔA) x_{k-1} + ΔB u_k, so Ā = I + ΔA. But that's the unstable choice — there's no guarantee the eigenvalues of I + ΔA sit inside the unit disk, which is exactly the stability condition for the discrete recurrence not to blow up, and I just spent all that effort getting the continuous dynamics right. I want a discretization that *preserves* stability: maps the stable continuous region (eigenvalues with negative real part, the left half-plane) into the stable discrete region (eigenvalues inside the unit circle).

The bilinear (trapezoidal) rule does that. Integrate x' over one step with the trapezoid rule:

    x_k − x_{k-1} = (Δ/2)[ (A x_k + B u_k) + (A x_{k-1} + B u_{k-1}) ].

Treating the driving input as held over the step (fold u_{k-1} into u_k for a single-input form), collect the x_k terms on the left:

    (I − Δ/2·A) x_k = (I + Δ/2·A) x_{k-1} + Δ B u_k.

Invert the left factor:

    x_k = Ā x_{k-1} + B̄ u_k,   y_k = C̄ x_k,
    Ā = (I − Δ/2·A)^{-1}(I + Δ/2·A),   B̄ = (I − Δ/2·A)^{-1} Δ B,   C̄ = C.

That map A ↦ (I − Δ/2·A)^{-1}(I + Δ/2·A) is the Cayley transform; it sends the left half-plane to the open unit disk, so a stable continuous system becomes a stable discrete one. Good — that's why bilinear and not Euler. Notice for later: as Δ → 0, Ā → I.

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

The most convenient basis is the one that diagonalizes A. If A = V Λ V^{-1} with Λ diagonal, then powers are free — Λ^k is elementwise — and the kernel entry C̄ Ā^k B̄ becomes Σ_n (C̄V)_n Λ_n^k (V^{-1}B̄)_n, which as a function of k is a *Vandermonde* form in the eigenvalues. A Vandermonde matrix-vector product is a known near-linear computation, O((N+L) log²(N+L)). That would solve everything.

So let me just diagonalize the HiPPO matrix. … And here it falls apart. Let me actually look at its eigenvectors. Up to sign and a diagonal rescaling the HiPPO matrix is the lower-triangular thing with A_nk = (−1)^{n−k}(2k+1) for n>k and k+1 on the diagonal. I'll guess that the eigenvectors are columns of V_{ij} = C(i+j, i−j), i.e. the j-th column has v^{(j)}_i = C(i+j, 2j) for i ≥ j and 0 for i < j, and check it's an eigenvector with eigenvalue j+1. I need (A v^{(j)})_k = (j+1) v^{(j)}_k for all k. For k < j every term has either k < i (so A_{ki}=0) or i < j (so v^{(j)}_i = 0), so both sides are 0. For k ≥ j, expand:

    (A v^{(j)})_k = Σ_{i=j}^{k-1} (−1)^{k−i}(2i+1) C(i+j, 2j) + (k+1) C(k+j, 2j).

Base case k = j: the sum is empty, leaving (j+1) C(2j,2j) = (j+1) v^{(j)}_j. Good. For k > j, the sum for index k is the sum for index k−1 with every sign flipped plus the two new edge terms, so

    (A v)_k = −(A v)_{k-1} − (2k−1) C(k−1+j, 2j) + k C(k−1+j, 2j) + (k+1) C(k+j, 2j).

Use the inductive hypothesis (A v)_{k-1} = (j+1) C(k−1+j, 2j):

    = −(j+1) C(k−1+j,2j) − (k−1) C(k−1+j,2j) + (k+1) C(k+j,2j)
    = −(j+k) C(k−1+j,2j) + (k+1) C(k+j,2j).

Now C(k−1+j,2j) = (k−1+j)! / [(k−1−j)!(2j)!], so (j+k)·that = (k+j)!/[(k−1−j)!(2j)!] = (k−j)·(k+j)!/[(k−j)!(2j)!] = (k−j)(k+1)·C(k+j,2j)/(k+1)… let me just push the factorials: (j+k)C(k−1+j,2j) = (k−j)·C(k+j,2j) after the cancellation, since (k+j)!/[(k−1−j)!(2j)!] = (k−j)·(k+j)!/[(k−j)!(2j)!]. Then

    (A v)_k = −(k−j) C(k+j,2j) + (k+1) C(k+j,2j) = (j+1) C(k+j,2j) = (j+1) v^{(j)}_k.

So yes, V diagonalizes it, eigenvalues 1,2,…,N. Lovely — except look at the entries of V. The (3i, i) entry is C(4i, 2i) ≈ 2^{4i}. So V has entries as large as 2^{4N/3}. That's catastrophic: to use this basis I'd have to compute C̄V = C V, multiplying by a matrix with entries near 2^{4N/3}. In floating point that's pure noise; the diagonalization is numerically infeasible. (And the alternative "fast" route some have tried — going through the characteristic polynomial of A and its inverse mod x^L — has the same disease: for Ā near the identity, p(x) = (1−x)^N has coefficients up to C(N, N/2) ≈ 2^N, and (1−x)^{−N} mod x^L has the even-larger C(N+L−2, L−1); and recall Ā → I is the *typical* small-Δ regime, so this isn't a corner case.) Diagonalization itself is the right idea; diagonalizing *this* matrix by *this* V is the problem.

So the real constraint is: only conjugate by *well-conditioned* V. The perfectly conditioned case is V unitary, and the spectral theorem says A is unitarily diagonalizable exactly when A is **normal** (A A^* = A^* A). If HiPPO were normal I'd be done — diagonalize by a unitary, get a Vandermonde kernel, go home. But it isn't normal; the lower-triangular structure with that diagonal is the opposite of normal.

Let me stare at the matrix and see how far it is from normal. Take the LegS form A_nk = −(2n+1)^{1/2}(2k+1)^{1/2} for n>k, −(n+1) on the diagonal, 0 above. What if I add a rank-one matrix to symmetrize the off-diagonal? Add ½(2n+1)^{1/2}(2k+1)^{1/2} to *every* entry. Off the diagonal that turns the n>k entries into −½(2n+1)^{1/2}(2k+1)^{1/2} and the (previously zero) n<k entries into +½(2n+1)^{1/2}(2k+1)^{1/2} — antisymmetric in n,k. On the diagonal −(n+1) + ½(2n+1) = −½. So the result is −½I plus a skew-symmetric matrix S. A skew-symmetric matrix is normal (its eigenvalues are pure imaginary, and it's unitarily diagonalizable); adding −½I just shifts the eigenvalues by a real constant and doesn't touch the eigenvectors. And the thing I added — ½(2n+1)^{1/2}(2k+1)^{1/2} = ½ p_n p_k with p_n = (2n+1)^{1/2} — is rank one, p p^*.

So the HiPPO matrix is **normal plus low-rank**: A = (normal) − P Q^*, with the low-rank term rank 1 here (rank 2 for some of the other HiPPO variants). Write the normal part by its unitary diagonalization, normal = V Λ V^*, and absorb V into the low-rank factors: conjugating by V^*,

    A = V Λ V^* − P Q^*  ⇒  V^*(A)V = Λ − (V^*P)(V^*Q)^*,

so in the well-conditioned (unitary) basis A becomes **diagonal plus low-rank** (DPLR), now over the complex numbers: A = Λ − P Q^*, Λ diagonal, P, Q ∈ C^{N×r} with r = 1. The conditioning problem is gone — V is unitary.

But I'm not home. Diagonal-plus-low-rank is not diagonal, and the kernel needs *powers* of Ā. Powering a diagonal matrix is free; powering Λ − PQ^* is not — the low-rank term entangles, and (Λ − PQ^*)^k has no simple form. So I still can't build K̄ from powers. I need to get rid of "powers" entirely.

Here's the lever. Powers show up because I'm summing Σ_k C̄ Ā^k B̄ z^k along the kernel. That sum is a geometric series in the matrix Ā. Geometric series of matrices collapse to an inverse: Σ_{i≥0} Ā^i z^i = (I − Ā z)^{-1}. So instead of computing K̄ in the time domain, let me compute its **generating function**, the polynomial whose coefficients are the kernel entries, evaluated at points z:

    K̂(z) = Σ_{i=0}^{L-1} C̄ Ā^i B̄ z^i.

A truncated geometric series, but still closed-form. Use Σ_{i=0}^{L-1} (Āz)^i = (I − (Āz)^L)(I − Āz)^{-1} = (I − Ā^L z^L)(I − Āz)^{-1}, so

    K̂(z) = C̄ (I − Ā^L z^L)(I − Āz)^{-1} B̄.

The matrix *power* Ā^k has become a matrix *inverse* (I − Āz)^{-1}. That's the move — inverses of diagonal-plus-low-rank are tractable, powers are not. There's still the Ā^L term, but I get to choose the evaluation points z, so let me choose them to kill it: the L-th roots of unity, z = ω_k = exp(−2πi·k/L) for k = 0…L−1. Then z^L = 1, and the factor (I − Ā^L z^L) = (I − Ā^L) is *the same constant for every node* — independent of z. So fold it into C̄: define C̃^* = C̄^*(I − Ā^L), or equivalently C̃ = (I − Ā^L)^* C̄ (and in practice I can just learn C̃ directly and skip recomputing it). Then at the roots of unity,

    K̂(ω_k) = C̃^* (I − Ā ω_k)^{-1} B̄.

And why the roots of unity specifically, beyond killing z^L? Because evaluating the generating function at them is *exactly a DFT* of the kernel: K̂_j = Σ_{k} K̄_k exp(−2πi·jk/L). So once I have K̂ at all L roots of unity, I recover K̄ by a single inverse FFT in O(L log L). The frequency-domain detour is free to undo. Two birds.

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

So the L powers of Ā are gone; what's left is, for each of the L nodes, one resolvent (g(z)I − A)^{-1} of the original A. And A is Λ − PQ^*.

Now the low-rank term. (g I − A) = (g I − Λ) + P Q^*. The first piece is diagonal, so its inverse is trivial: let R(z) = (g(z) I − Λ)^{-1}, an elementwise reciprocal. The full inverse is a diagonal-plus-low-rank inverse, and that is exactly what the **Woodbury identity** handles:

    (Λ' + P Q^*)^{-1} = Λ'^{-1} − Λ'^{-1} P (I + Q^* Λ'^{-1} P)^{-1} Q^* Λ'^{-1},

with Λ' = gI − Λ, so Λ'^{-1} = R(z). Sandwiching with C̃^* on the left and B on the right,

    C̃^* (g I − A)^{-1} B = C̃^* R B − C̃^* R P (I + Q^* R P)^{-1} Q^* R B.

For rank 1, (I + Q^* R P) is a scalar, so its inverse is just dividing. So

    K̂(z) = (2/(1+z)) [ C̃^* R(z) B − (C̃^* R(z) P)(Q^* R(z) B) / (1 + Q^* R(z) P) ].

Every term here is one of four bilinear forms of the same shape: u^* R(z) v = Σ_n u_n^* v_n / (g(z) − λ_n), for (u, v) ∈ {(C̃,B), (C̃,P), (Q,B), (Q,P)}. Look at what that sum is across all the nodes. Stack the nodes ω (equivalently g(ω)) along rows and the eigenvalues λ_n along columns; the matrix with entries 1/(g(ω_i) − λ_n) is a **Cauchy matrix**. Each bilinear form over all nodes is one Cauchy matrix-vector product against the weights u_n^* v_n. And Cauchy matrix-vector products are a classical, numerically stable problem: naively O(MN), but O((M+N) log²(M+N)) in exact arithmetic and O((M+N) log(M+N) log(1/ε)) numerically, via Fast-Multipole-style algorithms. With M = L nodes and N eigenvalues, four Cauchy multiplies cost Õ(N + L) time and O(N + L) space.

That's the whole chain. Conjugate A into Λ − PQ^* (well-conditioned, because the normal part is unitarily diagonalizable and only a rank-1 piece is split off). Take the generating function at the L roots of unity, turning powers of Ā into one inverse per node. Back the inverse out to (g(z)I − A)^{-1} on the original A. Use Woodbury to peel off the rank-1 term, leaving the diagonal resolvent R(z). Recognize the four resulting bilinear forms as Cauchy matrix-vector products — Õ(N+L). Scale by 2/(1+z). Inverse-FFT the result over the nodes to recover K̄. The O(N²L) kernel is now Õ(N + L). And nothing in this path ever forms an ill-conditioned matrix or an exponentially large coefficient: the only basis change was unitary, and the Cauchy kernel is numerically well-behaved.

I should double-check I haven't lost the recurrence. For inference I want the stepwise form, and I need Ā as a cheap matrix-vector map even though it's (I − Δ/2·A)^{-1}(I + Δ/2·A). Split it. The forward factor: I + Δ/2·A = I + Δ/2(Λ − PQ^*) = (Δ/2)[ (2/Δ)I + Λ − PQ^* ] — diagonal plus low-rank, so it's an O(N) matrix-vector multiply. The backward factor (I − Δ/2·A)^{-1} = (2/Δ)[ (2/Δ)I − Λ + PQ^* ]^{-1}, and by Woodbury the inverse of a diagonal-plus-low-rank matrix is again diagonal-plus-low-rank: with D = ((2/Δ) − Λ)^{-1} (diagonal), it equals (2/Δ)[ D − D P (1 + Q^* D P)^{-1} Q^* D ]. So Ā is a product of two DPLR matrices, each an O(N) apply; one recurrence step is O(N). Both faces — convolution for training, recurrence for inference — are cheap, off the same parameters.

Let me pin down the parameterization and the few design choices that remain. I keep A as the pair (Λ, P) — for the canonical scaled-Legendre case Q = P (the rank-1 factor is symmetric), so the low-rank term is P P^*, which also keeps the system stable by construction. Together with B, C (or rather C̃) and the step size Δ, that's about 5N real numbers per SSM — N for Λ, N for P, N each for B and C as complex vectors. Δ I make a *learnable per-feature timescale*, initialized log-uniformly over a range like [10^{-3}, 10^{-1}]: different features then specialize to different memory horizons, and because Δ is an explicit sampling step, I can change it at test time to handle a different input sampling rate without retraining — a genuine perk of having stayed in continuous time. Initialize A to the HiPPO matrix (then immediately conjugate to its DPLR form); a random A would reintroduce the exponential decay and the long-range memory would be gone, so the HiPPO initialization is doing real work, not cosmetics. One more practical point: A comes in complex-conjugate eigenvalue pairs, so I store half of them and take twice the real part of the kernel — halving the cost for free.

That handles one 1-D channel. A real model has H feature channels, so I just run H independent copies of this 1-D SSM — H separate (Λ, P, B, C, Δ) — producing H output sequences, then mix the H channels with a position-wise linear layer (the SSM never mixes channels itself). This is precisely a depthwise-separable convolution structure, except the per-channel filters are *global* (length L) and generated implicitly from the SSM rather than stored. Stack these with norm, residual, and a pointwise nonlinearity between layers, and the depth supplies the nonlinearity that the linear core lacks. Train the SSM parameters (Λ, P, B, Δ) with a smaller learning rate and no weight decay — they're not ordinary weights, they're a continuous dynamical system, and decaying them toward zero would corrupt the HiPPO structure — while the mixing layers train normally.

Now the code. The kernel routine mirrors the derivation step for step: nodes at the roots of unity, the bilinear node g(z), four Cauchy multiplies, the rank-1 Woodbury combination, the 2/(1+z) scale, the inverse FFT.

```python
import torch
import torch.nn as nn
import numpy as np

def cauchy_naive(v, z, w):
    """For each node z_i, sum_n v_n / (z_i - w_n).  (the four bilinear forms u^* R v)"""
    # v: (..., N) weights u_n^* v_n ;  w: (..., N) poles lambda_n ;  z: (L,) nodes g(omega)
    cauchy = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (..., N, L)
    return cauchy.sum(dim=-2)                                       # (..., L)

class S4Kernel(nn.Module):
    """Generates the global convolution kernel K-bar for one (group of) 1-D SSM(s)
    parameterized as A = Lambda - P P^*  (DPLR), the HiPPO matrix in its well-conditioned basis."""
    def __init__(self, Lambda, P, B, C, log_dt):
        super().__init__()
        # complex N-vectors stored as real (..., N, 2); halved by conjugate symmetry
        self.Lambda = nn.Parameter(torch.view_as_real(Lambda))  # diagonal of normal part
        self.P      = nn.Parameter(torch.view_as_real(P))       # rank-1 low-rank factor
        self.B      = nn.Parameter(torch.view_as_real(B))
        self.C      = nn.Parameter(torch.view_as_real(C))       # actually C-tilde
        self.log_dt = nn.Parameter(log_dt)                      # learnable per-feature timescale

    def forward(self, L):
        dt = torch.exp(self.log_dt)                             # (H,)
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P); Q = P.conj()         # Q = P  (symmetric rank-1)
        B = torch.view_as_complex(self.B); C = torch.view_as_complex(self.C)

        # roots of unity and the bilinear node g(z) = (2/dt) (1 - z)/(1 + z)
        omega = torch.exp(-2j * np.pi * torch.arange(L) / L)    # z = omega_k
        g = (2.0 / dt)[..., None] * (1 - omega) / (1 + omega)   # nodes for the resolvent

        # four bilinear forms  u^* R(z) v = sum_n (u_n^* v_n)/(g - Lambda_n)  -> Cauchy multiplies
        k_CB = cauchy_naive(C.conj() * B, g, Lambda)            # C-tilde^* R B
        k_CP = cauchy_naive(C.conj() * P, g, Lambda)            # C-tilde^* R P
        k_QB = cauchy_naive(Q.conj() * B, g, Lambda)            # Q^* R B
        k_QP = cauchy_naive(Q.conj() * P, g, Lambda)            # Q^* R P

        # rank-1 Woodbury correction  +  the 2/(1+z) bilinear factor
        K_hat = (2.0 / (1 + omega)) * (k_CB - k_CP * k_QB / (1 + k_QP))

        # frequency -> time: the generating function at roots of unity is a DFT of K-bar
        K = torch.fft.irfft(K_hat, n=L)                         # the convolution kernel K-bar
        return K

class S4Layer(nn.Module):
    """One S4 sequence layer: H independent SSMs -> global conv via FFT -> skip + nonlinearity."""
    def __init__(self, H, N=64, l_max=None):
        super().__init__()
        self.H, self.N = H, N
        # HiPPO-LegS initialization, conjugated to DPLR (Lambda, P), then H copies:
        Lambda, P, B, C, log_dt = hippo_dplr_init(H, N)         # (see HiPPO derivation above)
        self.kernel = S4Kernel(Lambda, P, B, C, log_dt)
        self.D = nn.Parameter(torch.randn(H))                   # the feedthrough skip we set aside
        self.activation = nn.GELU()
        self.out = nn.Conv1d(H, H, 1)                           # position-wise channel mixing

    def forward(self, u):                                       # u: (B, H, L)
        L = u.size(-1)
        K = self.kernel(L)                                      # (H, L) global filters
        # non-circular convolution y = K * u via length-2L FFT
        K_f = torch.fft.rfft(K, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * K_f, n=2 * L)[..., :L]        # (B, H, L)
        y = y + u * self.D[..., None]                           # D u skip connection
        return self.out(self.activation(y))                    # nonlinearity + mix H channels

    def step(self, u_k, state):
        # inference: x_k = A-bar x_{k-1} + B-bar u_k ; A-bar = A1 @ A0, both DPLR (O(N)/step)
        # A0 = (2/dt) I + (Lambda - P Q^*) ;  A1 = (2/dt)[D - D P (1 + Q^* D P)^{-1} Q^* D]
        ...                                                     # O(N) matrix-vector applies
```

The causal chain, end to end: I want one layer with an RNN's unbounded memory, a CNN's parallel training, and an RNN's cheap stepping. A linear state space model is simultaneously a recurrence and a convolution, so it could have all three — but only if the state matrix A doesn't cause exponential gradient decay, which the HiPPO matrix fixes by construction. Discretize with the bilinear transform to keep stability; the convolutional view needs a kernel K̄ whose naive construction costs O(N²L) because it powers Ā L times. Diagonalizing A would make the powers free, but HiPPO's eigenvectors are exponentially ill-conditioned — yet HiPPO is normal-plus-low-rank, so a *unitary* change of basis brings it to diagonal-plus-low-rank. Then, rather than powering, take the generating function at the L roots of unity, which turns the powers into a single resolvent per node, makes the truncation term constant, and lets an inverse FFT recover K̄; back the resolvent onto the original A, peel the rank-1 term with Woodbury to reach a diagonal resolvent, and recognize the four remaining forms as Cauchy matrix-vector products — Õ(N+L), numerically stable. The same DPLR structure gives an O(N)-per-step recurrence for inference. Stack H copies with channel mixing and nonlinearities, initialize A to HiPPO, learn Λ, P, B, C, Δ — and the long-range, parallel, fast sequence layer exists.
