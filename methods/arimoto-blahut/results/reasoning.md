I want a number. Given a discrete memoryless channel — input letters 1..N, output letters 1..M, and a transition matrix Q with Q(j|i) the probability of seeing output j when I send input i — I want to actually compute its capacity,

  C = max_p I(X;Y),   I(X;Y) = Σ_i Σ_j p_i Q(j|i) log( Q(j|i) / q_j ),   q_j = Σ_i p_i Q(j|i),

for a Q that has no symmetry I can exploit. Everyone knows the symmetric cases: if every row of Q is a permutation of every other row and the columns balance, the uniform input is optimal by symmetry and C = log M − H(row). Beautiful, useless for a general Q. The moment the channel is asymmetric the uniform input is wrong, the support of the optimal input might not even be all of the alphabet, and I have no formula. So I have to optimize numerically. Fine — what's the optimization actually like?

Let me get the shape of I(p,Q) as a function of p straight, because that decides everything. Split it:

  I(p,Q) = Σ_i p_i [ Σ_j Q(j|i) log Q(j|i) ] − Σ_j q_j log q_j.

The first piece is linear in p. The second is −Σ_j q_j log q_j with q = pQ a linear function of p, and t↦t log t is convex, so Σ_j q_j log q_j is convex in p, and its negative is concave. Linear plus concave is concave. Good: I is concave in p, and I'm maximizing a concave function over the probability simplex. That's a convex program. "Solvable in principle." But in principle is doing a lot of work here.

What does "solve it" concretely mean? Write the Lagrangian, take KKT. Stationarity says: at the optimal p*, for every letter i that's actually used (p*_i > 0),

  Σ_j Q(j|i) log( Q(j|i) / q*_j ) = γ,

a common constant γ, and for the unused letters (p*_i = 0) the left side is ≤ γ. And that γ is exactly the capacity. This is a clean characterization — the "information that the output gives about input letter i" is equalized across all the letters I bother to use. But as a *recipe* it's hopeless: I don't know the support in advance, q* depends on p* through q = pQ so the equations are all tangled, and there's no closed form for p* in terms of Q. I'd be guessing which letters are active and solving coupled nonlinear equations for the rest. For a 2×2 channel, sure. For anything real, no.

So drop the idea of solving it in one shot and think iteratively. Gradient ascent on the simplex? The gradient of I has that log(Q(j|i)/q_j) structure, but after each step I have to project back onto the simplex, which is a clumsy little optimization of its own, and the step size is a tuning headache, and — this nags at me — I get no cheap way to know how far I am from the top. Monotone improvement without an error certificate means I never actually know when to stop. Interior-point on the KKT system carries matrices of size growing with N·M and ignores the special log-sum structure entirely. All of this is generic machinery bolted onto a problem that has a lot of structure I'm not using.

Let me stop reaching for solvers and stare at the object itself. The thing inside the log is Q(j|i)/q_j. Multiply top and bottom by p_i:

  Q(j|i)/q_j = p_i Q(j|i) / (p_i q_j).

And p_i Q(j|i) is the joint P(X=i, Y=j), while p_i Q(j|i)/q_j is the *posterior* P(X=i | Y=j) whenever q_j > 0 — the probability that the input was i given that I observed output j. Call it φ*(i|j) = p_i Q(j|i)/q_j on those outputs; outputs with q_j = 0 carry zero weight in the sums. So

  log( Q(j|i)/q_j ) = log( φ*(i|j) / p_i ),

and

  I(p,Q) = Σ_i Σ_j p_i Q(j|i) log( φ*(i|j) / p_i ).

Interesting. The mutual information is the expected log-ratio of the posterior to the prior. That posterior φ* is a *derived* quantity — it's pinned down by p and Q through Bayes' rule — but what if I treat it as a free variable? Let me define, for *any* matrix of conditional probabilities φ(i|j) (any "reverse channel," input given output),

  Ĩ(p, Q, φ) = Σ_i Σ_j p_i Q(j|i) log( φ(i|j) / p_i ).

This is a function of p and of φ. When φ happens to equal the true posterior φ*, I get back I(p,Q) exactly. The question is what happens for other φ. Compute the difference:

  I(p,Q) − Ĩ(p,Q,φ) = Σ_i Σ_j p_i Q(j|i) log( φ*(i|j) / φ(i|j) ).

Group the sum by output j. The weight p_i Q(j|i) = q_j · φ*(i|j), so

  = Σ_j q_j Σ_i φ*(i|j) log( φ*(i|j)/φ(i|j) ) = Σ_j q_j D( φ*(·|j) ‖ φ(·|j) ).

Every term is a KL divergence between the true posterior at output j and my guessed posterior, weighted by how often that output occurs. KL divergence is non-negative and zero only when the two match. So

  Ĩ(p,Q,φ) ≤ I(p,Q),   with equality iff φ(i|j) = φ*(i|j) = p_i Q(j|i)/q_j for every output j with q_j > 0.

That is, I(p,Q) = max_φ Ĩ(p,Q,φ), and the maximizing φ is exactly the Bayes posterior. I've turned the mutual information itself into a maximization over a free reverse channel. Why would I want to *complicate* a function by writing it as a max of a bigger function? Because of what it does to the capacity:

  C = max_p I(p,Q) = max_p max_φ Ĩ(p,Q,φ).

The capacity is now a **double** maximization, over the input distribution p and over the reverse channel φ. And the whole game in optimization is that a double max over two blocks is wonderful exactly when each block, holding the other fixed, can be solved in closed form. The single max over p had no closed form. Let me check whether the two inner maxes here do.

First inner max — fix p, maximize over φ. I already did it: the answer is the posterior, in closed form,

  φ(i|j) = p_i Q(j|i) / Σ_k p_k Q(j|k).

A Bayes-rule application, nothing to iterate. One down.

Second inner max — fix φ, maximize Ĩ(p,Q,φ) over the input distribution p. Now p appears in two places: in the explicit p_i Q(j|i) weight and inside the log( φ(i|j)/p_i ). Expand:

  Ĩ = Σ_i Σ_j p_i Q(j|i) log φ(i|j) − Σ_i Σ_j p_i Q(j|i) log p_i
    = Σ_i p_i [ Σ_j Q(j|i) log φ(i|j) ] − Σ_i p_i log p_i,

using Σ_j Q(j|i) = 1 in the second term. So as a function of p this is

  Σ_i p_i a_i + H(p),   where a_i = Σ_j Q(j|i) log φ(i|j),

a linear term plus the entropy H(p) = −Σ_i p_i log p_i. Maximize over the simplex. This is the cleanest little problem in the world — maximize a linear functional plus entropy — and I know its answer is the Gibbs / softmax distribution. Let me actually do it so I trust the constant. Lagrangian with multiplier λ for Σ_i p_i = 1:

  ∂/∂p_i [ Σ p_i a_i − Σ p_i log p_i − λ(Σ p_i − 1) ] = a_i − log p_i − 1 − λ = 0
  ⟹ log p_i = a_i − 1 − λ ⟹ p_i ∝ exp(a_i).

So with r_i := exp(a_i) = exp( Σ_j Q(j|i) log φ(i|j) ),

  p_i = r_i / Σ_k r_k,

in closed form. And the optimal value is tidy: plug back, Ĩ = Σ_i p_i (a_i − log p_i) = Σ_i p_i ( a_i − (a_i − log Σ_k r_k) ) = log Σ_k r_k. The second inner max equals log Σ_i r_i. (I could also get the bound directly: for any p, Ĩ(p,Q,φ) = Σ_i p_i log( r_i / p_i ) = log Σ_k r_k − D( p ‖ r/Σr ) ≤ log Σ_k r_k, with equality iff p = r/Σr — same answer, and it shows the gap is a KL divergence again. Nice.) Two down.

Both inner maximizations are closed form. Starting from any strictly positive p, I can get the posterior φ from Bayes, get the new p from the softmax, and repeat. Each half-step is a single closed-form evaluation, no projection, no step size. Let me make sure it does what I want — that the mutual information actually goes *up* every round.

This is where the double-max framing pays off twice. Ĩ(p,Q,φ) is concave in φ for fixed p (it's Σ of p_i Q(j|i) log φ(i|j), a sum of logs, concave) and concave in p for fixed φ (linear plus entropy). It is *not* jointly concave — the log(φ/p) term couples them — but I don't need joint concavity. I need that alternating maximization climbs. And it does, by construction: when I fix p^t and set φ^{t+1} = argmax_φ Ĩ, by the variational identity that φ-max equals I, so Ĩ(p^t,Q,φ^{t+1}) = I(p^t,Q). Then I fix φ^{t+1} and set p^{t+1} = argmax_p Ĩ, which can only raise Ĩ further:

  I(p^t,Q) = Ĩ(p^t,Q,φ^{t+1}) ≤ Ĩ(p^{t+1},Q,φ^{t+1}) ≤ max_φ Ĩ(p^{t+1},Q,φ) = I(p^{t+1},Q).

So I(p^t,Q) is non-decreasing in t. It's bounded above by C. A bounded monotone sequence converges. The mutual information of the iterates climbs and settles. That's the engine.

Let me also collapse the two half-steps into one update on p, because carrying φ around is wasteful. With q^t = p^t Q the current output distribution, the posterior is φ^{t+1}(i|j) = p_i^t Q(j|i)/q_j^t, so

  a_i = Σ_j Q(j|i) log φ^{t+1}(i|j) = Σ_j Q(j|i) log( p_i^t Q(j|i)/q_j^t )
      = Σ_j Q(j|i) log( Q(j|i)/q_j^t ) + log p_i^t · Σ_j Q(j|i)
      = D( Q_i ‖ q^t ) + log p_i^t,

where Q_i is the i-th row of Q (the output distribution given input i) and D(Q_i‖q^t) = Σ_j Q(j|i) log(Q(j|i)/q_j^t). So r_i^{t+1} = exp(a_i) = p_i^t · exp( D(Q_i ‖ q^t) ), and the update is

  p_i^{t+1} = p_i^t · exp( D(Q_i ‖ q^t) ) / Σ_k p_k^t · exp( D(Q_k ‖ q^t) ).

This is lovely and tells me exactly what's happening. D(Q_i‖q^t) is how distinguishable input letter i's output distribution is from the current average output — how much "information" output gives about input i under the present guess. Letters whose outputs stand out from the crowd get their probability multiplied up; letters that look like everyone else get suppressed. It's a multiplicative reweighting that reinforces the discriminating letters, and it can never resurrect a letter I've zeroed (which is why I start full-support, uniform — the agnostic, maximum-entropy start).

Now the thing the gradient method couldn't give me: when do I stop, and how good is my current answer? Monotone increase tells me I'm climbing but not how far the summit is. I need an *upper* bound on C that I can compute from the current iterate. Look again at the quantity c_i^t := r_i^{t+1}/p_i^t = exp(D(Q_i‖q^t)), so ln c_i^t = D(Q_i‖q^t). The new value of Ĩ I'm achieving is log Σ_i r_i^{t+1} = log Σ_i p_i^t c_i^t — the log of the *average* of the c_i^t under p^t. By concavity of log (Jensen), log Σ_i p_i^t c_i^t ≥ Σ_i p_i^t log c_i^t, and clearly the average of the c_i^t is ≤ their max, so log Σ_i p_i^t c_i^t ≤ max_i ln c_i^t = max_i D(Q_i‖q^t). I claim this max is actually an upper bound on the capacity itself:

  C ≤ max_i D(Q_i ‖ q^t).

Let me prove it, because it's the certificate I've been missing. Let p* achieve capacity with output q* = p*Q. Then

  C = Σ_i Σ_j p*_i Q(j|i) log( Q(j|i)/q*_j ) = Σ_i Σ_j p*_i Q(j|i) log( Q(j|i)/q^t_j · q^t_j/q*_j ).

Split the log into the q^t term and the q^t/q* term:

  C = Σ_i p*_i D(Q_i‖q^t) + Σ_j Σ_i p*_i Q(j|i) log( q^t_j/q*_j )
    = Σ_i p*_i D(Q_i‖q^t) + Σ_j q*_j log( q^t_j/q*_j )
    = Σ_i p*_i D(Q_i‖q^t) − D( q* ‖ q^t ).

The last term is a KL divergence, so −D(q*‖q^t) ≤ 0, giving C ≤ Σ_i p*_i D(Q_i‖q^t) ≤ max_i D(Q_i‖q^t), since a weighted average is no bigger than the maximum. So at every iteration I have a sandwich:

  I(p^t,Q) ≤ C ≤ max_i D(Q_i ‖ q^t).

The lower bound is the current mutual information; the upper bound is the largest per-letter divergence to the current output. The gap between them is a *computable* error bar. Stop when max_i D(Q_i‖q^t) − I(p^t,Q) < ε and I've certified C to within ε. And notice what the sandwich becomes at the optimum: as the iterates settle, the upper and lower bounds must squeeze together, which forces D(Q_i‖q*) to be the *same* value for all the active letters and no larger for the inactive ones — that is exactly the KKT condition I wrote down at the very start, with γ = C. So the fixed point of this innocent multiplicative map is precisely the capacity-achieving distribution; the map didn't just find *a* stationary point, it found the global one, because the problem is concave and the KKT point of a concave program over a convex set is the global max. The convergence is to C, not merely to a local plateau.

There's a second way to see why this had to work that I find reassuring. Rewrite the p-update as

  p^{t+1} = argmax_p ( Σ_i p_i D(Q_i ‖ q^t) − D(p ‖ p^t) ).

Check: the unconstrained-up-to-normalization maximizer of Σ_i p_i b_i − Σ_i p_i log(p_i/p_i^t) is p_i ∝ p_i^t exp(b_i), exactly my update with b_i = D(Q_i‖q^t). And Σ_i p_i D(Q_i‖q^t) is the first-order Taylor expansion of I(p,Q) around p^t (the gradient of I in direction p), while D(p‖p^t) is a proximity penalty keeping the new p near the old. So each step is a proximal / mirror-ascent step: maximize a local linear model of I with a KL trust region. That's a textbook recipe for monotone ascent on a concave function, and it explains why no step size is needed — the entropy geometry supplies the trust region for free. (It also suggests a knob: weight the penalty by γ_t and shrink it as I near the top to take bigger steps, as long as γ_t stays large enough — at least the maximal KL-induced gain of Q — to keep the ascent monotone.)

The same pressure appears on the rate-distortion side. The sign flips: I am *minimizing* mutual information instead of maximizing it:

  R(D) = min_{ p(x̂|x): E[d] ≤ D } I(X;X̂).

The hard distortion constraint is annoying; Lagrange it away. For a slope s > 0, minimize the unconstrained

  I(X;X̂) + s·E[d(X,X̂)] = Σ_x Σ_{x̂} p(x) p(x̂|x) [ log( p(x̂|x)/r(x̂) ) + s·d(x,x̂) ],

where r(x̂) = Σ_x p(x) p(x̂|x) is the reproduction marginal. Sweeping s > 0 traces the whole R(D) curve, with s the magnitude of the slope at each point. Now the same observation: r(x̂) is *determined* by p(x̂|x), but treat it as a free variable. The functional, as a function of the marginal-like argument r and the test channel p(x̂|x), is convex in each separately (KL is convex in each of its arguments), so this is a double *minimization* over two blocks, and I expect both inner mins to be closed form.

Fix the test channel p(x̂|x): minimizing over a free r the quantity Σ p(x) p(x̂|x) log( p(x̂|x)/r(x̂) ) is minimizing Σ_{x̂} (Σ_x p(x)p(x̂|x)) · (−log r(x̂)) plus constants — a cross-entropy in r — and is minimized at r(x̂) = Σ_x p(x) p(x̂|x), the induced marginal. Closed form.

Fix the marginal r: minimize over p(x̂|x), per x, the inner Σ_{x̂} p(x̂|x)[ log(p(x̂|x)/r(x̂)) + s d(x,x̂) ] subject to Σ_{x̂} p(x̂|x)=1. This is min of a KL-like term plus a linear cost — the same Gibbs problem, now with a minus sign — and the Lagrangian gives log p(x̂|x) = log r(x̂) − s d(x,x̂) + const, i.e.

  p(x̂|x) = r(x̂) exp( −s d(x,x̂) ) / Σ_{x̂'} r(x̂') exp( −s d(x,x̂') ).

Closed form. So the rate-distortion algorithm is: fix r, reweight each reproduction symbol x̂ by exp(−s d(x,x̂)) — close reproductions (small distortion) keep weight near 1, far ones are exponentially suppressed — and renormalize to get the test channel; then recompute the marginal r = Σ_x p(x) p(x̂|x); repeat. Each step decreases the Lagrangian, and biconvexity + KL geometry give monotone convergence to the minimum, exactly mirroring the capacity case but with max→min and the exp(+D) reweight replaced by exp(−s d). At convergence read off R = I(X;X̂) in bits and D = E[d], one point on the curve; sweep s for the rest.

One last thing clicks. The reason both of these — the max for capacity, the min for R(D) — converge to the *global* optimum and not just a stationary point is that each is alternating optimization of a KL-type divergence between two *convex* families of distributions. The input-side family and the reverse-channel family are each convex (linear in their free parameters), and alternating minimization of relative entropy between two convex sets is the canonical case that converges monotonically to the global minimum, because relative entropy has the right Pythagorean geometry over linear families. The alternating step is an information projection, and projections onto convex sets compose to the joint optimum. That's why I trust it: it's not a trick that happens to work on this objective, it's the natural geometry of the problem.

The equations fit in a small loop:

```python
import numpy as np

def kl_rows_to_vec(Q, q):
    """D(Q_i || q) for every row i of the channel matrix Q. Returns shape (N,)."""
    nz = Q > 0
    # Q_ij log(Q_ij / q_j), summed over j
    ratio = np.zeros_like(Q)
    ratio[nz] = np.log(Q[nz] / q[np.where(nz)[1]])
    return (Q * ratio).sum(axis=1)

def blahut_arimoto_capacity(Q, max_iter=1000, tol=1e-9):
    """Capacity of a DMC. Q[i,j] = P(y=j | x=i)."""
    N = Q.shape[0]
    p = np.full(N, 1.0 / N)                 # uniform, full-support start
    C_lower = -np.inf
    for _ in range(max_iter):
        q = p @ Q                           # induced output distribution q = pQ
        d = kl_rows_to_vec(Q, q)            # D(Q_i || q) for each input letter
        # sandwich at the current iterate: I(p,Q) <= C <= max_i D(Q_i || q)
        I_lower = float((p * d).sum())      # mutual information I(p,Q), in nats
        C_upper = float(d.max())            # certified upper bound on C
        C_lower = I_lower
        if C_upper - I_lower < tol:         # stop once C is pinned to within tol
            break
        # fold the Bayes-posterior step and the softmax step into one update:
        # p_i <- p_i * exp(D(Q_i || q)) / Z      (reweight by per-letter divergence)
        p = p * np.exp(d)
        p /= p.sum()                        # renormalize over the simplex
    return C_lower / np.log(2), p           # capacity in bits, optimal input

def blahut_arimoto_rate_distortion(p_x, dist, s, max_iter=1000, tol=1e-9):
    """One point of R(D). p_x: source pmf (L,); dist[x, xhat] = d(x, xhat); s>0 slope."""
    L, Lhat = dist.shape
    p_x = p_x / p_x.sum()                      # normalize the source weights
    W = np.exp(-s * dist)                    # exp(-s d): closeness reweighting
    p_cond = np.tile(p_x, (Lhat, 1)).T       # initial test channel p(xhat|x)
    p_cond /= p_cond.sum(axis=1, keepdims=True)
    D_prev = np.inf
    for _ in range(max_iter):
        r = p_x @ p_cond                     # reproduction marginal r = sum_x p(x) p(xhat|x)
        p_cond = r * W                       # p(xhat|x) ~ r(xhat) exp(-s d(x,xhat))
        p_cond /= p_cond.sum(axis=1, keepdims=True)
        ratio = np.log(p_cond / r[None, :])
        I = float((p_x[:, None] * p_cond * ratio).sum())   # rate (nats)
        D = float((p_x[:, None] * p_cond * dist).sum())    # average distortion
        if abs(D - D_prev) < tol:
            break
        D_prev = D
    return I / np.log(2), D                   # (rate in bits, distortion) on the R(D) curve
```

The causal chain, start to end: I needed to compute max_p I over an arbitrary channel; the concave-max-over-the-simplex had no closed-form step and no error certificate, so the generic solvers were clumsy and blind. Recognizing Q(j|i)/q_j = φ*(i|j)/p_i exposed the *posterior* hiding inside the mutual information; promoting that posterior to a free reverse channel φ turned I into a maximum, I = max_φ Ĩ, with the gap an average KL divergence that vanishes at Bayes. That made capacity a double maximization C = max_p max_φ Ĩ whose two inner problems are both closed form — Bayes for φ, Gibbs softmax for p — so alternating between them monotonically increases I; the variational identity makes the increase automatic, boundedness forces convergence, and the dual bound C ≤ max_i D(Q_i‖q) both certifies the stopping gap and, by squeezing the sandwich shut, pins the limit to the KKT point, i.e. to the true global capacity. The exact same construction with minimization and an exp(−s d) reweight computes R(D), and the whole thing is one instance of alternating information projection between two convex families.
