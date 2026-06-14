Let me start from what actually goes wrong when I try to fix an orbit. I have a handful of unknowns — the elements of the orbit — and I can take as many observations as I like, each of which, once the geometry is reduced, hands me one approximately linear equation in those unknowns. So I have more equations than unknowns. If every observation were exact I could pick as many equations as I have unknowns, solve, and be done. But no observation is exact; every equation is off by some small error. So there is no set of values that satisfies all of them simultaneously, and the residuals can't all be zero. The whole difficulty is right here: I have to combine discordant equations into one definite answer, and I need a rule for *how* to split the unavoidable error among them.

The first thing I try is the obvious thing. I have, say, six unknowns; I keep six equations, solve that square system exactly, throw the rest away. It gives an answer, and a unique one once I've chosen which six. But the moment I look at it I'm uneasy. Which six? There's no reason to anoint any particular six as the true ones — and the answer changes if I pick a different six. Worse, the six I keep are forced to hold *exactly*, which means whatever errors happen to live in those six get baked straight into the solution, while the other observations, which I paid for and which carry just as much information, are simply discarded. I'm throwing away data and letting the noise in an arbitrary subset dominate. That can't be right. I want a rule that uses *all* the equations and lets no single one tyrannize the rest.

So I want to minimize the errors collectively, not zero out a chosen few. Fine — minimize what, exactly? The errors come with signs, and they'll be positive on some equations and negative on others, so I can't minimize their plain sum; that would let a huge positive error cancel a huge negative one and call the result good. I need a measure of the *size* of each error, blind to sign, and then make the whole collection of sizes small. The honest first candidate is the sum of absolute errors, Σ|Δ|. Boscovich had done something like this — minimize the total absolute deviation, sometimes with the side condition that the signed residuals sum to zero. It's a real principle, it uses all the data, and it's stubbornly resistant to a single wild observation, since one big error contributes only its first power. I like it on those grounds. But when I actually try to minimize it for several unknowns I hit a wall. The absolute value has a corner at zero; |Δ| isn't differentiable there. So I can't just set a derivative to zero and read off a clean equation for each unknown. The minimum is governed by *which* residuals happen to be exactly zero at the optimum — it's a combinatorial search over sign patterns, not a smooth solve, and that's exactly why in practice nobody pushes it past two unknowns. The same disease kills the minimax idea, "make the single largest error as small as possible": max|Δ| is also non-smooth, its optimum pinned by whichever few equations are tight, again combinatorial, again no derivative to exploit. Both principles are reasonable and both leave me without a uniform linear system to solve for any number of unknowns. Wall.

Let me step back and ask what property I actually need from the measure of error-size. I need something that (a) is blind to the sign of Δ, (b) is *smooth* through Δ = 0 so I get a derivative condition, and (c) grows with |Δ| so making it small means making the errors small. The absolute value has (a) and (c) but fails (b) at the one point that matters. The cure for the corner is to round it off — and the simplest smooth, even, increasing-in-|Δ| function I can put in place of |Δ| is Δ² itself. It's symmetric, it's flat-bottomed and differentiable at zero, its derivative is just 2Δ, and it's as elementary as it gets. So let me try: choose the unknowns to make the *sum of the squares* of the errors a minimum,

  Φ(x, y, z, …) = Σ_k E_k²,

where the k-th equation has residual E_k = a_k + b_k x + c_k y + f_k z + ⋯ , the a's, b's, c's being known coefficients that differ from equation to equation. Now watch what the smoothness buys me. Set the derivative with respect to each unknown to zero. For x,

  ∂Φ/∂x = Σ_k 2 E_k · ∂E_k/∂x = 2 Σ_k b_k E_k = 0,

so, dropping the 2 and substituting E_k = a_k + b_k x + c_k y + f_k z + ⋯,

  0 = Σ_k b_k(a_k + b_k x + c_k y + f_k z + ⋯) = Σab + x Σb² + y Σbc + z Σbf + ⋯,

writing Σab for a_1 b_1 + a_2 b_2 + ⋯ and Σb² for b_1² + b_2² + ⋯ and so on. This is *linear* in the unknowns. That's the whole point I was missing with absolute values: squaring makes the cost quadratic, so its stationarity condition is linear, and a linear condition is something I can actually solve. Do the same for y and for z:

  0 = Σac + x Σbc + y Σc² + z Σcf + ⋯ ,
  0 = Σaf + x Σbf + y Σcf + z Σf² + ⋯ .

One equation per unknown — exactly as many as I have unknowns — so the system is determinate, for *any* number of unknowns. And look at the coefficient matrix: the off-diagonal entries Σbc, Σbf, Σcf appear in two equations each. The thing multiplying y in the x-equation, Σbc, is the very same thing multiplying x in the y-equation. The matrix is symmetric. That's not a curiosity; it cuts the arithmetic roughly in half and it's a structural sign that I've stumbled onto something natural rather than arbitrary. So squaring gives me, all at once, a smooth objective, a derivative condition, and a single symmetric linear system in any number of unknowns. The non-smooth principles gave me none of that.

Is this minimum a genuine one? Suppose I've found values that make all the errors zero (the lucky exact case) and I perturb the unknowns by small δx, δy, δz. Each residual, which was zero, becomes b_k δx + c_k δy + f_k δz + ⋯ to first order, and its square is (b_k δx + c_k δy + ⋯)², a quantity of the *second* order in the perturbations. So Φ leaves its minimum only quadratically — the first-order variation vanishes — which is the defining signature of a true minimum. Squaring is what makes the cost sit in a smooth quadratic bowl. And there's a practical bonus I notice in the structure of the normal equations: each one is built by *adding up* per-equation products. If after solving I find one residual absurdly large — a botched observation — I don't have to start over. I just subtract that equation's contributions out of the sums and re-solve. The additive form makes editing the data cheap.

Now the test that decides whether I've found the right principle or merely a workable one: does it reproduce the arithmetic mean? Everyone already trusts that for many equally good direct measurements of a single quantity, the most probable value is the plain average. If my squared-error rule contradicts that, it's wrong, full stop. Take one unknown x and direct observations a', a'', a''', …; the residuals are a' − x, a'' − x, …; minimize Σ(a^(i) − x)². Derivative zero:

  0 = Σ (a^(i) − x) = (a' + a'' + ⋯) − n x  ⇒  x = (a' + a'' + ⋯)/n,

the arithmetic mean, with n the number of observations. It falls straight out. So least squares contains the mean as its one-unknown special case. That's the moment I trust the principle: the rule that gives me a clean linear system for many unknowns is the *same* rule that gives the universally accepted answer for one. The squaring wasn't a convenience I settled for; it's the choice that makes the general method agree with the special case nobody would give up.

I could stop here — I have a usable, justified, general method. But something nags. I chose the square because it was the simplest smooth even function, and "simplest" is an aesthetic argument, not a reason from the nature of the errors. Why squares and not fourth powers, which are also smooth and even? Both would give me a stationarity condition, though the fourth power's would be cubic, not linear — and there's a real difference, since the linear one is the only one I can solve as a clean system. But I want a deeper justification than "the algebra is nicest." Let me approach the whole thing from the other side, from probability, and see whether the square is *forced* on me rather than merely convenient.

Set it up properly. Each observed value M is the true value of some function V of the unknowns p, q, r, s, … plus a random error, so the error is Δ = M − V. The chance of an error of size Δ is given by some density φ(Δ). I don't know φ's exact form, but I can say a few things any honest error law must satisfy: it's largest at Δ = 0 (small errors beat large ones), it's even (an error +Δ is as likely as −Δ, since there's no reason for the instrument to favor one direction), and it falls toward zero as |Δ| grows, fast enough that enormous errors are effectively impossible. The observations are independent, so the probability of getting this whole *collection* of observed values together, for a given guess at the unknowns, is the product of the individual densities,

  Ω = φ(M − V) · φ(M' − V') · φ(M'' − V'') ⋯ .

And here's the inversion I need. Before observing, every system of values of the unknowns was equally likely — a flat prior over them. After observing, the probability that the unknowns take any particular system of values is proportional to the probability that *those* values would have produced the data I actually saw, which is Ω. So the most probable system of unknowns is the one that *maximizes Ω*. That's the principle: maximize the product of the error densities, equivalently set ∂(log Ω)/∂p = 0 for each unknown. Since log Ω = Σ_i log φ(Δ_i), and Δ_i depends on the unknowns through V_i, the condition is

  Σ_i [φ'(Δ_i)/φ(Δ_i)] · (∂Δ_i/∂p) = 0,   and similarly for q, r, s, … .

This is general and clean, but it's useless until I know φ. And I just admitted I can't write φ down from first principles — it depends on a tangle of physical and physiological causes I can't calculate. So the form of the estimator is hostage to a function I don't know. Wall.

I'll break it the way I broke the choice of error-measure before: by demanding consistency with the thing I already trust. I don't know φ, but I *do* know what answer the method must give in the simplest case. For n equally good direct observations M, M', …, M^(n) of a single quantity p, the most probable value has to come out to the arithmetic mean — that's non-negotiable, it's the one case where the answer is settled. So let me impose that and see what it forces φ to be. Here V is just p itself for every observation, so ∂Δ_i/∂p = −1, and the stationarity condition collapses to

  φ'(M − p)/φ(M − p) + φ'(M' − p)/φ(M' − p) + ⋯ + φ'(M^(n) − p)/φ(M^(n) − p) = 0,

and this must hold *whenever* p is the arithmetic mean of the M's. That's a strong constraint — it has to be satisfied for every possible set of observations whose mean is p. Let me write g(Δ) = φ'(Δ)/φ(Δ) to keep it readable; the condition is Σ_i g(M^(i) − p) = 0 with p = (1/n)Σ M^(i), i.e. Σ_i g(Δ_i) = 0 whenever Σ_i Δ_i = 0 (since Σ(M^(i) − p) = Σ M^(i) − n p = 0 by definition of the mean). So I need: g is a function with Σ g(Δ_i) = 0 for *every* tuple of residuals summing to zero.

Let me squeeze that without hiding the functional equation. With two residuals, Δ and −Δ, the condition gives g(Δ) + g(−Δ) = 0, so g is odd. With three residuals, a, b, and −(a+b), it gives

  g(a) + g(b) + g(−a − b) = 0.

Oddness turns the last term into −g(a+b), so g(a+b) = g(a) + g(b). The log-derivative of a differentiable positive density is continuous, and a continuous additive function can only be linear. Therefore g must be linear through the origin: g(Δ) = kΔ for some constant k. Check it against any residual tuple with zero sum: Σ kΔ_i = kΣΔ_i = 0. Any nonlinearity would violate the three-residual equation somewhere. So consistency with the arithmetic mean forces

  φ'(Δ)/φ(Δ) = k Δ.

That's a differential equation, and an easy one. d(log φ)/dΔ = k Δ, so log φ(Δ) = (k/2)Δ² + constant, so

  φ(Δ) = A · exp( (k/2) Δ² ).

For φ to be a density that's largest at zero and decays — for the product Ω to have a *maximum* rather than running off to infinity — the exponent must be negative, so k < 0. Write k = −2h² with h real, so the exponent is −h²Δ². The constant A is fixed by normalization: φ must integrate to one over all Δ, and the integral ∫_{−∞}^{∞} exp(−h²Δ²) dΔ = √π / h (Laplace's result), so A = h/√π. The error law is therefore

  φ(Δ) = (h/√π) · exp(−h² Δ²),

and it dropped out of one demand: that the mean be the most probable value in the direct-observation case. I didn't assume the bell curve; this consistency demand forces it inside that setup. The parameter h I read off physically — it sets how sharply errors cluster at zero. Compare two instruments with precisions h and h'; the probability of a given error band scales with the standardized product hΔ, so errors d in the first and d·(h/h') in the second have the same probability scale. If h' = 2h, then a double error in the first system has the same facility as a single error in the second; the second has double precision and half the typical error scale. h is the measure of precision: larger h, tighter observations.

Now close the loop. With φ(Δ) = (h/√π) exp(−h²Δ²), the product to maximize is

  Ω = Π_i (h/√π) exp(−h² Δ_i²) = (h/√π)^N exp( −h² Σ_i Δ_i² ).

The prefactor doesn't depend on the unknowns, and the exponential is monotone decreasing in Σ_i Δ_i². So *maximizing* the probability Ω is exactly the same as *minimizing* Σ_i Δ_i² — the sum of the squares of the errors. There it is. The squaring I had reached for on grounds of smoothness and simplicity is the very thing the probability calculus produces, once I insist the method agree with the arithmetic mean. Least squares is the most-probable-value estimator under the error law forced by that direct-observation consistency postulate. The two routes — minimize Σ Δ² because it's smooth and reduces to the mean, and maximize Ω which forces the Gaussian which forces Σ Δ² — meet at the same place.

I should be honest about a circularity that's nagging at me. I postulated the mean to derive the Gaussian, and then I use the Gaussian to justify least squares — but least squares *gives back* the mean in the one-unknown case. Am I just assuming what I set out to prove? Let me lay out the logical order carefully. The postulate is the mean — that's my one assumption, motivated by universal practice, not derived. From it I deduce, inside the direct-observation consistency setup, that the error law must be exp(−h²Δ²): the mean can be the most probable value there only when φ'/φ is linear in the error. Then, assuming that Gaussian law, maximizing the joint probability is minimizing the sum of squares for any configuration of unknowns, because the exponential is monotone. So the structure is: postulate (mean) ⇒ forces (Gaussian error law in the direct-observation argument) ⇒ yields (least squares for the general observation equations), and least squares happens to reproduce the mean in the special case where it started. It's not circular; it's a derivation of a general estimator from a postulate that only pins down a special case. What I've shown is not that errors are "really" Gaussian out in the world — I can't show that — but that if I accept the mean as the most-probable value for equal direct observations, and I use the smooth independent-error probability calculus, the Gaussian form and the squared-error rule come with it. The strength of the argument is exactly the strength of the one postulate.

Let me also handle observations of unequal quality, since real campaigns mix careful and rough measurements. If observation i has its own precision h_i, then φ_i(Δ_i) = (h_i/√π) exp(−h_i² Δ_i²), and the product gives exp(−Σ_i h_i² Δ_i²), so the thing to minimize is the *weighted* sum of squares Σ_i h_i² Δ_i², each residual weighted by the square of its precision. A precise observation (large h_i) pulls harder on the fit; a sloppy one (small h_i) is allowed to disagree more. Under this normalization Var(Δ_i) = 1/(2h_i²), so h_i² is proportional to inverse variance; the missing factor 2 is common and cannot move the minimizer. I don't have to invent the weighting, the same maximization hands it to me. When all observations are equally good the weights are equal and I'm back to the plain sum of squares.

Now I want to see the whole thing geometrically, because the picture makes the algebra less accidental. Stack the equations: write the residual vector as y − Xβ, where y holds the observed values, β the unknowns, and X the matrix of known coefficients (its columns are the coefficient-vectors b, c, f, … from before). The sum of squared residuals is just the squared Euclidean length ‖y − Xβ‖². Minimizing a squared length — that's a projection. As β ranges over all values, Xβ ranges over the column space of X, the subspace of all combinations of the coefficient-vectors. So I'm asking: which point ŷ = Xβ̂ in that subspace is closest to the data point y? The closest point is the foot of the perpendicular — the orthogonal projection of y onto the column space. The minimizing residual ε̂ = y − Xβ̂ must be perpendicular to the subspace, i.e. to every column of X, which is exactly Xᵀε̂ = 0, that is

  Xᵀ(y − Xβ̂) = 0  ⇒  Xᵀ X β̂ = Xᵀ y,

the normal equations again — and now I see why they're called *normal*: they're the statement that the residual is normal (perpendicular) to the model space. When the columns of X are independent, XᵀX is invertible and β̂ = (XᵀX)⁻¹Xᵀy, with fitted values ŷ = X(XᵀX)⁻¹Xᵀ y = P y, P the projection onto col(X). This is the deep reason the square is special and the absolute value isn't: the squared Euclidean norm makes the best fit over a subspace an orthogonal projection, a linear operation, which is why the solution is a linear system. Other powers lose these normal equations and become nonlinear best-fit problems. Squaring is the choice that turns "best fit" into "perpendicular projection," the most rigid and computable notion of closest there is.

One more thing bothers me, and resolving it makes the method stronger than the Gaussian argument alone. The Gaussian derivation rests on a postulate (the mean) and an assumed error law. What if I don't want to assume the law at all — can I still defend least squares? Suppose only the bare facts: errors have mean zero, common variance σ², and are uncorrelated — nothing about their shape. Restrict to estimators that are *linear* in the data, β̃ = C y, and *unbiased*, E[β̃] = β for all true β. Unbiasedness with E[y] = Xβ means C Xβ = β for every β, so C X = I. Write any such C as C = (XᵀX)⁻¹Xᵀ + D; then C X = I forces ((XᵀX)⁻¹Xᵀ + D)X = I + D X = I, so D X = 0. Compute the variance: with Var(y) = σ²I,

  Var(β̃) = σ² C Cᵀ = σ² ( (XᵀX)⁻¹Xᵀ + D )( (XᵀX)⁻¹Xᵀ + D )ᵀ.

Expand. The cross terms carry D X or its transpose, both zero: (XᵀX)⁻¹Xᵀ Dᵀ = (XᵀX)⁻¹(D X)ᵀ = 0. So the variance splits cleanly,

  Var(β̃) = σ²(XᵀX)⁻¹ + σ² D Dᵀ = Var(β̂_LS) + σ² D Dᵀ.

D Dᵀ is positive semidefinite, so Var(β̃) ⪰ Var(β̂_LS) — every linear unbiased estimator has variance at least that of the least-squares one, with equality only when D = 0, i.e. only for least squares itself. The squared-error estimator is the *best linear unbiased estimator*, and this needs no Gaussian assumption — only zero-mean, equal-variance, uncorrelated errors. So least squares is doubly justified: assume the Gaussian and it's the most-probable-value (maximum-likelihood) estimator; drop the Gaussian and it's still the minimum-variance estimator among all linear unbiased ones. The Gaussian upgrades "best linear unbiased" to "most probable"; without it, the optimality survives in the weaker but assumption-light form.

So where does this land as something I'd actually compute? The principle is: given an array of per-element discrepancies between what I predicted and what I observed, collapse it to a scalar by *averaging their squares*, and drive that scalar down. In a setting where the predictions and targets come as same-shaped multidimensional arrays, the residual is just their elementwise difference, and the aggregate I minimize is the mean of the squared differences — the empirical version of Σ Δ², divided by the count so the scale doesn't depend on how many elements there happen to be (dividing by a constant doesn't move the minimizer; it just keeps the number interpretable across array sizes). Every element is treated alike, which is precisely the equal-precision case where the weights are uniform. Filling the one empty slot — the rule that turns the array of residuals into a scalar:

```python
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Mean squared error: the squared Euclidean residual, averaged over
    elements. The empirical least-squares objective -- minimizing it makes the
    prediction the (equal-precision) best fit to the target. Symmetric, smooth
    through zero, with a linear stationarity condition; reduces to the mean in
    the one-unknown case and is the maximum-likelihood fit under Gaussian errors."""

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        # reduction="mean" returns (1/N) * sum_i (predicted_i - state_i)^2
        # over all elements of the same-shaped tensors.
        return F.mse_loss(predicted, state, reduction="mean")
```

Let me retrace the causal chain. I started stuck with an over-determined system whose errors can't all vanish, and the crude fix — solve a chosen square subsystem, discard the rest — was arbitrary and let the noise in the kept equations dominate. Minimizing the total absolute error or the maximum error were honest principles but non-smooth, so they gave no derivative condition and no uniform linear solve, stalling at two unknowns. Demanding a sign-blind error-measure that is smooth through zero pointed at the square; the sum of squares turned out to have a linear stationarity condition — one symmetric normal equation per unknown, determinate for any number of unknowns — and to reduce exactly to the arithmetic mean in the single-quantity case, which is the test that it's the right principle and not just a workable one. Pushing for a reason beyond convenience, I inverted to probability: maximize the product of error densities (flat prior), and impose that the mean be the most probable value for equal direct observations; that demand forces the error law to be Gaussian, exp(−h²Δ²), inside the direct-observation argument, and under that law maximizing the joint probability is minimizing the sum of squares — with unequal precisions weighting each residual by a factor proportional to inverse variance. The apparent circularity dissolves once the order is fixed: the mean is the postulate, the Gaussian is what it forces in that setup, least squares is what the Gaussian forces for the general equations. Geometrically the squared length makes the best fit an orthogonal projection onto the column space, which is why the solution is linear (the normal equations XᵀXβ̂ = Xᵀy say the residual is perpendicular to the model). And dropping the Gaussian entirely, least squares is still the minimum-variance estimator among all linear unbiased ones. The whole thing collapses, in code, to averaging the squared elementwise residual between prediction and target.
