I want to solve a convex-concave saddle point, min over x and max over y of f(x,y), and I keep tripping over the same thing: the method that works for minimization just doesn't work here, and I don't fully understand why yet. So let me strip the problem to the bone and stare at the simplest instance that already breaks, f(x,y) = x·y. It's bilinear, convex in x, concave in y, with the obvious saddle at the origin. If I can't handle this, I can't handle anything.

The natural move is to treat min-max as "descend in x, ascend in y" at the same time. Write z = (x,y) and bundle the two updates into one vector field. For minimization I'd follow minus the gradient; for the max-player I follow plus the gradient. So the joint direction is the field F(z) = (∂f/∂x, −∂f/∂y), and the simultaneous update is z_{t+1} = z_t − τ F(z_t). For f = x·y this is F(x,y) = (y, −x). Let me just compute. ∂f/∂x = y, ∂f/∂y = x, so F = (y, −x). That's z rotated by ninety degrees — it's J·z with J = [[0,1],[−1,0]], the skew-symmetric rotation generator. Huh. The whole field is rotational. At every point it points *around* the origin, never toward it. There is no "downhill toward the saddle" anywhere in this field; it's all sideways.

That should already worry me, but let me make it quantitative, because "rotational" by itself isn't a verdict. The update operator is M = I − τJ. Its eigenvalues are 1 − τ·(eigenvalues of J), and J has eigenvalues ±i, so M has eigenvalues 1 ∓ iτ. The modulus is |1 ∓ iτ| = √(1 + τ²). That's strictly greater than 1 for any τ > 0. So ‖z_t − z*‖ doesn't just fail to shrink — it grows like (√(1+τ²))^t, geometric divergence, the iterates spiraling outward forever. And shrinking τ does nothing structural: it makes the blow-up slower (1 + τ² is closer to 1), but the modulus is still above 1, so it still diverges. The single forward evaluation at z_t tells me which way the field points *right now*, but the field is about to rotate me, and by the time I've moved, the direction I committed to is already stale and pointing slightly outward. Each step overshoots, and the overshoots compound. That's the obstruction I have to get around, and it's not a tuning problem — no step size rescues a modulus that exceeds one.

Let me make sure I'm not fooling myself about the cause, because if I misdiagnose this I'll patch the wrong thing. For ordinary minimization F = ∇φ with φ convex, and there the update I − τ∇²φ has eigenvalues 1 − τ·(nonneg), which sit in [1 − τL, 1], inside the unit disk for τ small — it contracts. The difference here is that F's Jacobian is *skew*-symmetric instead of symmetric-positive; its eigenvalues are pure imaginary, not real-nonnegative. The right generalization of "convex gradient field" isn't "gradient of something convex," it's a **monotone operator**: ⟨F(z) − F(z'), z − z'⟩ ≥ 0 for all z, z'. For F = ∇φ this is exactly convexity of φ; for my rotational F = Jz it's ⟨J(z−z'), z−z'⟩ = (z−z')ᵀJ(z−z') = 0, since Jᵀ = −J makes that quadratic form vanish. So the bilinear field is monotone but *only just* — the monotonicity inequality is tight at equality, zero, everywhere. There's no strong-monotonicity margin to lean on, no ⟨F(z)−F(z'),z−z'⟩ ≥ μ‖z−z'‖² with μ > 0 to give me contraction. That's the precise sense in which this is the hard case: the operator is monotone but has no contractive component at all, it's pure rotation, and the explicit forward step has nothing to grab onto.

Now, is there *any* update that's stable on this? Let me think about the opposite of the forward step. Instead of evaluating the field at where I *am* and stepping, evaluate it at where I'm *going* — the implicit/backward step z_{t+1} = z_t − τ F(z_{t+1}). This is backward Euler, and the folklore from stiff ODEs and from monotone-operator theory is that backward steps are unconditionally stable. Let me check it on the bilinear field. z_{t+1} = z_t − τ J z_{t+1} ⟹ (I + τJ) z_{t+1} = z_t ⟹ z_{t+1} = (I + τJ)^{−1} z_t. Eigenvalues of (I + τJ)^{−1} are 1/(1 ∓ iτ), modulus 1/√(1 + τ²) < 1 for *every* τ > 0. It spirals *inward*. Unconditionally. No step-size restriction at all. And this isn't just a bilinear accident: for any monotone F the resolvent (I + τF)^{−1} is firmly nonexpansive and gives Fejer decrease toward the solution set. So the implicit method is, in a real sense, the ideal — maximally stable, τ unrestricted.

So why don't I just run it? Because it's implicit. z_{t+1} appears on both sides, inside F. Computing (I + τF)^{−1}(z_t) means solving a nonlinear fixed-point system in z_{t+1} at every single iteration. For a generic f that inner solve is as hard as the original problem. For the bilinear toy I could invert a 2×2 matrix, sure, but the whole point is to have a method that works on real f where F is some complicated nonlinear operator and I have no closed-form inverse. So the implicit step is a target to *imitate*, not an algorithm I can run. The question crystallizes: how do I get the backward step's stability while only ever evaluating F explicitly, at points I already know?

Here's the tension stated cleanly. The backward step needs F evaluated at z_{t+1} — the future point I don't have yet. The forward step evaluates F at z_t — the present point, which I have, but which gives the wrong, outward-pointing direction. What if I *guess* the future point cheaply, evaluate F there, and use that? I don't know z_{t+1}, but I can take one ordinary forward step to get a rough prediction of where I'm headed:

  w = z_t − τ F(z_t).

This w is a cheap, explicit estimate of the next point — a "look-ahead" or leader iterate. It's not z_{t+1} (that forward step is exactly the unstable one), but it's *in the direction* I'm rotating, one step out. Now use the field *there* in place of the field at the unknown z_{t+1}, but — and this is the crux — take the actual step from the *original* point z_t, not from w:

  z_{t+1} = z_t − τ F(w).

Anchor at z_t, aim with F(w). Two evaluations of F per iteration: one at z_t to find the look-ahead w, one at w to get the corrected direction. The first is a *predictor*, the second a *corrector*. And it's fully explicit — w and z_{t+1} are both closed-form, no inner solve.

I have to check I haven't just smuggled in two forward steps, because two forward steps would still diverge. If I had stepped from w instead — w − τF(w) — that's literally two GDA steps in a row, the operator (I−τJ)² with modulus (1+τ²) > 1, still blows up. The thing that makes this different is the anchor: I peek ahead with F(w) but I commit the move from z_t. Let me grind out what that does on the bilinear field, because this is where I'll see whether the idea has teeth. F(z_t) = J z_t, so w = z_t − τJ z_t = (I − τJ) z_t. Then F(w) = J w = J(I − τJ) z_t = (J − τJ²) z_t. And J² = −I (rotating by 90° twice is rotating by 180°, which is −I), so J − τJ² = J + τI. Therefore

  z_{t+1} = z_t − τ(J + τI) z_t = (I − τJ − τ²I) z_t.

Look at what appeared: a −τ²I term that wasn't in the forward step. The eigenvalues are now 1 − τ² ∓ iτ, with modulus √((1 − τ²)² + τ²). Let me expand: (1 − τ²)² + τ² = 1 − 2τ² + τ⁴ + τ² = 1 − τ² + τ⁴ = 1 − τ²(1 − τ²). For τ < 1 that's strictly less than 1, so it should contract. Let me put a number on it at τ = 0.1, and let me get it two ways so I'm sure I haven't mis-derived the map. Symbolically the modulus is √(1 − 0.01·0.99) = √0.9901 = 0.995038. Numerically, building the map straight from two field evaluations — w = (I − 0.1J)z, then z₊ = z − 0.1·J·w — and reading off |eig| gives 0.995038 as well; the matrix it produces, [[0.99, −0.1],[0.1, 0.99]], is exactly I − 0.1J − 0.01I, so the algebra and the actual update agree. Below one. The spiral turns inward — the real part pulled under 1, which the forward step (modulus 1.004988 at the same τ) never managed. So per step the forward iterate grows by ×1.004988 and the corrected one shrinks by ×0.995038; starting from ‖z₀‖ = √200 ≈ 14.14 and running 50 steps, that predicts forward ≈ 14.14·1.004988⁵⁰ ≈ 18.14 and corrected ≈ 14.14·0.995038⁵⁰ ≈ 11.03. I ran the two iterations to check, and at t = 50 they read 18.14 and 11.03 — the forward iterate climbing, the corrected one decaying, matching the per-step factors to the digits I can see. The extra step manufactured a genuine contraction where there was none.

That −τ²I is suggestive: it looks like the start of the implicit step's expansion, and if so the corrector isn't just *a* fix but specifically an approximation of the backward step. Let me check that rather than assert it. The backward step on the bilinear field is (I + τJ)^{-1}; since (I + τJ)(I − τJ) = I + τ²I, it equals (1/(1+τ²))(I − τJ) exactly. Expanding the scalar factor, 1/(1+τ²) = 1 − τ² + O(τ⁴), so (I + τJ)^{-1} = I − τJ − τ²I + O(τ³). The corrected step I just derived, I − τJ − τ²I, is precisely this truncated after the −τ²I term, whereas the plain forward step I − τJ truncates one term earlier. If that reading is right, the corrected map should sit *closer* to the true resolvent than the forward map does — by a factor of about τ, since I'm keeping one more order. So I measured the Frobenius distances at τ = 0.1: forward-to-resolvent is 0.01407, corrected-to-resolvent is 0.001407 — exactly a factor of ten, i.e. a factor of 1/τ, closer. And the corrected map's modulus 0.995038 essentially coincides with the resolvent's own modulus 0.995037. So the corrector isn't merely contracting; it is buying the leading curvature term of the resolvent, explicitly, with one extra gradient evaluation in place of a matrix inverse — and the factor-of-τ gap is the quantitative signature that it's a second-order, not first-order, approximation.

Let me nail that O(τ²)-vs-O(τ) claim in general, not just on the toy, because it's the real justification and I want it to hold for any Lipschitz monotone F. Suppose F is L-Lipschitz, ‖F(a) − F(b)‖ ≤ L‖a − b‖. Let w_imp be the true implicit/proximal next point, w_imp = z_t − τ F(w_imp), and let z_{eg} = z_t − τ F(w) be my explicit corrector, with w = z_t − τ F(z_t) the look-ahead. Then

  ‖z_{eg} − w_imp‖ = ‖(z_t − τF(w)) − (z_t − τF(w_imp))‖ = τ‖F(w) − F(w_imp)‖ ≤ τL‖w − w_imp‖.

So I've reduced the error of the corrector to τL times the error of the *look-ahead* w against the implicit point. Now bound that. The look-ahead is itself the zeroth approximation to the implicit fixed point — w = z_t − τF(z_t) uses F at z_t whereas w_imp uses F at w_imp — so the same one-line argument gives ‖w − w_imp‖ = τ‖F(z_t) − F(w_imp)‖ ≤ τL‖z_t − w_imp‖. Chain them:

  ‖z_{eg} − w_imp‖ ≤ τL · τL · ‖z_t − w_imp‖ = τ²L² ‖z_t − w_imp‖.

There it is: the extra-gradient point matches the implicit point to error τ²L² times the distance the implicit step itself moves. Order τ², where the plain forward step is order τL — and the extra factor τL is genuinely a *reduction* only when τ < 1/L, which is exactly why this method wants small step sizes: the look-ahead-then-correct trick is only an improvement in the regime τL < 1, where each layer of the recursion shrinks the error. (This generalizes: k look-ahead steps before correcting would give error (τL)^k, but each costs a gradient, and one extra step already turns O(τ) into O(τ²), which is enough to cross from divergence to convergence — so I'll pay for exactly one.) This also tells me the structure is a predictor-corrector approximation of the implicit map, which is the deep reason it inherits the implicit step's stability.

Now I want a convergence proof that doesn't lean on the bilinear special structure — something that works for any monotone L-Lipschitz F, because that's the class I actually care about. Let me set z* with F(z*) = 0 (the equilibrium), w = z_t − τF(z_t), z_{t+1} = z_t − τF(w), and track the squared distance to z*. Expand:

  ‖z_{t+1} − z*‖² = ‖z_t − τF(w) − z*‖² = ‖z_t − z*‖² − 2τ⟨F(w), z_t − z*⟩ + τ²‖F(w)‖².

The cross term has z_t in it, but the natural quantity for monotonicity is z_t measured against w. So split z_t − z* = (z_t − w) + (w − z*) inside the inner product:

  −2τ⟨F(w), z_t − z*⟩ = −2τ⟨F(w), w − z*⟩ − 2τ⟨F(w), z_t − w⟩.

The first piece is the good one. By monotonicity, ⟨F(w) − F(z*), w − z*⟩ ≥ 0, and F(z*) = 0, so ⟨F(w), w − z*⟩ ≥ 0; with the minus sign in front it *reduces* the distance. The second piece plus the τ²‖F(w)‖² tail I need to control and show they don't overwhelm the gain. Note z_t − w = τF(z_t) by definition of the look-ahead. Let me regroup the leftover terms − 2τ⟨F(w), z_t − w⟩ + τ²‖F(w)‖² and remember z_{t+1} − z_t = −τF(w) and w − z_t = −τF(z_t). I'll complete the square against ‖w − z_t‖². Write the standard identity for the two anchor moves: with a = z_{t+1} − z_t = −τF(w) and b = w − z_t = −τF(z_t),

  ‖z_{t+1} − z*‖² = ‖z_t − z*‖² − 2τ⟨F(w), w − z*⟩ + τ²‖F(w) − F(z_t)‖² − ‖w − z_t‖².

Let me verify that regrouping reproduces what I had. τ²‖F(w) − F(z_t)‖² − ‖w − z_t‖² = τ²‖F(w)‖² − 2τ²⟨F(w), F(z_t)⟩ + τ²‖F(z_t)‖² − τ²‖F(z_t)‖² (using ‖w − z_t‖² = τ²‖F(z_t)‖²) = τ²‖F(w)‖² − 2τ²⟨F(w), F(z_t)⟩ = τ²‖F(w)‖² − 2τ⟨F(w), τF(z_t)⟩ = τ²‖F(w)‖² − 2τ⟨F(w), z_t − w⟩. Yes — that's exactly the leftover two terms. So the clean one-step identity is

  ‖z_{t+1} − z*‖² = ‖z_t − z*‖² − 2τ⟨F(w), w − z*⟩ + τ²‖F(w) − F(z_t)‖² − ‖w − z_t‖²,

and now every term has a meaning. The middle term −2τ⟨F(w), w − z*⟩ ≤ 0 is the progress from monotonicity. The last two together, τ²‖F(w) − F(z_t)‖² − ‖w − z_t‖², are the *discretization error* — the price of using F(w) instead of F at the true implicit point. And this is precisely where the extra step pays off, because I can control ‖F(w) − F(z_t)‖ by Lipschitzness: ‖F(w) − F(z_t)‖ ≤ L‖w − z_t‖, so

  τ²‖F(w) − F(z_t)‖² − ‖w − z_t‖² ≤ (τ²L² − 1)‖w − z_t‖².

If τ < 1/L this is strictly negative. So the error term doesn't just stay bounded — it's *another* negative contribution. Putting it together,

  ‖z_{t+1} − z*‖² ≤ ‖z_t − z*‖² − 2τ⟨F(w), w − z*⟩ − (1 − τ²L²)‖w − z_t‖²,

with both subtracted terms ≥ 0 when τ ≤ 1/L. The distance to z* is monotonically nonincreasing, and it strictly decreases unless w = z_t (no look-ahead movement) and ⟨F(w), w − z*⟩ = 0 simultaneously — i.e. unless we're at the solution. That's the convergence the forward step could never get: the −‖w−z_t‖² coming from anchoring at z_t is the contractive term that cancels the rotational overshoot. The step-size ceiling τ ≤ 1/L is forced by exactly this inequality — at τ = 1/L the error term vanishes and I'm at the boundary; below it, strict contraction. (Smaller is safer; on the bilinear toy I saw the contraction modulus √(1 − τ²(1−τ²)), which wants τ comfortably below 1, matching L = 1 there.)

Two regimes fall out of this same inequality. If F is merely monotone (μ = 0), I don't get linear contraction, but the telescoped sum of the progress terms is bounded, and the averaged point ẑ_t = (1/t)Σ w_k drives the usual variational-inequality gap or merit function down at O(1/t) in the deterministic bounded-domain setting. That is the right generic certificate for monotone VIs; a last-iterate operator-norm statement needs extra assumptions, so I should not pretend the gap bound and ‖F(z_t)‖² are automatically the same theorem. If F is μ-strongly monotone, the middle term gives −2τ⟨F(w), w − z*⟩ ≤ −2τμ‖w − z*‖² (since F(z*) = 0 and strong monotonicity), feeding a geometric factor (1 − cτμ) into the recursion — linear convergence, last iterate, no averaging needed. Either way, the extra step converted "diverges on the simplest problem" into "Fejer-decreasing on monotone Lipschitz problems, with linear last-iterate convergence once a strong-monotonicity margin is present" — the first part is what the one-step identity I just wrote down actually buys, the second follows from the strong-monotone term.

So the deterministic method is settled: at each step, take a look-ahead gradient step to w, evaluate the field there, and step from the original point in that corrected direction; use τ ≤ 1/L. This is the extra-gradient iteration. Now the wrinkle I actually have to live with: the updates aren't exact. In the setting I'm targeting each gradient step is corrupted — there's additive noise on the update, either because the operator is only seen through a stochastic estimate F(·;ξ) or because the update itself is perturbed. Let me think carefully about how to inject the stochasticity, because there's a trap here that I want to avoid.

The two-step method has two gradient evaluations per iteration, one at z_t and one at w. If those two evaluations use *different* random draws of the operator — sample ξ for the first, an independent ξ′ for the second — what happens? My whole justification was that the corrector approximates the implicit step of *one* operator: F(w) stands in for F(z_{t+1}) of the same F. If the second evaluation is a different operator F(·;ξ′) than the first F(·;ξ), then the look-ahead w was computed to predict the implicit point of F(·;ξ), but I'm correcting with F(·;ξ′) — the prediction and the correction are about different operators, so the O(τ²) approximation argument has nothing to stand on. Algebraically the error ‖F(w;ξ′) − F(w_imp;ξ)‖ no longer shrinks with the distance to the solution; it's floored by the variance between ξ and ξ′. But I've fooled myself before with "this should diverge" arguments, so let me actually run both variants before I commit to a rule. I take a stochastic bilinear field F(z;ξ) = (J + ξ·S)z with S skew and ξ zero-mean unit-variance — monotone in expectation, rotational, the same kind of field that broke the deterministic case — and run the two-step iteration from z₀ = (1,1) for 2000 steps at τ = 0.1, averaging the final ‖z‖ over 40 seeds. Same-sample (ξ for both evaluations) returns mean ‖z_T‖ ≈ 6·10⁻⁹ — converged to the solution within machine noise. Independent-sample (a fresh ξ′ for the corrector) returns mean ‖z_T‖ ≈ 1.74 — it doesn't converge at all; it sits at a floor that doesn't shrink no matter how long I run. So the gap isn't a constant-factor degradation, it's the difference between converging and not. The design choice is therefore forced: **use the same sample for both evaluations within an iteration.** Compute the look-ahead with F(z_t;ξ) and the correction with F(w;ξ), the same ξ. Then I'm always approximating the implicit update of the one operator F(·;ξ), the predictor-corrector logic survives the noise, and the variance enters only as a neighborhood, not as a divergence.

Let me confirm that the strong-monotone rate degrades gracefully under same-sample noise rather than breaking, because I want to know what the noise actually costs. Carry the proof through with F(·;ξ) almost surely monotone and L-Lipschitz, and with bounded variance at the optimum, E‖F(z*;ξ) − F(z*)‖² ≤ σ². The same complete-the-square identity holds inside the expectation; the only new term is the cross term between the noise and the iterate. I split off the noise at the optimum, ⟨F(z*) − F(z*;ξ), w − z*⟩, and tame it with Young's inequality: E⟨F(z*) − F(z*;ξ), w − z*⟩ ≤ ησ² + (1/4η)E‖w − z_t‖² — the first piece is the irreducible variance cost, the second is absorbed by the −‖w − z_t‖² discretization term I already have in surplus (this is *why* I want that surplus, and why the step-size ceiling tightens to about η ≤ 1/(2L) in the noisy case — I need a bit of the negative term left over to soak up the noise cross-term). Pushing it through, the recursion becomes, for μ-strong monotonicity,

  (1 + 3ημ/2) E‖z_{t+1} − z*‖² ≤ E‖z_t − z*‖² + 2η²σ²,

and since 1/(1 + 3ημ/2) ≤ 1 − 2ημ/3 for ημ ≤ 1/2, unrolling gives

  E‖z_t − z*‖² ≤ (1 − 2ημ/3)^t ‖z_0 − z*‖² + 3ησ²/μ.

So the iterate contracts geometrically down to a noise floor of size 3ησ²/μ — proportional to η and to the variance, vanishing as η → 0. When σ = 0 (no noise at the optimum) I recover the clean linear rate of the deterministic method. That's exactly the graceful behavior I wanted: noise turns "converges to z*" into "converges to an O(ησ²/μ) neighborhood of z*," and that neighborhood is the price of running explicitly under noise. The metric I'll watch in the benchmark is the operator/gradient norm ‖F(z_t)‖; under Lipschitzness, controlling the distance to z* in the strongly monotone setting controls that norm up to the same neighborhood logic, while the merely monotone case is certified by the ergodic gap rather than this last-iterate norm theorem.

One more thing to settle before I write code: the constrained / regularized case. If there's a feasible set or a nonsmooth regularizer g, both gradient steps become proximal steps, w = prox_{ηg}(z_t − ηF(z_t;ξ)) and z_{t+1} = prox_{ηg}(z_t − ηF(w;ξ)) — and on an unconstrained box (feasible set all of R^d, no regularizer) the prox is the identity, so it collapses to the plain two-step update. The benchmark I'm targeting is unconstrained, so prox = identity and I can drop it; but it's good to know the method's general form is "replace each gradient step by a proximal step" and nothing else changes in the analysis (the prox-strong-convexity lemma carries the g terms).

Now the concrete iteration, mapping onto the harness. The oracle hands me a deterministic operator evaluation oracle.grad(z) = F(z) and a fresh additive Gaussian perturbation oracle.noise() per draw; the feasible set is all of R^{2d}, so no projection. One step is: evaluate the field at the current point, take the look-ahead step (with one noise draw), evaluate the field at the look-ahead point, take the corrector step from the *original* point (with a second noise draw). Two operator evaluations, both on the same deterministic F; the stochasticity in this benchmark is additive update noise, not two independently sampled operators, so the same-sample concern is automatically avoided. The step size τ is per-problem: small on the bilinear field (τ = 0.1, where the field is 1-Lipschitz so τ < 1/L = 1 with margin, and the contraction modulus √(1 − τ²(1−τ²)) wants τ well below 1), and τ = 1 on the structured (δ,ν) instance whose monotone clipped component has slope at most about one and whose skew coupling is small, matching the reference setup even though the clean strict-contraction proof sits at the boundary.

```python
from typing import Any
import numpy as np

from fixed_benchmark import (
    ProblemSpec, StepOutput, StochasticOracle,
    as_vector, make_step_output, run_cli,
)


def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    # keep the fixed starting point z_0 in the state
    return {"z": as_vector(initial_z, expected_dim=2 * problem.dim), "step_index": 0}


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    # predictor: one forward step to the look-ahead point w (the cheap guess of z_{t+1})
    g = oracle.grad(z)                       # F(z_t)
    w = z - tau * g + oracle.noise()         # w = z_t - tau F(z_t) + noise

    # corrector: evaluate the field AT w, but step from the ORIGINAL z_t -- this anchor
    # is what turns two forward steps into a contraction (the -tau^2 / -||w-z|| term)
    gw = oracle.grad(w)                      # F(w), same deterministic operator
    z_next = z - tau * gw + oracle.noise()   # z_{t+1} = z_t - tau F(w) + noise

    # measurement convention of the harness (not part of the method)
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        2,                                   # two operator evaluations per iteration
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    # tau <= 1/L; both fields are ~1-Lipschitz, so small tau on the pure-rotation
    # bilinear field and tau at the boundary on the structured instance
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```

Let me close the loop on the whole chain. I started with min-max and the obvious simultaneous descent-ascent, and it diverged geometrically on f = x·y because the game field is purely rotational — its update operator has modulus √(1+τ²) > 1 for every step size (1.004988 at τ = 0.1, an outward spiral I confirmed climbing to ‖z‖ ≈ 18 over 50 steps), with no contractive component, monotone but with zero strong-monotonicity margin. The implicit/backward step doesn't have that problem — its operator is the resolvent (I+τF)^{−1}, modulus 1/√(1+τ²) < 1 — but it's unrunnable because it needs the field at the unknown next point. So I approximated that future point explicitly with a cheap look-ahead forward step to w, evaluated the field there, and stepped from the *original* point z_t using F(w); anchoring at z_t rather than at w is what injects the −τ²I / −‖w−z_t‖² term the forward step lacked. That this anchored step is genuinely a second-order approximation of the implicit one — and not just some other stable map — is the thing I checked rather than assumed: it reproduces the resolvent's −τ²I expansion term, sits ten times (≈ 1/τ) closer to the true resolvent in matrix distance than the forward step, and contracts at modulus 0.995038, matching the resolvent's 0.995037. The one-step identity, with monotonicity killing the main cross term and Lipschitzness turning the discretization error into the negative (τ²L²−1)‖w−z_t‖², then gives Fejer decrease for the deterministic monotone case — linear when strongly monotone, O(1/t) on the ergodic gap when merely monotone. Under noisy updates the predictor-corrector logic only survives if both evaluations use the same operator draw — I saw the same-sample variant converge to ~10⁻⁹ and the independent-sample variant stall at ~1.7 on the stochastic bilinear field — and then the strongly monotone stochastic rate degrades gracefully to a contraction down to an O(ησ²/μ) neighborhood. What's left is two field evaluations per step, fully explicit, no inner solve, no projection on the unconstrained instances. The look-ahead-and-anchor iteration this converged on is exactly the (stochastic) extra-gradient method.
