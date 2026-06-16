Let me start from what I actually want and what actually hurts. I have a smooth convex-concave saddle problem, min over x and max over y of f(x,y), and I can only touch it through a noisy first-order oracle: I hand it a point z=(x,y), it gives me back F(z;xi) = F(z) + noise, an unbiased estimate of the saddle gradient operator F(z) = (grad_x f, -grad_y f), with the noise having bounded variance sigma^2. The thing I am graded on is the norm of the true operator, ||F(z)||: I want to return a point where the gradient is small. Not where the duality gap is small — the gap is the textbook measure but it is a nuisance here. For the bilinear f(x,y)=xy the gap is max over y' of x*y' minus min over x' of x'*y, which is plus infinity at every point except the saddle, so it is not even finite to chase. ||F(z)|| is always well defined and easy to read off. So my target is unambiguous: out of a stream of noisy saddle-gradient queries, manufacture an iterate whose true operator norm is small.

The first move anyone would make is to descend in x and ascend in y simultaneously: z_{t+1} = z_t - eta F(z_t). Let me just check it on the easiest convex-concave instance there is, f(x,y)=xy, where F(z) = (y, -x). Watch the distance to the saddle, which is the origin. ||z_{t+1}||^2 = ||z_t - eta F(z_t)||^2 = ||z_t||^2 - 2 eta z_t^T F(z_t) + eta^2 ||F(z_t)||^2. The cross term is z_t^T F(z_t) = x*y + y*(-x) = 0. So ||z_{t+1}||^2 = ||z_t||^2 (1 + eta^2). The iterates spiral strictly outward, for any step size, from any nonzero start. This is not a step-size bug I can tune away — the update direction is exactly orthogonal to the vector pointing at the saddle, so simultaneous descent-ascent has no component that ever pulls me in. The operator is rotational and the naive method rotates with it, drifting out. Wall, immediately, on the simplest possible problem.

So I need a step that sees the rotation coming. The trick that fixes this is to not trust F(z) at z. Take a tentative half-step to a lookahead point, evaluate the operator *there*, and then step from the original z with that re-evaluated gradient: z_{1/2} = z - eta F(z), then z+ = z - eta F(z_{1/2}). Why would the second evaluation help? Intuitively the half-step moves me a little way around the rotation, and the gradient at the moved-to point already has a component that the gradient at z did not — the part that, after one rotation tick, points back toward the center. Let me make that precise on the bilinear case to be sure it is real and not a story. F(z) = (y,-x). The half-step is z_{1/2} = (x - eta y, y + eta x). Now F(z_{1/2}) = ( y + eta x, -(x - eta y) ) = (y + eta x, -x + eta y). Step from z with this: z+ = (x,y) - eta (y + eta x, -x + eta y) = ( x - eta y - eta^2 x, y + eta x - eta^2 y ) = ( (1-eta^2) x - eta y, (1-eta^2) y + eta x ). Distance: ||z+||^2 = ((1-eta^2)x - eta y)^2 + ((1-eta^2)y + eta x)^2. Expand and the cross terms -2 eta(1-eta^2)xy and +2 eta(1-eta^2)xy cancel, leaving ((1-eta^2)^2 + eta^2)(x^2 + y^2) = (1 - 2 eta^2 + eta^4 + eta^2)(x^2+y^2) = (1 - eta^2 + eta^4)||z||^2. For small eta, 1 - eta^2 + eta^4 < 1, so now the iterates contract. The lookahead converted the outward spiral into an inward one. That is the extragradient idea, and on reflection it is exactly a cheap approximation of the implicit proximal-point step z+ = z - eta F(z+): proximal point contracts for any monotone operator because it is firmly nonexpansive, and the extragradient lookahead is its explicit first-order surrogate. So I will build on the two-evaluation extragradient step, not on the single-gradient one. Good.

Now bring in the two real complications: my oracle is noisy, and my problem is only convex-concave, i.e. F is monotone but not necessarily strongly monotone. Let me think about what the noisy extragradient — query the operator at z and at the lookahead, each with its own independent noise — can actually promise, and let me try to derive the per-step inequality rather than quote one, because the shape of that inequality is going to tell me exactly where this method breaks. Suppose for the moment the operator is lambda-strongly monotone (I will come back to the fact that mine is not), with z* the unique zero of the operator. Write the two steps with noisy queries: z_{1/2} = z_t - eta F(z_t; xi_i), z_{t+1} = z_t - eta F(z_{1/2}; xi_j), and I will track ||z_{t+1} - z*||^2.

The clean way in is to start from the quantity strong monotonicity controls and decompose it. Consider 2 F(z_{1/2})^T (z_{1/2} - z*). I want to write z_{1/2} - z* in terms of the actual iterates the algorithm produced, because those are what telescope. Split it as

  2 F(z_{1/2})^T (z_{1/2} - z*)
   = 2 ( F(z_{1/2}) - F(z_{1/2}; xi_j) )^T (z_{1/2} - z*)        [the noise on the second query, dotted with distance]
   + 2 F(z_{1/2}; xi_j)^T (z_{t+1} - z*)                          [the actual full-step gradient, dotted with the new distance]
   + 2 ( F(z_{1/2}; xi_j) - F(z_t; xi_i) )^T (z_{1/2} - z_{t+1})  [difference of the two queries, dotted with half-to-full move]
   + 2 F(z_t; xi_i)^T (z_{1/2} - z_{t+1}).                        [the half-step gradient, dotted with the same move]

This is just an algebraic identity once you check the cross terms: F(z_{1/2}) appears on the left; on the right F(z_{1/2};xi_j) cancels between line one and line two/three groupings, F(z_t;xi_i) cancels between line three and four, and the distances regroup z_{1/2}-z* = (z_{1/2}-z_{t+1}) + (z_{t+1}-z*). Let me handle the four pieces.

Piece one, the second-query noise against the distance: take expectations. The half-step z_{1/2} is already fixed before the second oracle query is drawn, and the oracle is unbiased, so conditionally on z_{1/2}, E[F(z_{1/2}) - F(z_{1/2};xi_j) | z_{1/2}] = 0. Therefore this whole cross term vanishes in the unbiased setting I care about. For a biased oracle, the same line would be bounded by (lambda/2)E||z_{1/2}-z*||^2 plus 2b_{t+1/2}^2/lambda, but with an unbiased SFO there is no sigma^2/lambda contribution here; the variance must enter through the difference between the two noisy queries.

Piece two, the full-step gradient against the new distance: here is where the geometry pays off. By construction z_{t+1} = z_t - eta F(z_{1/2};xi_j), so F(z_{1/2};xi_j) = (z_t - z_{t+1})/eta. Then 2 F(z_{1/2};xi_j)^T (z_{t+1} - z*) = (2/eta)(z_t - z_{t+1})^T (z_{t+1} - z*). Now use the polarization identity 2 (a-b)^T(b-c) = ||a-c||^2 - ||a-b||^2 - ||b-c||^2 with a=z_t, b=z_{t+1}, c=z*: this equals (1/eta)( ||z_t - z*||^2 - ||z_t - z_{t+1}||^2 - ||z_{t+1} - z*||^2 ). There it is — the telescoping pair ||z_t - z*||^2 - ||z_{t+1} - z*||^2, the thing that will collapse when I sum over t. The price is a -(1/eta)||z_t - z_{t+1}||^2 I will have to keep track of.

Piece four, the half-step gradient against the half-to-full move: same substitution, F(z_t;xi_i) = (z_t - z_{1/2})/eta, so 2 F(z_t;xi_i)^T (z_{1/2}-z_{t+1}) = (2/eta)(z_t - z_{1/2})^T (z_{1/2} - z_{t+1}) = (1/eta)( ||z_t - z_{t+1}||^2 - ||z_t - z_{1/2}||^2 - ||z_{1/2} - z_{t+1}||^2 ), again by polarization. Notice the +(1/eta)||z_t - z_{t+1}||^2 here cancels exactly the -(1/eta)||z_t - z_{t+1}||^2 left over from piece two. That cancellation is the whole reason the extragradient bookkeeping is clean: the two evaluation points conspire so the intermediate distance drops out.

Piece three, the difference of the two queries against the half-to-full move: this is where Lipschitzness and the step-size restriction earn their keep. Young again, 2 a^T b <= 2 eta ||a||^2 + (1/(2 eta)) ||b||^2 with a the query difference and b = z_{1/2} - z_{t+1}: at most 2 eta ||F(z_{1/2};xi_j) - F(z_t;xi_i)||^2 + (1/(2 eta)) ||z_{1/2} - z_{t+1}||^2. Split the operator difference into its noiseless part and the two noises: ||F(z_{1/2};xi_j) - F(z_t;xi_i)||^2 <= 3||F(z_{1/2}) - F(z_{1/2};xi_j)||^2 + 3||F(z_t) - F(z_t;xi_i)||^2 + 3||F(z_{1/2}) - F(z_t)||^2. Taking expectations gives 6 eta sigma^2 + 6 eta sigma^2 from the two noisy queries and 6 eta L^2 E||z_t - z_{1/2}||^2 from Lipschitzness of the exact operator. The negative terms already present are -(1/eta)||z_t - z_{1/2}||^2 and -(1/eta)||z_{1/2}-z_{t+1}||^2. Since eta <= 1/(4L), I have 6 eta L^2 <= 1/(2 eta), so the Lipschitz term consumes only half of the negative ||z_t-z_{1/2}||^2 coefficient. The Young term consumes half of the negative ||z_{1/2}-z_{t+1}||^2 coefficient. Both intermediate squares remain nonpositive and can be dropped, and the only positive contribution from this piece is 12 eta sigma^2.

Put the four pieces together and use the actual property I assumed, strong monotonicity, on the left: F(z_{1/2})^T (z_{1/2} - z*) >= lambda ||z_{1/2} - z*||^2 (since F(z*) = 0, this is (F(z_{1/2}) - F(z*))^T(z_{1/2}-z*) >= lambda||z_{1/2}-z*||^2). The identity started with 2 F(z_{1/2})^T(z_{1/2}-z*), so the left side gives 2 lambda E||z_{1/2}-z*||^2. I only need one copy of lambda, and after the cancellations and absorption above the right side is the telescoping pair plus at most 12 eta sigma^2, which I can loosen to the standard 16 eta sigma^2 statement:

  lambda E||z_{1/2} - z*||^2 <= (1/eta) E[ ||z_t - z*||^2 - ||z_{t+1} - z*||^2 ] + 16 eta sigma^2,

for any 0 < eta < 1/(4L). The sign is important: the distance at time t is first and the distance at t+1 is second, so the term telescopes in the decreasing direction. Sum over t = 0..T-1, telescope the distance pair, divide by T, and apply strong monotonicity once more to convert the averaged half-iterate distance into something I can report:

  E||zbar - z*||^2 <= ||z_0 - z*||^2 / (lambda eta T) + 16 eta sigma^2 / lambda.

Stare at this, because it tells me both the good news and exactly where I am stuck. Good news: with strong monotonicity the optimization error ||z_0 - z*||^2/(lambda eta T) decays like 1/T, the iterates actually contract toward z*, and with a fixed step they reach a neighborhood of radius governed by 16 eta sigma^2/lambda. Two pieces of bad news that I have to confront. First, the whole thing is multiplied by lambda on the left and divided by lambda on the right — every useful term has a lambda in it, and my problem is only convex-concave, so lambda = 0 and this inequality says nothing at all. The contraction I just derived literally does not exist for a merely monotone operator. Second, even granting strong monotonicity, the fixed-step noise floor 16 eta sigma^2/lambda does not go away as T grows; I would have to shrink eta (or batch) to push it down, trading off against the 1/(eta T) optimization term.

So I have two diseases. Let me take the more fundamental one first: I have a method that works beautifully when the operator is strongly monotone, and an operator that is not. The honest options are to find a method that doesn't need strong monotonicity, or to manufacture the strong monotonicity I am missing. The first route is the hard research road; the second is suspiciously cheap, and cheap is worth checking. What does it cost to add curvature? If I take the operator F and add lambda times (z - a) for some anchor point a and some lambda > 0,

  G(z) = F(z) + lambda (z - a),

then for any z, z', (G(z) - G(z'))^T(z - z') = (F(z)-F(z'))^T(z-z') + lambda||z-z'||^2 >= 0 + lambda||z-z'||^2. G is lambda-strongly monotone *by construction*, for free, regardless of F being merely monotone. And G is still L+lambda-Lipschitz, basically as smooth as F. So I can run the extragradient method I just analyzed on G and get the contraction. G is the gradient operator of the regularized saddle objective f(x,y) + (lambda/2)||x - a_x||^2 - (lambda/2)||y - a_y||^2 — a strongly-convex penalty on x, a strongly-concave one on y — which is exactly the Tikhonov / regularization device that is known to help convert hard small-gradient questions into easy ones in convex minimization. The catch I have to face squarely: running the method on G drives me toward G's zero, call it w*, which is *not* z*, the zero of F. I have solved a different problem. So the question that decides whether this trick is legitimate is: how far is w* from being a near-stationary point of the *original* F, and can I control that gap?

Let me bound ||F(z~)|| for an arbitrary point z~ in terms of ||G(z~)||, because that is the quantity my SEG run can make small. Just from the definition, F(z~) = G(z~) - lambda(z~ - a), so by the triangle inequality

  ||F(z~)|| <= ||G(z~)|| + lambda ||z~ - a||.

The first term I can shrink by running the method; the second term, lambda||z~ - a||, is the price of the regularization, and I need it to be small too. ||z~ - a|| could be large in general, so I should pick the anchor a and split that distance through w*, G's solution: ||z~ - a|| <= ||z~ - w*|| + ||w* - a||. The first of these, ||z~ - w*||, is controlled by strong monotonicity again — since G is lambda-strongly monotone and G(w*)=0, lambda||z~ - w*|| <= ||G(z~) - G(w*)|| = ||G(z~)||, so lambda||z~ - w*|| <= ||G(z~)||. That converts another copy of ||G(z~)|| out of the anchor term. The remaining piece is lambda||w* - a||, the distance from G's solution to the anchor, which depends on where I put a and I have not yet pinned down.

Where should the anchor go? The only special points I have are the start z_0 and the (unknown) solution z*. I cannot anchor at z* — I do not know it. Anchor at the initial point, a = z_0. Now I need to control ||w* - z_0||, the distance from the regularized solution to the start. This is where I should check whether the regularized solution can wander off somewhere far. Claim: w* sits no farther from z_0 than z* does, and in fact w* lands between them in a precise sense. Let me prove the non-expansiveness. G is lambda-strongly monotone with G(w*) = 0, so lambda||w* - z*||^2 <= (G(w*) - G(z*))^T(w* - z*) ... wait, I should orient this to use what I know, namely F(z*) = 0. Start from lambda||w* - z*||^2 <= G(z*)^T(z* - w*) (this is strong monotonicity of G at z* and w*, using G(w*)=0: (G(z*)-G(w*))^T(z*-w*) >= lambda||z*-w*||^2, and G(w*)=0). Now G(z*) = F(z*) + lambda(z* - z_0) = lambda(z* - z_0) because F(z*) = 0. So

  lambda||w* - z*||^2 <= lambda (z* - z_0)^T (z* - w*).

Divide by lambda and expand the right side with the polarization identity (z* - z_0)^T(z* - w*) = (1/2)( ||w* - z*||^2 + ||z* - z_0||^2 - ||w* - z_0||^2 ):

  ||w* - z*||^2 <= (1/2)( ||w* - z*||^2 + ||z* - z_0||^2 - ||w* - z_0||^2 ),

so (1/2)||w* - z*||^2 + (1/2)||w* - z_0||^2 <= (1/2)||z* - z_0||^2, which gives me *both* facts at once: ||w* - z*|| <= ||z* - z_0|| and ||w* - z_0|| <= ||z* - z_0||. The regularized solution is no farther from the anchor than the true solution is — anchoring at z_0 cannot send w* off to infinity. Good, the trick is geometrically safe.

Now assemble the anchoring bound. With a = z_0,

  ||F(z~)|| <= ||G(z~)|| + lambda||z~ - z_0||
           <= ||G(z~)|| + lambda||z~ - w*|| + lambda||w* - z_0||
           <= ||G(z~)|| + ||G(z~)|| + lambda||z* - z_0||
           = 2||G(z~)|| + lambda||z_0 - z*||,

using lambda||z~-w*|| <= ||G(z~)|| and the non-expansiveness lambda||w*-z_0|| <= lambda||z*-z_0||. So

  ||F(z~)|| <= 2||G(z~)|| + lambda ||z_0 - z*||.

This is the lever. It says: if I make G's gradient norm small at z~, then F's gradient norm at z~ is small up to an irreducible additive term lambda||z_0 - z*||. The first term I can crush by running stochastic extragradient on the strongly-monotone G; the second term is the cost of having regularized, and it is governed by lambda and by how far my start was from the solution.

So the design reduces to choosing lambda, and there is a genuine tension I can see directly in the two terms. If lambda is large, G is very strongly monotone, the SEG contraction is fast and the noise floor 16 eta sigma^2/lambda is small — but the irreducible term lambda||z_0 - z*|| is large, so even a perfect solve of G leaves F's gradient norm stuck at lambda||z_0 - z*||. If lambda is tiny, the irreducible term vanishes — but G is barely strongly monotone, its condition number L/lambda blows up, SEG crawls, and the noise floor 16 eta sigma^2/lambda explodes. So lambda trades the regularization bias against the conditioning and the noise floor, and I should balance them. Let D be an upper bound on ||z_0 - z*||. To make the irreducible term at most of order eps I want lambda ||z_0 - z*|| <= lambda D ~ eps, i.e. lambda ~ eps/D. With that choice the bias term is O(eps), and I separately drive 2||G(z~)|| down to O(eps) by running SEG on G long enough and with small enough (or epoched) step to beat the noise floor; the conditioning becomes L/lambda ~ L D/eps, finite and the price I pay for the convex-concave (non-strongly-monotone) regime. So lambda = min(eps/D, L) — clip at L because there is no point regularizing harder than the smoothness — is the right setting, and the whole thing reduces finding a near-stationary point of the hard monotone F to finding a near-stationary point of the easy strongly-monotone G, which I already know how to do.

Let me now make G concrete and write the iteration. G(z) = F(z) + lambda(z - z_0). One stochastic extragradient step on G with step size eta = tau:

  z_{1/2} = z - tau G(z; xi) = z - tau [ F(z; xi) + lambda(z - z_0) ] = z - tau F(z) + tau lambda (z_0 - z) + (noise),
  z+      = z - tau G(z_{1/2}; xi') = z - tau F(z_{1/2}) + tau lambda (z_0 - z_{1/2}) + (noise).

So each iteration evaluates the true operator F at z and at the lookahead z_{1/2}, adds a fixed pull toward the initial point z_0 with weight tau*lambda at both sub-steps, and injects the oracle noise at both sub-steps (two independent SFO queries per iteration). The anchor stays fixed at z_0 forever, because the transfer bound I just proved uses the single distance ||z_0 - z*|| and needs no additional anchor state.

Two small implementation points remain. The analysis wants tau < 1/(4 L_G), where L_G = L + lambda is the smoothness of the regularized operator, and lambda is chosen on the scale eps/D. In the fixed benchmark harness I use exactly the provided constants: for the bilinear instance, tau = 0.1 and lambda = 0.1; for the (delta,nu) instance, tau = 1.0 and lambda = 0.01. The metric also follows the harness. For the bilinear case F(z) = (y, -x), so ||F(z)|| = ||z|| exactly, and the reported point is the post-step iterate. For the (delta,nu) case the operator norm is computed directly at the pre-step iterate. Each iteration spends exactly two SFO calls, one at z and one at the lookahead.

Everything now lines up. I want a small *gradient norm* on a noisy convex-concave saddle problem. Simultaneous descent-ascent diverges because the operator is rotational and the step is orthogonal to the pull toward the saddle — on f=xy, ||z|| grows by a factor (1+eta^2) each step. The extragradient lookahead fixes that by evaluating the operator at a tentative point so the step inherits the toward-the-saddle component — on f=xy, ||z|| shrinks by (1 - eta^2 + eta^4). Making that extragradient step stochastic and analyzing it gives a per-step descent inequality whose every useful term carries the strong-monotonicity constant lambda, so on a merely convex-concave problem (lambda = 0) it promises nothing, and even when lambda > 0 a fixed step lands in a noise ball of size ~ eta sigma^2/lambda. The cure for the missing strong monotonicity is to add it: regularize F into G(z) = F(z) + lambda(z - z_0), strongly monotone by construction. Running stochastic extragradient on G is well-behaved, but it solves for G's zero w*, not F's zero z*; the anchoring inequality ||F(z~)|| <= 2||G(z~)|| + lambda||z_0 - z*|| — proved via the triangle inequality, strong monotonicity, and the non-expansiveness ||w* - z_0|| <= ||z* - z_0|| — shows the gap is exactly an irreducible lambda||z_0 - z*||. Balancing that bias against the conditioning and the noise floor fixes lambda ~ eps/D. The result is stochastic extragradient run on the initial-point-anchored, Tikhonov-regularized operator: a fixed pull toward z_0 added to each of the two extragradient sub-steps. The code is one regularized extragradient iteration, two operator queries, two noise draws, and the fixed z_0 anchor:

```python
from __future__ import annotations
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
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {
        "z": z0,
        "anchor_z": z0.copy(),     # a = z_0: the fixed Tikhonov anchor, never updated
        "step_index": 0,
    }


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])       # extragradient step size eta (< 1/(4 (L+lambda)))
    lam = float(hyperparameters["lambda"])    # regularization weight lambda ~ eps / D
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    # half-step (lookahead) of extragradient on G(z) = F(z) + lambda (z - anchor):
    #   w = z - tau F(z) + tau lambda (anchor - z) + noise
    g = oracle.grad(z)
    w = z - tau * g + tau * lam * (anchor_z - z) + oracle.noise()

    # full step of extragradient on G, using the operator at the lookahead point:
    #   z+ = z - tau F(w) + tau lambda (anchor - w) + noise
    gw = oracle.grad(w)
    z_next = z - tau * gw + tau * lam * (anchor_z - w) + oracle.noise()

    # report ||F(z)||: post-step iterate for the bilinear case (||F||=||z||), pre-step otherwise
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
        metric_iterate,
        2,  # two stochastic operator queries this iteration
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    # lambda ~ eps/D balances the irreducible bias lambda||z_0 - z*|| against
    # conditioning L/lambda and the noise floor; tau < 1/(4(L+lambda)).
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01}
    raise KeyError(f"Unknown problem: {problem_name}")


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```
