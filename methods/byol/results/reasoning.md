OK, let me think this through from scratch. I want representations of images learned without labels, and the way the field is winning right now is contrastive learning. Take an image, make two augmented views, push their representations together; take views of *different* images, push those apart. The pushing-apart part — the negatives — is doing all the heavy lifting, and I want to understand exactly why before I decide whether I can get rid of it.

So why are negatives there at all? Strip the method down to its honest core: I have two views of the same image, and I want one view's representation to be predictive of the other's. That's the whole idea, "cross-view prediction." But if I write that down literally as a loss — make the representation of view A close to the representation of view B in feature space, both networks free to move — then there's a solution that's perfect and useless: output the same constant vector for every image. A constant is trivially predictive of itself, the loss is zero, and the features carry nothing. Collapse. So a pure "predict the other view's features" objective has this degenerate global optimum sitting right in the middle of it.

The contrastive trick is to refuse to let prediction be prediction. Instead of "make A close to B," it says "from A, identify which of these many vectors is B" — discrimination against a crowd of negatives. Now a constant can't win, because a constant can't tell B apart from the negatives. The negatives are a repulsive force that holds the representations spread out and rules out the collapsed solution. That's the actual job of the negatives: they're the anti-collapse mechanism, nothing more mysterious than that. Concretely it's the InfoNCE / NT-Xent loss — l2-normalize the projections, take cosine similarities, softmax with a temperature, positive in the numerator, all the other in-batch examples in the denominator.

And here's where it gets expensive. For the discrimination to be hard enough to teach the network anything, I need *many* negatives, ideally hard ones. So I either run enormous batches — thousands of images, tens of thousands of negatives per positive — and watch performance fall off a cliff when the batch shrinks; or I keep a memory bank / queue of old features and fight their staleness as the encoder drifts; or I do hand-tuned hard-negative mining. All of this machinery exists to feed the anti-collapse term. There's also a brittleness I keep seeing: contrastive methods are weirdly sensitive to the augmentation set. If I only crop, the task becomes nearly solvable by matching color histograms — two crops of one image share a color histogram, different images differ in color — so the representation never has to encode anything but color statistics, and people patch this by adding strong color distortion. The whole thing leans hard on a carefully chosen augmentation recipe.

So the question I actually want to answer: are the negatives *indispensable*? They're the collapse-preventer, fine — but is there another way to prevent collapse that doesn't drag in large batches, memory banks, mining, and augmentation fragility? Can I keep the honest prediction objective and find a different reason for it not to collapse?

Let me think about what makes a prediction objective collapse in the first place. It collapses because *both sides can move*. The thing I'm predicting (the target) and the thing doing the predicting (the online network) are the same network, so they can race to the bottom together — meet at a constant. What if the target couldn't move? Freeze a network, call its output the target, and train a second network to predict that frozen target. Now collapse is impossible by construction: the target is fixed, it's not a constant, the second network has to actually match a non-trivial, varied target to lower the loss. No negatives needed.

The obvious objection is that a frozen target sounds useless — surely you just learn to reproduce whatever junk the frozen net outputs. So let me push it to the extreme and freeze a *randomly initialized* network as the target. Predict the random network's features. What happens? The surprising thing — and this is the seed of everything — is that the network trained to predict the random target ends up with a representation that's *much better than the random target itself*. By the linear-evaluation yardstick the trained predictor lands far above where the frozen random network sits. Stare at that for a second. I took a bad representation, asked a second network to predict it, with no negatives at all, and out came a *better* representation than the one I was predicting.

Why would that even happen? Predicting a fixed mapping of an image forces the online network to compute features from which that mapping is recoverable; to do that across all the augmentation noise, it has to find the stable, shared content of the image, and that shared content is more semantic than the random target's raw output. So the act of prediction, plus the invariance demanded by augmentations, distills something better than the target. The target was just a scaffold.

That immediately suggests an iteration. If predicting target #0 gives me a better network #1, then make network #1 the new target and predict *it* — that should give an even better network #2. A sequence of representations, each trained to predict the previous, each better than the last, and none of them needing a single negative. The fixed random target was never the point; it was round zero of a ladder. Whether the ladder actually keeps climbing rather than stalling or sinking is exactly what I'll have to check below — for now it's a hypothesis, not a result.

Now, doing this with discrete checkpoints — train to convergence against a frozen target, snapshot, repeat — is clunky. I'd rather make it continuous: let the target be a slowly trailing version of the online network, updated a little every step. The online network θ races ahead, the target ξ follows behind. How should ξ follow? I've seen exactly this problem in deep RL: bootstrapped targets (the Bellman target depends on the network's own estimate) are unstable if the target chases the online network too fast, and the fix is a slow copy — either a periodically frozen copy or a soft exponential moving average of the weights. The EMA version is θ_target ← τ θ_target + (1−τ) θ, with τ near 1. That's exactly the "delayed, stable version of myself" I want. So:

  ξ ← τ ξ + (1−τ) θ.

When τ = 1 the target never moves — that's the frozen-target case, round zero forever, no iteration. When τ = 0 the target instantly equals the online network every step — that's predicting myself with both sides free, back to the collapse danger. Somewhere in between, the target is a stable trailing average that still slowly incorporates the online network's improvements. That's the bootstrap made continuous.

Let me write the prediction loss carefully. Two views v = t(x), v' = t'(x). Online network on v produces a projection z_θ; target network on v' produces a projection z'_ξ. Following the contrastive practice of not predicting the raw representation but a lower-dimensional projection of it — apply the loss on z = g(y) where g is a small MLP head, but keep the pre-projection y as the actual representation, because empirically the loss-on-projection / evaluate-on-representation split works better — I want z_θ to predict z'_ξ. l2-normalize both and take squared error:

  L = ‖ z̄_θ − z̄'_ξ ‖²  where x̄ = x/‖x‖.

Expanding, ‖z̄_θ − z̄'_ξ‖² = ‖z̄_θ‖² + ‖z̄'_ξ‖² − 2⟨z̄_θ, z̄'_ξ⟩ = 1 + 1 − 2⟨z̄_θ, z̄'_ξ⟩ = 2 − 2·⟨z_θ, z'_ξ⟩ / (‖z_θ‖‖z'_ξ‖). So it's just (twice) the negative cosine similarity, shifted. And critically, I take the gradient of this *only with respect to θ*, never ξ — stop-gradient on the target. ξ moves only through the EMA. That's deliberate: I do not want to give the target a reason to come meet the online network at a constant. The only force on ξ is "trail θ."

Now I have to be honest and check: have I actually escaped collapse, or have I just hidden it? Let me try the most naive version of this and see if it holds up. Predict myself: target = online network, gradient flowing through both, no predictor, no negatives. Does it collapse? It collapses — instantly. The two branches agree on a constant and the loss is zero. The stop-gradient and the EMA are clearly load-bearing, but are they enough?

Let me reason about the EMA version with stop-gradient but nothing else. There's still a worry. The loss L(θ, ξ) is minimized over θ each step, and ξ is dragged toward θ. Over time ξ → θ, and then I'm predicting a near-copy of myself again. Naively it *seems* like the whole system should drift toward a joint minimum of L over both θ and ξ — and a constant representation is such a joint minimum. So why doesn't it sink into the constant?

Here's the thing I have to get precise about: the dynamics are *not* gradient descent on any single loss jointly over θ and ξ. The ξ update is not −∇_ξ L; it's "move toward θ." So there's no joint objective being descended. This is the same structure as a GAN — there's no single loss minimized in both the generator's and the discriminator's parameters; it's a dynamical system, not an optimization. So I can't argue "it converges to a minimum of L over (θ, ξ)," because nothing is minimizing L over ξ. Good — but "it's not obviously a minimization of the bad thing" is not the same as "it provably avoids the bad thing." The collapsed configuration is still an equilibrium of the dynamics. I need a reason it's an *unstable* equilibrium — something that actively pushes the online network away from constant features.

Let me look harder at what the online update is actually doing, and to make the analysis tractable, suppose the predictor I add to the online branch — wait, I haven't motivated the predictor yet. Let me get there honestly, because it turns out to be the crux.

Try the EMA-target-with-stop-gradient method without any extra piece. What is it? Online encoder+projector trained to match a slow EMA of itself, l2 consistency, no labels, no negatives. That's *exactly* an unsupervised mean teacher. Mean teacher works in the semi-supervised setting — but there it has a supervised classification loss on real labels doing the grounding, and the EMA-consistency term is just a regularizer riding on top. Take the labels away — remove the classification loss — and what's left is precisely my "predict a slow copy of myself" objective. Does that survive on its own? It collapses. The classification loss was the thing holding it up; without it, student and teacher happily agree on a constant. So the slow EMA target, by itself, is *not* a sufficient anti-collapse mechanism. I need something more, and it has to be the thing that's different from mean teacher.

The asymmetry. In mean teacher the student and teacher have the same architecture and the loss just asks them to agree. What if I break that symmetry by putting an extra little network — a *predictor* q — only on the online branch, and ask q(z_θ) to predict z'_ξ, with no corresponding head on the target side? Then the online branch is not being asked to *equal* the target; it's being asked to *predict* it through an extra learned map. Let me see if that asymmetry is the missing anti-collapse force, and why.

Assume the predictor is optimal for the current online and target networks — q = q*, the best possible predictor of z'_ξ from z_θ in squared error. The minimizer of E‖q(z_θ) − z'_ξ‖² over functions q is the conditional expectation:

  q*(z_θ) = E[ z'_ξ | z_θ ].

(That's the standard fact: the function of z_θ that minimizes expected squared error to z'_ξ is the conditional mean.) Plug it back in. The loss the online network is then minimizing, in expectation, is

  E‖q*(z_θ) − z'_ξ‖² = E‖ E[z'_ξ | z_θ] − z'_ξ ‖² = E[ Σ_i Var(z'_{ξ,i} | z_θ) ],

because E‖E[Y|X] − Y‖² summed over coordinates is exactly the sum of conditional variances of Y given X. So with an optimal predictor, the online network is being pushed to minimize the **expected conditional variance of the target projection given the online projection.** That reframing is the whole game.

Why would minimizing conditional variance prevent collapse? The monotonicity I'd need is the expected one: conditioning on more information cannot *increase* the expected conditional variance. For scalar X, the law of total variance gives Var(X | Y) = E[Var(X | Y, Z) | Y] + Var(E[X | Y, Z] | Y); the second term is ≥ 0, so after averaging, E Var(X | Y, Z) ≤ E Var(X | Y). For a vector I apply this coordinate by coordinate and sum. Read it in my setting: let X be the target projection z'_ξ, let Y be the current online projection z_θ, and let Z be any extra variability the online network *could* carry about the image. Then E[Σ_i Var(z'_{ξ,i} | z_θ, Z)] ≤ E[Σ_i Var(z'_{ξ,i} | z_θ)] — the online network lowers the loss by carrying *more* information about the image, never by carrying less.

I want to actually see this rather than trust the algebra, because it's the whole load-bearing claim, so let me build the smallest world that has the structure and put numbers on it. Four equally-likely image types, target projection a scalar drawn around a per-type mean μ = (−2, −0.5, 0.5, 2) with noise σ = 0.5 (so the irreducible Var(z'|type) = σ² = 0.25). The optimal predictor is the conditional mean E[z'|z_θ], and the profiled loss is the resulting expected conditional variance, which I can just measure for three choices of how informative the online projection is:
- online = full image type (maximally informative): loss measures out to 0.2499 ≈ σ² = 0.25, the irreducible floor;
- online = one coarse bit, type≥2 vs type<2 (one bit of image info thrown away): loss = 0.8124;
- online = constant (fully collapsed, carries nothing): loss = 2.3761, which is exactly the marginal variance of z' I get from Var(z') = 2.3761.

The ordering 0.25 < 0.81 < 2.38 comes out clean: every bit of image information the online projection discards *raises* the loss, and the constant — the collapse — sits at the very top, pinned to the full marginal variance. So discarding information can never help here, and that is the opposite of collapse: collapse means throwing information away, and throwing it away strictly *increases* this objective in the example.

Make it sharp at the collapsed point in general. If the online projection is a constant c, conditioning on it tells you nothing, so E[Σ_i Var(z'_{ξ,i} | z_θ = c)] is just the summed marginal variance of z'_ξ — exactly the 2.38 I measured. For any informative z_θ the same quantity is no larger (the monotonicity, which I also re-checked numerically: E[Var | the fine 4-way label] = 0.2499 ≤ E[Var | the coarse 2-way label] = 0.811). So the constant online projection is the *worst* point of the profiled objective, and any perturbation that makes z_θ carry real information about z'_ξ lowers it. The collapsed equilibrium is therefore unstable — and that, not an appeal to some loss being minimized, is what lets this avoid collapse without negatives.

And now I can see exactly why I must *not* let ξ chase this objective. If I were to minimize that same E[Σ_i Var(z'_{ξ,i} | z_θ)] with respect to ξ, the target side, the minimizer is a *constant* z'_ξ — variance is smallest when the target itself is constant. So the target, left to minimize the loss, would collapse the whole thing. The asymmetry in the dynamics is essential: θ minimizes the loss (good, it's pushed to be informative), but ξ does *not* minimize the loss — ξ only trails θ via the EMA. That's why "there's no joint loss being descended" isn't a hand-wave; it's the precise reason collapse is avoided. The online direction increases information; the target direction is forbidden from taking the collapse step and instead just inherits the online network's growing information.

Let me make the "online update follows the conditional-variance gradient" claim rigorous, because I waved at it. Give the predictor its own parameters a, and write J(θ, a; ξ) = E‖q_a(z_θ) − z'_ξ‖². For fixed θ and ξ, let a*(θ, ξ) minimize J. The profiled objective is F(θ; ξ) = J(θ, a*(θ, ξ); ξ). When I differentiate it, the chain rule gives

  dF/dθ = ∂J/∂θ |_{a=a*} + ∂J/∂a |_{a=a*} · da*/dθ.

The second term is the dependence of the optimal predictor's parameters on θ. At the optimum, ∂J/∂a = 0 by the first-order condition, so the envelope theorem deletes exactly that term. What remains treats the predictor parameters as fixed at their optimum and differentiates only through the input z_θ:

  ∂J/∂θ |_{a=a*} = E[ ∂L/∂q · ∂q_{a*}(z_θ)/∂z_θ · ∂z_θ/∂θ ].

That surviving term is exactly the direction my actual algorithm approximates when q is kept near its optimum: the θ-gradient flows through z_θ and through the predictor's input-output map, while the separate movement of the optimal predictor's parameters with θ is not an extra term in the profiled gradient. So in the optimal-predictor idealization, the online step is descending the conditional variance — which, combined with the numbers I just put on that conditional variance, is what makes the collapse direction uphill. The load-bearing word is *idealization*: this is clean only while q sits at its optimum, and the next thing I have to worry about is whether that assumption survives contact with a predictor I'm training online rather than solving exactly.

Of course the predictor isn't *exactly* optimal at every step — it's a learned MLP being trained alongside everything else. So the clean "loss = conditional variance" identity only holds when q is near-optimal. This tells me something about why the *slow* target matters, beyond the bootstrap story. Why not just hard-copy θ into ξ every step (τ = 0) instead of an EMA? A hard copy would also propagate new variability into the target — the bootstrap would still work in principle. But a sudden jump in the target changes the prediction problem abruptly, and then my predictor, which was near-optimal for the *old* target, is suddenly stale and far from optimal for the *new* one. When q is far from optimal, the loss is no longer the conditional variance, and my anti-collapse guarantee evaporates. So the real role of the slow-moving target is to keep the prediction problem changing slowly enough that the predictor can stay near-optimal throughout training. The EMA isn't just "smoother bootstrap," it's "keep q optimal."

That's a checkable prediction, and a sharp one, because it decouples two things the EMA had bundled together. If predictor-near-optimality really is the mechanism — and not the slow target per se — then I should be able to throw the EMA away entirely (hard-copy θ into ξ, τ = 0) and *still* avoid collapse, provided I keep the predictor near-optimal by some other means. Two ways to keep it near-optimal that I'd want to run: give the predictor a much larger learning rate than the rest of the network so it tracks the moving optimum; or skip gradient descent on it altogether and solve the optimal *linear* predictor in closed form each batch, Q* = (ZθᵀZθ)⁻¹ ZθᵀZ'ξ for predictions ZθQ (with a pseudoinverse or small ridge if ZθᵀZθ is singular — and it can be singular, so I'd need that guard). I expect both to hold up under τ = 0; if they did, the EMA would be revealed as one convenient way to keep q optimal rather than a necessary ingredient. The control that the same story predicts should *fail* is cranking the projector's learning rate up alongside the predictor's — then the target side (fed by the projector) is also racing, the optimum q chases is itself moving fast, and the predictor can't stay near it. If "predictor faster than the rest" survives while "everything fast together" collapses, that asymmetry is the signature I'm after, and it would pin the target network's role to keeping the predictor optimal rather than to anything about negatives.

Let me line up what the mechanism predicts for the two single-piece ablations, since they're the cleanest test of the story. Remove the predictor but keep the target network, no negatives (β = 0): now nothing creates the conditional-variance dynamics — the online branch is just asked to *equal* a slow copy of itself, which is the unsupervised-mean-teacher case I already argued collapses. So I predict collapse. Remove the target network but keep the predictor, no negatives: now the predictor exists, but the thing it predicts is moving as fast as the online network, so it can never sit near its optimum, and the variance identity never kicks in — I predict collapse here too. Only *both together* should stay up. If the experiments come out that way, it's consistent with exactly the division of labor I derived: the predictor supplies the conditional-variance dynamics that make collapse unstable, the target network keeps the predictor near-optimal so that dynamics actually holds, and neither alone is enough.

Now let me make sure I haven't quietly smuggled negatives back in or made this just a re-skinned contrastive method. Let me cast both in one frame as a score to maximize; minimizing the usual loss is the same as maximizing this score up to constants and a sign. For one ordered anchor direction, write

  InfoNCE^{α,β} = (2/B) Σ_i S(v_i, v'_i)  −  β · (2α/B) Σ_i ln( Σ_{j≠i} exp(S(v_i,v_j)/α) + Σ_j exp(S(v_i,v'_j)/α) ),

with similarity S(u₁,u₂) = ⟨φ(u₁), ψ(u₂)⟩ / (‖φ(u₁)‖‖ψ(u₂)‖). This really is the log-softmax contrastive score with a tunable negative term. Start from the positive logit f_i = S(v_i,v'_i)/α and the denominator containing the positive plus the in-batch negatives from both views:

  (1/B) Σ_i ln[ exp f(v_i,v'_i) / ((1/B)( Σ_{j≠i} exp f(v_i,v_j) + Σ_j exp f(v_i,v'_j) )) ]
   = ln B + (1/B) Σ_i f(v_i,v'_i) − (1/B) Σ_i ln( Σ_{j≠i} exp f(v_i,v_j) + Σ_j exp f(v_i,v'_j) ).

Drop ln B (constant in the parameters), multiply through by 2α, set f = S/α, and weight the log-denominator term by β; that's the general score above. Setting β = 1 and dividing by 2α gives the standard InfoNCE score for this anchor direction, so minimizing its negative gives the ordinary contrastive loss. Now read off the two endpoints. The contrastive baseline is φ = ψ = z_θ (no predictor, no target network) with β = 1 — the negative term fully present. My method is φ = q_θ(z_θ) (predictor on the online side) and ψ = z_ξ (target network on the other side) with **β = 0** — the entire negative-examples term removed. At β = 0 the score is just (2/B)Σ_i S, and minimizing the normalized squared error 2 − 2S is exactly the same update direction. I add the swapped view direction in the actual loss, just as the symmetric contrastive baseline adds the opposite anchor direction. So in this shared language my method is literally "drop the negative term, and replace plain similarity with predictor-vs-target similarity." The β = 0 column is the one I need to survive, and the only configuration that survives at β = 0 is the one with *both* the predictor and the target network. Adding the negatives back (β = 1) to my method, without re-tuning, actually *hurts* — they're not just unnecessary, they're in the way at the operating point I've tuned for.

A nice side-observation falls out of this frame: just bolting a target network onto the plain contrastive loss (still β = 1, still negatives) already helps a bit. That recasts what the momentum encoder was doing in the queue-based contrastive method — it was credited with supplying consistent negatives, but here, with the same number of negatives, the mere *stabilization* from the slow target improves things. The target network's value is stabilization, somewhat independent of negatives.

Let me also pin down why my method should be *more robust to augmentations* than the contrastive baseline, since that was one of the original pain points. The contrastive method, when augmentations are weak (say crop-only), can solve its discrimination task through a shortcut — color histograms — and is then never forced to encode more. My objective is different: the online network is rewarded for retaining *any* information the target captured, because more retained information lowers the conditional variance of the target given the online projection. There's no shortcut that lets it off the hook — even if two views share a color histogram, predicting the full target projection still demands more than color. So I expect a much smaller drop when color distortion is removed, and a usable representation even with crop-only augmentations. That's the mechanism, and it's the same conditional-variance logic.

Now the concrete design, with each piece earned. Two views from each image, and I'll use the same strong augmentation family the contrastive method uses — random resized crop, flip, color jitter, optional grayscale, Gaussian blur, solarization — but I'll make the two view-distributions slightly asymmetric (e.g., blur almost always on one view, rarely on the other; solarization only on one), which pairs naturally with the asymmetric online/target roles. Encoder f is a ResNet-50; the representation y is the post-global-average-pool feature (2048-d at width 1×), and that y is all I keep at the end — everything else is scaffolding I throw away.

The projector g: I want the loss to live in a smaller space than y, as in the contrastive baseline, so g is an MLP — Linear to 4096, BatchNorm, ReLU, Linear to 256. The hidden BatchNorm helps optimization; I deliberately do *not* batch-norm the 256-d output (unlike the contrastive baseline's head), and l2-normalization will handle the output scale instead. The predictor q has the same MLP shape as g, and it sits *only* on the online branch — that asymmetry is the anti-collapse ingredient, so it must not appear on the target side. Why this depth and width? A depth-2 MLP (one hidden layer) is the sweet spot; a depth-1 linear head is clearly worse (the predictor needs enough capacity to approximate the conditional expectation), and depth-3 buys nothing. The projection dimension is forgiving — anything from ~64 to 512 plateaus — so 256 is a fine default.

The loss, normalized and symmetrized. For the pair (v → online, v' → target):

  L = ‖ q̄_θ(z_θ) − z̄'_ξ ‖²  = 2 − 2 · ⟨q_θ(z_θ), z'_ξ⟩ / (‖q_θ(z_θ)‖ · ‖z'_ξ‖),

with stop-gradient on z'_ξ. l2-normalization matters: it's the best-performing choice; without any normalization the projection norm runs away (it drifts up to enormous magnitudes) yet still works passably, and batch-norm-in-the-loss is notably worse — so l2 it is, which is exactly why the squared error collapses to the clean 2 − 2·cosine form above. Then symmetrize: also feed v' to the online branch and v to the target branch, compute the same loss L̃, and use L^total = L + L̃ so both views are used in both roles and the signal is doubled.

The update. One gradient step on L^total with respect to θ only — the predictor, projector, and encoder of the online branch all move; the target parameters are untouched by the gradient:

  θ ← optimizer(θ, ∇_θ L^total, η),
  ξ ← τ ξ + (1 − τ) θ.

For the τ schedule: early in training the online network is improving fast, so I want the target to move reasonably quickly to keep up (and to keep the predictor's problem fresh); late in training the online network has converged, so I want the target nearly frozen for stability. So anneal τ from a base value up toward 1 over training — τ = 1 − (1 − τ_base)·(cos(πk/K) + 1)/2 with step k of K, starting around τ_base = 0.996. The exact value is forgiving: anything in the 0.9–0.999 range works; τ = 1 (never update) leaves me stuck at round-zero quality, τ = 0 (instant copy) destabilizes — which is exactly the trade-off the bootstrap-vs-stability argument predicted.

Optimizer and regularization, for completeness, since this is meant to run at large batch. LARS, because layer-wise adaptive scaling is what makes very large batches train stably; cosine-decayed learning rate over ~1000 epochs with a short warmup; base learning rate scaled linearly with batch size (0.2 × batch/256). A small global weight decay (~1.5e-6), and — this matters — exclude the biases and batch-norm parameters from both the LARS adaptation and the weight decay. Weight decay can't be dropped: with no weight decay the network diverges, so some weight regularization is needed in this self-supervised setting. The only batch-size dependence left in the whole pipeline is the batch-norm in the encoder; there are no negatives drawing on the batch, which is precisely why I expect this to stay stable as the batch shrinks where the contrastive method falls apart.

Let me write it the way it actually runs, in JAX/Haiku, mirroring the structure above.

```python
from typing import NamedTuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax

class TrainState(NamedTuple):
    online_params: hk.Params
    target_params: hk.Params
    online_state: hk.State
    target_state: hk.State
    opt_state: optax.OptState

class Encoder(hk.Module):
    """Encoder slot; full ImageNet training uses a ResNet-50 torso here."""
    def __init__(self, name=None):
        super().__init__(name=name)
    def __call__(self, x, is_training):
        bn = dict(create_scale=True, create_offset=True, decay_rate=0.9)
        x = x.astype(jnp.float32)
        for channels in (64, 128, 256, 512):
            x = hk.Conv2D(channels, 3, stride=2, padding='SAME', with_bias=False)(x)
            x = hk.BatchNorm(**bn)(x, is_training=is_training)
            x = jax.nn.relu(x)
        return jnp.mean(x, axis=(1, 2))

# --- Network: encoder f -> projector g -> predictor q (online has all three;
#     the target reuses the same definition but its prediction head is unused).
def network(inputs, is_training):
    embedding = Encoder(name='encoder')(inputs, is_training)    # representation y; this is all we keep
    projection = MLP(name='projector')(embedding, is_training)  # z = g(y)
    prediction = MLP(name='predictor')(projection, is_training) # q(z), used only on the online branch
    return {'projection': projection, 'prediction': prediction}

class MLP(hk.Module):
    """Linear(4096) -> BN -> ReLU -> Linear(256); output is NOT batch-normed."""
    def __init__(self, name):
        super().__init__(name=name)
    def __call__(self, x, is_training):
        x = hk.Linear(4096)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)
        return hk.Linear(256, with_bias=False)(x)

net = hk.without_apply_rng(hk.transform_with_state(network))

# --- Loss: predict the (stop-gradient) target projection from the online prediction.
def regression_loss(x, y, eps=1e-12):
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)
    y = y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), eps)
    return 2 - 2 * jnp.sum(x * y, axis=-1)               # = || x_bar - y_bar ||^2

def apply_two_views(params, state, view_1, view_2):
    out_1, state = net.apply(params, state, view_1, is_training=True)
    out_2, state = net.apply(params, state, view_2, is_training=True)
    return out_1, out_2, state

def loss_fn(online_params, target_params, state_o, state_t, view_1, view_2):
    o1, o2, state_o = apply_two_views(online_params, state_o, view_1, view_2)
    t1, t2, state_t = apply_two_views(target_params, state_t, view_1, view_2)
    # stop_gradient on the target side: the gradient flows into the online network only.
    loss  = regression_loss(o1['prediction'], jax.lax.stop_gradient(t2['projection']))
    loss += regression_loss(o2['prediction'], jax.lax.stop_gradient(t1['projection']))  # symmetrized
    return jnp.mean(loss), (state_o, state_t)

# --- Update: gradient step on theta only; EMA the target toward the online net.
def update_fn(state, step, view_1, view_2, optimizer, total_steps, base_ema=0.996):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (state_o, state_t)), grads = grad_fn(
        state.online_params, state.target_params,
        state.online_state, state.target_state, view_1, view_2)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.online_params)
    online_params = optax.apply_updates(state.online_params, updates)
    tau = 1 - (1 - base_ema) * (jnp.cos(jnp.pi * step / total_steps) + 1) / 2
    target_params = jax.tree_util.tree_map(
        lambda t, o: tau * t + (1 - tau) * o,                  # xi <- tau*xi + (1-tau)*theta
        state.target_params, online_params)
    return TrainState(online_params, target_params, state_o, state_t, opt_state), loss

def keep_encoder(tree):
    return {name: value for name, value in tree.items() if name.startswith('encoder')}

# --- At the end of training, discard the projector, predictor, and target;
#     keep only the online encoder f as the representation.
```
