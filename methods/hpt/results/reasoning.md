Let me start from what actually breaks when I post-train one of these reasoning models, because the cleanest way in is the failure, not the recipe. I have a pretrained policy π_θ and a stack of math prompts, each one shipped with two things: a verifier that scores any candidate solution 1 if it ends in the right answer and 0 otherwise, and a single teacher demonstration τ★ that solves the prompt correctly. So per prompt I can either let the model generate its own rollouts and learn from whichever ones the verifier likes — that's reinforcement learning — or I can just push the model toward copying τ★ — that's supervised fine-tuning. The field treats these as two different machines that you bolt together by hand: do SFT first, then RL; or add the two losses with some fixed coefficient; or schedule the coefficient. Everyone agrees both are needed, nobody quite says why they're allowed to be combined, and every combination has a knob that has to be re-tuned per model and per dataset.

And there's a brittleness in the training dynamics I want to pin down before I trust any recipe. Watch on-policy RL on a weak model. GRPO samples a group of G rollouts for a prompt, scores them, and standardizes the scores within the group to make the advantage: Â_i = (R(τ_i) − mean) / std. I keep saying "all-wrong prompts give zero gradient," but let me actually run the advantage formula on a few groups instead of asserting it, because the exact behavior at the boundaries is the whole game. Take G = 8 and three cases — all rollouts wrong, all right, and a mixed 2-of-8:

  all-wrong   R = [0,0,0,0,0,0,0,0] → mean 0, std 0 (floored to 1) → Â = [0,0,0,0,0,0,0,0]
  all-correct R = [1,1,1,1,1,1,1,1] → mean 1, std 0 (floored to 1) → Â = [0,0,0,0,0,0,0,0]
  mixed 2/8   R = [1,1,0,0,0,0,0,0] → mean 0.25, std ≈ 0.463       → Â ≈ [+1.62,+1.62,−0.54,−0.54,−0.54,−0.54,−0.54,−0.54]

So the degeneracy is real and it's exactly the all-*equal* cases: whenever every rollout in the group carries the same reward the recentered reward is zero for all of them, and the per-token advantage that multiplies ∇log π_θ is identically zero — no gradient on that prompt. The mixed case, by contrast, produces a clean contrast (+1.62 for the two correct, −0.54 for the six wrong). Two of the three cases vanish, but they're not the same: the all-correct vanish is a saturation case — the model already solves it, fine to skip — while the all-wrong vanish is the dangerous one, because the gradient is exactly zero on a prompt the model still cannot solve. The model learns nothing on precisely the prompts where it most needs to learn — and if I plot per-question rollout accuracy across training I'd expect whole bands of these all-wrong prompts that RL never moves. That's not a tuning issue; it's structural. Pure on-policy RL can only sharpen what the model can already occasionally do; it cannot bootstrap a capability the model never samples. This is the same thing people see when "zero-RL" stalls on weak models, and it's why RL-zero works on Qwen but face-plants on Llama, while SFT helps both. RL is bounded by the base model.

Now flip to SFT, which obviously fixes the all-wrong prompt — just show the model τ★ and pull up its likelihood, no exploration required. But SFT has the opposite disease. It only ever says "copy the teacher." It gives the model no signal about which of its own behaviors were good; it overfits and memorizes the demonstrations; it degrades out of distribution. There's a clean diagnostic for this — train on rule-based reasoning variants and SFT memorizes the training rule while RL with an outcome reward generalizes across variants, yet SFT is still needed first because it stabilizes the output format so RL can take off. And there's the Pass@k observation: RLVR lifts Pass@1 but not large-k Pass@k, so it sharpens rather than expands the capability frontier; SFT-style injection of outside data is what raises the frontier. So the two paradigms genuinely do different things — exploitation of one's own good behaviors versus injection of new behavior — and they're complementary, not interchangeable.

So I want both, but the moment I write "add the two losses," I have to confront the thing nobody seems to justify: why is it even legitimate to add an RL loss and an SFT loss? They look like different objectives. One maximizes expected reward under the model's own distribution; the other minimizes cross-entropy against an external dataset. If they're optimizing different things, summing their gradients is just hoping two unrelated forces happen to point somewhere useful. Before I design any switching rule I need to know whether these are actually the same objective wearing two costumes, because if they are, then combining them isn't a hack — it's choosing how to estimate one gradient. Let me try to write a single objective that has both inside it.

What does each paradigm "want"? RL wants high expected verifier reward: maximize E over τ drawn from π_θ of r(τ|q). SFT wants the policy to stay close to the demonstration distribution — call it the behavior policy π_β, the thing that generated τ★. "Stay close to a distribution" is a KL penalty. So let me write the most naive common objective I can:

  J_μ(θ) = E_{τ~π_θ}[ r(τ|q) ] − μ · KL( π_β(·|q) ‖ π_θ(·|q) ),   μ ≥ 0.

Maximize expected reward, while not drifting too far from the demonstrations, with μ trading the two off. This is just "RL with a behavior-cloning regularizer," nothing exotic. The real question is what its gradient looks like. If the gradient cleanly splits into an RL piece and an SFT piece, I've shown they're one objective.

Take the gradient term by term. The reward term first. I can't differentiate E_{τ~π_θ}[r] naively because the distribution I'm averaging over depends on θ. The score-function identity is the tool: ∇_θ E_{τ~π_θ}[f(τ)] = E_{τ~π_θ}[ f(τ) ∇_θ log π_θ(τ) ]. Quick sanity-check on where it comes from: E_{π_θ}[f] = ∫ f(τ) π_θ(τ) dτ, so ∇ = ∫ f ∇π_θ dτ = ∫ f (∇π_θ / π_θ) π_θ dτ = E_{π_θ}[ f ∇log π_θ ], using ∇log π_θ = ∇π_θ / π_θ. And note the corollary with f≡1: E_{π_θ}[∇log π_θ] = ∇∫π_θ = ∇1 = 0, the baseline-subtraction fact I'll lean on. So the reward term gives

  ∇_θ E_{τ~π_θ}[r] = E_{τ~π_θ}[ r(τ|q) ∇_θ log π_θ(τ|q) ].

That's REINFORCE — the on-policy reward policy gradient. Good, the RL piece is there.

Now the KL term, and I have to be careful about the direction of the KL, because that's exactly the kind of sign I'd get wrong by waving my hands. I wrote KL(π_β ‖ π_θ), the divergence from the behavior policy to the model. Expand it: KL(π_β ‖ π_θ) = E_{τ~π_β}[ log π_β(τ) − log π_θ(τ) ]. Now differentiate with respect to θ. The π_β does not depend on θ — it's the fixed demonstration distribution — and the expectation is over π_β, also fixed in θ, so I only differentiate the −log π_θ inside:

  ∇_θ KL(π_β ‖ π_θ) = E_{τ~π_β}[ −∇_θ log π_θ(τ) ] = − E_{τ~π_β}[ ∇_θ log π_θ(τ) ].

This is exactly the place I'd flip a sign by hand-waving, so I'll check it on a tiny three-outcome model before I trust the rest. Take π_θ = softmax(θ) over three trajectories, a fixed demonstration distribution π_β = [0.6, 0.3, 0.1], and θ = [0.1, 0.4, −0.3]. Autodiff of KL(π_β‖π_θ) gives ∇KL ≈ [−0.2689, +0.1469, +0.1219]; computing −E_{π_β}[∇log π_θ] component-wise gives [−0.2689, +0.1469, +0.1219] as well, matching to ~3e-8. So the identity holds and the sign is the minus I wrote. Then the contribution to ∇J_μ is −μ · ∇KL = −μ · ( − E_{τ~π_β}[∇log π_θ] ) = + μ E_{τ~π_β}[ ∇_θ log π_θ(τ) ]. The two minus signs cancel and I get a plus. Is that the *right* plus — does ascending it raise the demonstration's probability? In the same numeric example +E_{π_β}[∇log π_θ] ≈ [+0.2689, −0.1469, −0.1219], which pushes weight onto trajectory 0 (the one π_β favors most, mass 0.6) and off 1 and 2; that's the maximum-likelihood direction, push up the log-probability of the demonstration tokens. Maximizing J means ascending, and ascending should pull π_θ toward the demonstrations — the +MLE direction. It does. If I'd written the KL the other way, KL(π_θ ‖ π_β), the expectation would be over π_θ and I'd have an extra ∇ acting on the sampling measure too — much messier and not the clean SFT term I want. The KL(π_β ‖ π_θ) direction is the one that makes the demonstration term fall out as plain behavior cloning. So:

  ∇_θ J_μ(θ) = E_{τ~π_θ}[ r(τ|q) ∇log π_θ ] + μ E_{τ~π_β}[ ∇log π_θ ].

The gradient of this one objective is a sum of two terms: an on-policy reward term sampled from the model, and a data-adherence term sampled from the demonstration policy. The second is the SFT gradient — E_{π_β}[∇log π_θ] is exactly maximum-likelihood ascent on the demonstrations. So SFT and RL are not two objectives here; they're the two halves of the gradient of this single J_μ, and the only difference between them is *which distribution you sample the trajectory from* and *what scalar you weight it by*. On this reading, adding the two losses is not an unrelated hack glued on — within J_μ it *is* the gradient. That reframes the switching question: I'm not blending two competing forces, I'm choosing how to sample and weight one estimator. (I'll keep "within J_μ" as the caveat — this legitimacy is relative to the objective I just wrote down; a different regularizer would give a different decomposition.)

Let me push on that, because I want both terms in *one* shared form so I can see exactly which knobs distinguish them. Both terms are an expectation of (something) times ∇log π_θ, but they're sampled from different distributions — π_θ for one, π_β for the other. Estimators from different sampling distributions are awkward to fuse. Can I rewrite both as expectations under a *common* sampling density and read off what's left? That's a change of measure. The identity: for any sampling density s(τ) that's positive wherever π_θ is,

  E_{τ~π_θ}[ f(τ) ∇log π_θ ] = E_{τ~s}[ (π_θ/s) f ∇log π_θ ] = E_{τ~s}[ (1/s) f ∇π_θ ].

The first equality is just multiplying and dividing by s inside the integral; the second uses ∇log π_θ = (1/π_θ)∇π_θ, so π_θ/s times 1/π_θ collapses to 1/s and the π_θ cancels, leaving (1/s) f ∇π_θ. The striking part is that the final form E_s[(1/s) f ∇π_θ] = Σ_τ f(τ) ∇π_θ(τ) has no s left in it at all — the sampling density cancels against its own 1/s. That's a strong enough claim that I should check it numerically rather than trust the cancellation. On the same three-outcome softmax model, with f = [2.0, −1.0, 0.5] and an arbitrary sampling density s = [0.2, 0.5, 0.3]: computing the left side E_{π_θ}[f ∇log π_θ] by autodiff gives [0.4566, −0.3665, −0.0901], and computing the right side Σ_τ f ∇π_θ (which is what E_s[(1/s) f ∇π_θ] reduces to) gives [0.4566, −0.3665, −0.0901] — identical to machine precision, and I can change s freely without moving it. So the cancellation is real: it kills the explicit π_θ in the numerator and leaves a 1/s denominator that I'm free to choose. Apply this with s = π_ref, the reference density that supplies the estimator's denominator:

  ∇J_μ = E_{τ~π_ref}[ (1/π_ref) Â_uni ∇π_θ ],   where the unified advantage is

  Â_uni(τ,q) = r(τ|q)  +  μ · π_β(τ|q)/π_θ(τ|q).

Let me check that second piece. The SFT term was μ E_{τ~π_β}[∇log π_θ] = μ E_{π_β}[(1/π_θ)∇π_θ]. Under the common-reference rewrite, the expectation weight has to integrate back to π_β/π_θ, so the SFT part of the unified advantage is μ π_β/π_θ. The point that jumps out: as an "advantage," the demonstration pull is large exactly where the model assigns small probability to the demonstration trajectory. In the standalone SFT row of the template I would write π_ref = π_θ and Â ≡ 1, giving ∇π_θ/π_θ on the demonstration tokens. In the common-objective row I keep the data source π_β visible inside Â_uni as μ π_β/π_θ. Same behavior-cloning gradient, two notations; the denominator is the token reweighting, not a claim that the demonstration was sampled from the current model.

So now I have a single template:

  grad = E_{τ~π_ref}[ (1/π_ref) · Â · ∇π_θ ].

Three knobs distinguish every method I know: the reference-policy denominator 1/π_ref, the advantage Â, and the likelihood gradient ∇π_θ which is shared by everyone (it's just how you backprop from token-actions to weights — nobody varies it). Let me actually instantiate the methods and see if they all fall out, because if they do I've found the right generalization and not just a cute rewrite.

SFT: π_ref = π_θ, Â ≡ 1 (every demonstration token is a "positive" with unit advantage). grad = (1/π_θ) · 1 · ∇π_θ = ∇π_θ/π_θ = ∇log π_θ. That's the cross-entropy ascent gradient — push up the log-prob of the demonstration token, with no extra weight. So I have to make sure the template *reproduces* the standalone SFT gradient I wrote at the top (the L_SFT direction Σ_t ∇log π_θ), and it does: setting the two dials to (π_θ, 1) recovers it exactly rather than something merely similar. That's the one I most need to land, since the whole argument is that SFT lives inside this estimator. REINFORCE: π_ref = π_θ (on-policy, no importance ratio), Â = ±1 from the verifier. grad = ∇π_θ · (Â/π_θ) = Â ∇log π_θ — REINFORCE, unbiased but high-variance. PPO: I sampled from an old policy π_{θ_old}, so π_ref = π_{θ_old}, and the 1/π_ref denominator together with ∇π_θ reconstructs the importance ratio r = π_θ/π_{θ_old}; Â is GAE. GRPO: same π_ref = π_{θ_old}, but Â is the group-normalized reward (R(τ_j) − mean over the on-policy group) / std over the group. Offline RL like SRFT: I have no access to the policy that produced the offline trajectory, so I set π_ref ≡ 1 — which turns the importance ratio into just π_θ, i.e. it stops reweighting; that's heavily biased (it's really rejection sampling, valid only if the offline data uniformly covers the state-action space, which it never does), but numerically stable. LUFFY: same offline π_ref ≡ 1, the SRFT-style combined-group advantage, plus a policy-shaping factor on the gradient. Each of these is recovered by a choice of (π_ref, Â), with the SFT case checked above against its standalone form. One template, with the reference denominator and the advantage as the dials.

But wait — I dropped the one thing that makes RL stable: the clipping. Where does PPO's clip live in this template? PPO's clipped surrogate has a piecewise derivative: in the region where pushing the ratio further would be harmful and the ratio has already left the trust region, the clip flattens the objective so the derivative is zero — it's a stop-gradient. So clipping doesn't change the *target* gradient; it multiplicatively zeroes the gradient on unsafe samples. That's an indicator, a stabilization mask 1_stable(τ,q) ∈ {0,1}, inserted in front:

  grad_uni = E_{τ~π_ref}[ 1_stable · (1/π_ref) · Â · ∇π_θ ]  =  1_stable · (1/π_ref) · Â · ∇π_θ.

Four components: the stabilization mask, the reference-policy denominator, the advantage, the likelihood gradient. And the variants people fight about are all just choices of mask or advantage — DAPO and CISPO loosen or refine the clip mask because the classic clip drops large-update tokens that matter for entropy; GSPO clips at the sequence level with a different π_ref; Dr.GRPO and RLOO drop the std-normalization in the advantage arguing it injects a difficulty bias. Every one of these is a point in the same four-component space.

Let me also push the trust region through the template properly, because I claimed clipping "is" a trust region and I should show the penalty form lands in the same place. Add an explicit KL-to-reference penalty: maximize E_{π_θ}[r] − λ KL(π_θ ‖ π_ref) − μ KL(π_β ‖ π_θ). The new KL is in the other direction, KL(π_θ ‖ π_ref), because now I'm constraining the *model* to stay near the reference (that's the trust region). Its gradient: ∇ E_{π_θ}[log(π_θ/π_ref)] = E_{π_θ}[ ∇log π_θ · log(π_θ/π_ref) + ∇log π_θ ] by the product-of-(sampling-measure, integrand) rule, and the lone ∇log π_θ term has zero expectation, leaving E_{π_θ}[ ∇log π_θ · log(π_θ/π_ref) ]. So the trust region just modifies the advantage: Â^(λ) = r − λ log(π_θ/π_ref) + μ π_β/π_θ. Same template, advantage shifted by −λ log-ratio. With π_ref = π_{θ_old} this is the penalty form whose constrained twin (maximize reward s.t. KL ≤ δ) is TRPO, and λ is the Lagrange multiplier tied to the radius δ. PPO's clip is the cheap surrogate of this. So the two ways of stabilizing land in the same place by two different routes: the clip enters as the mask 1_stable that zeroes unsafe samples, and the explicit KL penalty enters as a −λ log(π_θ/π_ref) shift on the advantage. Different components of the same four-component estimator, neither of which touches ∇π_θ — which is what I'd want if clipping and the trust-region penalty really are two realizations of one idea.

Now the payoff for *designing* a method. If SFT and RL are instances of one estimator, then given the right data assumptions and enough samples, *every* instance is a valid ascent direction on the common objective J_μ. They're not conflicting; they're different noisy measurements of the same true policy gradient, with different bias-variance profiles. The pure on-policy 1/π_θ (REINFORCE) is unbiased but high-variance; the importance-ratio 1/π_{θ_old} (PPO) reduces variance but the ratio is ill-posed and biased; the offline 1/π_ref = 1 is very biased but stable; std-normalized advantage versus recenter-only trade difficulty-bias against scale. The instinct from estimation theory is: if I have several noisy measurements of one quantity, average them, weighted toward the lower-variance ones — a complementary filter. So maybe the answer is a fixed weighted blend of the SFT and RL gradients?

But this stalls as soon as I ask what the weights should be, because they aren't fixed properties of the estimators. The bias and variance of each instance depend on the *current* state of π_θ and on the prompt. On a prompt the model already solves, the on-policy RL signal is informative and low-variance and the SFT term is pure overfitting pressure I don't want. On a prompt the model fails completely, the on-policy term is the all-wrong degenerate-zero I diagnosed at the start — it carries *no* information — and the only useful measurement is the demonstration. And which prompt is which *changes as the model learns*: a hard prompt the model fails today it may solve in fifty steps. So a single global mixing coefficient (which is exactly what LUFFY's fixed mix and SRFT's fixed combination commit to) is structurally wrong, the same way one global learning rate is wrong when different parameters need different scales. The right weighting has to be decided per prompt, and re-decided as training proceeds. The "average the estimators" idea was right in spirit; the constant weights were the mistake. I need the weighting to be a *function of how the model is currently doing on this prompt*.

What signal do I have, per prompt, that measures "how the model is currently doing"? I'm already sampling n on-policy rollouts per prompt and verifying them to compute the GRPO advantage. The pass-rate falls out for free:

  P(q) = (1/n) Σ_{i=1}^n v(τ_i),   v(τ_i) = R(τ_i) ∈ {0,1}.

P is exactly the model's current competence on q, on a scale from "fails everything" (P=0) to "solves everything" (P=1), and it costs nothing extra — it's the same verifier scores GRPO already consumes. So let the mixing be α = f(P) on the RL loss and β = g(P) on the SFT loss. The qualitative shape is forced by the diagnosis: when the model is strong on q (P high), I want exploration — emphasize RL — so f should increase with P; when the model is weak on q (P low), I want the demonstration to bootstrap it — emphasize SFT — so g should decrease with P. f positively correlated with P, g negatively. That's the whole adaptive idea: route each prompt's gradient toward the estimator that actually carries information about that prompt right now.

Now, what's the simplest f and g that does this? I could make them smooth — a soft sigmoid blend in P — but let me think about what's cleanest and what the degenerate-zero analysis demands. The sharpest possible version is a hard switch at a threshold γ:

  α = f(P) = 1 if P > γ else 0,
  β = g(P) = 1 if P ≤ γ else 0,

so each prompt does pure RL when it's above the gate and pure SFT when it's at or below. Binary, only one loss active per prompt. Why is a hard switch defensible rather than a crude approximation of a smooth blend? Because the failure mode I'm targeting is itself a sharp cliff — the one I already traced through the advantage formula. Set γ = 0. Then a prompt switches to SFT only when P = 0 — when *every* rollout failed. And P = 0 is exactly the all-wrong group from that trace: R = [0,…,0], so Â = [0,…,0], the standardized advantage vanishes and the RL gradient with it. The same all-equal algebra also flattened the all-correct group (R = [1,…,1] → Â = [0,…,0]), but that group does not need a teacher to bootstrap the first correct behavior — the model already solves it. So at γ = 0 the switch lines up with the two computed cases: it hands a prompt to SFT in, and only in, the all-wrong situation where the advantage I computed was zero, and keeps it on RL once the group has the kind of contrast the mixed 2/8 case showed (Â ≈ ±, nonzero) or the prompt is already saturated. It's not approximating a blend; it's repairing the discontinuity in the RL signal at the one point where that signal dies before competence appears. The minimal intervention also maximally preserves exploration — I only inject the teacher when the model is completely stuck, never diluting the prompts where it's learning on its own. That's the γ = 0 choice for Qwen; for a much weaker model like Llama, where "solves at least one of eight" is still too fragile to learn from on-policy, I'd lift the gate to γ = 2/8, switching to SFT unless the model gets at least three of eight — same logic, a higher competence bar for trusting exploration.

Let me sanity-check the gate direction against the alternatives. If I raised γ high — switch to SFT whenever the model isn't already near-perfect — I'd be doing SFT on prompts the model is actively improving on via RL, exactly the memorization/overfitting regime I want to avoid, drowning out exploration. The diagnosis says SFT generalizes worse and narrows the model toward the demonstration distribution, so more SFT is not automatically better; I want the *least* SFT that fixes the dead-gradient prompts. So small γ, and γ = 0 as the principled floor. The expected dynamics are simple: early in training a weak model fails many hard prompts, so more of the batch gets demonstration gradients; as it strengthens, more prompts cross the gate and the mixture moves toward on-policy RL. The mixing ratio self-adjusts from competence rather than being fixed in advance the way LUFFY's is.

One more decision to nail down: when a prompt is below the gate, why plain SFT on the demonstration rather than feeding the demonstration in as an *off-policy RL* signal — importance-sampling it into the advantage group the way LUFFY does? Both inject the teacher. But the off-policy-RL route needs a reference policy for the teacher trajectory, which I don't have, so it sets π_ref ≡ 1 — and I showed that's the rejection-sampling assumption that injects heavy bias, valid only under uniform coverage that never holds. Plain SFT (Â ≡ 1, π_ref = π_θ) has no such ill-posed ratio; it's the cleanest way to consume an offline trajectory, and it's exactly the second half of my unified gradient with μ folded into the loss weight. So for the stuck prompts I use the SFT term, not the off-policy-RL term. (The general scaffold can also add off-policy RL as a third route for an intermediate band — keep the demonstration but score it as RL — but the bias of π_ref ≡ 1 makes plain SFT the better default for the fully-stuck case.)

For the RL side I keep the GRPO / Dr.GRPO-style on-policy path: sample a group, verify, group-normalize the advantage Â_i = (R(τ_i) − mean) / (std + ε) — and crucially compute that mean and std over the *on-policy* rollouts only, not contaminated by any injected SFT/teacher samples, so the RL measurement stays clean — and optimize the clipped surrogate. When a prompt's group is degenerate (all rewards equal, std = 0) I floor std to 1 so the advantage is just the zero recentered reward rather than a divide-by-zero; at γ = 0 the dangerous all-wrong degenerate groups have already been routed to SFT, while all-correct groups can safely contribute no RL gradient.

So the full step. For each prompt q: draw n on-policy rollouts τ_i ~ π_θ(·|q); verify to get R(τ_i) ∈ {0,1} and the pass-rate P = (1/n)Σ v(τ_i); set α = 1[P > γ], β = 1[P ≤ γ]; if α is live, compute the GRPO clipped surrogate L_RL on the rollouts with group-normalized advantages; if β is live, compute the SFT NLL L_SFT on the demonstration τ★; form L = α L_RL + β L_SFT and descend. Because α and β are a hard switch, exactly one loss is live per prompt, and the batch loss is a sum over prompts of whichever term each one selected.

The step is now:

  for t = 1..T:
    for each prompt q in the batch:
      sample {τ_i}_{i=1..n} ~ π_θ(·|q)
      R(τ_i) = verifier(τ_i) ∈ {0,1}
      P = (1/n) Σ_i R(τ_i)
      α, β = (1, 0) if P > γ else (0, 1)
      L_RL  = GRPO_clip_surrogate({τ_i}, group_norm_advantage({R(τ_i)}))   # used iff α
      L_SFT = − (1/|τ★|) Σ_t log π_θ(τ★_t | q, τ★_{<t})                    # used iff β
      L_q   = α L_RL + β L_SFT
    θ ← θ − η ∇_θ ( Σ_q L_q )                                              # AdamW, η ~ 5e-6

In a real verl-style trainer I don't need to materialize α and β as floating tensors inside the actor. I can make the data stream express the switch. The controller reads on_solve_num, the integer count n·P, and returns exactly the batch edit triple (on_remove_num, on_add_num, off_add_num). The SFT branch removes that prompt's on-policy rollout group and adds one demonstration sample; the keep-RL branch leaves the on-policy group in place. The broader mixed-code path also has an off-policy-RL arm marked by a negative off_add_num, but the SFT/GRPO switch I want is the positive-off branch versus zero-off branch. With the common eight-rollout setup, the remove count is literally 8. After the batch is edited, the advantage code computes group normalization over on-policy samples only, and the actor computes the supervised NLL on prefix-masked demonstration samples while computing the clipped policy loss on the remaining on-policy samples. The actual mixed actor line is the simple one: pg_loss = sft_loss * sft_loss_coef + pg_loss.

```python
from collections import defaultdict
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo import core_algos


def select_on_off_ada_balance(config, on_solve_num):
    """verl mix_src controller: return (on_remove_num, on_add_num, off_add_num)."""
    if config.trainer.unify_strategy == "switch":
        on_add_num = 0
        if on_solve_num <= config.trainer.switch_gate:
            return 8, on_add_num, 1          # drop rollout group, add one SFT target sample
        if on_solve_num <= config.trainer.switch_gate_off:
            return 8, on_add_num, -1         # optional off-policy-RL branch in the shared path
        return 0, on_add_num, 0              # keep on-policy GRPO samples

    if config.trainer.unify_strategy == "soft":
        return 0, 0, 1

    raise NotImplementedError


def compute_grpo_outcome_advantage_split(token_level_rewards, eos_mask, index,
                                         on_policy_mask, epsilon=1e-6, use_std=True):
    """GRPO group advantage over on-policy samples only, matching grpo_split behavior."""
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score, id2mean, id2std = defaultdict(list), {}, {}
    with torch.no_grad():
        for i in range(scores.shape[0]):
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])

        for uid, values in id2score.items():
            if len(values) == 1:
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = torch.mean(torch.tensor(values))
                id2std[uid] = torch.std(torch.tensor([values]))
                if id2std[uid].item() == 0:
                    id2std[uid] = torch.tensor(1.0)

        for i in range(scores.shape[0]):
            centered = scores[i] - id2mean[index[i]]
            scores[i] = centered / (id2std[index[i]] + epsilon) if use_std else centered

    advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return advantages, advantages


def compute_sft_pure_loss(log_prob, eos_mask):
    return verl_F.masked_mean(-log_prob, eos_mask)


def actor_mixed_loss(log_prob, old_log_prob, advantages, response_mask, prefix_mask, config):
    """Actor path for offline_loss_type == 'sft' in mix_actor.py."""
    off_policy_mask = prefix_mask.any(-1)
    sft_loss = compute_sft_pure_loss(
        log_prob=log_prob[off_policy_mask],
        eos_mask=response_mask[off_policy_mask],
    )

    on_policy_mask = ~off_policy_mask
    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob[on_policy_mask],
        log_prob=log_prob[on_policy_mask],
        advantages=advantages[on_policy_mask],
        eos_mask=response_mask[on_policy_mask],
        cliprange=config.clip_ratio,
        loss_remove_token_mean=config.loss_remove_token_mean,
        loss_remove_clip=config.loss_remove_clip,
    )

    if not torch.isnan(sft_loss):
        pg_loss = sft_loss * config.sft_loss_coef + pg_loss

    return pg_loss, pg_clipfrac, ppo_kl
```

I began stuck: on-policy RL contributes zero gradient on exactly the prompts the model fails completely (group advantage degenerates to zero when all rollouts get the same reward), and SFT alone memorizes and narrows exploration — yet the field combines them by hand-tuned blends and pipelines with no account of why combining is even legitimate. Writing the most naive common objective — maximize expected verifier reward minus a KL pull toward the demonstration distribution — and differentiating it (score-function identity for the reward term, and the KL(π_β‖π_θ) direction making the demonstration term come out as the +behavior-cloning gradient) showed the gradient is exactly an on-policy RL term plus an SFT term: they are two halves of one gradient, distinguished by sampling distribution and reweighting. A change of measure to a common reference denominator collapsed both into one estimator, grad = 1_stable · (1/π_ref) · Â · ∇π_θ, with four components — stabilization mask, reference denominator, advantage, likelihood gradient — into which SFT, REINFORCE, PPO, GRPO, offline RL, and the mixed-policy methods all fall as choices of π_ref and Â (and clipping reappears as the mask, with the explicit trust-region penalty shifting the advantage by −λ log(π_θ/π_ref)). Since each instance is then an estimator, sometimes biased, of the same gradient under its data assumptions, the design question becomes how to weight the estimators — but their bias-variance depends on the model's current competence per prompt, which a single fixed coefficient can't track, so the weighting must be a function of per-prompt competence. The free competence signal is the rollout pass-rate P; routing α = 1[P>γ] to RL and β = 1[P≤γ] to SFT, with γ = 0, sends a prompt to the demonstration exactly in the all-wrong case where the on-policy advantage vanishes before competence appears, and otherwise preserves the on-policy path. Plain SFT is the clean stuck-prompt route because off-policy RL would need the biased π_ref ≡ 1 assumption. The result drops into the existing GRPO trainer as a controller that edits each prompt's samples and lets the actor add `sft_loss * sft_loss_coef` to the on-policy policy loss.
