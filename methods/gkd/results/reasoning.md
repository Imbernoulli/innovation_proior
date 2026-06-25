Let me start from what actually breaks when I distill one of these big autoregressive language models into a small one. I have a teacher I can query — not just sample from, but read its full next-token distribution `p_T(·|y_{<n}, x)` at any context — and I want a student `p_S^θ` with a tenth or a fortieth of the parameters to behave like it. The obvious thing, and what everyone does, is supervised KD: take a fixed dataset of sequences, run the teacher over each one to get its softened token-level distribution, and train the student to match it with forward KL at every position. Hinton's argument for why this is better than training on hard labels is good and I believe it — the teacher's soft distribution carries the relative probabilities of the wrong tokens, a similarity structure that a one-hot target throws away, the gradient is lower-variance, and you can train on less data at a higher learning rate. And there's the clean fact that in the high-temperature limit `∂C/∂z_i ≈ (z_i − v_i)/(N T²)`, so matching the softened distribution is matching logits, which is why you scale the soft loss by `T²` to keep it commensurate with any hard-label term. Fine. So why doesn't this just work?

Here is the thing that nags at me. The student is autoregressive. It factorizes `p(y|x) = Π_n p(y_n | y_{<n}, x)` and at inference it generates one token at a time, each conditioned on the tokens *it* produced. During supervised KD, every prefix `y_{<n}` it ever conditions on comes from the fixed dataset — ground-truth text, or text the teacher wrote. But at deployment, the prefixes come from the student itself, and the student is small and imperfect, so the moment it emits a token the teacher wouldn't have, it lands in a context that never appeared in training, where its behavior was never constrained, and from there things tend to spiral. People call this exposure bias and there's plenty of evidence it produces exactly the degenerate, drifting text you'd expect. So the training distribution of contexts and the inference distribution of contexts are different, and the difference is concentrated precisely where the student is weakest. I'm optimizing the right loss on the wrong inputs.

I want to know how bad this actually is, not just hand-wave "mismatch." Let me think about it as a horizon problem, because that's what an autoregressive rollout is — a length-`T` sequence of decisions where each decision changes the next state. Suppose the student, trained to match the teacher under the *teacher's* (or the data's) distribution of prefixes, makes a "mistake" — emits a token out of line with the teacher — with probability `ε` on the prefixes it was trained on. If I only ever evaluated it on those prefixes, its expected number of mistakes over `T` steps would be about `T·ε`, linear, fine. But that's not the test distribution. The first mistake takes it off the trained-on distribution of prefixes; now on this new prefix I have no guarantee at all — call the per-step error there as bad as 1 — and it can keep compounding for the rest of the horizon. Counting it carefully: a mistake at step `t` can corrupt all `T − t` steps after it, and summing the chance of a first deviation at each step against the tail it spoils gives something that scales like `T²ε`, not `T·ε`. That quadratic-in-horizon blow-up is the exposure-bias picture with a number attached: training only on in-distribution prefixes earns you a cost that grows with the *square* of the sequence length. For a model that emits hundreds of tokens, the gap between `T²ε` and `T·ε` is the whole problem.

That framing is familiar — it's exactly the imitation-learning story. A learner cloned from an expert under the expert's state distribution suffers `T²ε`; the cure is to train it under *its own* state distribution. And the cleanest way I know to do that without training `T` separate policies, one per step, is the dataset-aggregation idea: roll the current learner out, collect the states *it* actually visits, ask the expert to label those states, and train on them; iterate. Mixing the expert and learner during rollout, with the expert fraction decaying to zero, and folding the whole thing into a no-regret online-learning argument gives a bound of the form `J(π̂) ≤ J(π*) + O(uTε_N) + O(1)` — linear in the horizon when the cost-to-go penalty `u` is controlled. The mechanism is the only part I actually need: *the contexts the learner trains on must be the contexts the learner generates.* In control they call the expert an interactive oracle because you keep querying it on the learner's own trajectories.

So map it onto distillation. The student is the learner. The teacher is the interactive oracle — and a *better* oracle than usual, because I can query its full distribution, not just its preferred action. The "states" are partial token sequences. The fix the imitation-learning analysis prescribes is to stop training on a fixed dataset of prefixes and instead train on prefixes the student itself generates, with the teacher labeling them. Concretely: given an input `x`, let the student sample its own output `y ~ p_S(·|x)`, and at each intermediate context `y_{<n}` along that self-generated sequence, pull the teacher's distribution `p_T(·|y_{<n}, x)` and push the student toward it. Write the on-policy objective as

  L_OD(θ) = E_{x∼X} [ E_{y ∼ p_S(·|x)} [ KL(p_T ‖ p_S^θ)(y|x) ] ],

where the inner KL is the per-token divergence averaged over the sequence,
`KL(p_T ‖ p_S^θ)(y|x) = (1/L_y) Σ_n KL(p_T(·|y_{<n},x) ‖ p_S^θ(·|y_{<n},x))`. Now the prefixes `y_{<n}` are drawn from the student's own rollout, which is the kind of rollout it must survive at inference. I am no longer asking the loss to fix behavior only on prefixes from a static corpus. And it's *cheap* in the way that matters: the expensive object at deployment is serving the model, and here I generate the rollouts from the small student, not the big teacher, so generating training data is far cheaper than it would be to sample from the teacher. There's a bonus the imitation-learning view predicts: as the student improves, the sequences it generates improve, so the data quality climbs along with the model — a moving, self-improving target rather than a frozen corpus.

Now I hit a wall, and it's a subtle one. If `y ~ p_S(·|x)`, then the loss depends on `θ` in *two* places: through the divergence `KL(p_T ‖ p_S^θ)`, and through the sampling distribution `p_S(·|x)` that produced `y`. If I want the true gradient of `E_{y~p_S}[...]`, I have to differentiate through the sampling too. Writing it out: `∇_θ E_{y~p_S^θ}[f_θ(y)] = E_{y~p_S^θ}[ ∇_θ f_θ(y) ] + E_{y~p_S^θ}[ f_θ(y) ∇_θ log p_S^θ(y) ]`, where `f_θ(y)` is the per-sequence divergence. The first term is the ordinary "treat the data as fixed" gradient; the second is the score-function / policy-gradient term, and it is the troublesome one — `f_θ(y) ∇_θ log p_S^θ(y)` with `f` a divergence over a length-`T` sequence has variance that grows with the horizon, and I've seen what it takes to keep it from diverging in practice: variance reduction, fighting reward hacking, correcting for the model gaming sequence length. That's a pile of machinery and instability I'd rather not buy.

So the question is whether I can drop the second term and keep what I came for. Let me be honest about what dropping it costs. The *only* reason I went on-policy was to fix the *distribution of contexts* — to evaluate and improve the student on the prefixes it really visits. That fix lives entirely in *where `y` is drawn from*, i.e. in the sampling distribution being `p_S^θ` rather than the dataset; it does not live in differentiating through that sampling. The imitation-learning prescription is the same: collect the states the learner visits, label them, train — it never asks me to push gradient back through the rollout that produced the states. So if I sample `y` from the current student and then *freeze it* — stop-gradient on the generated tokens — I keep the first term, which is exactly a supervised-KD gradient evaluated on those particular self-generated sequences, and I discard the second term, which is precisely the high-variance piece I don't want. What I lose is honesty about the chain rule: this is not the exact gradient of `E_{y~p_S^θ}[f_θ]`, because the distribution of contexts shifts with `θ` and I'm not accounting for that shift. What I keep is the entire mechanism the imitation-learning analysis says matters — on this step the contexts *are* the student's own — at the cost and stability of ordinary distillation. Given that the second term's whole reputation is instability and that the distribution-of-contexts fix survives without it, dropping it is the move; I'll keep the stop-gradient.

Good — but I've quietly committed to the forward KL as the divergence, and the second failure of supervised KD was never about the data distribution at all; it was about *which* divergence. So let me reopen that. Forward KL from teacher to student, `KL(p_T ‖ p_S) = Σ_c p_T(c) log(p_T(c)/p_S(c))`. Look at where this blows up: at any token `c` where `p_T(c) > 0` but `p_S(c) → 0`, the ratio `p_T(c)/p_S(c) → ∞` and the term diverges. So minimizing forward KL forces the student to put *some* mass everywhere the teacher has mass — it is mass-covering, mean-seeking. If the student had infinite capacity that would be fine, it could just match the teacher exactly. But the student is small; it cannot represent the teacher's distribution faithfully. Forced to cover every mode the teacher touches, a low-capacity student smears probability across tokens the teacher only weakly likes, including genuinely bad ones, and that smeared mass is what you sample at inference — hallucinated, low-quality continuations. The very property that makes forward KL safe for a powerful student (don't drop any of the teacher's mass) is what makes it dangerous for a weak one.

What's the opposite behavior? Reverse KL, `KL(p_S ‖ p_T) = Σ_c p_S(c) log(p_S(c)/p_T(c))`. Now the divergence is large where `p_S(c) > 0` but `p_T(c) → 0` — the student is punished for putting mass where the teacher has *none*. So reverse KL is zero-forcing: the student is driven to keep its support inside the teacher's, which means concentrating on one (or a few) high-probability regions and abandoning the rest. Mode-seeking. For an underpowered student that's often what I actually want — pick a mode the teacher genuinely likes and do it well, rather than spread thin and hallucinate. The cost is diversity: a mode-seeking student generates less varied text for a given input. So neither direction is universally right. It depends on how badly the student is outmatched and on whether the task rewards faithful coverage or crisp high-quality single answers. I don't want to pick one and bake it in; I want to be able to *choose*, even slide between them.

So I need a one-parameter family of divergences that has forward KL at one end and reverse KL at the other, and behaves sensibly in between. Anything built from a mixture is a natural candidate, because a mixture coefficient is exactly the kind of dial I want; the Jensen-Shannon construction is the obvious one. Introduce a mixing coefficient `β ∈ (0,1)`, form the mixture `M = β p_T + (1−β) p_S`, and define

  D_{JSD(β)}(p_T ‖ p_S) = β · KL(p_T ‖ M) + (1−β) · KL(p_S ‖ M).

Does this actually interpolate to the two KLs at the ends? I shouldn't take it on faith, so let me pin down the `β → 0` behavior carefully. As `β → 0`, `M = p_S + β(p_T − p_S) → p_S`. The first term is `β · KL(p_T ‖ M)`; since `M → p_S`, `KL(p_T ‖ M) → KL(p_T ‖ p_S)`, so this term behaves like `β · KL(p_T ‖ p_S)` — linear in `β`. The second term `(1−β) · KL(p_S ‖ M)` has `KL(p_S ‖ M) → KL(p_S ‖ p_S) = 0`; the leading correction of a KL when its two arguments differ by `O(β)` is *quadratic*, so this term is `O(β²)`. That means the unscaled `D_{JSD(β)}` itself vanishes like `β` at the boundary — it does *not* limit to forward KL literally — and the right statement is the scaled one, `lim_{β→0} D_{JSD(β)}/β = KL(p_T ‖ p_S)`.

I want to actually see this rather than trust the bookkeeping, so take a concrete two-token case `p_T = (0.7, 0.3)`, `p_S = (0.4, 0.6)`. By hand the forward KL is `KL(p_T ‖ p_S) = 0.7 ln(0.7/0.4) + 0.3 ln(0.3/0.6) = 0.183787`. Now evaluate `D_{JSD(β)}/β` shrinking `β`:

  β = 0.01:  D_JSD = 1.819e−3,  D_JSD/β = 0.181913
  β = 1e−3:  D_JSD = 1.836e−4,  D_JSD/β = 0.183599
  β = 1e−4:  D_JSD = 1.838e−5,  D_JSD/β = 0.183768
  β = 1e−5:  D_JSD = 1.838e−6,  D_JSD/β = 0.183785

The ratio is climbing straight to `0.183787` — the forward KL — and the unscaled value really is collapsing toward zero (`1.8e−6` at `β = 1e−5`), exactly as the second-order argument said. Splitting the two terms confirms the mechanism: at `β = 1e−4`, `term1/β = 0.18375` (tracking forward KL) while `term2/β² ≈ 0.1875` settles to a constant, so `term2` is genuinely `O(β²)` and drops out of the scaled limit. So small `β` gives the forward-KL direction — mass-covering. By the symmetry `D_{JSD(β)}(p_T ‖ p_S) = D_{JSD(1−β)}(p_S ‖ p_T)`, the `β → 1` end must scale to reverse KL; checking it the same way, `D_{JSD(β)}/(1−β)` runs `0.18990 → 0.19183 → 0.19202 → 0.192041` against the hand value `KL(p_S ‖ p_T) = 0.192042` — so `β` near one is mode-seeking, and `β = 0.5` is the symmetric Jensen-Shannon divergence sitting between. Good: the family does span forward to reverse KL, and I now know precisely how — through scaled limits, not literal endpoint values, which is a fact I'll have to respect when I implement the endpoints.

There's a second property I get from the mixture form that I didn't go looking for. Because `M` is a convex combination of `p_T` and `p_S`, each KL inside `D_{JSD(β)}` is a KL *to the mixture*, and a KL to a mixture is finite even when `p_T` and `p_S` have disjoint support — wherever `p_T(c) > 0` the mixture has `M(c) ≥ β p_T(c) > 0`, so no `log 0` ever appears. (In fact `D_{JSD(β)}` is bounded; the symmetric `β = 0.5` case is bounded by `log 2`.) Contrast plain forward KL, which goes to infinity the instant the student assigns zero to a token the teacher likes. Early in training the student's distribution can be wildly off the teacher's, with near-disjoint support on some contexts; forward KL there produces an enormous, possibly destabilizing gradient, while the JSD stays finite. So generalizing to JSD(β) gives me the mode-seeking/covering knob and, for free, tames the early-training blow-up of raw KL.

Now I can step back and see the shape of the whole thing, because the two moves are orthogonal and compose into one object. I have one axis for *which sequences I train on* — fixed dataset versus the student's own on-policy rollouts — and one axis for *which divergence I minimize* — anywhere on the forward-to-reverse family. Let me parameterize the data axis by a fraction `λ ∈ [0,1]`, the share of on-policy student-generated sequences, and let the divergence be any `D` from the family. The objective is

  L_GKD(θ) = (1 − λ) · E_{(x,y) ∼ (X,Y)} [ D(p_T ‖ p_S^θ)(y|x) ]
           +    λ   · E_{x ∼ X} [ E_{y ∼ p_S(·|x)} [ D(p_T ‖ p_S^θ)(y|x) ] ],

with the stop-gradient on the student's sampling, as I argued. Now let me check what this object reduces to at its corners, because if it's the right generalization the prior methods should fall out of it rather than sit beside it. Set `λ = 0`: the second line vanishes and `L_GKD = E_{(x,y)~(X,Y)}[ D(p_T ‖ p_S^θ)(y|x) ]`, fixed-dataset expectation only; further set `D =` forward KL and the per-token term is `KL(p_T ‖ p_S^θ)` on fixed data — that is the supervised-KD objective verbatim, `L_SD`. Set `λ = 1`: the first line vanishes and the data is entirely `y ~ p_S(·|x)`; with `D =` forward KL that is the pure on-policy KD I derived from the imitation-learning argument, `L_OD`. So both endpoints land exactly on objectives I already had, which is reassuring — I haven't quietly changed the loss, only the data fraction and the divergence around it. The mixed-data, forward-KL-at-the-token-level method that interpolated dataset and student samples with a decaying schedule sits at an interior point — some `0 < λ < 1`, forward KL — so it was already moving along the data axis, but it never left forward KL and never went fully on-policy, which marks the two directions it left on the table: the divergence knob (for when capacity mismatch wants mode-seeking or a bounded loss) and the all-on-policy corner. That the existing methods are recovered as specific `(λ, D)` settings, rather than needing a separate mechanism each, is the evidence I'd want that the two knobs are the right axes — though it's consistency, not proof, and the real test is whether moving off the corners actually helps, which only the experiments can settle.

Let me settle the operational details, because the abstraction has to become a per-batch loop and a per-token loss. For the data axis I don't need a fancy schedule; I can realize the fraction `λ` by a coin flip per batch. Draw `u ~ Uniform(0,1)`; if `u ≤ λ`, sample inputs and generate the outputs from the student to form the batch; otherwise take inputs and outputs from the fixed dataset. Over many steps that's exactly a `λ` fraction of on-policy batches, and it cleanly degenerates to pure supervised (`λ=0`) or pure on-policy (`λ=1`) at the ends. There's a precondition I should be honest about: this only works if the student can already generate adequate sequences for the teacher to give meaningful feedback on. A randomly initialized student would generate garbage, the teacher's per-token corrections on garbage prefixes would be nearly useless, and the early gradients would be junk. So I start from a student that's already been supervised-fine-tuned — it generates reasonable sequences, the teacher's feedback is informative, and on-policy distillation then refines it. This is the same two-stage shape as supervised-fine-tune-then-RL pipelines, which is convenient: I can borrow their tuning intuitions, and as I'll see, the on-policy structure makes the two fit together.

That last point deserves following, because it falls right out and it's useful. On-policy GKD needs nothing from the student but samples and a per-sequence loss. Reward-based fine-tuning of a language model — optimizing some scalar reward `r(y)`, maybe from human or AI feedback — also needs nothing but student samples and a per-sequence signal. So they compose with no friction. If I want to maximize a reward while staying close to the teacher, I write

  E_{x∼X} [ (1−α) · E_{y∼p_S^θ(·|x)}[ r(y) ]  −  α · E_{y∼p_S(·|x)}[ D(p_T ‖ p_S^θ)(y|x) ] ],

where `α ∈ [0,1]` trades the reward against the distillation term and `α = 1` is pure distillation. The usual reward-fine-tuning recipe regularizes the learned policy back toward the initial supervised policy with a reverse KL; here the regularizer is instead the distillation toward the *teacher*. That's a meaningful change in what you're anchored to — you're pulled toward a strong teacher rather than toward your own starting point — and because reverse KL is the conventional regularizer in that setting, if I'm grafting GKD onto an existing reward-fine-tuning workflow with minimal disruption I'd reach for the reverse-KL end, `β` near 1, say JSD(0.9), to match the regularizer the pipeline already expects.

Now the per-token loss in code, which is where I have to be careful about conventions and numerics, because the family has endpoints that the interior formula can't touch. For the interior `0 < β < 1` I need the mixture `M = β p_T + (1−β) p_S`, then `β · KL(p_T ‖ M) + (1−β) · KL(p_S ‖ M)`. I have the logits, so I work in log-probabilities throughout for stability. `log M` is the log of a sum of two scaled probability vectors; computing it as `log( (1−β) exp(log p_S) + β exp(log p_T) )` directly would underflow, so I form it as a log-sum-exp of the two shifted log-prob tensors: `log M = logsumexp( [ log p_S + log(1−β), log p_T + log(β) ] )` stacked along a new axis. That's numerically the right way to get `log M` from `log p_S` and `log p_T` without ever leaving log space.

Then there's the KL primitive, and its argument order is the kind of thing I should not guess at, because getting it backwards minimizes the wrong direction silently — no error, just a quietly wrong objective. The library's `kl_div(input, target)` with `log_target=True` computes `Σ target · (log target − input)`, which on paper reads as `input` being the log of the denominator and `target` the log of the numerator, i.e. `kl_div(log q, log p) = KL(p ‖ q)`. Rather than trust my reading of the docstring, let me trace it on the same `p_T = (0.7,0.3)`, `p_S = (0.4,0.6)`. By hand `KL(p_T ‖ p_S) = 0.183787` and `KL(p_S ‖ p_T) = 0.192042` (the two values from before). Feeding logs: `kl_div(log p_S, log p_T)` returns `0.183787` and `kl_div(log p_T, log p_S)` returns `0.192042`. So `kl_div(log p_S, log p_T) = KL(p_T ‖ p_S)` — the *input* slot is the denominator `q`, the *target* slot is the numerator `p`, confirming `kl_div(log q, log p) = KL(p ‖ q)`. So for `KL(p_T ‖ M)` I want `p = p_T` as target and `q = M` as input: `kl_div(log M, log p_T)`; likewise `KL(p_S ‖ M) = kl_div(log M, log p_S)`. First arg is the mixture log-probs, second is the distribution the KL is "from." Sum over the vocabulary axis to a per-token value, then combine: `per_token = β · kl_teacher + (1−β) · kl_student`. To make sure the whole interior path is wired right, I also run it end to end at `β = 0.5`: building `log M` by the logsumexp recipe gives `exp(log M) = (0.55, 0.45)`, which is exactly `0.5·p_T + 0.5·p_S`, and `0.5·kl_div(log M, log p_T) + 0.5·kl_div(log M, log p_S) = 0.046201`, matching the symmetric JSD computed directly by hand on the same vectors. The mixture and both KL directions are correct.

The endpoints can't go through the mixture formula: at `β = 0` or `β = 1` the code forms `log(β)` or `log(1−β)`, which is `log 0`, and even if it didn't, I already saw the unscaled JSD collapse toward zero at the boundary (`1.8e−6` at `β = 1e−5`), so the literal formula gives a vanishing loss there, not the KL I want. The useful thing to *put* at those two trainer settings is therefore the scaled-limit object the family points to: `β = 0` should mean forward KL `KL(p_T ‖ p_S)`, and `β = 1` reverse KL `KL(p_S ‖ p_T)` — the very limits I verified the ratios `D_JSD/β` and `D_JSD/(1−β)` converge to. So I branch: at `β = 0`, `kl_div(log p_S, log p_T) = KL(p_T ‖ p_S)`, forward (which I just traced returns `0.183787` on the test vectors); at `β = 1`, `kl_div(log p_T, log p_S) = KL(p_S ‖ p_T)`, reverse. The interior branch is the actual generalized JSD; the endpoint branches are the scaled-limit continuations, and they sidestep the `log 0`. The loss can also divide both logit tensors by a temperature before the softmax; separately, when generating student rollouts, I sample with temperature rather than greedy decoding so the student visits a useful spread of its own continuations.

Reduction and masking. The divergence contributions live on the completion positions only — the prompt and padding positions must not contribute. The label tensor marks them with `-100`, so I build a mask `labels != -100`, keep only those positions, and average. In the trainer path where labels are present, "batchmean" means summing the kept vocabulary contributions and dividing by the number of kept tokens, which is the masked token average corresponding to the `1/L_y` normalization from the per-sequence definition. I mirror that reducer directly: no extra protective clamp or alternate denominator, because the training batches are expected to contain completion tokens.

Two more alignment details that are easy to get wrong. The teacher runs in eval mode under no-grad — it's frozen, I only read its distribution, and I don't want its dropout or its gradient. And the logits have to be shifted to line up with the tokens they predict: position `n`'s logits predict token `n+1`, so over a (prompt + completion) sequence I slice the student and teacher logits from `prompt_len − 1` to `−1` and take the labels from `prompt_len` onward, so that the sliced student logits, teacher logits, and labels all index the same completion tokens before the divergence touches them.

And the on-policy generation step itself: with probability `λ`, before computing the loss, I take the prompts, generate completions from the student under sampling (temperature on, so the rollouts are diverse), build a fresh attention mask and a label tensor that marks padding as `-100`, and swap these self-generated sequences in as the batch. Then the ordinary distillation forward-and-loss runs on them. That's the whole on-policy mechanism — generate, relabel, then treat as data — and because I stopped the gradient at sampling, the loss machinery downstream is identical whether the batch is on-policy or from the fixed dataset.

So let me write the trainer pieces that fill the two slots — the per-token divergence and the on-policy data selection — in the shape I would actually ship.

```python
import random
import torch
import torch.nn.functional as F
from trl.models.utils import unwrap_model_for_generation


def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                         beta=0.5, temperature=1.0, reduction="batchmean"):
    """Per-token divergence between teacher and student over completion tokens.
    beta interpolates: 0 -> forward KL(p_T||p_S), 1 -> reverse KL(p_S||p_T),
    in between -> generalized JSD with mixture M = beta*p_T + (1-beta)*p_S."""
    # temperature softening, same knob as classic distillation
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # work in log-probs throughout for numerical stability
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        # forward KL = KL(p_T || p_S): mass-covering end
        jsd = F.kl_div(student_log_probs, teacher_log_probs,
                       reduction="none", log_target=True)
    elif beta == 1:
        # reverse KL = KL(p_S || p_T): mode-seeking end
        jsd = F.kl_div(teacher_log_probs, student_log_probs,
                       reduction="none", log_target=True)
    else:
        # interior: log M via logsumexp of the two shifted log-prob tensors,
        # M = (1-beta)*p_S + beta*p_T, computed without leaving log space
        beta = torch.tensor(beta, dtype=student_log_probs.dtype)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log(1 - beta),
                         teacher_log_probs + torch.log(beta)]),
            dim=0,
        )
        # F.kl_div(input=log_q, target=log_p, log_target=True) = KL(p || q)
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs,
                              reduction="none", log_target=True)   # KL(p_T || M)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs,
                              reduction="none", log_target=True)   # KL(p_S || M)
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    # keep only completion positions (-100 marks prompt/padding), then average
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    if reduction == "batchmean":           # per-token mean over real completion tokens
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    return jsd


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Student + frozen teacher forward over the same tokens; slice to completion
    positions (logit at n predicts token n+1); take the generalized-JSD loss."""
    student_outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
    self.teacher_model.eval()
    with torch.no_grad():                  # teacher is frozen: read its distribution only
        teacher_outputs = self.teacher_model(input_ids=inputs["input_ids"],
                                             attention_mask=inputs["attention_mask"])

    # shift so logits at position n align with the token they predict (n+1)
    prompt_len = inputs["prompts"].shape[1]
    student_logits = student_outputs.logits[:, prompt_len - 1 : -1, :]
    teacher_logits = teacher_outputs.logits[:, prompt_len - 1 : -1, :]
    labels = inputs["labels"][:, prompt_len:]

    loss = self.generalized_jsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        beta=self.beta,
    )
    return (loss, student_outputs) if return_outputs else loss


@staticmethod
def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
    """Roll out a generator on the prompts and relabel its generations as the batch.
    The generated token ids are reassigned as fixed training data."""
    generated_outputs = model.generate(
        input_ids=inputs["prompts"],
        attention_mask=inputs.get("prompt_attention_mask", None),
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    generated_tokens = generated_outputs.sequences
    new_attention_mask = torch.ones_like(generated_tokens)
    new_labels = generated_tokens.clone()
    if pad_token_id is not None:
        new_labels[new_labels == pad_token_id] = -100
        new_attention_mask[generated_tokens == pad_token_id] = 0
    return generated_tokens, new_attention_mask, new_labels


def training_step(self, model, inputs, num_items_in_batch=None):
    """If requested, replace fixed completions by generated completions, then
    let the ordinary trainer path run compute_loss/backward/optimizer handling."""
    if self.seq_kd:
        with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
            new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
            )
        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_attention_mask
        inputs["labels"] = new_labels

    if random.random() <= self.lmbda:
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
            )
        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_attention_mask
        inputs["labels"] = new_labels

    loss = super().training_step(model, inputs, num_items_in_batch)
    return loss
```

Let me trace the causal chain end to end. I started with supervised KD failing for two distinct reasons: it trains the student on a fixed distribution of prefixes while the student walks its *own* prefixes at inference, and the imitation-learning analysis says that train-inference mismatch costs `T²ε` over a horizon `T` rather than the linear-in-horizon cost available when learning is controlled on the learner's induced state distribution; and it uses forward KL, which is mass-covering and forces an underpowered student to smear probability onto teacher-unlikely tokens, producing low-quality, hallucinated generations. The first failure is exactly the kind of problem dataset-aggregation imitation learning addresses — collect the contexts the learner actually visits, label them with the interactive oracle — so I train the student on its own on-policy rollouts with the teacher labeling each token-level distribution. That follows the same route toward a linear-horizon regime and is cheap because the small student does the generating. To avoid the high-variance policy-gradient term, I stop the gradient through the student's sampling and treat each rollout as fixed data, keeping a supervised-style gradient. The second failure is about divergence direction, so I generalize forward KL to a one-parameter Jensen-Shannon family — `M = β p_T + (1−β) p_S`, with small `β` taking the forward-KL direction through the scaled limit and `β` near one taking the reverse-KL direction by symmetry — which lets a low-capacity student be mode-seeking when coverage is hopeless, and is bounded even on disjoint supports, taming early-training KL blow-ups. Composing the data fraction `λ` and the divergence `D` gives one objective in which supervised KD (`λ=0`, forward KL) and pure on-policy KD (`λ=1`, forward KL) are corners. The data fraction is realized by a per-batch coin flip; the method needs an SFT student so the teacher's feedback on generations is meaningful; and because it only consumes student samples, it grafts directly onto reward fine-tuning, regularizing toward the teacher rather than the initial policy. The loss lands as the masked, length-normalized generalized-JSD over completion tokens with careful log-space mixture and KL-direction handling, and the on-policy step is generate-relabel-then-treat-as-data feeding the same distillation loss.
