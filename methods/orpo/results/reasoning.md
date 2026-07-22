Let me start from what actually costs me when I align a language model, because the pain is concrete and it's not in the loss formula, it's in the pipeline. To make a base model into something that prefers the answers people want, I run two trainings back to back. First supervised fine-tuning on the chosen responses, to drag the model into the target domain and teach it the format. Then a preference stage on top of it — RLHF or DPO — that takes the pairs (x, y_w, y_l) and teaches the model to favor y_w over y_l. And that second stage drags a frozen copy of the SFT model around with it the whole time, because both RLHF and DPO measure how far the policy has drifted from where SFT left it, so the reference model is literally the SFT checkpoint held fixed. So I'm paying twice: two training runs, and during the second one I'm holding two full copies of a multi-billion-parameter model in memory and pushing tokens through both. DPO per batch is four forward passes — chosen and rejected, each through the policy and through the reference. I keep asking myself: is the reference model actually load-bearing, or is it scaffolding I've never questioned? If I could fold the preference signal directly into SFT, one stage, no reference, I'd halve the memory and most of the compute. So let me see whether a single loss, applied straight to the pre-trained model, can do the domain adaptation *and* the preference contrast at once, with nothing frozen on the side — and whether such a thing is even coherent, or whether the reference is doing work I'll regret removing.

But before I reach for a clever contrast, I should understand exactly why SFT alone doesn't already do the job — because if SFT-on-chosen-only secretly suppressed rejected responses, I'd be done. Write the SFT loss out, the plain causal-LM negative log-likelihood for a sequence of length m over the vocabulary V:

  L = -(1/m) Σ_k Σ_i y_i^(k) · log p_i^(k),

where y_i^(k) is the one-hot indicator that token i is the label at position k, and p_i^(k) the model's probability there. Stare at the inner sum. It only survives where y_i = 1 — the single label token at each position. For every other token in the vocabulary, including every token that would build a rejected response, y_i = 0 and the term vanishes. So cross-entropy *rewards* the label tokens and is completely silent about all the others. There is no machinery here that pushes any specific continuation *down*. That's the structural hole: SFT is a one-sided objective, all reward, no penalty.

And I can predict what that one-sidedness does, because it's a known behavior of SFT: training a model to raise the probability of chosen responses in a domain raises the probability of the *whole neighborhood* of that domain, and rejected responses live in exactly that neighborhood — same topics, same register, often overlapping phrasing. The log-probability of rejected responses climbs right alongside the chosen ones during SFT, and can end up at or above the chosen response's likelihood. The cross-entropy has no term that could pull them apart. So plain SFT, on its own, doesn't just fail to align — it actively makes the disfavored style *more* generatable as a side effect of domain adaptation. So if I want one stage, I need a penalty term running during SFT that discriminates y_l from y_w, because SFT will never produce that contrast by itself.

What does a penalty term look like that *appends* to NLL rather than replacing it? I've seen this shape before in degeneration work — to stop repetition you take the unwanted tokens and add a term that penalizes their probability, roughly Σ log(1 − p_i) over the unwanted set, on top of the usual likelihood. That's the right idea structurally: keep the NLL doing domain adaptation, bolt a penalty onto it that pushes down the bad stuff. But there the unwanted set was hand-crafted (recent tokens, to kill repeats). I don't want to craft anything per example. I have the rejected response y_l sitting right there in every pair — that *is* my dynamically-supplied "unwanted set," one per query, no crafting. So the plan crystallizes: L = L_SFT (on the chosen) + λ · (a penalty term that contrasts y_w against y_l), all in one stage.

Now the real question — what's the contrast? The obvious move, the one DPO and IPO use, is a probability ratio: compare P(y_w|x) to P(y_l|x), maximize P(y_w)/P(y_l). And the natural way to turn "make this ratio big" into a smooth loss is to wrap it in a log-sigmoid: minimize −log σ(log[P(y_w)/P(y_l)]). As the ratio grows the loss shrinks. Clean. So why don't I just use that? Let me think hard about what happens when I run a log-ratio contrast *during* SFT, on a model that hasn't been domain-adapted yet — because that's the regime I've put myself in by going single-stage. DPO uses the probability ratio, but DPO runs *after* SFT, on an already-adapted model, and against a reference. I'm running it *from scratch*, fused with SFT. That's a different animal and I should check it behaves.

Here's the worry I want to make precise. For any ratio R, minimizing −log σ(log R) doesn't just want R > 1, it wants R *large*: the loss keeps pulling until the typical log R is well into the saturating tail of the sigmoid. So the *scale* of the ratio I feed in sets the margin the model will try to force between chosen and rejected. If the log-ratio is one whose values are tightly concentrated near zero across examples, then to move the loss the model has to push individual examples *hard* — it has to overshoot. And in the probability-ratio case, overshoot means crushing P(y_l|x) toward zero, which means slamming the logits of the rejected tokens way down. On a model that's still learning the domain, those rejected tokens overlap heavily with perfectly good tokens it still needs; nuking their logits would degrade generation. So the question I actually need to answer is: how concentrated is log[P(y_w)/P(y_l)] — and is there a different ratio whose log is naturally more spread out, so the model doesn't have to overshoot?

I can settle the "how concentrated" part by just simulating it, instead of asserting "concentrated" and moving on. Take two independent probabilities from a flat prior, X_1, X_2 ~ Unif(0,1), as stand-ins for P(y_w) and P(y_l) before the model has any opinion. The log probability ratio is log PR = log X_1 − log X_2. Compare it to the log of the *odds* ratio, where odds(P) = P/(1−P) and log-odds is the logit, log(P/(1−P)): log OR = logit(X_1) − logit(X_2). Same uniform inputs, two different transforms. Draw two million samples of each and look at the spread:

```
logPR    std=1.413   IQR=1.387   P(|.|<1)=0.632   P(|.|>3)=0.050   [p1,p99]=[-3.91, 3.90]
logOR    std=2.564   IQR=3.266   P(|.|<1)=0.323   P(|.|>3)=0.226   [p1,p99]=[-6.27, 6.26]
```

That's the check, and it comes out clearly on the side of the worry. The log probability ratio really is concentrated: 63% of its mass sits inside |log PR| < 1, and only 5% past |.| > 3. The log odds ratio, from the *same* input probabilities, is about 1.8× wider in standard deviation and 2.35× wider in interquartile range — only 32% of its mass inside ±1, and 23% past ±3, with the 1st/99th percentiles roughly ±6 versus ±4. The reason is structural, and the simulation matches it: log X piles up near 0 because X near 1 gives log X near 0, so the difference of two log-probs concentrates; but the logit explodes toward ±∞ as P approaches 1 or 0 (the log(1−P) piece blows up near P=1), so the difference of two logits is stretched out.

Now I can turn the worry into a design choice rather than a feeling. With the wide-ranging odds ratio, a given target sigmoid output is reached by a *modest* per-example margin, because |log OR| is already large across the data — the model doesn't have to overshoot any single example. With the tightly-concentrated probability ratio, the only way to move the loss is to force each example to an *extreme* margin, which is exactly the rejected-logit-crushing I feared. So the odds ratio gives a *mild* discrimination of the rejected response, the probability ratio a *harsh* one — and during fused SFT, mild is what keeps me out of degeneration. So I'll build the contrast on the odds ratio, not as an arbitrary pick but because its natural scale matches "penalize the disfavored style a little, adapt to the favored style a lot."

Let me pin down the objects cleanly. For a response y of length m, I take the length-normalized sequence log-probability,

  log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t | x, y_<t),

the geometric mean of the per-token probabilities. I want it length-normalized for two reasons: it keeps P_θ(y|x) genuinely in (0,1) so the odds P/(1−P) is well-defined and finite, and it makes responses of different lengths comparable instead of letting a long rejected response be "unlikely" merely by being long. The odds is

  odds_θ(y|x) = P_θ(y|x) / (1 − P_θ(y|x)),

read as "how many times more likely the model is to generate y than not." And the odds ratio of chosen over rejected,

  OR_θ(y_w, y_l) = odds_θ(y_w|x) / odds_θ(y_l|x).

The penalty term wraps its log in a negative log-sigmoid, so that driving the loss down *is* driving the log odds ratio up:

  L_OR = −log σ( log[ odds_θ(y_w|x) / odds_θ(y_l|x) ] ).

And the full single-stage objective, expectation over the dataset of pairs:

  L_ORPO = E_{(x,y_w,y_l)} [ L_SFT + λ · L_OR ].

The L_SFT term is the ordinary NLL on the chosen response — it's doing the domain adaptation, increasing the likelihood of the reference tokens, exactly as before. The L_OR term is the new penalty, contrasting the two styles. λ trades the two off. And I notice what's *not* in here: no π_ref. Nowhere does this objective reference a frozen model. The contrast is between y_w and y_l under the *same, current* parameters θ, and the "don't generate y_l" pressure comes from comparing each response probability with its own complement, 1 − P. One model, one stage; the preference signal is internal to the current policy.

I should be honest about λ before I move on, because it controls how aggressively the penalty can fight the NLL. If λ is too small, the extra term is almost decorative and SFT keeps pulling both styles upward. If λ is too large, the contrast can dominate the domain-adaptation signal and I am back in the harsh-suppression regime I wanted to avoid. So λ is genuinely a knob, not a free win; it should start small, matching the idea that a minor penalty for the disfavored style is enough, and only move upward when the data actually needs a sharper separation.

Now I need to convince myself this loss does the right thing dynamically, not just at the level of "it looks like a contrast." The way to do that is the gradient — if I differentiate L_OR and keep the signs straight, I'll trust it. Let u = log g, with g = odds_θ(y_w|x)/odds_θ(y_l|x). I first compute the derivative of the thing I want to increase, log σ(u), because the loss is its negative. Using σ' = σ(1−σ),

  ∇_θ log σ(log g) = [σ'(log g)/σ(log g)] · ∇_θ log g = (1 − σ(log g)) · ∇_θ log g = σ(−log g) · ∇_θ log g.

And σ(−log g) = 1/(1 + e^{log g}) = 1/(1 + g) = [1 + odds_w/odds_l]^{−1}. Call that δ(d):

  δ(d) = [ 1 + odds_θ(y_w|x)/odds_θ(y_l|x) ]^{−1}.

Sweeping g across orders of magnitude shows the range this weight covers:

```
g=1e-3   delta=0.99900
g=0.1    delta=0.90909
g=1.0    delta=0.50000
g=10     delta=0.09091
g=1e+3   delta=0.00100
```

When the model already strongly prefers the chosen response, odds_w ≫ odds_l, so g is huge and δ → 0 (the g=1e3 row) — this example stops pushing. When the model is getting it wrong, preferring the rejected, odds_w < odds_l, g < 1 and δ > 1/2, approaching 1 as g approaches 0 (the g=1e−3 row) — the update fires strongly. So δ is an automatic difficulty weight: it accelerates the parameter update on examples where the model is currently more likely to generate the rejected response, and goes quiet once the example is solved. That self-pacing falls out of the log-sigmoid wrapping; I didn't have to design it separately.

Now the direction, ∇_θ log g. Since g is a ratio of odds, log g = log odds_w − log odds_l, and log odds(y) = log P(y) − log(1 − P(y)), so

  ∇_θ log g = [∇_θ log P(y_w) − ∇_θ log(1−P(y_w))] − [∇_θ log P(y_l) − ∇_θ log(1−P(y_l))].

The awkward pieces are the ∇_θ log(1 − P(y)) terms; let me simplify one. Write it out by the chain rule:

  ∇_θ log(1 − P(y)) = ∇_θ(1 − P(y)) / (1 − P(y)) = −∇_θ P(y) / (1 − P(y)).

I'd rather have everything in terms of ∇_θ log P(y), the quantity backprop actually gives me, so use ∇_θ P = P · ∇_θ log P:

  ∇_θ log(1 − P(y)) = −P(y)/(1 − P(y)) · ∇_θ log P(y) = −odds_θ(y) · ∇_θ log P(y),

recognizing P/(1−P) = odds. So the bracket for each response collapses:

  ∇_θ log odds(y) = ∇_θ log P(y) − ∇_θ log(1−P(y)) = ∇_θ log P(y) + odds(y)·∇_θ log P(y) = (1 + odds(y))·∇_θ log P(y).

And 1 + odds(y) = 1 + P/(1−P) = (1−P+P)/(1−P) = 1/(1−P). So

  ∇_θ log odds(y) = ∇_θ log P(y) / (1 − P(y)).

That last step folded two non-obvious identities together — ∇log(1−P) = −odds·∇logP, and the collapse to 1/(1−P). Finite-differencing both sides confirms it: model P as a sigmoid of a scalar parameter θ, at θ = 0.7 where P = 0.6682, odds = 2.0138:

```
grad log(1-P)        = -0.66819      -odds * grad logP = -0.66819   (claim: equal)  ✓
grad log odds        =  1.00000      grad logP / (1-P) =  1.00000   (claim: equal)  ✓
```

Both identities check out to five-plus digits, so the algebra is sound and I can use ∇_θ log odds(y) = ∇_θ log P(y)/(1 − P(y)).

Putting it together, ∇_θ log g = ∇_θ log P(y_w)/(1−P(y_w)) − ∇_θ log P(y_l)/(1−P(y_l)). Call that h(d). The sign is the part I cannot blur:

  ∇_θ log σ(log g) = δ(d) · h(d),
  ∇_θ L_OR = ∇_θ[−log σ(log g)] = −δ(d) · h(d).

So the descent step moves in the +δ(d)h(d) direction, which raises log g. The structure of h(d) is a contrast: a +∇log P(y_w) component for the chosen response and a −∇log P(y_l) component for the rejected one — exactly the discrimination SFT could not do — and the 1/(1−P(y)) factors are the log-odds sensitivity, near 1 when P is small and growing as P approaches 1, so a rejected response that has become too plausible receives a sharper negative push. That's what I want it to do; let me confirm the actual sign on a point where the model is *wrong*. Take chosen and rejected each driven by a scalar logit, P_w = σ(0.3), P_l = σ(0.8) (so P_l > P_w, the model currently prefers the rejected response), and finite-difference L_OR with respect to each logit:

```
dL_OR/d(chosen logit)   = -0.6225   (negative: gradient descent raises the chosen logit)   ✓
dL_OR/d(rejected logit) = +0.6225   (positive: gradient descent lowers the rejected logit) ✓
delta at this point = 1/(1+g) = 0.6225,  g = 0.6065
```

Both signs come out the way the contrast demands — chosen up, rejected down — and the magnitude is exactly δ(d) = 0.6225 for this point, which is δ pulling harder than 1/2 precisely because the model is on the wrong side.

The gradient also settles the reference-free claim, which is where the compute savings live: nothing in δ(d) or h(d) involves a second model — they're all functions of the current θ's probabilities on y_w and y_l and their complements. So I genuinely need only one model in memory, and per batch only two forward passes — y_w and y_l through the single policy — versus DPO's four. No SFT warm-up either, since L_SFT is right there in the objective doing the adaptation from the pre-trained checkpoint. The efficiency isn't a separate trick; it's a consequence of making the chosen/rejected contrast internal to the current policy.

Now let me get the loss into the actual shape the training code wants, because there are numerical traps. The harness hands me, per response, a sum of token log-probs and a valid length; I want the *length-normalized* log-prob, so I divide the summed log-probs by the valid length to get c = log P_θ(y_w|x) and r = log P_θ(y_l|x), each a mean-per-token log-prob, hence ≤ 0 and corresponding to P ∈ (0,1). The log odds ratio I need is

  log OR = log odds_w − log odds_l = [log P_w − log(1−P_w)] − [log P_l − log(1−P_l)]
         = (c − r) − [ log(1 − e^{c}) − log(1 − e^{r}) ],

since P = e^{c} and 1 − P = 1 − e^{c}. The term log(1 − e^{c}) is exactly the kind of thing that underflows or hits log(0) if computed naively — when c is near 0 (P near 1), 1 − e^{c} is a tiny positive number, and when c is very negative, e^{c} underflows. The stable primitive for log(1 − e^{c}) with c ≤ 0 is log1p(−exp(c)) — compute exp(c) (safe, it's in (0,1)), negate, and use log1p which is accurate for arguments near 0. So:

  log_odds = (c − r) − ( log1p(−exp(c)) − log1p(−exp(r)) ).

Let me trace this on a concrete pair to make sure the log1p path reproduces the math definition, and to see the danger it's avoiding. Take c = −0.5 (P_w = 0.6065) and r = −1.2 (P_l = 0.3012):

```
log_odds via log1p path = 1.27437
log_odds via direct math = (log P_w - log(1-P_w)) - (log P_l - log(1-P_l)) = 1.27437   ✓
```

Same value to full precision. And to see why the stable primitive matters: at c = −1e−7 (P_w ≈ 1), the naive 1 − exp(c) evaluates to 9.9999995e−08 — a near-cancellation that a direct log would handle badly, while log1p(−exp(c)) returns −16.118 cleanly. Then the penalty is L_OR = −log σ(log_odds), and the stable primitive for −log σ(·) is logsigmoid, built not to overflow for large-magnitude inputs. The SFT term is just the negative length-normalized chosen log-prob, L_SFT = −c. Assembling the per-example loss on that same c, r pair with λ = 0.1: L_SFT = 0.5, L_OR = −log σ(1.27437) = 0.2466, so loss = 0.5 + 0.1·0.2466 = 0.5247 — a small finite number, dominated by the SFT term, with the penalty contributing a modest nudge, which is exactly the balance I argued λ small should give. The full per-example loss is

  loss = L_SFT + λ · L_OR = −c + λ · ( −logsigmoid(log_odds) ).

λ is the weight I called λ in the objective — the single knob, default small. Wiring that into the harness, the empty pair-loss slot gets filled in the same shape as the practical trainer: average the per-response log-probs for this loss type, compute the odds-ratio loss, and return the chosen/rejected reward metrics from the current policy only.

```python
import torch
import torch.nn.functional as F


def get_batch_logps(logits, labels):
    """Per-response SUMMED label log-probs and valid (non-pad) lengths. Provided by harness."""
    # gather log P(label_t | x, y_<t) over valid positions, sum per response
    return summed_logps, valid_length  # both shape (batch,)


class PreferenceTrainer:
    def __init__(self, model, beta):
        self.model = model
        self.beta = beta  # lambda: weight on the odds-ratio penalty (small, e.g. 0.1)

    def concatenated_forward(self, batch):
        labels = batch.pop("labels")
        logits = self.model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        summed_logps, valid_length = get_batch_logps(logits, labels)
        # ORPO uses average log-probs: log of the geometric-mean token probability.
        all_logps = summed_logps / valid_length
        bsz = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(bsz, dim=0)
        chosen_logits, rejected_logits = logits.split(bsz, dim=0)
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps,
        }

    def odds_ratio_loss(self, chosen_logps, rejected_logps):
        # log OR = (c - r) - [log(1-P_w) - log(1-P_l)].
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        return sft_loss + self.beta * odds_ratio_loss

    def compute_preference_loss(self, policy_chosen_logps, policy_rejected_logps):
        losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(self, batch):
        output = self.concatenated_forward(batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            output["chosen_logps"], output["rejected_logps"]
        )
        sft_loss = -output["chosen_logps_avg"]
        metrics = {
            "sft_loss": sft_loss.mean().item(),
            "odds_ratio_loss": ((losses - sft_loss) / self.beta).mean().item(),
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
        }
        return losses.mean(), metrics
```
