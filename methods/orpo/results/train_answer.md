What actually costs me when I align a language model is not the loss formula, it is the pipeline. To turn a pre-trained base model into an assistant that prefers the answers people want, I run two trainings back to back: first supervised fine-tuning (SFT) on the chosen responses, to drag the model into the target domain and teach the format, then a preference stage on top of it — RLHF or DPO — that takes the pairs $(x, y_w, y_l)$ and teaches the model to favor $y_w$ over $y_l$. That second stage drags a frozen copy of the SFT model around the whole time, because both RLHF and DPO measure how far the policy has drifted from where SFT left it, so the reference model is literally the SFT checkpoint held fixed. I pay twice: two training runs, and during the second one I hold two full copies of a multi-billion-parameter model in memory and push tokens through both. DPO is four forward passes per batch — chosen and rejected, each through the policy and through the reference. The question that will not leave me alone is whether the reference model is actually load-bearing or just scaffolding I have never questioned. If I could fold the preference signal directly into SFT — one stage, nothing frozen on the side — I would halve the memory and most of the compute.

Before reaching for a clever contrast, I need to understand why SFT alone does not already do the job. Writing out the plain causal-LM negative log-likelihood for a length-$m$ response over vocabulary $V$, $L_{\text{SFT}} = -\frac{1}{m}\sum_k \sum_i y_i^{(k)} \log p_i^{(k)}$, where $y_i^{(k)}$ is the one-hot indicator that token $i$ is the label at position $k$, the inner sum survives only where $y_i = 1$ — the single label token at each position. For every other token, including every token that would build a rejected response, $y_i = 0$ and the term vanishes. Cross-entropy rewards the label tokens and is completely silent about all the others; there is no machinery that pushes any specific continuation down. That one-sidedness has a known consequence: raising the probability of chosen responses in a domain raises the probability of the whole neighborhood of that domain, and rejected responses live in exactly that neighborhood — same topics, same register, often overlapping phrasing. Their log-probability climbs right alongside the chosen ones during SFT and can end up at or above the chosen response's likelihood. So plain SFT does not merely fail to align; as a side effect of domain adaptation it actively makes the disfavored style more generatable. I need a penalty term that runs during SFT and discriminates $y_l$ from $y_w$, because SFT will never produce that contrast by itself. Unlikelihood training showed the right shape — append a term that penalizes the probability of an unwanted token set, of the form $\sum \log(1 - p_i)$, on top of the usual likelihood — but there the unwanted set was hand-crafted. I do not want to craft anything per example: the rejected response $y_l$ is sitting in every pair and is itself the dynamically supplied unwanted set, one per query.

I propose ORPO, Odds Ratio Preference Optimization: a reference-free, single-stage loss that appends a log-odds-ratio penalty to the ordinary SFT negative log-likelihood, so domain adaptation and chosen-vs-rejected preference learning happen in one run with no frozen reference and no separate warm-up. The design hinge is which ratio to contrast with. The obvious move, the one DPO and IPO use, is a probability ratio wrapped in a log-sigmoid. But DPO runs after SFT, against a reference; I am running the contrast from scratch, fused with SFT, on a model that has not yet adapted to the domain, and that regime behaves differently. Minimizing $-\log \sigma(\log R)$ for any ratio $R$ does not just want $R > 1$, it keeps pulling until the typical $\log R$ is well into the saturating tail of the sigmoid, so the scale of the ratio I feed in sets the margin the model will try to force. If $\log R$ is tightly concentrated near zero, the only way to move the loss is to push individual examples hard — and for the probability ratio that means crushing $P_\theta(y_l\mid x)$ toward zero, slamming down the logits of rejected tokens that, on a still-adapting model, overlap heavily with perfectly good tokens it still needs. That is degeneration. I can make the worry quantitative: take $X_1, X_2 \sim \text{Unif}(0,1)$ as stand-ins for the two response probabilities before the model has an opinion. The log probability ratio $\log\text{PR} = \log X_1 - \log X_2$ is a difference of two log-uniforms, each piling up near $0$ with a tail toward $-\infty$, so it is symmetric and concentrated around $0$. The log odds ratio $\log\text{OR} = [\log X_1 - \log(1-X_1)] - [\log X_2 - \log(1-X_2)]$ replaces each $\log X$ with a logit, and the $\log(1-X)$ piece explodes near $X=1$, so each term has heavy spread and the difference of two logits ranges over a far wider interval than $\log\text{PR}$ for the very same inputs. That flips the worry into a choice: with the wide-ranging odds ratio, a given target sigmoid output is reached by a modest per-example margin, so I never have to overshoot any single example; the odds ratio gives a mild discrimination of the rejected response where the probability ratio gives a harsh one, and during fused SFT mild is exactly what keeps me out of degeneration.

Pinning the objects down, for a response $y$ of length $m$ I use the length-normalized sequence log-probability $\log P_\theta(y\mid x) = \frac{1}{m}\sum_t \log P_\theta(y_t\mid x, y_{<t})$, the log of the geometric-mean token probability. Length normalization does double duty: it keeps $P_\theta(y\mid x) \in (0,1)$ so the odds $P/(1-P)$ is finite and well-defined, and it makes responses of different lengths comparable instead of penalizing a long rejected response merely for being long. The odds is $\text{odds}_\theta(y\mid x) = P_\theta(y\mid x)/(1 - P_\theta(y\mid x))$, the odds ratio of chosen over rejected is $\text{OR}_\theta(y_w, y_l) = \text{odds}_\theta(y_w\mid x)/\text{odds}_\theta(y_l\mid x)$, and the penalty wraps its log in a negative log-sigmoid so that driving the loss down is driving the log odds ratio up. The full single-stage objective is

$$L_{\text{ORPO}} = \mathbb{E}_{(x,y_w,y_l)}\big[\, L_{\text{SFT}} + \lambda\, L_{\text{OR}} \,\big], \qquad L_{\text{SFT}} = -\log P_\theta(y_w\mid x), \qquad L_{\text{OR}} = -\log \sigma\!\left( \log \frac{\text{odds}_\theta(y_w\mid x)}{\text{odds}_\theta(y_l\mid x)} \right).$$

The $L_{\text{SFT}}$ term is the ordinary NLL on the chosen response, doing domain adaptation from the pre-trained checkpoint directly; $L_{\text{OR}}$ is the new penalty contrasting the two styles; and crucially no $\pi_{\text{ref}}$ appears anywhere — the contrast is between $y_w$ and $y_l$ under the same current parameters $\theta$, with the "do not generate $y_l$" pressure coming from comparing each response probability with its own complement $1-P$. The weight $\lambda$ is a genuine knob, not a free win: too small and the penalty is decorative while SFT pulls both styles upward; too large and the contrast dominates the adaptation signal and recreates the harsh-suppression regime the odds ratio was meant to avoid. It should start small, e.g. $\lambda = 0.1$, matching the idea that a minor penalty for the disfavored style is enough.

The gradient confirms the loss does the right thing dynamically. With $g = \text{odds}_\theta(y_w\mid x)/\text{odds}_\theta(y_l\mid x)$ and $L_{\text{OR}} = -\log\sigma(\log g)$, using $\sigma' = \sigma(1-\sigma)$ gives $\nabla_\theta \log\sigma(\log g) = (1-\sigma(\log g))\,\nabla_\theta \log g = \sigma(-\log g)\,\nabla_\theta \log g$, and $\sigma(-\log g) = 1/(1+g) = [1 + \text{odds}_\theta(y_w\mid x)/\text{odds}_\theta(y_l\mid x)]^{-1}$, which I call $\delta(d)$. This $\delta$ is an automatic difficulty weight: when the model already strongly prefers the chosen response $g$ is huge and $\delta \to 0$, so the example stops pushing; when the model is getting it wrong, $g < 1$ and $\delta > \tfrac12$, approaching $1$ as $g \to 0$, so the update fires strongly. The direction is $\nabla_\theta \log g = \nabla_\theta \log\text{odds}(y_w) - \nabla_\theta \log\text{odds}(y_l)$, and the awkward complement terms simplify cleanly: $\nabla_\theta \log(1-P(y)) = -\nabla_\theta P(y)/(1-P(y)) = -\text{odds}_\theta(y)\,\nabla_\theta \log P(y)$ using $\nabla_\theta P = P\,\nabla_\theta \log P$, so

$$\nabla_\theta \log\text{odds}(y) = \nabla_\theta \log P(y) - \nabla_\theta \log(1-P(y)) = (1 + \text{odds}_\theta(y))\,\nabla_\theta \log P(y) = \frac{\nabla_\theta \log P(y)}{1 - P(y)},$$

since $1 + P/(1-P) = 1/(1-P)$. Collecting, $\nabla_\theta L_{\text{OR}} = -\delta(d)\,h(d)$ with $h(d) = \nabla_\theta \log P_\theta(y_w\mid x)/(1-P_\theta(y_w\mid x)) - \nabla_\theta \log P_\theta(y_l\mid x)/(1-P_\theta(y_l\mid x))$, so gradient descent moves in the $+\delta(d)\,h(d)$ direction — raising the chosen side, lowering the rejected side, exactly the discrimination SFT could not do. The $1/(1-P)$ factors are the log-odds sensitivity: near $1$ when $P$ is small and growing as $P$ approaches $1$, so a rejected response that has become too plausible receives a sharper negative push. And since $\delta(d)$ and $h(d)$ are functions only of the current $\theta$'s probabilities on $y_w$, $y_l$ and their complements, no second model ever appears — one model in memory, two forward passes per batch, no SFT warm-up.

Getting this into the training code raises two numerical traps. The harness hands me per response a sum of token log-probs and a valid length; dividing gives the length-normalized $c = \log P_\theta(y_w\mid x)$ and $r = \log P_\theta(y_l\mid x)$, each $\le 0$ with $P = e^{c} \in (0,1)$. The log odds ratio is $\log\text{OR} = (c - r) - [\log(1 - e^{c}) - \log(1 - e^{r})]$, and $\log(1 - e^{c})$ is precisely what underflows or hits $\log(0)$ when computed naively — near $c=0$ the argument $1-e^{c}$ is a tiny positive number. The stable primitive for $c \le 0$ is $\text{log1p}(-\exp(c))$: compute $\exp(c) \in (0,1)$, negate, and use $\log1p$, accurate for arguments near $0$. The penalty $-\log\sigma(\cdot)$ uses $\text{logsigmoid}$, built not to overflow for large-magnitude inputs. The SFT term is just $-c$, and the per-example loss is $-c + \lambda\,(-\text{logsigmoid}(\log\text{OR}))$. The code below carries this out, with $\beta$ playing the role of $\lambda$.

```python
import torch
import torch.nn.functional as F


def get_batch_logps(logits, labels):
    """Per-response SUMMED label log-probs and valid (non-pad) lengths."""
    # gather log P(label_t | x, y_<t) over valid positions, sum per response
    return summed_logps, valid_length  # both shape (batch,)


class PreferenceTrainer:
    def __init__(self, model, beta=0.1):
        self.model = model
        self.beta = beta  # lambda: weight on the odds-ratio penalty

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

    def compute_loss(self, batch):
        output = self.concatenated_forward(batch)
        losses, _, _ = self.compute_preference_loss(output["chosen_logps"], output["rejected_logps"])
        return losses.mean()
```
