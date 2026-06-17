# Context: single-stage preference alignment of language models (circa 2023-2024)

## Research question

A pre-trained language model is not directly usable as an assistant. It must be tuned
twice: first **supervised fine-tuning (SFT)** on demonstration data to teach the target
domain and format, then **preference alignment** on pairwise data `(x, y_w, y_l)` — an
input `x` with a chosen response `y_w` and a rejected response `y_l` — to teach it which
of two plausible answers humans prefer. The dominant alignment recipes (RLHF, DPO) share
two costs that this two-stage pipeline forces on you:

1. **A separate frozen reference model.** Both RLHF and DPO measure how far the policy has
   moved from the SFT checkpoint, so they keep a second, frozen copy of the model in memory
   and run extra forward passes through it every step. For a multi-billion-parameter model
   that is a large, unavoidable memory and FLOP tax.
2. **A mandatory SFT warm-up phase.** Preference alignment is run *on top of* an
   already-SFT'd model; the reference model *is* that SFT checkpoint. So you pay for two
   training runs, not one.

The precise goal is a single loss, applied in **one** stage directly to the pre-trained
model, that simultaneously (1) does the domain adaptation that SFT does, (2) teaches the
chosen/rejected preference, and (3) needs **no reference model and no separate warm-up** —
so it is cheaper in both memory and compute than the two-stage methods, while at least
matching their alignment quality.

## Background

**The SFT loss and what it cannot do.** SFT minimizes the causal-LM negative
log-likelihood (NLL / cross-entropy) of the chosen response. For one example of length `m`
over vocabulary `V`,

```
L = -(1/m) Σ_{k=1}^{m} log P(x, y^{(k)})
  = -(1/m) Σ_k Σ_{i=1}^{|V|} y_i^{(k)} · log p_i^{(k)}
```

where `y_i^{(k)}` is 1 iff token `i` is the label at position `k` and `p_i^{(k)}` is the
model's probability for token `i`. The structural fact about this loss: the inner sum only
touches the *label* token (the term survives only where `y_i = 1`); for every non-label
token `y_i = 0`, so the loss neither rewards nor **penalizes** it. Cross-entropy is an
excellent domain-adaptation signal but has no mechanism to push *down* the probability of
any specific undesirable continuation.

**The diagnostic phenomenon that motivates everything.** A pilot observation about plain
SFT: fine-tune a model (OPT-350M) on the *chosen responses only* of a preference dataset
(HH-RLHF) and track, batch by batch, the log-probability of the held-out *rejected*
responses. The log-probability of the rejected responses rises
**together** with that of the chosen responses — sometimes the rejected response ends up
*more* likely than the chosen one. SFT moves the whole neighbourhood of the target domain
up, including the disfavored styles, exactly because the NLL has no penalty term. This is
the gap a single-stage alignment loss must close.

**Unlikelihood training.** Prior work on degeneration (Welleck et al. 2019; Li et al. 2020)
showed you can suppress unwanted continuations by *appending* a penalty: for an unwanted
token set `C_recent` (e.g. recently emitted tokens, to stop repetition), add a term that
penalizes the model for assigning them high probability, of the form `Σ log(1 - p_i)` over
the unwanted tokens. The lesson carried forward is that a penalty *added to* the NLL,
rather than a replacement for it, can curb undesired generation while leaving the main
likelihood objective intact — but it required hand-crafting the unwanted-token set.

**Sequence likelihoods for pairwise data.** A preference trainer normally reduces the
token-level causal-LM log-probabilities of each response to a scalar before comparing the
chosen and rejected responses. The raw sum favors shorter continuations mechanically, while
an average per valid token is comparable across different response lengths. Any single-stage
preference loss has to decide how to reduce these token log-probabilities before forming
the chosen-vs-rejected contrast.

## Baselines

**RLHF with a reward model (Ziegler et al. 2019; Stiennon et al. 2020; Ouyang et al. 2022).**
Fit a reward model `r(x, y)` from pairwise data under the Bradley-Terry model
(maximize `log σ(r(x, y_w) − r(x, y_l))`), then optimize the policy with PPO to maximize the
reward, regularized by a KL penalty to the SFT reference. It works at scale but is
unstable: PPO is sensitive to many hyperparameters, the reward model can be exploited
(reward over-optimization), and the pipeline carries a reward model *and* a reference
policy. The cost it pays is the full apparatus of online RL plus two auxiliary models.

**DPO — Direct Preference Optimization (Rafailov et al. 2023).** Removes the explicit
reward model and PPO loop by reparametrizing the optimal RLHF reward as a log-ratio against
the reference policy, `r(x,y) = β · log( π_θ(y|x) / π_ref(y|x) ) + const`, which turns the
Bradley-Terry objective into a direct classification loss on the pairs:

```
L_DPO = −log σ( β [ (log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )
```

It is far more stable than PPO, but it still uses a **probability ratio relative to a frozen
reference model** `π_ref`, so it inherits both two-stage costs: it needs an SFT warm-up to
produce `π_ref`, and it keeps `π_ref` in memory and runs extra forward passes through it.
Per batch, DPO requires four forward passes (chosen and rejected, through both the policy
and the reference).

**IPO — Identity Preference Optimization (Azar et al. 2023).** Diagnoses that DPO's
log-sigmoid objective can drive the policy to a near-deterministic preference and overfit
when the empirical preference is clean; replaces the log-sigmoid with a squared loss that
targets a finite margin, `( h − 1/(2β) )²` where `h` is the same reference-relative
log-ratio difference. It regularizes DPO's tendency to push the margin to infinity, but it
is still reference-based and two-stage.

Across these, the common pattern of the prior art is a **probability-ratio** contrast
between chosen and rejected, applied either against a frozen reference or as its own stage.
Where they leave room: the contrast is built to run *after* domain adaptation, and the
probability ratio's behavior, when used during SFT on a not-yet-adapted model, is not
characterized.

## Evaluation settings

- **Preference / instruction datasets**: Anthropic HH-RLHF and binarized UltraFeedback for
  the alignment studies, filtering out pairs with `y_w = y_l` or empty responses.
- **Models**: a scale ladder of base/pre-trained models (OPT 125M–1.3B for controlled
  comparisons; Phi-2 2.7B, Llama-2 7B, Mistral 7B for leaderboard runs).
- **Metrics / protocol**: for general alignment, win rate judged by a held-out reward model,
  and instruction-following leaderboards (AlpacaEval 1.0/2.0, MT-Bench, IFEval), plus lexical
  diversity (per-input and across-input cosine similarity of sampled generations).
  Optimization with AdamW, cosine LR schedule with warm-up; training with DeepSpeed ZeRO /
  FSDP.

## Code framework

The training harness already exists: a Trainer that, for each batch of concatenated chosen
and rejected sequences, runs one forward pass to get per-token logits, gathers the
log-probabilities of the label tokens, and reduces them to a per-response scalar. The only
empty slot is the preference loss that maps those scalars to a training loss.

```python
import torch
import torch.nn.functional as F


def get_batch_logps(logits, labels):
    """Per-response log-probabilities and valid (non-pad) lengths. Already provided."""
    # ... gather log P(label_t | x, y_<t) over valid positions ...
    return summed_logps, valid_length  # both shape (batch,)


class PreferenceTrainer:
    """Existing harness: shared encoder forward, then a pluggable preference loss."""

    def __init__(self, model, beta):
        self.model = model
        self.beta = beta  # weight on the preference term

    def concatenated_forward(self, batch):
        labels = batch.pop("labels")
        logits = self.model(**batch).logits.float()
        summed_logps, valid_length = get_batch_logps(logits, labels)
        # TODO: reduce token log-probabilities to one scalar per response.
        all_logps = self.reduce_logps(summed_logps, valid_length)
        bsz = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(bsz, dim=0)
        return chosen_logps, rejected_logps

    def reduce_logps(self, summed_logps, valid_length):
        # TODO: choose the response-level reduction.
        pass

    def pair_loss(self, chosen_logps, rejected_logps):
        # TODO: design the chosen/rejected contrast for this batch.
        pass

    def get_batch_loss_metrics(self, batch):
        chosen_logps, rejected_logps = self.concatenated_forward(batch)
        losses = self.pair_loss(chosen_logps, rejected_logps)
        return losses.mean()
```
