# Prompt Tuning (The Power of Scale for Parameter-Efficient Prompt Tuning)

## Problem

Specialize one frozen pre-trained text-to-text Transformer (T5) to many downstream tasks by
learning only a tiny task-specific signal at the input, with no change to the model's weights, and
close the accuracy gap to full fine-tuning ("model tuning"). Full fine-tuning stores a complete
model copy per task (prohibitive at 11B parameters); discrete prompt design is search-bound and caps
each slot at a real word's embedding; prefix-tuning tunes activations at every layer.

## Key idea

Prepend a **soft prompt** — a block of free, trainable embeddings P_e ∈ R^{p×e} — to the embedded
input X_e ∈ R^{n×e}, run [P_e; X_e] through the frozen encoder-decoder as usual, and train **only**
P_e (the model weights θ stay frozen):

  maximize Pr_{θ; θ_P}(Y | [P; X]),  updating only P_e.

This tunes only the *input layer* — no per-layer activations, no reparametrization MLP, no inserted
modules. Parameter cost is exactly E·P (embedding dim × prompt length): <0.01% of the model per task
for billion-parameter models. The central finding is the **power of scale**: the larger the frozen
model, the more competitive this minimal intervention becomes, until at XXL (11B) it matches even
the strong multi-task model-tuning baseline despite >20,000× fewer task-specific parameters.

Design choices (each matters at small/mid scale, then washes out at XXL):
- **LM-adapt the base model first.** T5.1.1 is pre-trained only on span corruption (targets begin
  with sentinel tokens), so its decoder has a strong sentinel prior that a frozen prompt cannot
  override — mid-sized models often fail to emit a legal class label (0% accuracy; copying input
  spans or empty strings). Continue T5's self-supervised training under the LM objective (natural
  prefix → natural continuation) for ~100K steps *once*, yielding a reusable frozen base that writes
  natural text. Merely prepending a sentinel to targets barely helps.
- **Initialization.** random uniform [−0.5, 0.5] < sampled common-vocabulary embeddings < class-label
  embeddings (verbalizer-style; average multi-token labels; fall back to sampled vocab for extra
  slots). Class-label init primes the model toward legal outputs and is best — but the gap vanishes
  at XXL.
- **Prompt length.** Cost is E·P, so pick the minimal length that performs: 1→20 tokens gives a large
  boost, >20 is marginal, >100 is mildly detrimental for large models; XXL does well even with a
  single-token prompt.

Free consequences: one batch can mix different prompts against the same frozen backbone (multi-task
serving), and several prompts for one task form a cheap ensemble. Optimizer: Adafactor, constant
lr 0.3, batch 32, ~30K steps, weight decay 1e-5, β₂ decay 0.8, parameter scaling off; cross-entropy;
early stopping on dev. Evaluated on SuperGLUE across T5 sizes Small→XXL.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPrompt(nn.Module):
    """Trainable virtual-token embeddings prepended to the input. Cost = length * E."""
    def __init__(self, length, embed_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.empty(length, embed_dim))

    @torch.no_grad()
    def init_from(self, strategy, frozen_embed, class_label_ids=None):
        if strategy == "random":
            self.prompt.uniform_(-0.5, 0.5)
        elif strategy == "sampled_vocab":
            idx = torch.randint(0, 5000, (self.prompt.size(0),))      # common tokens
            self.prompt.copy_(frozen_embed(idx))
        elif strategy == "class_label":
            rows = [frozen_embed(ids).mean(0) for ids in class_label_ids]
            while len(rows) < self.prompt.size(0):                    # fall back to vocab
                rows.append(frozen_embed(torch.randint(0, 5000, (1,))).squeeze(0))
            self.prompt.copy_(torch.stack(rows[: self.prompt.size(0)]))

    def forward(self, x_e):
        p = self.prompt.unsqueeze(0).expand(x_e.size(0), -1, -1)
        return torch.cat([p, x_e], dim=1)                            # [P_e ; X_e]


def loss_fn(frozen_t5, soft_prompt, input_ids, target_ids):
    x_e = frozen_t5.embed(input_ids)
    inputs_embeds = soft_prompt(x_e)
    logits = frozen_t5(inputs_embeds=inputs_embeds, decoder_target=target_ids).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))


def train(frozen_t5, soft_prompt, loader):
    for p in frozen_t5.parameters():
        p.requires_grad = False                                       # frozen, LM-adapted base
    optim = Adafactor(soft_prompt.parameters(), lr=0.3, weight_decay=1e-5,
                      beta2_decay=0.8, parameter_scaling=False)
    for batch in loader:                                              # batch 32, ~30k steps
        loss = loss_fn(frozen_t5, soft_prompt, batch["input_ids"], batch["target_ids"])
        loss.backward(); optim.step(); optim.zero_grad()
```
