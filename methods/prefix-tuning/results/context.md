# Context: Steering a Frozen Language Model for Conditional Generation

## Research question

A large pre-trained autoregressive Transformer LM (GPT-2) or encoder-decoder (BART) can be adapted
to a conditional-generation task — table-to-text, summarization — by full fine-tuning, which
updates *all* of the model's parameters and stores a complete model copy per task. When one wants to
serve many tasks (or many users), that per-task cost is the bottleneck: a fresh several-hundred-
million-parameter checkpoint for every task, with no sharing.

The precise problem: **adapt one frozen pre-trained generation model to a downstream task by
learning only a tiny task-specific increment, while keeping accuracy close to full fine-tuning.** A
good solution should store a small per-task object (so many tasks share one frozen backbone),
support a different object per example in a batch (so requests for different tasks/users can be
batched against the same backbone), and not require touching the pre-trained weights at all.

## Background

**Conditional generation with Transformers.** For input context x and output sequence y, an
autoregressive LM models p_φ(y | x) over the concatenation z = [x; y]. At each position i the model
produces an activation h_i = LM_φ(z_i, h_{<i}); the top layer's activation is mapped by a fixed
matrix to a softmax over the vocabulary. Crucially, h_i can be viewed as the stack of all layers'
activations at position i. In a self-attention Transformer, each layer's activation at position i is
computed by attending to left-context keys and values from that layer, so the cached state needed by
later tokens is a per-layer key/value pair. Information thus flows *rightward* (to later positions,
through attention) and *upward* (to higher layers). An encoder-decoder works the same way: a
bidirectional encoder computes activations for x, and an autoregressive decoder produces y
conditioned on encoded x and its own left context.

**Full fine-tuning.** Initialize from φ and maximize Σ_{i∈y} log p_φ(z_i | h_{<i}) by gradient
updates on *all* of φ. The accuracy target, but |delta| = |φ| per task.

**Prompting / in-context steering.** A striking observation: a frozen LM can be *steered* by its
context without changing any weights. Prepend the right tokens and the next-token distribution
shifts toward the desired continuation — prepend "Barack" and the LM raises the probability of
"Obama". This suggests that for a whole task there might exist a context that steers the LM to solve
it. But two problems: (1) natural-language task instructions ("summarize the following table in one
sentence") mostly *fail* to steer pre-trained LMs (GPT-2 and BART do not follow them; only the very
largest models do); and (2) searching for a better instruction means optimizing over *discrete*
tokens, which is combinatorially hard. A discrete prompt is also fundamentally limited: every prompt
position must be the embedding of an actual vocabulary word.

**AutoPrompt / discrete prompt search** (Shin et al. 2020) optimizes a discrete trigger sequence by
gradient-guided search over the vocabulary. It removes hand-design but stays inside the discrete
constraint: each slot must equal a real word's embedding, capping expressiveness.

**Adapters** (Houlsby et al. 2019) take a different route to parameter efficiency — insert small
trainable bottleneck modules between the frozen Transformer's layers and tune only those (plus layer
norms). They show ~3% of parameters per task can nearly match fine-tuning. But adapters *add layers
into* the network (extra depth/compute in the backbone) and modify the model's internal computation
path; they do not exploit the "steer-by-context" mechanism.

**Lightweight fine-tuning precedents** (e.g. tuning only top layers, or only bias/normalization
terms) reduce the per-task footprint but generally trade away accuracy.

## Baselines

- **Full fine-tuning** of GPT-2 / BART. Tune all of φ. Accuracy target; gap: |delta| = |φ| per task,
  no sharing, no per-example task batching.
- **FT-top-k** (fine-tune only the top k layers). Smaller footprint; gap: accuracy drops as k
  shrinks — a coarse accuracy/size knob.
- **Adapter tuning** (Houlsby et al. 2019). Frozen backbone + small bottleneck modules + layer-norm
  params, ~2–4% params/task. Gap: inserts modules into the backbone (extra depth), changes the
  internal compute path rather than steering via context.
- **Discrete prompting / AutoPrompt** (Shin et al. 2020). A frozen LM steered by an optimized
  discrete token sequence. Gap: discrete optimization is hard, and each slot is constrained to a
  real word's embedding — limited expressiveness, and instructions mostly fail to steer GPT-2/BART.

## Evaluation settings

- **Table-to-text**: E2E (~50K examples, 1 domain), WebNLG (~22K, 14 domains, with a held-out set of
  unseen categories for extrapolation), DART (~82K, open-domain). Inputs are linearized tables /
  (subject, property, object) triples. Metrics: BLEU, NIST, METEOR, ROUGE-L, CIDEr (E2E); BLEU,
  METEOR, TER (WebNLG); plus MoverScore, BERTScore, BLEURT (DART). Base models: GPT-2 MEDIUM/LARGE,
  tables linearized.
- **Summarization**: XSUM (~225K news articles; avg article 431 words, avg summary 23.3). Metrics:
  ROUGE-1/2/L. Base model: BART LARGE, source articles truncated to 512 BPE tokens.
- **Low-data and extrapolation** settings: subsampled training sets; WebNLG's unseen categories.
- **Optimization/decoding protocol**: AdamW, linear LR schedule; default 10 epochs, batch size 5,
  lr 5e-5. Beam search at decoding (beam 5 for table-to-text; beam 6 + length-normalization 0.8 for
  summarization).
- Implementation built on the Hugging Face Transformers library.

## Code framework

The primitives that already exist: a pre-trained autoregressive/encoder-decoder Transformer with the
standard attention/caching interface of the Hugging Face Transformers library, AdamW, and the
standard token-level log-likelihood loss. The slot to fill is *what task-specific object to learn*
and *how to feed it into the frozen model*.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# A frozen pre-trained generation model (Hugging Face Transformers).
frozen_lm = load_pretrained_lm()           # all parameters frozen
for p in frozen_lm.parameters():
    p.requires_grad = False

n_layers = frozen_lm.config.n_layer
d_model  = frozen_lm.config.n_embd
n_head   = frozen_lm.config.n_head


class TaskParameters(nn.Module):
    """The small per-task object to be designed."""
    def __init__(self):
        super().__init__()
        # TODO: what trainable object steers the frozen LM toward this task,
        #       and in what form must it enter the model's activations?
        pass

    def materialize(self, batch_size):
        # TODO: produce whatever form the frozen LM needs to consume this object
        pass


def loss_fn(frozen_lm, task_params, x_ids, y_ids):
    # TODO: condition the frozen LM on the task object + x, score y autoregressively
    #       (objective unchanged from fine-tuning; only the trainable set changes)
    pass


def train(frozen_lm, task_params, loader, opt):
    optim = torch.optim.AdamW(task_params.parameters(), lr=opt.lr)   # only task params
    for batch in loader:
        loss = loss_fn(frozen_lm, task_params, batch["x_ids"], batch["y_ids"])
        loss.backward(); optim.step(); optim.zero_grad()
```
