# Context: Conditioning a Frozen Text-to-Text Model on a Learned Prompt

## Research question

Adapting a large pre-trained Transformer to a downstream task by full fine-tuning ("model tuning")
updates *all* of the model's weights and stores a separate copy of the entire model per task. For an
11-billion-parameter model served across many tasks, that per-task cost dominates everything. The
question: **can one frozen pre-trained model be specialized to many tasks by learning only a tiny
task-specific signal at the input, with no change to the model's weights — and how good can such a
method get?** A solution must (i) leave the pre-trained weights untouched and shared across tasks,
(ii) add a minimal number of new parameters per task, and (iii) ideally close the accuracy gap to
full fine-tuning. A central sub-question is how this trade-off depends on model *scale*.

## Background

**Text-to-text framing.** A unified way to handle every NLP task (T5, Raffel et al. 2020) is to
cast it as conditional text generation: an encoder-decoder Transformer models Pr_θ(Y | X), where X
is the input token sequence and Y is a token sequence representing the answer (including class
labels rendered as words). Classification becomes "generate the label string." The encoder embeds
the n input tokens into X_e ∈ R^{n×e} (e the embedding dimension), and the encoder-decoder produces
Y.

**Prompting / prompt design.** A frozen LM can be conditioned by *prepending* tokens P to the input
so the model maximizes Pr_θ(Y | [P; X]) with θ fixed. In GPT-3-style prompting, P is a sequence of
real vocabulary tokens, so the prompt's representation is drawn from the model's frozen embedding
table — finding a good prompt means *selecting discrete tokens*, by hand or by non-differentiable
search (Jiang et al. 2020; AutoPrompt, Shin et al. 2020). Prompt design is extremely parameter-
efficient (the prompt is just token IDs) and needs no training, but its quality is limited and it
caps each prompt slot at a real word's embedding. GPT-3 showed large frozen models are strong
"few-shot learners" via such prompts — hinting that scale matters for promptability.

**Prefix-tuning** (Li & Liang 2021) makes the prompt continuous and learnable, but goes further:
it prepends trainable activations at *every* layer of the network (overwriting per-layer key/value
activations throughout the stack), with an MLP reparametrization for stable optimization. It works
well, but it tunes parameters across all layers and modifies the model's internal activations, not
just the input.

**Adapters** (Houlsby et al. 2019) insert small trainable bottleneck modules between frozen
Transformer layers — another parameter-efficient route, but one that adds modules into the backbone.

**T5's pre-training objective (a complication).** T5.1.1 is pre-trained *only* on span corruption:
masked spans in the input are replaced by unique sentinel tokens, and the target is the masked
content delimited by those sentinels (every target *begins* with a sentinel). Such a model has never
seen fully natural input text nor produced fully natural targets — its decoder has a strong prior
toward emitting sentinels. This "unnatural" prior is easy to override by fine-tuning (which can move
the decoder weights) but plausibly hard to override by a prompt alone (which cannot adjust decoder
priors). So a span-corruption model may be a poor *frozen* base for prompting.

**Motivating observations about scale and base model.** Mid-sized T5 models prompted off-the-shelf
on span corruption are *unreliable*: on many tasks they never emit a legal class label (scoring 0%),
with the common failure modes being copying sub-spans of the input or predicting the empty string —
and this is consistent across runs, not variance. GPT-3 (a left-to-right LM that always outputs
natural text) responds far better to prompts. This suggests two pre-method facts: prompting works
best on a model whose pre-training taught it to read and write *natural* text, and promptability
improves with scale.

## Baselines

- **Model tuning (full fine-tuning)** of T5.1.1, per task. Accuracy target; gap: stores a full model
  copy per task; no sharing; default lr 1e-3, Adafactor with restored pre-training parameter states.
- **Model tuning, multi-task.** One model tuned on all tasks jointly with a task-name prefix; a
  stronger accuracy baseline. Gap: still a full model, and needs all tasks at once.
- **Prompt design / GPT-3 few-shot** (Brown et al. 2020). Frozen model conditioned on a hand-built
  discrete prompt. Gap: limited quality; discrete slots capped to real words; very long prompts.
- **Prefix-tuning** (Li & Liang 2021). Learnable continuous activations at *every* layer + MLP
  reparametrization. Gap: more task-specific parameters than tuning the input alone; modifies
  internal activations across the stack.
- **Discrete prompt search / AutoPrompt** (Shin et al. 2020). Gradient-guided discrete token search.
  Gap: non-differentiable search; real-word constraint.

## Evaluation settings

- **SuperGLUE** (Wang et al. 2019): eight English language-understanding tasks (BoolQ, CB, COPA,
  MultiRC, ReCoRD, RTE, WiC, WSC), each cast into T5 text-to-text format (omitting the task-name
  prefix). Report each dataset's default dev-set metric (or the average when a dataset has several).
- **Base models**: public T5.1.1 checkpoints at all sizes (Small, Base, Large, XL, XXL = 11B), so
  the method can be studied *as a function of scale*. T5.1.1 differs from original T5 (no supervised
  data in pre-training, adjusted d_model/d_ff, GeGLU instead of ReLU).
- **Pre-training-objective variants** to compare as frozen bases: off-the-shelf span corruption;
  span corruption with a sentinel prepended to downstream targets; and "LM adaptation" (continue T5's
  self-supervised training with the LM objective — natural prefix → natural continuation — for up to
  100K extra steps, *once*, producing a single reusable frozen model).
- **Optimization protocol**: Adafactor, batch size 32, constant learning rate 0.3, 30,000 steps,
  weight decay 1e-5, β₂ decay 0.8, parameter scaling off; cross-entropy loss; early stopping on dev.
  Implemented in JAX/Flax.
- **Comparison axis**: dev accuracy vs. number of task-specific parameters, swept across model size.

## Code framework

The primitives that already exist: a frozen pre-trained text-to-text Transformer with a token
embedding lookup feeding an encoder-decoder, Adafactor, and the standard cross-entropy generation
loss. The slot to fill is the small task-specific object inserted at the input, and how it is
initialized.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# A frozen pre-trained text-to-text (encoder-decoder) model, weights shared across tasks.
frozen_t5 = load_pretrained_t5()            # possibly after a one-time objective adaptation
for p in frozen_t5.parameters():
    p.requires_grad = False
E = frozen_t5.config.d_model                 # token embedding dimension


class SoftPrompt(nn.Module):
    """The small per-task object to be designed, living at the input."""
    def __init__(self, length, embed_dim):
        super().__init__()
        # TODO: what trainable object, and in what space, conditions the frozen model?
        pass

    def init_from(self, strategy, frozen_embeddings, class_label_ids=None):
        # TODO: how should it be initialized so the frozen model is primed for the task?
        pass


def loss_fn(frozen_t5, soft_prompt, input_ids, target_ids):
    x_e = frozen_t5.embed(input_ids)         # [batch, n, E]
    # TODO: combine the task object with the embedded input, run the frozen model,
    #       score the target with cross-entropy
    pass


def train(frozen_t5, soft_prompt, loader, opt):
    optim = Adafactor(soft_prompt.parameters(), lr=opt.lr)   # only the task object trains
    for batch in loader:
        loss = loss_fn(frozen_t5, soft_prompt, batch["input_ids"], batch["target_ids"])
        loss.backward(); optim.step(); optim.zero_grad()
```
