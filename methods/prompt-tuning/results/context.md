# Context: Conditioning a Frozen Text-to-Text Model with a Tiny Input Signal

## Research question

Adapting a large pre-trained Transformer to a downstream task by full fine-tuning ("model tuning")
updates *all* of the model's weights and stores a separate copy of the entire model per task. For an
11-billion-parameter model served across many tasks, that per-task cost dominates everything. The
question: **can one frozen pre-trained model be specialized to many tasks by learning only a tiny
task-specific signal at the input, with no change to the model's weights — and how good can such a
method get?** A central sub-question is how any such approach depends on model *scale*.

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
search (Jiang et al. 2020; AutoPrompt, Shin et al. 2020). GPT-3 showed large frozen models are
strong "few-shot learners" via such prompts.

**Prefix-tuning** (Li & Liang 2021) makes the prompt continuous and learnable, prepending trainable
activation prefixes at *every* transformer layer, with an MLP reparametrization for stable
optimization.

**Adapters** (Houlsby et al. 2019) insert small trainable bottleneck modules between frozen
Transformer layers.

**T5's pre-training objective.** T5.1.1 is pre-trained *only* on span corruption:
masked spans in the input are replaced by unique sentinel tokens, and the target is the masked
content delimited by those sentinels (every target *begins* with a sentinel). Such a model has never
seen fully natural input text nor produced fully natural targets — its decoder has a strong prior
toward emitting sentinels. Full fine-tuning can move the decoder weights; an approach that leaves
the weights frozen cannot.

**Motivating observations about scale and base model.** Mid-sized T5 models prompted off-the-shelf
on span corruption produce unexpected outputs on many tasks, including copying sub-spans of the
input or predicting the empty string, scoring 0%. GPT-3 (a left-to-right LM that always outputs
natural text) responds far better to prompts.

## Baselines

- **Model tuning (full fine-tuning)** of T5.1.1, per task. Accuracy target; default lr 1e-3,
  Adafactor with restored pre-training parameter states.
- **Model tuning, multi-task.** One model tuned on all tasks jointly with a task-name prefix.
- **Prompt design / GPT-3 few-shot** (Brown et al. 2020). Frozen model conditioned on a
  hand-built discrete prompt.
- **Prefix-tuning** (Li & Liang 2021). Learnable continuous activations at *every* layer plus MLP
  reparametrization.
- **Discrete prompt search / AutoPrompt** (Shin et al. 2020). Gradient-guided discrete token
  search.

## Evaluation settings

- **SuperGLUE** (Wang et al. 2019): eight English language-understanding tasks (BoolQ, CB, COPA,
  MultiRC, ReCoRD, RTE, WiC, WSC), each cast into T5 text-to-text format (omitting the task-name
  prefix). Report each dataset's default dev-set metric (or the average when a dataset has several).
- **Base models**: public T5.1.1 checkpoints at all sizes (Small, Base, Large, XL, XXL = 11B), so
  the method can be studied *as a function of scale*. T5.1.1 differs from original T5 (no supervised
  data in pre-training, adjusted d_model/d_ff, GeGLU instead of ReLU).
- **Frozen base.** The default base is the off-the-shelf span-corruption T5.1.1 checkpoint;
  the harness also allows swapping in alternative once-prepared frozen bases to compare as the
  starting point for prompting.
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
frozen_t5 = load_pretrained_t5()
for p in frozen_t5.parameters():
    p.requires_grad = False
E = frozen_t5.config.d_model                 # token embedding dimension


class TaskInputSignal(nn.Module):
    """The small per-task object, living at the input."""
    def __init__(self, length, embed_dim):
        super().__init__()
        # TODO: what trainable object, and in what space, conditions the frozen model?
        pass

    def init_from(self, strategy, frozen_embeddings, class_label_ids=None):
        # TODO: how should it be initialized so the frozen model is primed for the task?
        pass


def loss_fn(frozen_t5, task_signal, input_ids, target_ids):
    x_e = frozen_t5.embed(input_ids)         # [batch, n, E]
    # TODO: combine the task object with the embedded input, run the frozen model,
    #       score the target with cross-entropy
    pass


def train(frozen_t5, task_signal, loader, opt):
    optim = Adafactor(task_signal.parameters(), lr=opt.lr)   # only the task object trains
    for batch in loader:
        loss = loss_fn(frozen_t5, task_signal, batch["input_ids"], batch["target_ids"])
        loss.backward(); optim.step(); optim.zero_grad()
```
