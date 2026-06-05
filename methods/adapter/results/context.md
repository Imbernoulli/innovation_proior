# Context: Parameter-Efficient Transfer Learning for NLP

## Research question

A large pre-trained Transformer (BERT) transfers extremely well to downstream NLP tasks, but the
dominant way to transfer — full fine-tuning — copies the *entire* model's weights and adjusts all
of them for each task. So serving N tasks costs N× the full model: a fresh ~110M-parameter (or
larger) checkpoint per task. In a setting where tasks arrive in a stream — a cloud service taking
new customer tasks one after another — this is untenable on two counts. Storage/serving cost grows
linearly with the number of tasks, and there is no sharing: every task owns a complete copy of the
network.

The precise problem: **adapt one frozen pre-trained Transformer to many tasks while adding only a
tiny number of new parameters per task, with no loss of accuracy, and in a way that supports
sequential (online) arrival of tasks.** A solution must be (i) *compact* — small marginal parameter
cost per added task, so total size stays ≈1× the base model even for many tasks; (ii) *extensible*
— a new task can be added without retraining or degrading any previous task; and (iii) *good* —
within a small margin of full fine-tuning. Compactness and extensibility together imply the bulk of
the weights must be *shared and frozen*, with each task contributing only a small private delta.

## Background

**Two standard transfer recipes, both per-task-heavy.** Feature-based transfer pre-trains
embeddings (word/sentence/paragraph level) and feeds them to a fresh task-specific model
χ_v(φ_w(x)) — only v is trained, but a whole new top model per task. Fine-tuning copies the
pre-trained weights w and tunes all of them per task; it beats feature-based transfer on accuracy,
but each task now owns its own full set of weights. Both produce a per-task model the size of the
base network (or larger). Fine-tuning can be made more efficient by sharing lower layers and tuning
only the top, but that trades accuracy for compactness.

**The abstract framing.** A pre-trained network is a function φ_w(x) with parameters w. Transfer
defines some new function and trains a small set of new parameters:
- feature-based: χ_v(φ_w(x)), train v;
- fine-tuning: re-tune w itself, |delta| = |w| per task;
- a third option: define ψ_{w,v}(x) with w *copied and frozen* from pre-training and v a small set
  of new parameters, with the initial v₀ chosen so ψ_{w,v₀}(x) ≈ φ_w(x). If |v| ≪ |w|, then many
  tasks together need only ≈|w| parameters, and since w is fixed, adding a task never disturbs the
  others. This is the structure a compact+extensible method must have; the open question is *what
  the new function ψ should be* and *where to inject v*.

**Adapters in vision.** Rebuffi et al. (2017) introduced "residual adapters" for multi-domain image
classification: small per-domain modules added into a frozen shared convolutional backbone, so one
network serves many visual domains with a small per-domain parameter cost. The idea — inject small
trainable modules into a frozen backbone — had not been carried over to text Transformers, where
the sub-layer structure (attention + feed-forward, each with a residual and a layer norm) is
different and the right placement/parameterization is unknown.

**Conditioning-by-modulation precedents.** Conditional batch normalization (de Vries et al. 2017),
FiLM (Perez et al. 2018), and self-modulation (Chen et al. 2019) adapt a network cheaply by
learning per-task affine parameters (scales/shifts) of normalization layers — on the order of a few
parameters per channel. These show that very small interventions can re-purpose a network, and they
motivate also tuning the per-task layer-normalization parameters; but tuning normalization alone is
known to be too weak for strong task adaptation.

**Multi-task and continual learning.** Multi-task learning yields compact shared models but needs
simultaneous access to all tasks. Continual learning targets a stream of tasks but suffers
catastrophic forgetting — re-training on a new task degrades old ones (McCloskey & Cohen 1989;
French 1999). An approach that freezes the shared weights and gives each task its own non-
interacting module sidesteps forgetting by construction: previous tasks are exactly preserved.

**The Transformer sub-layer structure (what we're injecting into).** Each Transformer layer has two
sub-layers — multi-head attention and a position-wise feed-forward network. Each sub-layer ends
with a projection back to the model dimension d, then a residual add, then layer normalization
(post-LN): output = LayerNorm(x + Sublayer(x)). Any injected module must respect this residual +
LN structure.

## Baselines

- **Full fine-tuning of BERT** (Devlin et al. 2018). Copy all pre-trained weights, add a linear
  classifier on the [CLS] token, tune everything. Strong accuracy; the target to match. Gap: |delta|
  = |w| per task (100% of parameters), so N tasks cost N× the model — not compact, not naturally
  extensible.
- **Feature-based transfer** (e.g. ELMo/word/sentence embeddings; Howard & Ruder 2018 context).
  Frozen embeddings into a fresh task model. Gap: typically lower accuracy than fine-tuning, and
  still a full new model per task.
- **Top-layer-only fine-tuning** (variable number of top layers). Share lower layers, tune the top
  k. Gap: a knob that trades accuracy against compactness; tuning fewer layers loses accuracy,
  tuning more loses compactness.
- **Tuning layer-norm / modulation parameters only** (CBN/FiLM-style, ~2d params per layer). Very
  compact. Gap: insufficient for good downstream performance on its own.

## Evaluation settings

- **GLUE** (Wang et al. 2018): nine English text-classification/inference tasks (MNLI, CoLA, SST-2,
  MRPC, STS-B, QQP, QNLI, RTE, etc.). The standard transfer-learning yardstick; report the official
  test metrics from the submission server.
- **17 additional public text-classification datasets** plus **SQuAD v1.1** extractive question
  answering, to confirm breadth beyond GLUE.
- **Base model**: public pre-trained BERT (BASE and LARGE). Classification via a linear layer on the
  first ([CLS]) token, following Devlin et al. (2018).
- **Optimization protocol**: Adam, learning rate warmed up linearly over the first 10% of steps then
  decayed linearly to zero, batch size 32, per-dataset hyperparameter sweep selecting on validation
  accuracy.
- **Primary axis of comparison**: downstream accuracy vs. *number of trained parameters per task*
  (the marginal model-size increase per added task) — i.e. the accuracy/compactness trade-off curve.

## Code framework

The primitives that already exist: a pre-trained BERT Transformer with the standard post-LN
sub-layer pattern, Adam with linear warmup/decay, and a linear classification head on the [CLS]
token. What is missing is the small per-task module and the rule for *where* it goes and *how* it is
initialized so the adapted network starts out equal to the pre-trained one.

```python
import torch
import torch.nn as nn

# --- The small per-task module to be designed (frozen backbone, trainable module) ---
class TaskModule(nn.Module):
    def __init__(self, d_model, ...):
        super().__init__()
        # TODO: what parameterization gives a small per-task module that can
        #       be initialized to (approximately) the identity?
        pass

    def forward(self, x):
        # TODO: transform sub-layer output, starting from ~identity at init
        pass


# --- One Transformer sub-layer's output path (post-LN), with a slot for the module ---
def sublayer_output(sublayer_fn, x, layer_norm, task_module=None):
    h = sublayer_fn(x)                 # attention or feed-forward, then projection back to d
    h = dropout(h)
    # TODO: where does the per-task module attach, relative to the residual and the LayerNorm?
    return layer_norm(h + x)


# --- Which parameters are trainable for a given task ---
def trainable_parameters(model):
    # TODO: freeze the pre-trained weights; train only the new module(s),
    #       and (which existing parameters, if any, should also be trained?)
    pass


# --- Training (already standard) ---
def train(model, loader, opt):
    optim = torch.optim.Adam(trainable_parameters(model), lr=opt.lr)  # linear warmup + decay
    for batch in loader:
        loss = classification_loss(model(batch["input_ids"]), batch["labels"])
        loss.backward(); optim.step(); optim.zero_grad()
```
