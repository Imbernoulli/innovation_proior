# Context

## Research question

The reigning recipe in natural language processing is to pre-train one large Transformer
language model on general-domain text and then *adapt* it to each downstream task. Adaptation is
almost always done by **full fine-tuning**: initialize at the pre-trained weights and run
gradient descent on every parameter. This works, but it has a structural cost that grows with
model size. The fine-tuned model has *exactly as many parameters as the original*, and you get a
*fresh* copy for every task. For a 175-billion-parameter model, each adapted task is a separate
~350 GB checkpoint. Hosting a handful of tasks means hosting a handful of full models;
serving N specialized variants costs N times the storage and turns "switch task" into "load a
different 350 GB model." Training is just as heavy: an adaptive optimizer such as Adam must keep
running statistics (first and second moments) for every trainable parameter, roughly tripling the
memory beyond the weights themselves, and a gradient must be produced for all of them.

The question is: how can a frozen, very large pre-trained model be adapted to many downstream
tasks while drastically reducing per-task storage and training memory cost?

## Background

**The pretrain–adapt paradigm.** A Transformer language model (Vaswani et al. 2017; the
decoder-only autoregressive variant of Radford et al.; the large-scale instances BERT, Devlin
et al. 2019; RoBERTa, Liu et al. 2019; GPT-2, Radford et al. 2019; GPT-3 175B, Brown et al. 2020)
defines an autoregressive distribution P_Φ(y_t | x, y_<t). Adapting it to a task with data
{(x_i, y_i)} means maximizing the conditional log-likelihood Σ_t log P_Φ(y_t | x, y_<t). Larger
pre-trained models keep getting better, but each increment in size makes the per-task storage and
the hardware barrier worse. With GPT-3, few-shot prompting is possible, but weight-level
adaptation remains a common approach even at 175B.

**The conventions used throughout.** d_model is the layer input/output width; the self-attention
module has four projection matrices W_q, W_k, W_v, W_o; the MLP block uses an inner dimension
d_ffn = 4·d_model. W (or W_0) denotes a pre-trained weight matrix and ΔW its accumulated change
during adaptation. Optimization uses Adam / AdamW.

**Motivating finding — adaptation lives on a low-dimensional manifold.** Li et al. (2018), and
specifically for language models Aghajanyan et al. (2020), measure the *intrinsic dimension* of
fine-tuning. They reparameterize the whole parameter vector as θ = θ₀ + P θ_d, where θ₀ is the
frozen pre-trained vector, P is a *fixed* random projection from a low-dimensional space R^d into
the full parameter space R^D (a Fastfood-style transform), and only the small vector θ_d ∈ R^d is
trained. They define d₉₀ as the smallest d that reaches 90% of full-fine-tuning performance. The
finding is striking: d₉₀ is tiny — on the order of a couple hundred trainable parameters lets
RoBERTa reach 90% of full performance on MRPC — and, counter-intuitively, *larger* pre-trained
models have *lower* intrinsic dimension. So although the ambient parameter space is enormous, the
solution found by fine-tuning can sit in a very low-dimensional subspace. That strongly suggests
the *change* in weights need not be a full-dimensional object.

**On structure in trained networks.** Over-parameterized networks are known to develop
low-dimensional structure after training (Oymak et al. 2019); some methods even impose structural
constraints during training. Aghajanyan's projection is into a random subspace of the *flattened*
parameter vector: it needs a large (if implicit) projection matrix, is not tied to any individual
weight matrix, and gives no special deployment structure — the trained subspace cannot be folded
back into the model's existing weights to recover a plain forward pass.

## Baselines

**Full fine-tuning (FT).** Initialize Φ = Φ₀, update all parameters by gradient ascent on
Σ_{(x,y)} Σ_t log P_Φ(y_t | x, y_<t). A weaker variant trains only a subset of layers (e.g. the
top two, FT^Top2). The learned delta ΔΦ has |ΔΦ| = |Φ₀|.

**Adapter tuning (Houlsby et al. 2019; Lin et al. 2020; Rebuffi et al. 2017; Pfeiffer et al.
2021).** Insert small modules *between* existing layers. The original Houlsby design (Adapter^H)
puts two adapters per Transformer block, each a down-projection d_model → r, a nonlinearity, an
up-projection r → d_model, biases, and a residual connection. Lin et al. (Adapter^L) use one
adapter per block plus a LayerNorm. Parameter count per adapter is 2·d_model·r + r + d_model
(plus any LayerNorm). The bottleneck dimension r keeps the parameter count well under 1% of the
model.

**Prefix / prompt-style tuning (Li & Liang 2021; Lester et al. 2021; Hambardzumyan et al. 2020;
Liu et al. 2021).** Instead of changing weights, optimize input activations. *Prefix-embedding
tuning* (PreEmbed) prepends l_p (and optionally infixes l_i) trainable "virtual token"
embeddings not in the vocabulary; the trainable count is d_model·(l_p + l_i). *Prefix-layer
tuning* (PreLayer) additionally replaces the activations after every Transformer layer with
trainable vectors; the count is L·d_model·(l_p + l_i).

**Bias-only tuning (BitFit, Zaken et al. 2021).** Train only the bias vectors, freezing
everything else. Extremely few parameters.

## Evaluation settings

Natural-language *understanding* is measured on the GLUE benchmark (Wang et al. 2019): MNLI,
SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B, reporting accuracy (Matthew's correlation for CoLA,
Pearson for STS-B). Pre-trained backbones for these are RoBERTa base (125M) and large (355M) and
DeBERTa XXL (1.5B), taken from the HuggingFace Transformers library. Natural-language
*generation* is measured on the E2E NLG
Challenge, WebNLG, and DART (BLEU / NIST / METEOR / ROUGE-L / CIDEr / TER), and for large-scale
adaptation on WikiSQL (natural-language-to-SQL logical-form accuracy) and SAMSum (conversation
summarization, ROUGE-1/2/L), with backbones GPT-2 medium/large and GPT-3 175B. The standard
adaptation protocol is AdamW with a linear learning-rate schedule and warmup, reporting the
median or mean over several random seeds. A central operational axis for comparison is the number
of trainable (and stored) parameters per task, and — separately — single-forward-pass inference
latency on a fixed GPU.

## Code framework

The pieces below already exist: a frozen pre-trained linear layer (`nn.Linear` with weights
loaded from a checkpoint), the optimizer (AdamW), the conditional-language-model loss, and a
standard training loop. A minimal harness wraps existing linear layers, freezes everything except
the small task-specific parameters, and saves only those task-specific parameters.

```python
import torch.nn as nn
import torch.nn.functional as F


class AdaptationLayer:
    def __init__(self, merge_weights=True):
        self.merged = False
        self.merge_weights = merge_weights


class AdaptedLinear(nn.Linear, AdaptationLayer):
    def __init__(self, in_features, out_features, merge_weights=True,
                 fan_in_fan_out=False, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        AdaptationLayer.__init__(self, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        # TODO: allocate the small trainable parameters of the mechanism.
        # TODO: freeze the pre-trained weight.
        # TODO: handle implementations whose linear weight is stored transposed.
        pass

    def reset_parameters(self):
        super().reset_parameters()
        # TODO: initialize the task-specific adaptation parameters.
        pass

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # TODO: restore any train-time state needed before training resumes.
            pass
        else:
            # TODO: prepare the layer for inference / evaluation mode.
            pass
        return self

    def forward(self, x):
        # TODO: produce the layer output for the current task.
        return F.linear(x, self.weight, self.bias)


class AdaptedFusedLinear(nn.Linear, AdaptationLayer):
    def __init__(self, in_features, out_features, enabled_slices,
                 merge_weights=True, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        AdaptationLayer.__init__(self, merge_weights=merge_weights)
        self.enabled_slices = enabled_slices
        # TODO: allocate adaptation parameters only for selected slices when a
        #       single projection stores several logical matrices.
        pass


def mark_only_adaptation_as_trainable(model, bias="none"):
    # TODO: freeze every parameter except the task-specific adaptation
    #       parameters, with optional bias handling.
    pass


def adaptation_state_dict(model, bias="none"):
    # TODO: return ONLY the small per-task adaptation parameters, so each task's
    #       checkpoint does not duplicate the shared base model.
    pass
```
