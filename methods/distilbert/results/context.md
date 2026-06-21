## Research question

Large pre-trained Transformer language models are the default tool in NLP, carrying
hundreds of millions of parameters. Pre-training such a model from scratch consumes
large amounts of compute, and running a several-hundred-million-parameter model under
tight latency or memory budgets — on a phone, in a real-time service — is demanding.
How can one obtain a *general-purpose* pre-trained language representation model that is
substantially smaller and faster at inference and can be fine-tuned across the full
range of downstream tasks like its larger counterpart?

## Background

**Transfer learning with pre-trained Transformers.** The dominant recipe pre-trains a
Transformer encoder on a large unlabeled corpus with a self-supervised objective,
then fine-tunes it on each downstream task. The reference encoder here is a 12-layer
Transformer (hidden size 768, 12 heads, ~110M parameters) pre-trained on the
masked-language-modeling (MLM) objective: 15% of input tokens are corrupted and the
model is trained to reconstruct them from bidirectional context, with token-type
("segment") embeddings and a next-sentence-prediction auxiliary task. A robust
follow-up recipe (Liu et al. 2019, RoBERTa) showed the pre-training improves with very
large batches, *dynamic* masking (a fresh mask each time a sequence is seen), and
*dropping* next-sentence prediction. The same corpus — English Wikipedia plus the
Toronto BookCorpus (Zhu et al. 2015) — is the standard pre-training data.

**Why a model's output distribution carries more than the top-1 label.** In ordinary
supervised classification, the target is a one-hot label and training minimizes
cross-entropy against it. A well-trained model nonetheless places small but
*structured* probability mass on the non-target classes: for a masked-token prediction
like "I think this is the beginning of a beautiful [MASK]", a strong model puts high
probability on *day* and *life* and a meaningful tail on *future*, *story*, *world*.
Those relative magnitudes encode how the model generalizes — which wrong answers are
"almost right." The soft output distribution of a strong model is a far richer
training signal than a hard label because it reveals the function the teacher actually
learned.

**Knowledge distillation (Bucila et al. 2006; Hinton et al. 2015).** A compact
*student* is trained to reproduce the behavior of a larger *teacher* (or ensemble) by
matching the teacher's soft predictions rather than (or in addition to) the hard
labels. To expose the tail of the distribution, both teacher and student logits are
divided by a temperature `T` before the softmax, `p_i = exp(z_i/T) / Σ_j exp(z_j/T)`;
larger `T` flattens the distribution and amplifies the relative weight of the small
non-target probabilities. At inference `T` is set back to 1.

**Where the compute goes in a Transformer.** Linear layers and layer-normalization,
which dominate the per-layer cost, are highly optimized in modern linear-algebra
libraries. Profiling on this kind of hardware shows that varying the last (hidden-size)
dimension of the tensors has a smaller effect on wall-clock efficiency, for a fixed
parameter budget, than varying the number of layers.

## Baselines

**The large MLM-pretrained Transformer encoder (the teacher).** 12 layers, hidden 768,
12 heads, ~110M parameters; MLM (+ NSP) pre-training on Wikipedia + BookCorpus; fine-tuned
per task. Core idea: a single general-purpose representation that transfers everywhere.

**Task-specific distillation (Tang et al. 2019; Chatterjee et al. 2019; Turc et al.
2019).** Distill a *fine-tuned* large model into a small task-specific model — e.g.
into a BiLSTM classifier, or into a small Transformer initialized from the large one,
once a target task is fixed. Core idea: transfer the teacher's behavior on one task.
Turc et al. pre-train a small model with the original objective and then distill at
fine-tuning time.

**Multi-teacher / multilingual distillation (Yang et al. 2019; Tsai et al. 2019).**
Combine an ensemble of teachers, or pre-train a small multilingual model purely from a
teacher's signal. Core idea: richer or broader supervision.

**Other compression: pruning and quantization (Michel et al. 2019; Gupta et al.
2015).** Remove attention heads or reduce numerical precision of an existing model.
Core idea: shrink a trained model post hoc; these can be applied on top of a compact
pre-trained encoder.

## Evaluation settings

Pre-training data: concatenation of English Wikipedia and Toronto BookCorpus (the same
corpus as the teacher). General language understanding: the GLUE benchmark (Wang et al.
2018) — 9 datasets (CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B, WNLI) — reported on
the dev sets by single-task fine-tuning, with the macro-average as the headline number;
an ELMo-encoder-plus-BiLSTM serves as a published baseline. Downstream tasks under
efficient-inference constraints: IMDb sentiment classification (test accuracy) and
SQuAD v1.1 question answering (EM/F1 on dev). Efficiency: parameter count, and inference
wall-clock for a full pass over a task's dev set on CPU at batch size 1, plus on-device
timing on a mobile phone. The natural yardstick is the teacher model on the identical
tasks and corpus.

## Code framework

The harness loads a large pre-trained teacher (frozen) and a smaller student of the
same family, streams masked batches from the pre-training corpus, and runs a standard
gradient-accumulation training loop. How the student is configured, set up, and trained
is the empty slot.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """Standard MLM Transformer encoder. Configurable depth/width; exposes both the
    MLM logits over the vocabulary and the final-layer hidden states."""
    def forward(self, input_ids, attention_mask):
        # returns (mlm_logits[B,L,V], hidden_states[B,L,H])
        pass

def build_student(teacher: TransformerEncoder):
    # TODO: configure the smaller student and set its starting weights.
    pass

def training_objective(student_out, teacher_out, lm_labels, attention_mask):
    # TODO: design the training loss for the student.
    pass

def mask_tokens(input_ids, mask_prob=0.15):
    # dynamic masking: corrupt ~15% of tokens, return (corrupted_ids, lm_labels)
    pass

def train(teacher, student, corpus, steps, grad_accum):
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters())
    for step, batch in enumerate(corpus):
        ids, labels = mask_tokens(batch.input_ids)
        with torch.no_grad():
            t_out = teacher(ids, batch.attention_mask)
        s_out = student(ids, batch.attention_mask)
        loss = training_objective(s_out, t_out, labels, batch.attention_mask)
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            opt.step(); opt.zero_grad()
```
