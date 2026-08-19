## Research question

Self-training language model pretraining (ELMo, GPT, BERT, XLM) has produced large downstream gains,
but it is hard to tell which parts of a given recipe are doing the work: pretraining is expensive,
groups train on private data of different sizes, and hyperparameter choices turn out to matter a lot,
so published comparisons across papers routinely conflate several changes at once. This project holds
the model architecture fixed (the standard Transformer encoder BERT introduced) and treats the
**pretraining procedure itself** as the object of study: masking strategy, input/segment format and
the auxiliary NSP loss, batch size, subword vocabulary, and pretraining data size/duration. The goal
is to reproduce BERT's numbers under a controlled reimplementation, then test each procedural choice
one at a time against measured downstream performance, to find out whether BERT's published numbers
represent the ceiling of the masked-language-modeling objective or whether the objective was simply
undertrained.

## Prior art / Background / Baselines

- **BERT (Devlin et al. 2018).** Two pretraining objectives on a Transformer encoder: masked language
  modeling (MLM) — uniformly select 15% of input tokens, replace 80% with `[MASK]`, 10% with a random
  token, leave 10% unchanged, cross-entropy loss on recovering the originals — and Next Sentence
  Prediction (NSP), a binary classifier over whether two input segments are contiguous in the source
  text, positive/negative pairs sampled with equal probability. Input format: two concatenated
  segments `[CLS] x_1...x_N [SEP] y_1...y_M [EOS]`, each segment possibly spanning multiple sentences,
  combined length under 512 tokens. Masking is computed once during data preprocessing (a *static*
  mask); to vary the mask seen across the 40 training epochs, the training data is duplicated 10x so
  each sequence is masked 10 different ways (so any one masking of a sequence is still seen 4 times).
  Optimizer: Adam (β1=0.9, β2=0.999, ε=1e-6), weight decay 0.01, learning rate warmed up over the
  first 10,000 steps to a peak of 1e-4 then linearly decayed, dropout 0.1 on all layers and attention
  weights, GELU activations. Pretrained for S=1,000,000 steps, batch size B=256 sequences, max
  sequence length T=512. Trained on BookCorpus + English Wikipedia, 16GB of uncompressed text.
  BERT-base: L=12 layers, H=768 hidden, A=12 attention heads, 110M params. BERT-large: L=24, H=1024,
  A=16, 355M params. NSP was hypothesized by the original authors to matter: they report that removing
  it hurts performance, with significant degradation on QNLI, MNLI and SQuAD 1.1.
  Published BERT-base dev numbers (the reference this study targets): SQuAD 1.1/2.0 F1 88.5/76.3,
  MNLI-m accuracy 84.3, SST-2 accuracy 92.8, RACE accuracy 64.3. Published BERT-large: SQuAD 1.1/2.0
  F1 90.9/81.8, MNLI-m 86.6, SST-2 93.7.
- **Text encoding.** The original BERT implementation uses a character-level Byte-Pair Encoding (BPE)
  vocabulary of 30K subword units, learned after heuristic input tokenization. A more recent GPT-2-style
  BPE variant instead operates over raw *bytes* rather than unicode characters, letting a modest
  vocabulary (50K units) encode arbitrary input text with no "unknown token" fallback, at the cost of
  roughly 15-20M extra embedding parameters depending on model size.
  - **Recent challenges to NSP.** Several concurrent efforts (cross-lingual pretraining, autoregressive
  permutation-based pretraining, span-based pretraining) have questioned whether the NSP loss is
  actually necessary, in tension with the original BERT ablation.
- **Large-batch optimization.** Prior work in neural machine translation has shown that training with
  very large mini-batches, with an appropriately increased learning rate, can improve both optimization
  speed and end-task performance. Concurrent work has shown BERT itself is amenable to large-batch
  training, up to 32K sequences.
- **More data, more compute.** A recently proposed alternative pretraining architecture (permutation
  language modeling) is trained on nearly 10x more data than original BERT, with a batch size 8x
  larger for half as many optimization steps — roughly 4x as many sequences seen in total — making it
  unclear how much of its reported improvement over BERT is architecture/objective versus scale.

## Fixed substrate

- **Model architecture** is not touched at any point in this study: a standard Transformer encoder,
  either the BERT-base shape (L=12, H=768, A=12, 110M params) or, from the point the recipe is
  assembled and scaled, the BERT-large shape (L=24, H=1024, A=16, 355M params). Both match BERT's
  published architectures exactly, so any downstream difference is attributable to the pretraining
  procedure, not the model.
- **Reimplementation framework:** fairseq. The original BERT optimization hyperparameters (Adam
  betas/epsilon, weight decay, dropout, activation) are followed by default; only peak learning rate
  and warmup steps are tuned per setting, plus, where noted, the Adam ε term and β2 (found to matter
  for training stability, β2=0.98 helps stability at large batch sizes).
  Sequences are packed to a fixed maximum length T=512 tokens throughout — no short-sequence warm-up
  and no reduced-length prefix of training, unlike the original BERT schedule; only full-length
  sequences are used.
- **Hardware:** mixed-precision training on DGX-1 machines, 8x32GB V100 GPUs per machine,
  Infiniband-interconnected.
- **Downstream evaluation protocol is fixed across every rung:**
  - **GLUE** (9 tasks: CoLA, SST, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI). For the design-choice
    study, models are finetuned single-task (no multi-task training, no ensembling) on each task's
    training data, following BERT's own finetuning procedure, and evaluated on the task's development
    set. Reported numbers below are, where the source specifies it, medians over five random
    initializations (seeds); accuracy is the metric for MNLI-m and SST-2.
  - **SQuAD** v1.1 and v2.0. Span-extraction QA: v1.1 always has an answer in context; v2.0 adds
    unanswerable questions, requiring an additional binary answerability classifier trained jointly
    with the span predictor (loss terms summed). Metric: F1 on the development set, reported as
    v1.1/v2.0.
  - **RACE.** Reading comprehension from English exams (middle/high school), 4-way multiple choice per
    question; each candidate answer is concatenated with the question and passage, encoded, and the
    four `[CLS]` representations are scored by a shared fully-connected layer. Metric: accuracy.
  - **Held-out perplexity (ppl).** For pretraining-only comparisons (e.g. batch size), perplexity on
    held-out training data measures the masked-language-modeling objective directly, alongside
    downstream accuracy.

## What is under test (the free variable across rungs)

Everything about the *pretraining procedure* is open to revision, one decision at a time, each
justified by the previous rung's measured numbers: the masking strategy (static vs. dynamic), the
input segmentation and whether to keep the NSP loss, the training batch size (holding total compute /
sequences-seen fixed via gradient accumulation), the subword vocabulary (character-level vs.
byte-level BPE), and, once a combined recipe is settled, the scale of pretraining itself — how much
text to pretrain over and for how many steps.
