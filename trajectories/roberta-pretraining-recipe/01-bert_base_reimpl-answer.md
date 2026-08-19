**Problem.** Every later rung in this study attributes a downstream-score change to exactly one
procedural decision. That attribution is only meaningful if the recipe I'm changing decisions *from*
is itself a faithful, self-consistent reproduction of BERT — not a silently-diverged framework port.
So before proposing any design change, I need a validated instrument.

**Proposal.** Reimplement BERT-base under fairseq, matching the original specification: 12-layer,
768-hidden, 12-head Transformer encoder (110M params); joint masked-language-modeling + Next Sentence
Prediction objectives; Adam (β1=0.9, β2=0.999, ε=1e-6), weight decay 0.01, LR warmed up linearly over
the first 10,000 steps to a peak of 1e-4 then linearly decayed, dropout 0.1, GELU; batch 256 sequences
of up to 512 tokens, 1,000,000 steps, over BookCorpus + English Wikipedia (16GB). Masking is *static*:
computed once at preprocessing, with the corpus duplicated 10x so each sequence is seen under 10
different masks across 40 epochs. Input format is `segment-pair+NSP`: two concatenated multi-sentence
segments delimited by `[CLS]`/`[SEP]`/`[EOS]`, combined length under 512 tokens.

**What can legitimately differ from the original.** Peak learning rate and warmup step count are
tuned per-setting rather than copied verbatim, since different implementations respond differently to
the same nominal schedule; the Adam ε term is tuned where training proves sensitive to it. Everything
else — objective, masking rate/split, architecture, data, batch size, step count — is held to the
original specification.

**Evaluation.** The fixed protocol: GLUE single-task dev-set finetuning (no multi-task training, no
ensembling), SQuAD 1.1/2.0 dev F1, RACE accuracy.

**What would count as success vs. failure.** A large, systematic shortfall spread across every
downstream task relative to BERT-base's published numbers (SQuAD 1.1/2.0 F1 88.5/76.3, MNLI-m 84.3,
SST-2 92.8, RACE 64.3) would flag a real implementation bug — tokenizer mismatch, masking error, a
broken NSP head, a schedule that never reaches its stated peak — and would need to be fixed before any
ablation could be trusted. Landing in the neighborhood of the published numbers, whether a little above
or below and spread evenly rather than concentrated in one task, is the signature of a working
reimplementation and becomes this study's actual baseline: the single fixed point every later rung's
one-variable-at-a-time change will be measured against, not the originally published numbers
themselves (produced under different code and hardware I cannot exactly recreate).

**Configuration under test (rung 1):**
```
architecture:      BERT-base (L=12, H=768, A=12, ~110M params)
objective:         MLM (15% select; 80% [MASK] / 10% unchanged / 10% random) + NSP
masking:           static (fixed at preprocessing, 10x data duplication over 40 epochs)
input format:      segment-pair + NSP
optimizer:         Adam, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.01
lr schedule:       linear warmup (10,000 steps, per-original) to peak, then linear decay
                    [peak LR / warmup steps tuned per-implementation if needed]
dropout:           0.1 (all layers + attention)
activation:        GELU
batch size:        256 sequences
max seq length:    512 tokens
steps:             1,000,000
data:              BookCorpus + English Wikipedia, 16GB
```
