I want to be careful about what this first step actually is, because it would be easy to mistake it
for a design choice when it is really a calibration. The whole plan for this study is to isolate one
procedural decision at a time — masking strategy, input format, batch size, vocabulary, data scale —
and read any downstream movement as caused by that one change. That kind of attribution only means
something if the recipe I'm changing decisions *from* is itself trustworthy: a faithful, self-consistent
reproduction of BERT, not a framework port that has quietly drifted from the specification in some way
I haven't noticed. So before I let myself touch anything, I need to build the instrument and check that
it reads correctly against a known reference.

The target is fully specified already, which is exactly what makes this step tractable rather than
open-ended. BERT-base is a 12-layer, 768-hidden, 12-head Transformer encoder, about 110M parameters.
Two objectives are trained jointly: masked language modeling, where 15% of input tokens are selected
and then split 80/10/10 into `[MASK]` / left-unchanged / replaced-with-a-random-token, with a
cross-entropy loss on recovering the originals; and Next Sentence Prediction, a binary classifier over
whether two concatenated segments are genuinely contiguous in the source text or paired from different
documents, sampled with equal probability. The input itself concatenates two segments — each segment
possibly spanning several natural sentences — delimited by `[CLS]`, `[SEP]`, `[EOS]`, with combined
length under 512 tokens; I'll call this `segment-pair+NSP` once I have an alternative to contrast it
with, but for now it is simply the format, because I haven't proposed a change to it yet. Optimization
follows Adam with β1=0.9, β2=0.999, ε=1e-6, weight decay 0.01; the learning rate warms up linearly over
the first 10,000 steps to a peak of 1e-4 and then decays linearly to zero; dropout is 0.1 throughout,
activations are GELU. Training runs for one million steps at a batch size of 256 sequences of up to
512 tokens, over 16GB of BookCorpus plus English Wikipedia text. And the masking itself is *static*:
computed once during preprocessing, with the corpus duplicated tenfold so that across 40 training
epochs, each sequence is seen under ten distinct masks — each particular masked view recurring four
times over the course of training.

None of these numbers are mine to invent; the value I'm adding at this step is entirely in the
faithfulness of the reproduction, not in the recipe itself. I'm building it in fairseq rather than the
original codebase, and that choice is not neutral — a framework port can diverge from the original in
ways that have nothing to do with the stated hyperparameters: how sequences get batched and padded, how
Adam's epsilon interacts with mixed-precision arithmetic, tokenizer edge cases, how the corpus gets
sharded and shuffled. None of that shows up by reading the spec; it only shows up by training the thing
and comparing outcomes. That is precisely why I am treating two hyperparameters as legitimately
re-tunable rather than copied verbatim: peak learning rate and warmup step count, because different
implementations are known to respond differently to the same nominal schedule even when every other
setting matches on paper. I'll also watch the Adam epsilon term and retune it if training proves
sensitive to it. Everything else — the objective, the masking rate and its 80/10/10 split, the
architecture, the data, the batch size, the step count — stays exactly at the original specification.

I want to say plainly what I'll treat as this rung succeeding versus failing, because "close to the
published numbers" is doing real work as a standard, not a hand-wave. A large, systematic shortfall
spread evenly across SQuAD, MNLI, SST-2, and RACE relative to BERT-base's published dev numbers — F1
88.5/76.3 on SQuAD 1.1/2.0, 84.3 accuracy on MNLI-m, 92.8 on SST-2, 64.3 on RACE — would be the
signature of a real implementation bug: a broken NSP head, a masking rate that isn't actually landing
at 15%, a schedule that never reaches its stated peak, a tokenizer subtly mismatched to the original.
That would mean stopping here and debugging before anything downstream of this rung could be trusted.
A shortfall or surplus that's small and spread roughly evenly across tasks, rather than concentrated
in one — which would instead point at a task-specific finetuning bug rather than a pretraining one — is
the signature I'd read as "the reimplementation basically works, and the residual gap is optimization
noise plus the minor hyperparameter retuning I already flagged as legitimate." I am not committing here
to which of those two pictures I'll see; that is exactly what running the reimplementation will tell
me. What I am committing to is the standard itself, and to using whatever number this rung actually
produces — not the originally published number, which was generated under different code on different
hardware I cannot exactly recreate — as the literal fixed point that every one-variable-at-a-time
change in the rungs after this one gets measured against.

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
