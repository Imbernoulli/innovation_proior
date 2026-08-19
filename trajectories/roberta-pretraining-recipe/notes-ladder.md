# Rung ladder — roberta-pretraining-recipe

Research question: which BERT pretraining design choices actually matter, holding architecture
fixed, and does a corrected recipe close the gap to (or beat) later published models? Source: full
paper LaTeX in methods/roberta/src/*.tex (numbered sections) and methods/roberta/src/tables/*.tex.

## Starting state (00-initial-context, not a rung)
BERT-base published numbers (the reference this replication targets/exceeds), from
`tables/base_apples_to_apples.tex:15` and `tables/static_vs_dynamic_masking.tex:7`:
- SQuAD 1.1/2.0 dev F1: 88.5/76.3; MNLI-m dev acc: 84.3; SST-2 dev acc: 92.8; RACE: 64.3.
Original BERT hyperparameters (`02-background.tex`): Adam β1=0.9, β2=0.999, ε=1e-6, weight decay
0.01, warmup 10,000 steps to peak LR 1e-4, dropout 0.1, GELU, S=1,000,000 steps, B=256 sequences,
T=512 tokens max. Static masking (mask once at preprocessing, data duplicated 10x over 40 epochs).
Input: segment-pair with NSP loss. Data: BookCorpus + English Wikipedia, 16GB.
Reimplementation framework: fairseq (`03-exp_setup.tex`, "Implementation").

## Rung 1 — BERT-base reimplementation (fairseq, static masking, segment-pair+NSP)
- Variant: reproduce the original BERT-base recipe verbatim (same hyperparams, static mask,
  segment-pair+NSP, bsz 256, 1M steps) inside fairseq, to validate the harness before changing
  anything.
- Numbers (`tables/base_apples_to_apples.tex:8` "our reimplementation (with NSP loss) — segment-pair"
  vs `:15` BERT-base published): SQuAD 1.1/2.0 90.4/78.7 (vs 88.5/76.3 published), MNLI-m 84.0 (vs
  84.3), SST-2 92.9 (vs 92.8), RACE 64.2 (vs 64.3).
- Also cross-checked via `tables/static_vs_dynamic_masking.tex:10` ("static" row, SQuAD2.0/MNLI/SST-2
  78.3/84.3/92.5) against `:7` ("reference" row 76.3/84.3/92.8) — same conclusion, reimplementation
  matches or exceeds the published baseline.
- Source: `03-exp_setup.tex` §Implementation, `02-background.tex` full section.

## Rung 2 — dynamic masking
- Variant: generate the MLM mask fresh every time a sequence is fed to the model, instead of fixing
  it once at preprocessing (10x duplication).
- Numbers: `tables/static_vs_dynamic_masking.tex:10-11`: static 78.3/84.3/92.5 (SQuAD2.0/MNLI-m/SST-2)
  vs dynamic 78.7/84.0/92.9.
- Source: `04-design.tex` §Static vs. Dynamic Masking.

## Rung 3 — FULL-SENTENCES without NSP
- Variant: drop the NSP loss; pack each input with contiguous full sentences (crossing document
  boundaries with an extra separator) up to 512 tokens, instead of two NSP segment-pairs.
- Numbers: `tables/base_apples_to_apples.tex:8` segment-pair+NSP 90.4/78.7, 84.0, 92.9, 64.2 vs
  `:12` full-sentences (no NSP) 90.4/79.1, 84.7, 92.5, 64.8. (Also on record: `:9` sentence-pair+NSP
  88.7/76.2,82.9,92.1,63.0 — single sentences hurt; `:13` doc-sentences 90.6/79.7,84.7,92.7,65.6 —
  slightly better than full-sentences but variable batch size, so full-sentences is kept for the rest
  of the paper's experiments "for easier comparison with related work.")
- Source: `04-design.tex` §Model Input Format and Next Sentence Prediction.

## Rung 4 — large-batch training (256 → 2K → 8K)
- Variant: increase batch size via gradient accumulation while holding compute (total sequences
  seen) fixed: 256×1M steps ≈ 2K×125K steps ≈ 8K×31K steps; tune LR per setting.
- Numbers: `tables/large_batches.tex:10-12`: bsz256/1M steps/lr1e-4 → ppl 3.99, MNLI-m 84.7, SST-2
  92.7; bsz2K/125K/lr7e-4 → ppl 3.68, MNLI-m 85.2, SST-2 92.9 (best); bsz8K/31K/lr1e-3 → ppl 3.77,
  MNLI-m 84.6, SST-2 92.8.
- Source: `04-design.tex` §Training with large batches.

## Rung 5 — combine (RoBERTa recipe: dynamic mask + full-sentences/no-NSP + 8K batch + byte-BPE),
scaled to BERT-large architecture, trained on original 16GB data for 100K steps
- Variant: aggregate rungs 2-4 plus a byte-level BPE vocabulary (50K units, `04-design.tex` §Text
  Encoding) instead of the original char-level 30K BPE; move to the BERT-large architecture (L=24,
  H=1024, A=16, 355M params, `05-roberta.tex`); train for 100K steps on the same Books+Wiki 16GB data
  used by original BERT, to isolate the effect of the recipe from data/compute scale-up.
- Numbers: `tables/ablation.tex:9` "with Books+Wiki, 16GB/8K/100K": SQuAD 93.6/87.3, MNLI-m 89.0,
  SST-2 95.3 — vs BERT-large published `tables/ablation.tex:15` 13GB/256/1M: SQuAD 90.9/81.8, MNLI-m
  86.6, SST-2 93.7.
- Source: `05-roberta.tex` first two paragraphs + `tables/ablation.tex`.

## Rung 6 — + additional pretraining data (16GB → 160GB)
- Variant: add CC-News (76GB), OpenWebText (38GB), Stories (31GB) to Books+Wiki, for 160GB total;
  same 100K steps, same 8K batch.
- Numbers: `tables/ablation.tex:10`: SQuAD 94.0/87.7, MNLI-m 89.3, SST-2 95.6.
- Source: `05-roberta.tex` §RoBERTa, `03-exp_setup.tex` §Data.

## Rung 7 — pretrain longer (100K → 300K steps)
- Variant: same 160GB data, same 8K batch; extend training to 300K steps.
- Numbers: `tables/ablation.tex:11`: SQuAD 94.4/88.7, MNLI-m 90.0, SST-2 96.1.
- Source: `05-roberta.tex` §RoBERTa, final paragraphs.

## Rung 8 — pretrain even longer (300K → 500K steps) — final/published RoBERTa
- Variant: same setup, extend to 500K steps.
- Numbers: `tables/ablation.tex:12`: SQuAD 94.6/89.4 (bold/best), MNLI-m 90.2 (bold/best), SST-2 96.4
  (bold/best). No overfitting observed at 500K.
- Source: `05-roberta.tex` §RoBERTa, "We note that even our longest-trained model does not appear to
  overfit our data and would likely benefit from additional training."

## Gaps
- No number-bearing ablation exists in the source for byte-level vs char-level BPE alone (paper says
  only "slight differences...slightly worse on some tasks", explicitly deferred to future work) — so
  BPE is folded into rung 5's combined recipe rather than given its own measured rung.
- large_batches.tex table doesn't state seed count (unlike the two 5-seed-median tables); reported as
  plain dev numbers, no median/single-run claim invented.
