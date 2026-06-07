# Synthesis — T5 (Text-to-Text Transfer Transformer)

## The pain point (research question, in-frame)

By ~2018-2019, transfer learning had taken over NLP: pre-train a big model on unlabeled
text, then fine-tune on the target task. It worked spectacularly (BERT, GPT, ELMo,
ULMFiT, XLNet, RoBERTa, ALBERT, UniLM...). But the **methodology had fragmented**:

- Different **pre-training objectives**: causal LM (GPT/ULMFiT), masked LM / denoising
  (BERT), permutation LM (XLNet), seq2seq masking (MASS), span masking (SpanBERT),
  prefix LM (UniLM/Liu et al.).
- Different **architectures**: encoder-only (BERT), decoder-only (GPT), encoder-decoder
  (original Transformer / MASS), single-stack-with-prefix (UniLM).
- Different **unlabeled corpora**: Wikipedia+BooksCorpus (BERT), WebText (GPT-2), CC-News
  (RoBERTa), Common Crawl variants. Datasets rarely released; rarely compared.
- Different **fine-tuning recipes**: full fine-tune, adapter layers, gradual unfreezing,
  multi-task.
- Different **task interfaces**: a classification head, a span-pointer head, a per-token
  tagging head, a seq2seq decoder — each task got a bespoke output module.

Each new paper changed *several* of these axes at once and reported a win, so you could
not attribute the gain to any one decision. The field could not answer "which of these
choices actually matters?" because there was **no controlled testbed**: changing the
objective also meant changing the architecture also meant changing the head also meant
changing the data. The goal: build a single framework general enough that *every* one of
these axes can be varied **one at a time** while everything else is held fixed, and then
do the controlled sweep. The thing that unlocks that is a **single, uniform task
interface** so that the model/loss/decoding never have to change between tasks.

## The central unification insight: cast every task as text → text

The blocker to a controlled comparison is the **output interface**. As long as
classification needs a softmax-over-labels head, span extraction needs a start/end
pointer, tagging needs per-token logits, and translation needs a decoder, you can never
swap architectures or objectives freely — the head is entangled with everything.

Resolve it by removing the head entirely: make **every** task map a text string to a
text string. Classification → emit the label *word* ("entailment"). Regression (STS-B,
scores 1-5) → round to nearest 0.2 and emit the *number as a string* ("3.8"), recast as
21-way classification. Summarization / translation / QA → emit the target text directly.
A task is selected purely by a **text prefix** prepended to the input ("translate English
to German: ", "stsb sentence1: ...", "summarize: ").

Consequences:
- One **loss** for everything: teacher-forced maximum likelihood / cross-entropy over the
  output token sequence. Same loss for pre-training (denoising) and every downstream task.
- One **decoding** procedure: autoregressive generation (greedy at test time).
- The same model can be **multi-task** trained by just mixing examples — no per-task
  modules to juggle.
- Now objective, architecture, corpus, and fine-tuning recipe become **free, orthogonal
  knobs** that you can ablate one at a time. That is the whole point: the unification is
  what makes the systematic study *possible*, not a performance trick.

Lineage / prior unifications it reacts to (cite as ancestors):
- McCann et al. 2018 "Natural Language Decathlon (decaNLP)": cast all tasks as
  **question answering**, but *required* simultaneous multi-tasking and an explicit
  Q/A format. T5 relaxes to short prefixes and allows separate fine-tuning per task.
- Radford et al. 2019 (GPT-2): cast tasks as **language modeling**, evaluated zero-shot
  by priming with a prompt ("TL;DR:"). T5 keeps the text I/O idea but focuses on
  transfer (fine-tuning) with a dedicated encoder, not zero-shot from a decoder-only LM.
- Keskar et al. 2019: unify tasks as **span extraction** (append candidate answers,
  point at the right span). Fails for generative tasks (translation, summarization)
  where you can't enumerate outputs. Text-to-text covers generative + extractive uniformly.

## Architecture: why encoder-decoder (the controlled-experiment answer)

Three structural candidates, all built from the same Transformer self-attention block,
differing only in the **attention mask**:

1. **Encoder-decoder** (original Transformer, Vaswani 2017): encoder uses *fully-visible*
   mask (every input token sees every other), decoder uses *causal* mask + cross-attention
   into the encoder.
2. **Decoder-only LM** (GPT): single stack, causal mask throughout; input and target are
   concatenated and the whole thing is modeled left-to-right.
3. **Prefix LM** (UniLM / Liu et al.): single stack, but *fully-visible* over the prefix
   (input) region and causal over the target region.

Insight that motivates the choice: in the text-to-text setting the model is given a
**prefix/context** and must condition its output on *all* of it. A fully causal mask
forces token i's representation to depend only on tokens ≤ i, so the representation of the
input the decoder attends to is **needlessly crippled** (same argument once made against
unidirectional RNN encoders in seq2seq, Bahdanau 2014). So you want *bidirectional*
(fully-visible) processing of the input → rules out the pure decoder-only LM; favors
encoder-decoder or prefix-LM.

The parameter/compute accounting (load-bearing derivation):
- Let an L-layer BERT-base-sized stack have P parameters. An L+L encoder-decoder has ≈2P
  parameters but the **same FLOPs** as an L-layer decoder-only model, because each of its
  stacks only runs over *one* sequence (encoder over input, decoder over output), whereas
  a decoder-only LM runs all L layers over the *concatenation* of input+output.
- So you cannot match a decoder-only LM to an encoder-decoder on *both* params and FLOPs.
  T5 compares fairly by holding FLOPs (M) fixed and considering several param budgets:
  enc-dec (2P, M); shared-param enc-dec (P, M); half-depth enc-dec (P, M/2);
  decoder-only LM (P, M); prefix LM (P, M).
- Encoder-decoder attention layers are ~10% of params, and L-layer LM vs L+L enc-dec have
  nearly identical step times empirically → treat L+L enc-dec ≈ 2L-layer LM in params.

Diagnostic finding (controlled): with a denoising objective, **encoder-decoder wins** on
every task. Sharing encoder/decoder params (→P) costs almost nothing (echoes ALBERT's
cross-layer sharing). Halving depth (→M/2) hurts. Shared enc-dec beats prefix-LM →
the explicit **encoder-decoder cross-attention is itself beneficial**, beyond just
bidirectional input. Denoising > LM objective universally. Prefix-LM under text-to-text
is basically BERT-for-classification with the classifier folded into the output layer.

## The pre-training objective: deriving span corruption

Start from the known landscape:
- **Causal LM** objective (predict next token): historic pre-training (GPT, ULMFiT). Only
  left-to-right context.
- **BERT MLM / denoising** (Devlin 2018): corrupt 15% of tokens — of those, 90% → [MASK],
  10% → random token; predict the *originals* at those positions, bidirectionally. Shown
  to beat LM objectives. But BERT is encoder-only; its target is a per-position
  reconstruction at the encoder output, not a generated sequence.

Step-by-step derivation toward T5's baseline, each step motivated:

1. **Pick denoising over LM.** Controlled comparison (prefix-LM obj vs BERT-style vs
   deshuffling) → BERT-style denoising best; deshuffling (reconstruct a shuffled
   sentence) much worse. Confirms the field's belief that denoising > causal LM for
   downstream transfer. Adopt a BERT-style corrupt-and-reconstruct objective.

2. **Adapt BERT MLM to a text-to-text (seq2seq) model.** BERT predicts at masked
   positions of the encoder; here we must *generate* a target sequence. Naive port:
   feed the corrupted sequence, generate the *entire original* sequence as target
   ("BERT-style" row). Works, but the target is as long as the input → expensive decoder
   self-attention over long sequences.

3. **Drop the random-token swap.** BERT's 10%-random-token trick exists to reduce the
   pretrain/finetune mismatch of an encoder that always sees [MASK]. Test removing it
   (just mask, no random swap) = "MASS-style". No loss in performance → drop it.
   Simpler.

4. **Only predict the corrupted tokens, not the whole sequence.** Reconstructing the
   full text wastes the decoder on copying the un-corrupted majority. Two ways to shorten
   the target:
   - **Replace each consecutive corrupted span with one unique sentinel token** (`<X>`,
     `<Y>`, ...; sentinels are new vocab IDs). Target = concatenation of the dropped
     spans, each prefixed by its sentinel, ending with a final sentinel `<Z>`.
   - **Drop corrupted tokens entirely** from input, predict them in order.
   Both perform ~equally and both **shorten input and target → faster training**. The
   sentinel-replace variant is slightly safer on SuperGLUE (drop-tokens helps CoLA, since
   detecting missing tokens ≈ detecting acceptability, but hurts elsewhere). Choose
   **replace-spans-with-sentinels** as baseline.

5. **Corruption rate.** Sweep 10/15/25/50%. Rate barely matters except 50% degrades
   GLUE/SQuAD and lengthens targets. Keep **15%** (BERT precedent, computationally cheap).

6. **Corrupt spans, not i.i.d. tokens.** i.i.d. masking rarely produces long consecutive
   runs, so the sentinel trick doesn't shorten things much. Explicitly corrupt
   contiguous spans (parametrized by corruption rate + #spans ⇒ average span length;
   SpanBERT, Joshi 2019, showed span masking helps). Sweep avg span length 2/3/5/10:
   avg length **3** slightly but significantly beats i.i.d. on most non-translation tasks
   *and* gives a speedup (shorter sequences). Final baseline: **span corruption, 15%,
   mean span length 3, sentinel replacement, predict only corrupted spans.**

Key meta-finding: among denoising variants the *downstream* differences are small;
**choose by computational cost** (target length). The objective's main job is just
"reconstruct corrupted text"; the exact flavor matters little.

## Architectural details (Transformer modifications) — every knob and its why

Base = Vaswani 2017 encoder-decoder, with deliberate simplifications:

- **Block** = (LayerNorm → sublayer → dropout → residual-add), i.e. **pre-norm**:
  LayerNorm is applied to the *input* of each sublayer and the residual skip is *outside*
  the normalized path. Why: pre-norm stabilizes training of deep Transformers (cleaner
  residual signal path; avoids the warmup-fragility of post-norm). Sublayers per block:
  encoder = [self-attn, FFN]; decoder = [self-attn, cross-attn, FFN].
- **RMSNorm (LayerNorm without mean-subtraction or bias)**: normalize by RMS only,
  `x * rsqrt(mean(x^2)+eps) * weight`, no centering, no additive bias. Why: the
  re-centering term contributes little; dropping mean-subtraction and bias is cheaper and
  just as stable (this is Zhang & Sennrich RMSNorm, 1910.07467). Fewer params.
- **No biases anywhere** in dense/attention projections. Simplicity + parameter saving;
  empirically harmless.
- **FFN**: `Linear(d_model→d_ff) → ReLU → Linear(d_ff→d_model)`, d_ff = 4·d_model
  (3072 for d_model=768). The 4× width is the standard Transformer ratio: gives the
  pointwise MLP enough capacity to be the model's main "compute/memory" while attention
  mixes. (Later T5 variants use gated GeGLU; baseline uses ReLU.)
- **Multi-head attention**, 12 heads, d_kv=64 per head (inner_dim = 12·64 = 768 = d_model).
  Multiple heads let the model attend to multiple positions/subspaces at once.
- **No 1/√d_k attention scaling.** The standard Transformer divides QKᵀ by √d_k to keep
  pre-softmax logits at unit scale. T5 omits it and instead **folds the scaling into the
  weight initialization** (Mesh-TF init): the query projection is initialized with std ∝
  (d_model·d_kv)^(−1/2) — the extra d_kv^(−1/2) is exactly the √d_k factor moved from the
  forward pass into init. k,v init std ∝ d_model^(−1/2); output proj std ∝
  (n_heads·d_kv)^(−1/2). Net effect: same well-scaled logits, one fewer op in the hot loop.
- **Shared vocab embedding**: input embedding, decoder input embedding, and output softmax
  projection share one matrix (weight tying) → saves params, regularizes.
- **Relative position bias** (replaces sinusoidal/learned absolute positions):
  - Self-attention is permutation-invariant; you must inject position. Absolute position
    embeddings (added to token embeddings) tie the model to absolute indices and
    generalize poorly to lengths unseen in training.
  - Use **relative** positions (Shaw et al. 2018): the bias depends on the *offset*
    (key_pos − query_pos), not absolute index. T5 **simplifies** Shaw: instead of a full
    learned vector per offset added inside the key/query computation, use a single learned
    **scalar** per (offset-bucket, head), added directly to the attention logit.
  - **Bucketing**: map each offset to one of 32 buckets; half the buckets are exact for
    small |offset| (<8), the other half grow **logarithmically** up to max_distance=128,
    and everything ≥128 collapses to one bucket. Why log buckets: nearby positions need
    fine resolution, far positions need only coarse "far-ness"; log spacing + a catch-all
    bucket lets the model **generalize to longer sequences** than seen in training. A
    layer is insensitive past 128, but stacking layers builds longer-range sensitivity.
  - For the (bidirectional) encoder, offsets are signed (separate buckets for
    past/future); for the causal decoder, only non-positive offsets are valid.
  - **Shared across all layers** (but distinct per head within a layer). Why: position
    relationships are reusable across depth; sharing slashes the parameter count of the
    position scheme to near-nothing (32×n_heads scalars per direction).
- **Optimizer**: Adafactor (Shazeer 2018) — memory-efficient (no full second-moment
  matrix), important at 11B scale. Inverse-square-root LR schedule with 10k warmup
  (constant 0.01 for warmup then 1/√step decay) — chosen over triangular because it
  doesn't require knowing total steps in advance (steps vary across experiments).
- **Tokenizer**: SentencePiece / WordPiece, 32k vocab, trained on 10:1:1:1 mix of
  En:De:Fr:Ro so the fixed vocab covers the translation target languages (model is
  English-pretrained only). Sentinel tokens added as extra IDs.

## Data: C4 (Colossal Clean Crawled Corpus)

Motivating fact: Common Crawl yields ~20TB/month of web text, but most is junk (menus,
boilerplate, gibberish, code, offensive text). To study the *effect of data*, they need a
big, clean, English corpus and they need to release it (prior pre-training corpora were
rarely public/comparable). Heuristic cleaning of the April-2019 dump:
retain only lines ending in terminal punctuation; drop pages <3 sentences / lines <5
words; drop pages with bad-words-list hits; drop "javascript"/"lorem ipsum"/curly-brace
(code) pages; strip citation markers; drop policy-boilerplate lines; dedup 3-sentence
spans; keep only langdetect-English ≥0.99. Result ≈750GB. Pre-train on 2^35≈34B tokens
(never repeating data) for the base study — far less than BERT(137B)/RoBERTa(2.2T), a
deliberate "reasonable budget" so the sweep is affordable.

Diagnostic findings (context): in-domain unlabeled data can help a specific downstream
task, but constraining domain shrinks the dataset; and a too-small corpus repeated many
times degrades performance → motivates a large diverse corpus like C4.

## Evaluation settings (pre-method facts, no outcomes)

Benchmarks/datasets that already existed and form the yardstick: GLUE & SuperGLUE
(classification meta-benchmarks: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, CB,
COPA, WiC, MultiRC, ReCoRD, BoolQ, WSC/DPR), CNN/DailyMail (abstractive summarization,
ROUGE), SQuAD (extractive QA, EM/F1), WMT EnDe/EnFr/EnRo (translation, BLEU via
SacreBLEU). All from TensorFlow Datasets. Protocol: pre-train on C4 then separately
fine-tune per task; report validation scores; greedy decoding at test; checkpoint
selection by best validation.

## Canonical code (grounding for final code)

HuggingFace `transformers` `modeling_t5.py` (snapshot in code/modeling_t5_hf.py) and the
original `google-research/text-to-text-transfer-transformer` (Mesh TF). Final code mirrors
HF structure: `T5LayerNorm` (RMSNorm), `T5DenseActDense` (FFN), `T5Attention` (with
`_relative_position_bucket` + `compute_bias`, no logit scaling), `T5LayerSelfAttention` /
`T5LayerCrossAttention` / `T5LayerFF` (pre-norm wrappers), `T5Block`, `T5Stack` (with
relative bias computed in first layer and threaded through), `T5ForConditionalGeneration`
(shared embeddings, tied LM head, `_shift_right` for teacher forcing). Plus the
span-corruption data preprocessing (noise mask → sentinel insertion → input/target split).

## Design-decision → why table (compact)

- text-to-text I/O → enables one loss/decoder + orthogonal ablation of every axis.
- prefix to select task → cheap, hyperparameter-like, no extra modules.
- encoder-decoder → bidirectional input + cross-attention; wins controlled FLOP-matched test.
- shared enc/dec params → halve params, ~no loss (ALBERT-style).
- denoising objective → beats causal LM / deshuffling for transfer.
- sentinel-replace + predict-only-corrupted → short targets ⇒ faster, ~no quality loss.
- drop random-token swap → simpler, no loss (target is generated, not encoder-position).
- span corruption, mean len 3, 15% → small quality gain + speedup; SpanBERT precedent.
- RMSNorm, no bias, pre-norm → cheaper, stable deep training.
- d_ff=4·d_model → standard capacity ratio for the pointwise MLP.
- no 1/√d_k scaling → folded into Mesh-TF init (q-init carries the d_kv^−½).
- relative position scalar bias, log buckets, shared across layers → length generalization,
  near-zero positional params, per-head expressivity.
- Adafactor + inverse-sqrt LR → memory-efficient at scale; schedule independent of #steps.
- shared 32k multilingual-aware vocab → fixed vocab must cover translation targets.
- C4 (heuristically cleaned Common Crawl, English) → large, clean, diverse, public;
  big-and-diverse beats small-repeated.
