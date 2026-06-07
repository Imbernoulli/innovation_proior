# BERT — Synthesis notes (Phase 1)

## The pain point (what existed and where it fell short)

By 2018, language-model pre-training had become the dominant recipe for transfer in NLP. Two strands:

1. **Feature-based (ELMo, Peters et al. 2018).** Train a deep biLM = a forward LSTM LM (predicts x_t from x_{<t}) and an independent backward LSTM LM (predicts x_t from x_{>t}). For each token, concatenate the two directions' hidden states at each layer, learn a task-specific softmax-weighted sum of the 2L+1 layer vectors, and feed these as *frozen features* into a task-specific architecture. Gains across QA/NLI/NER/SST. Limitation: the two LMs are trained **independently** and only **concatenated** — no single representation is jointly conditioned on both sides. It is "shallowly bidirectional": at every internal layer a forward unit still only sees the left, a backward unit only the right. Feature-based means each task still needs a hand-built architecture on top.

2. **Fine-tuning (OpenAI GPT, Radford et al. 2018).** Train a left-to-right Transformer-decoder LM: maximize Σ log P(x_t | x_{<t}) with causal (triangular) self-attention masking. Then fine-tune the *whole* network on a downstream task with one tiny output head — minimal task-specific params. Best GLUE results at the time. Limitation: it is **unidirectional**. Every token's representation only encodes left context. For token-level tasks (QA span extraction, NER) and even sentence-level tasks, ignoring right context is sub-optimal, even "harmful." The unidirectionality is forced by the LM objective, not chosen.

The common cause: **the standard LM objective (predict the next token) is intrinsically directional.** You cannot just "turn on" full bidirectional self-attention under an LM objective, because of the leakage problem below. So the architecture is hostage to the objective.

## The core difficulty (why bidirectional + LM don't mix) — the central derivation

We want a deep Transformer encoder where, **in every layer**, each token's representation attends to *all* other tokens (both sides). That is strictly more expressive than (a) one direction or (b) two separate directions concatenated only at the top.

But train it with a standard conditional LM and it collapses. In a multi-layer bidirectional encoder, the representation of position i at layer ℓ is a function of *all* positions at layer ℓ−1, including position i itself. If the target is "predict token i," then after even one bidirectional layer the model can route the identity of token i through its neighbors back to position i — each word can **indirectly "see itself."** The prediction becomes trivial; the model learns nothing about context. A causal mask is exactly what prevents this for the L→R model, but the mask is the very thing that destroys bidirectionality.

**Resolution (Masked LM / Cloze).** Decouple "which positions get rich bidirectional context" from "which positions are targets." Corrupt the input: hide a random subset of tokens, let the full bidirectional encoder process the corrupted sequence, and predict only the hidden tokens from their (uncorrupted) surrounding context. Because a hidden target token is **not present in the input**, there is no path for it to see itself — leakage is structurally impossible — yet every layer can be fully bidirectional. This is the Cloze task (Taylor 1953). The bidirectional objective *forces* the masked formulation: it is the only way to have full bidirectional attention and a non-trivial token-prediction target simultaneously.

- Mask rate **15%**: trade-off. Too few masked → too few prediction signals per sequence, expensive (you only learn from masked positions). Too many → not enough context left to reconstruct, and you destroy the sentence. 15% is the chosen middle.
- Loss only on the 15% masked positions (cross-entropy over vocab). Unlike a denoising autoencoder (Vincent 2008) which reconstructs the *entire* input, MLM predicts only the masked subset.

### The [MASK] mismatch and the 80/10/10 patch
The token `[MASK]` appears during pre-training but **never during fine-tuning** → train/test input distribution mismatch. Patch: of the 15% chosen positions, replace with `[MASK]` 80%, with a random vocab token 10%, leave unchanged 10%. Rationale:
- Keeping some real/random tokens means the encoder **cannot tell which positions are corrupted or will be queried**, so it is forced to build a good contextual representation of *every* token, not just the masked slots.
- The 10%-unchanged biases the representation toward the actually-observed word (otherwise the model could learn that an input token is "always wrong" and ignore it).
- Random replacement is only 1.5% of tokens (10% of 15%), too small to damage language understanding.
- Ablation: fine-tuning is robust to the mix; but MASK-only hurts the *feature-based* setting (mismatch is unhealable when you can't adapt the weights), and RND-only is much worse.

### MLM converges slower
Only 15% of tokens produce a loss signal per sequence vs. 100% for an L→R LM → MLM needs more steps to converge (marginally slower), but absolute downstream accuracy beats L→R almost immediately. Worth it.

## Next Sentence Prediction (NSP) — rationale
MLM is a token-level objective. But many downstream tasks (QA, NLI, paraphrase) hinge on the **relationship between two sentences** — something an LM never explicitly models. Add a second self-supervised task that requires no labels: build each example from two spans A and B; 50% of the time B truly follows A (IsNext), 50% B is a random sentence from the corpus (NotNext). Use the `[CLS]` pooled vector to do binary classification. Trivially generated from any monolingual corpus. Ablation: removing NSP hurts QNLI, MNLI, SQuAD. (Related to sentence-ranking objectives of Jernite et al. 2017 / Logeswaran & Lee 2018, but those transfer only sentence embeddings; here *all* parameters transfer.)

## Input representation (engineering choices that make one model serve all tasks)
- **WordPiece** tokenization (Wu et al. 2016), 30k vocab. Subword units solve OOV / open vocabulary; fixed small softmax. Split pieces marked `##`.
- A "sequence" = one or two "sentences" (arbitrary contiguous spans) packed together. `[CLS]` prepended (its final hidden state = aggregate sequence rep for classification). `[SEP]` separates the two sentences.
- Input embedding = **token + segment (A/B) + position** embeddings, summed. Segment embeddings (learned A/B) let the single sequence encode pairs; position embeddings are **learned** (not sinusoidal), supported length 512.
- One unified format handles single-sentence and pair tasks: (paraphrase pairs, premise-hypothesis, question-passage, or text-∅ for single-sentence) — so fine-tuning = swap inputs/outputs, add one head, tune everything end-to-end. Self-attention over a packed pair *is* bidirectional cross-attention between the two, unifying the "encode separately then cross-attend" pattern (Parikh 2016, BiDAF) into one stage.

## Architecture (inherited, not invented — it's the Transformer encoder)
- Multi-layer **bidirectional Transformer encoder** (Vaswani et al. 2017), tensor2tensor. No causal mask (vs GPT's triangular mask — the *only* architectural difference from GPT at BASE size).
- BASE: L=12, H=768, A=12, 110M params (matched to GPT for comparison). LARGE: L=24, H=1024, A=16, 340M.
- FFN inner size = 4H (3072 / 4096). GELU activation (Hendrycks & Gimpel 2016), following GPT, not ReLU.
- Self-attention scales QK^T by 1/√d_k (d_k = H/A), softmax, multi-head concat — all inherited Transformer mechanics.

## Pre-training procedure (appendix)
- Corpus: BooksCorpus (800M words) + English Wikipedia (2.5B words, text passages only). **Document-level** corpus is critical (not shuffled-sentence like Billion Word Benchmark) for long contiguous spans / NSP.
- Sample two spans, combined length ≤ 512. Mask after WordPiece at 15%.
- Adam, lr 1e-4, β1=0.9, β2=0.999, L2 wd 0.01, warmup 10k steps then linear decay, dropout 0.1, GELU.
- Batch 256 sequences × 512 = 128k tokens/batch, 1M steps ≈ 40 epochs.
- Speed trick: 90% of steps at seq len 128, last 10% at 512 (attention is O(n²); the long-sequence phase just learns the high position embeddings).
- Loss = mean MLM likelihood + mean NSP likelihood (summed).

## Fine-tuning
- Initialize from pre-trained params, add one output layer, tune all params end-to-end. Cheap (~1 hr on a TPU).
- Classification: `[CLS]` rep C → softmax(C W^T), W ∈ R^{K×H}, only new params.
- SQuAD span: learn start vector S and end vector E; P_start(i) = softmax_i(S·T_i); span score S·T_i + E·T_j, max over j≥i; loss = sum of start/end log-likelihoods. SQuAD 2.0: no-answer = span at [CLS]; predict non-null iff ŝ_{i,j} > s_null + τ.
- NER: token rep of first sub-token → tag classifier (no CRF).
- Hyperparam ranges: batch {16,32}, lr {5e-5,3e-5,2e-5}, epochs {2,3,4}. LARGE unstable on small datasets → random restarts.

## Canonical implementation (google-research/bert) — structure the final code mirrors
- `BertModel`: embedding_lookup (word) → embedding_postprocessor (add token_type + learned position, LayerNorm, dropout) → create_attention_mask_from_input_mask (padding mask only, **no causal mask** — this is the bidirectionality) → transformer_model (L layers of attention_layer with 1/√d_k scaling + FFN(4H, GELU) + residual + LayerNorm) → sequence_output (last layer) and pooled_output ([CLS] → dense + tanh).
- `get_masked_lm_output`: gather masked positions → dense(H, GELU) + LayerNorm ("transform") → logits = hidden · E^T (weights **tied** to input embedding table) + per-token output bias → log_softmax → cross-entropy averaged over masked positions (with label_weights for padding).
- `get_next_sentence_output`: pooled_output → dense to 2 logits → log_softmax → CE. (0 = IsNext, 1 = NotNext.)
- Total loss = MLM loss + NSP loss.

## Design-decision → why table (with rejected alternatives)
| Choice | Why / alternative rejected |
|---|---|
| Bidirectional encoder (no causal mask) | Strictly more expressive than L→R or shallow L→R⊕R→L concat; needed for token-level tasks with right context |
| Masked LM objective | Only way to get full bidirectional attention without target self-leakage; standard LM forces causal mask |
| 15% mask rate | Balance prediction signal vs. enough remaining context; too high destroys sentence, too low wastes compute |
| 80/10/10 mask mix | Fix [MASK]-never-at-fine-tune mismatch; force contextual rep of every token; bias toward observed word; random share tiny (1.5%) |
| Predict only masked (not full reconstruct) | Targets must be the hidden tokens; differs from denoising AE |
| NSP task | LM doesn't model inter-sentence relation needed by QA/NLI/paraphrase; cheap self-supervision; ablation confirms |
| [CLS] pooled by tanh dense | Aggregate fixed-length sequence rep for sentence/pair classification & NSP |
| Segment A/B embeddings | Encode sentence pairs in one sequence so self-attn = cross-attn; unify single & pair tasks |
| Learned position embeddings | Simpler; 512 cap fine; follows GPT (vs sinusoidal) |
| WordPiece 30k | Open vocab / OOV handling, small fixed softmax |
| Tied output↔input embeddings (MLM) | Parameter saving + regularization; standard LM practice |
| GELU, FFN=4H, 1/√d_k | Inherited Transformer/GPT choices: GELU smoother than ReLU; 4H is Transformer default; 1/√d_k keeps softmax out of saturation |
| Document-level corpus | Need long contiguous spans for NSP & long context |
| 90% len-128 / 10% len-512 | Attention O(n²); cheap to learn most of model short, only learn high position embeddings long |
| Match BASE to GPT size | Clean apples-to-apples comparison isolating bidirectionality+tasks |
