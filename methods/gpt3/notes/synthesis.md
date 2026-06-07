# Synthesis — GPT-3 / "Language Models are Few-Shot Learners"

## The pain point (what existed and where it fell short)

The dominant paradigm circa 2019–2020: **pre-train a Transformer LM on a big corpus, then fine-tune on a task-specific labeled dataset.** GPT-1 (Radford 2018), BERT (Devlin 2018), ULMFiT (Howard 2018), then RoBERTa, XLNet, ALBERT, T5. This removed task-specific *architectures* (one model, swap the head) but NOT task-specific *data + gradient updates*. Three concrete complaints:

1. **Practical**: every new task needs thousands–hundreds of thousands of labeled examples. The space of useful language tasks (correct grammar, give an example of a concept, critique a story) is unbounded; collecting a supervised set per task doesn't scale.
2. **Generalization**: a large model fine-tuned on a narrow distribution exploits spurious correlations of that distribution; out-of-distribution generalization can be poor, and benchmark scores at "human level" overstate true capability (Hendrycks 2020 — bigger models don't necessarily generalize OOD better; McCoy 2019, Gururangan 2018, Niven 2019 — annotation artifacts).
3. **Comparison to humans**: humans do a new language task from a short instruction or a couple of demonstrations — no gradient descent on 10k examples. Current NLP can't. This is both a conceptual gap and a practical one (fluidly switching tasks mid-dialogue).

## The two threads that get braided into the idea

**Thread A — in-context learning as task specification (GPT-2, Radford 2019).** GPT-2 showed a pure autoregressive LM, prompted in zero-shot with a natural-language framing ("TL;DR:" for summarization), could do downstream tasks with *no* fine-tuning. The input text itself is the task spec. This is the seed. But GPT-2's zero-shot numbers were far below fine-tuning (e.g. 4% on Natural Questions; 55 F1 CoQA, ~35 points behind SOTA). As a practical method, in-context learning looked dead on arrival.

**Thread B — smooth power-law scaling of LM loss (Kaplan 2020, "Scaling Laws for Neural Language Models").** Test cross-entropy loss falls as a power law in model size, data, and compute over many orders of magnitude:
- L(N) ≈ (Nc/N)^αN, αN ≈ 0.076, Nc ≈ 8.8e13 non-embedding params
- L(D) ≈ (Dc/D)^αD, αD ≈ 0.095, Dc ≈ 5.4e13 tokens
- L(Cmin) ≈ (Ccmin/Cmin)^αCmin, αCmin ≈ 0.050
- Compute-optimal training: **train very large models on relatively modest data and stop well before convergence.** Optimal model size grows with compute as N_opt ∝ C^0.73 (so most extra compute → bigger model, much less → more tokens/steps). Loss is insensitive to architectural shape (depth/width/aspect ratio) over a broad range; N dominates.

**The braid (the central bet):** capability tracks loss; in-context learning is *one of* the capabilities packed into the parameters by next-token prediction. If loss improves smoothly and predictably with scale, in-context-learning ability should too — and GPT-2's weakness might simply be that it was too small. So: don't redesign the method; scale the GPT-2 recipe by ~100x and measure whether few-shot in-context learning crosses into usefulness. The motivating diagnostic finding (the paper's own, but pre-method-fact about *scaling behavior*): aggregate few-shot accuracy rises faster with model size than zero-shot accuracy — the gap between 0/1/few-shot *widens* with scale, i.e. larger models are better in-context learners (better "meta-learners").

## Why next-token prediction should yield few-shot learning without gradient updates (the key derivation)

The training objective is just maximize Σ log p(x_t | x_<t) over web text. Why would that produce learning-from-examples at inference?

- Natural text is full of **implicit task demonstrations**: lists of translated pairs, Q→A patterns, "word: definition" entries, arithmetic worked out, repeated sub-tasks inside one document. To minimize loss on the *next* token of such a passage, the model is forced to infer "what regular relationship governs this sequence so far" and apply it. That inference *is* in-context learning.
- Frame it as meta-learning with an **outer loop** = SGD over the corpus (slow, updates weights, learns a broad repertoire of skills + pattern recognition) and an **inner loop** = the forward pass over one sequence (fast, no weight update, recognizes/adapts to the task implied by the prefix using only activations across positions). The autoregressive objective, averaged over a sufficiently diverse corpus, is implicitly an average over a huge distribution of tasks — exactly the meta-learning setup (Hochreiter 2001 "learning to learn by gradient descent in the recurrent net's activations"; RL² Duan 2016 — adaptation lives in activations, outer loop in weights). Stuffing the context with K demonstrations is structurally RL²-like.
- Therefore a few-shot "prompt" — K (context, completion) pairs then a final context — is not a new mechanism. It's the same conditional-density model p(completion | context, demonstrations). The K demonstrations sharpen which task the model thinks it's in; no fine-tuning needed. K is bounded by the context window (n_ctx=2048), typically 10–100 examples.

So the method *is*: don't change the objective, change the scale, and reinterpret evaluation as conditioning the LM on a text-encoded task. Zero/one/few-shot are points on a "how much task data" spectrum, parallel to fine-tuning but with **zero gradient updates at test time**.

## The architecture (= GPT-2, scaled) and every design knob → why

Same as GPT-2 (Radford 2019): decoder-only Transformer, with the GPT-2 tweaks:
- **Pre-normalization** (LayerNorm before each sublayer, not after) + a final LN. Why: post-LN deep Transformers have unstable gradients at depth; pre-LN keeps a clean residual identity path so 96-layer training is stable without delicate warmup. (Original Transformer was post-LN; GPT-2 moved to pre-LN to scale depth.)
- **Modified (scaled) initialization**: residual-path projection weights scaled by 1/√(2·n_layers) (i.e. 1/√N_residual_paths). Why: each residual add accumulates variance across layers; without down-scaling the residual stream variance grows ~linearly in depth and blows activations up at 96 layers. The 2× is because each block has two residual adds (attn + MLP). (minGPT applies exactly std=0.02/√(2·n_layer) to c_proj.weight.)
- **Reversible BPE tokenization** (byte-level BPE, vocab ~50257). Why byte-level: no UNK, can represent any string, no language-specific preprocessing — necessary for a task-agnostic model that must read arbitrary prompts (arithmetic digits, scrambled words, novel nonsense words).
- **FFN width = 4·d_model.** Inherited Transformer ratio; Kaplan shows loss is insensitive to aspect ratio so they keep the standard 4×.
- **d_head = 128** (mostly fixed), n_heads scales so n_heads·d_head = d_model. Choices set "based on computational efficiency and load-balancing in the GPU layout," and Kaplan says loss is insensitive to these within a broad range — so pick what maps cleanly onto the parallelism.
- **Context window n_ctx = 2048** (GPT-2 used 1024). Doubled so few-shot prompts (10–100 demonstrations) fit; the context window literally bounds K.
- **Alternating dense and locally-banded sparse attention** (Sparse Transformer, Child 2019). The *only* architectural change vs GPT-2. Why: full attention is O(n²) in sequence length; at n_ctx=2048 and 96 layers that's expensive. Alternating dense layers with banded-sparse layers cuts attention cost while keeping enough global mixing. (Note: minGPT/nanoGPT implement only dense causal attention — the canonical clean impl; the sparse pattern is an efficiency layer on top, contributes <10% of compute per the paper's own compute accounting which ignores attention entirely.)

**8 model sizes**, 125M → 175B (3 orders of magnitude), specifically to *trace the scaling curve* of in-context learning, not just train one big model. Param table:
| size | n_layers | d_model | n_heads | d_head | batch (tok) | LR |
|---|---|---|---|---|---|---|
|125M|12|768|12|64|0.5M|6.0e-4|
|350M|24|1024|16|64|0.5M|3.0e-4|
|760M|24|1536|16|96|0.5M|2.5e-4|
|1.3B|24|2048|24|128|1M|2.0e-4|
|2.7B|32|2560|32|80|1M|1.6e-4|
|6.7B|32|4096|32|128|2M|1.2e-4|
|13B|40|5140|40|128|2M|1.0e-4|
|175B|96|12288|96|128|3.2M|0.6e-4|

Pattern: **bigger model → larger batch, smaller LR.** Why: Kaplan + McCandlish 2018 (gradient noise scale) — the critical batch size grows as loss falls, so big models tolerate (need) bigger batches; and bigger models need a smaller LR for stability. They measured the gradient noise scale to set batch size.

## Data and training knobs → why

- **Corpus**: filtered Common Crawl (410B tok, 60% weight) + WebText2 (19B, 22%) + Books1 (12B, 8%) + Books2 (55B, 8%) + Wikipedia (3B, 3%). Trained 300B tokens total.
- **Why not just raw Common Crawl**: unfiltered CC is low quality. Quality filter = logistic-regression classifier (Spark HashingTF features) trained to distinguish curated text (WebText, Wikipedia, books) from raw CC; keep a doc iff `np.random.pareto(α) > 1 − doc_score`, α=9. Why the Pareto-noise threshold instead of a hard cutoff: keeps mostly high-scoring docs but still admits some out-of-distribution docs (diversity); α=9 chosen to match the score distribution on WebText.
- **Fuzzy dedup** (MinHashLSH, 10 hashes) within/across datasets, and remove WebText from CC. Why: a high-capacity model memorizes; duplicates inflate effective epochs on repeated content and corrupt the held-out validation signal. ~10% size reduction.
- **Non-proportional sampling**: high-quality sets are up-weighted (CC seen 0.44 epochs, Wikipedia 3.4 epochs). Accepts a little overfitting on small high-quality sets for better average quality.
- **Sequence packing**: always train on full 2048-token sequences; pack multiple short docs separated by an end-of-text token; no special cross-document masking (the eot token tells the model the segments are unrelated). Why: efficiency — no wasted padding.

## Optimizer / schedule → why

- **Adam**, β1=0.9, β2=0.95, ε=1e-8. (β2=0.95 vs the usual 0.999 — shorter second-moment memory, more stable for large-batch LM training.)
- **Global grad-norm clip at 1.0.** Stability at scale.
- **Weight decay 0.1** (decoupled, AdamW-style; Loshchilov 2017) — light regularization. minGPT's `configure_optimizers` applies decay to Linear weights only, not to biases/LayerNorm/embeddings.
- **LR schedule**: linear warmup over first 375M tokens, then cosine decay to 10% of peak over 260B tokens, then flat at 10%. Warmup: Adam's second-moment estimate is noisy early → large early steps destabilize; warmup avoids it. Cosine decay: standard smooth annealing.
- **Batch-size warmup**: ramp batch 32k → full over first 4–12B tokens. Early in training the gradient noise scale is small, so a small batch is compute-optimal; grow it as training proceeds.
- **Data sampled without replacement within an epoch** to minimize overfitting.

## Compute accounting (the bridge to scaling laws)

Forward+backward ≈ 6N flops per parameter per token (2 for fwd add+multiply per active param, ×3 for backward). Total ≈ 6·N·D. 175B over 300B tokens ≈ 3.14e23 flops ≈ 3640 PF-days. The point of training on "only" 300B tokens (rather than to convergence) is exactly Kaplan's compute-optimal prescription: at fixed compute, spend it on a bigger model, fewer tokens, stop early. (GPT-3 3B ≈ 10× RoBERTa-Large's params but ~same 50 PF-days, because it sees far fewer tokens.)

## Evaluation protocol (pre-method yardstick + the new in-context protocol)

- Multiple choice: give K (context, correct completion) demos, then context; compare LM likelihood of each candidate completion. Per-token-length-normalized likelihood usually; for ARC/OpenBookQA/RACE normalize by unconditional P(completion | "Answer:") to remove length/frequency bias.
- Binary classification: rename labels to meaningful words ("True"/"False") and treat as multiple choice.
- Free-form: beam search (width 4, length penalty α=0.6, per T5); score with F1 / BLEU / exact-match per dataset convention.
- K from 0 to context-window max; tune K on dev, report on test.
- Benchmarks that exist as the yardstick: SuperGLUE, LAMBADA, CoQA, TriviaQA/NaturalQuestions/WebQuestions, SQuAD/RACE/QuAC/DROP, Winograd/Winogrande, HellaSwag/StoryCloze/PIQA, WMT translation, ANLI, plus synthetic probes (arithmetic, word-scramble, SAT analogies, novel-word use).

## Limitations / honest uncertainties (knowable at derivation time, not posterior results)

- Autoregressive-only (no bidirectionality/denoising) → likely weaker on compare-two-spans tasks (WIC, ANLI, some reading comprehension). Chose AR because it's straightforward to sample and to compute likelihoods.
- The pretraining objective weights every token equally — no notion of importance; pure prediction may eventually hit a ceiling; no grounding in non-text modalities.
- Poor pre-training sample efficiency (sees far more text than a human lifetime).
- Open question whether few-shot "learns from scratch at inference" or "recognizes a task seen in pretraining" — a spectrum; synthetic tasks (scrambling, nonsense words) look more de-novo, translation must be from pretraining.
- Inference cost/inconvenience of a 175B model.

## Canonical implementation for grounding the code

minGPT (karpathy) — `mingpt/model.py`: NewGELU, CausalSelfAttention (dense), Block (pre-LN, MLP 4×), GPT (wte+wpe, n_layer blocks, ln_f, lm_head, scaled c_proj init std=0.02/√(2·n_layer)), forward = cross-entropy next-token loss, configure_optimizers (decay/no-decay split, AdamW β + weight decay), generate (autoregressive sampling, crop to block_size). This is GPT-2/GPT-3 dense architecture exactly; GPT-3 adds the alternating sparse-attention layers and scales the config. nanoGPT confirms the same structure.

The few-shot interface is *not* a model change — it's a prompt builder + a likelihood scorer wrapped around `model.forward`/`model.generate`. The final code shows: the GPT module (grounded in minGPT), the prompt-construction for K-shot, and the likelihood/generation-based scoring — i.e. the method = architecture + the in-context-learning evaluation harness.

## context.md scaffold plan (pre-method, piece-for-piece with final code)

Pre-method skeleton known before the method: byte-level BPE tokenizer, an embedding, *some* sequence model (the slot), output projection over vocab, next-token cross-entropy, Adam training loop, autoregressive sampler. Empty slots: `class SequenceModel` (the architecture we'll design — TODO), and `def specify_task(...)` / `def score(...)` (how we'll turn a task into model input and read an answer out — TODO; pre-method we only know "feed text, read text"). Final code fills: SequenceModel → decoder-only pre-LN Transformer (+scaled init, +context window, +optional sparse attention); specify_task → K-shot prompt builder; score → likelihood-compare / beam generate.
