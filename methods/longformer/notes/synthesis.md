# Longformer — synthesis (pre-Phase-2 notes)

## Pain point / research question
Self-attention is O(n²) in both time and memory because Q·Kᵀ produces an n×n score matrix. With 512-token pretrained encoders (BERT/RoBERTa), long documents (QA contexts of thousands–tens of thousands of tokens, coref over whole documents, char-LM over 32K chars) cannot be processed directly. Workarounds — truncation, chunk-encode-then-combine, two-stage retrieve-then-read — lose cross-chunk information or cascade errors and need bespoke architectures per task. Goal: an attention that scales linearly with n, works for both autoregressive LM and the pretrain/finetune (bidirectional, MLM) paradigm, and is a drop-in replacement for full attention in an existing pretrained model.

## Key insight chain (discovery order)
1. Most linguistic dependence is local — adjacent tokens, like a CNN's local receptive field. So most of the n² entries are wasted. Replace the dense score matrix with a **sliding window**: each token attends to w/2 tokens each side. Cost O(n·w), linear.
2. But local-only loses long-range. Resolve via depth, like a CNN: one window layer reaches w/2 each side; stacking L layers grows the one-sided reach to L·w/2, so receptive field ≈ **L·w**. Top layers see (almost) the whole document.
3. L·w may still be too small for 32K sequences without huge w or L. **Dilate** the window (gaps of size d), like dilated/à-trous CNNs (WaveNet). Same number of attended positions (same compute), but one layer now spans w/2·d each side → receptive field **L·d·w**. Multi-head: give some heads dilation (long reach) and keep some heads d=1 (preserve local detail).
4. Local attention can build token reps but cannot, in a few layers, aggregate the whole sequence into a single decision token, nor let every doc token see the question. Add **global attention** on a few task-chosen tokens ([CLS] for classification; all question tokens for QA): they attend to everything and everything attends to them (symmetric). g is constant ⇒ still O(n).
5. Global attention plays a different role than local; reusing the local Q/K/V projections is too rigid. Use **separate Q_g/K_g/V_g** projections, initialized as copies of the local ones.
6. Banded/dilated matmul isn't in PyTorch/TF: implement it (loop = test-only; chunked overlapping-blocks single matmul = pretrain/finetune, no dilation; TVM CUDA kernel = char-LM, supports dilation + autoregressive).
7. To get a long-doc encoder cheaply: **continue pretraining from RoBERTa** with window=512 (matches RoBERTa compute), extend learned position embeddings to 4096 by **copying** the 512-block repeatedly (preserves the learned local-position bias), MLM for a small number of updates. Char-LM uses **staged training**: start short seq/window, double both and halve LR each phase; **increasing window with depth** (small at bottom, large at top); dilation only on 2 heads of upper layers.

## Receptive field / complexity derivations (self-verified)
- Window, 1 layer: token attends w/2 each side ⇒ field = w+1 ≈ w. Compute = n·w ⇒ O(n·w).
- L layers: one-sided reach = L·(w/2) ⇒ total field ≈ L·w. Correct (paper: ℓ×w).
- Dilation d: one-sided reach per layer = (w/2)·d ⇒ field after L layers = L·d·w. Correct.
- Global: g tokens attend to all n, all n attend to g ⇒ 2·g·n ops; g constant ⇒ O(n). Combined O(n·w + n·g) = O(n). Correct.

## Design-decision → why (with rejected alternatives)
- **Why window, not full attention?** Local context dominates; full n² is mostly wasted and infeasible for long n. Rejected: full attention (O(n²) OOM); truncation/chunking (info loss).
- **Why depth gives global reach?** CNN analogy — stacked small kernels build large receptive field. Cheaper than one wide window.
- **Why dilation?** Enlarge receptive field L·d·w without extra compute. Rejected: just widening w (raises compute linearly); more layers (raises compute/params).
- **Why dilation only on some heads / upper layers?** Heads with d=1 keep fine local detail; dilated heads grab distant tokens. Lower layers need full local capacity. Ablation: dilation on 2 heads beats none; all-heads dilation hurts.
- **Why increasing window with depth (char-LM)?** Lower layers build local features (cheap, small w); upper layers integrate globally (large w). Ablation: increasing > fixed > decreasing.
- **Why global attention?** Local-only can't form a whole-sequence decision vector or let doc see question in few layers. [CLS]/question-token global fixes this with constant cost.
- **Why symmetric global?** A global token must both read all and be read by all to act as an information hub.
- **Why separate global projections?** Global serves a different function than local; sharing projections is too constrained — separate Q_g/K_g/V_g is critical for downstream. Init = copy of local to start from a good point.
- **Why window=512 for pretraining?** Matches RoBERTa's seq length ⇒ same compute, smooth continuation. No dilation here (incompatible with pretrained weights — empirically hurt).
- **Why copy position embeddings?** RoBERTa has learned a strong local-position bias (attend prev/next token); copying the 512 block preserves it everywhere except partition boundaries; random init destroys it (BPC 10.3 → 1.96 with copy).
- **Why continue from RoBERTa not train from scratch?** MLM pretraining is expensive; minimal changes + few updates suffice.
- **Why staged training (char-LM)?** Model must learn local context first; start short/cheap, grow seq & window, halve LR — keeps the expensive long-seq phase short.
- **Why three implementations?** loop (correct, dilation, but slow → test); chunks (single matmul, fast, 2× memory, no dilation → pretrain/finetune); TVM CUDA (memory-optimal, dilation, autoregressive → char-LM).

## Canonical implementation (allenai/longformer) — structure to mirror
- `LongformerSelfAttention(nn.Module)`: local q/k/v + global query_global/key_global/value_global; q /= sqrt(head_dim); banded QKᵀ via `sliding_chunks_matmul_qk` (or tvm); `mask_invalid_locations`; add global columns (einsum 'blhd,bshd->blhs'); softmax in fp32; PV via `sliding_chunks_matmul_pv`; then recompute global rows with bmm using global projections; attention_mask convention: −ve=no attn, 0=local, +ve=global.
- `sliding_chunks.py`: `_chunk` (overlapping 2w blocks via as_strided), `_skew`/`_skew2` (diagonals↔columns), `sliding_chunks_matmul_qk` returns (bsz, seqlen, heads, 2w+1), `sliding_chunks_matmul_pv`, `pad_to_window_size`.
- `diagonaled_mm_tvm.py`: banded matmul, offset = D[head]*(k−w), per-head dilation D.

## Ancestors (cite as prior art)
- Vaswani et al. 2017 (Transformer, O(n²) self-attention, Q/K/V, softmax(QKᵀ/√d)V).
- Devlin et al. 2018 (BERT, MLM, [CLS], 512 limit); Liu et al. 2019 (RoBERTa checkpoint).
- Dai et al. 2019 Transformer-XL; Sukhbaatar et al. 2019 Adaptive Span; Rae et al. 2019 Compressive — ltr recurrence, char-LM only, not bidirectional/MLM.
- Child et al. 2019 Sparse Transformer (dilated sliding window of 8×8 blocks, BlockSparse kernels, generation/LM only); Kitaev et al. 2020 Reformer (LSH attention).
- CNN analogies: dilated convs (van den Oord WaveNet 2016); Wu et al. 2019 (lightweight/dynamic conv ~ attention with local span).
- Clark et al. 2019 (BERT attention heads attend local/prev-next); Kovaleva et al. 2019 (importance of local context).
- TVM (Chen et al. 2018) for the kernel.

## Out of scope (do not write)
Proposed-method eval results: text8/enwik8 BPC, finetune accuracy/F1 tables, LED ROUGE. Motivating findings IN scope: full attention OOM/quadratic curve, local-context head studies, copy-init BPC as a diagnostic of the *initialization*, ablation rationale (used to justify design, but the trace ends before reporting them as wins).
