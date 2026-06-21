# Reformer Synthesis Notes

## Evidence Spine

- Primary: Kitaev, Kaiser, and Levskaya, "Reformer: The Efficient Transformer", arXiv:2001.04451; local source in `refs/primary/source/` and PDF/text in `refs/primary/`.
- Canonical implementation: Google Trax, commit `31022d6cd7dd525ed11a04d84cd3936228499173`, especially `trax/layers/research/efficient_attention.py`, `trax/layers/reversible.py`, and `trax/models/reformer/reformer.py`.
- Self-account: Google Research blog post by Nikita Kitaev and Lukasz Kaiser, January 16, 2020. This is a lab/author technical account, not a personal discovery memoir.
- Ancestors: Transformer attention, angular LSH/cross-polytope hashing, RevNet reversible residual layers, Sparse Transformer, and Adafactor.
- Explainers: Hugging Face Reformer docs and a Weights & Biases Reformer report, used only as secondary checks.

## Corrected Math

- The initial parameter example is a largest reported Transformer layer, not an attention layer.
- Standard attention is $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. The LSH derivation omits the scale for clarity, but the implementation equivalent is query unnormalized and key L2-normalized.
- The paper's chunk relation is `chunk_len = m = 2l/n_buckets`; therefore a code path using a fixed `chunk_len` should default to `n_buckets = 2 * length // chunk_len`. The stale deliverable code used `length // chunk_len`, which made the average bucket the same size as the chunk instead of half the chunk.
- The sorted chunk extension is exactly one previous chunk in the basic causal LSH setup:
  $$\widetilde{\mathcal{P}}_i=\{j:\lfloor s_i/m\rfloor-1\le\lfloor s_j/m\rfloor\le\lfloor s_i/m\rfloor\}.$$
- Multi-round union formula signs are:
  $$o_i=\sum_r \exp(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i))o_i^{(r)}.$$
  The duplicate correction is a positive mask term $\log N_{i,j}$, subtracted in the logit exponent.
- Self attention is excluded by a large finite positive penalty in the mask, $10^5$, which is subtracted from the dot product. The finite value preserves the first-token/no-other-target case.
- RevNet inversion signs are subtraction in reverse order:
  $$x_2=y_2-G(y_1),\qquad x_1=y_1-F(x_2).$$
- Feed-forward chunking is functionally identical because the FFN is position-wise; it changes peak memory, not the computed mapping.

## Canonical Code Corrections

- The previous notes used a third-party PyTorch port as the code map. That is not canonical for this paper. The paper footnote points to Google Trax.
- Trax `hash_vecs` implements angular LSH by random rotations, concatenating signs, and `argmax`. For large bucket counts it factorizes the bucket space into even factors.
- Trax `PureLSHSelfAttention.hash_vectors` defaults to `n_buckets = 2 * max(1, length // chunk_len)` and offsets bucket IDs per hash round before sorting.
- Trax `attend(..., k=None)` uses shared-QK. Its `length_normalized(k) / sqrt(d)` is equivalent to L2-normalized keys; applying an additional `dim**-0.5` after `F.normalize` in PyTorch would be too much scaling.
- Trax masks subtract `1e9` for future/padding and `1e5` for self. The stale deliverable code set self logits to `-5e4`; that is not the same as the paper mask case or the Trax implementation.
- Trax combines hash rounds by softmaxing the per-round log partitions and summing outputs. Current Trax code does not explicitly compute the paper appendix's `log N_{i,j}` duplicate correction, so the deliverable now separates exact paper math from code-faithful behavior.
- Trax `ReformerLM` uses `Dup()` before the reversible stack and `Concatenate()` after it. The stale PyTorch adaptation averaged the two halves, which is a third-party-port behavior, not the canonical Trax shape.

## Hindsight and Scaffold Pass

- `context.md` keeps exactly five `##` sections and does not name the target method.
- The context code is a plain pre-method self-attention scaffold with stubs; no LSH, reversible, or chunking implementation is smuggled into the setup.
- `reasoning.md` is now continuous first-person present-tense reconstruction with no markdown headers and no source/paper/codebase meta-commentary.
- `answer.md` is allowed to be final-method-facing and therefore names Reformer and the canonical code commit.

## Remaining Implementation Caveat

The code skeleton in `answer.md` is intentionally a compact reference-shaped artifact, not a full Trax copy. It captures the faithful mechanics that matter for the paper-to-reasoning deliverable: bucket count, signed-axis hash, sort key, shared-QK normalization, masks, round combination, and reversible inversion.
