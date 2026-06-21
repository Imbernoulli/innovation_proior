# FlashAttention-2 Synthesis

This note is superseded by `notes/discovery_synthesis.md`, which is the strict source-grounded synthesis used for the repaired deliverables.

Key audited points:

- Exact target operator: `O = softmax(QK^T) V` with optional score scale and masks.
- Correct online-softmax accumulator factor: `alpha = exp(m_old - m_new) <= 1`.
- Save `L = m + log(ell)` for backward; recompute `P = exp(S - L)`.
- Forward parallelism: query row blocks are independent and form the sequence-length launch axis.
- Backward parallelism: key/value column blocks own `dK,dV` and accumulate into `dQ`.
- Warp partition: split query rows across warps so each warp owns complete output rows.
- Causal cases: skip above-diagonal blocks, do no elementwise mask below diagonal, mask only the boundary block.

The previous draft contained a confusing discussion of inverse notation around the accumulator update. The deliverables now state the code-faithful factor directly.
