# S4 Review/Fix Changelog

## Deliverable Fixes

- `methods/s4/results/context.md:41` - Replaced the misleading "one FFT pair" phrasing with "FFT-based non-circular convolution" so the scaffold states the convolution cost without implying a single transform pair.
- `methods/s4/results/reasoning.md:29` - Corrected the bilinear-discretization derivation to use the held-input term `Delta B u_k` matching the paper's `Bbar=(I-Delta/2 A)^-1 Delta B`.
- `methods/s4/results/reasoning.md:83` - Fixed the HiPPO-LegS rank-correction constant: `P_n=sqrt(n+1/2)`, so `P_n P_k = 1/2 sqrt(2n+1)sqrt(2k+1)` and `A+PP^*=-1/2 I+S`.
- `methods/s4/results/reasoning.md:101` - Clarified that roots of unity make `(I-Abar^L z^L)` independent of the node, allowing it to fold into `Ctilde`; they do not remove the truncation factor outright.
- `methods/s4/results/reasoning.md:105` - Removed loose "killing z^L" language and kept the DFT/iFFT recovery claim exact.
- `methods/s4/results/reasoning.md:158` - Rechecked the recurrence constants against the appendix: `(I-Delta/2 A)^-1=(2/Delta)A_1`, while `Abar=A_1 A_0` and `Bbar=2A_1B`.
- `methods/s4/results/reasoning.md:160` - Distinguished the paper's general `Lambda-PQ^*` derivation from the stabilized public-code half-state convention.
- `methods/s4/results/reasoning.md:162` - Replaced underspecified per-channel parameter wording with "DPLR factors, Ctilde, and Delta" and removed the incomplete `(Lambda,P,B,C,Delta)` list.
- `methods/s4/results/reasoning.md:171` - Rewrote the code block to follow the official DPLR kernel structure: conjugate-pair Cauchy expansion, HiPPO-LegS NPLR initialization, half-state storage, `Q=P.conj()`, `z=2(1-omega)/(1+omega)`, `dt` scaling, rank-1 Woodbury, and `irfft`.
- `methods/s4/results/reasoning.md:186` - Copied `np.diag(T)` and stored diagonalized parameters as `cfloat`, keeping high-precision diagonalization without producing float/double mismatches in the illustrative layer.
- `methods/s4/results/reasoning.md:262` - Updated the closing causal chain to name the learned DPLR factors, `Ctilde`, and `Delta` instead of the incomplete parameter list.
- `methods/s4/results/answer.md:19` - Removed the inaccurate `Q (=P for LegS)` shortcut and stated the paper/code convention split explicitly.
- `methods/s4/results/answer.md:31` - Corrected the DPLR recurrence formula and the placement of the `(2/Delta)` factor.
- `methods/s4/results/answer.md:35` - Replaced the inaccurate `~5N`/missing-`Q` parameter summary with the paper and public-code parameterization.
- `methods/s4/results/answer.md:44` - Replaced the naive Cauchy helper with the reference-style conjugate-pair expansion.
- `methods/s4/results/answer.md:51` - Replaced the full-state diagonalization sketch with the official HiPPO-LegS NPLR/half-state initialization pattern and the correct `P=sqrt(.5+arange(N))` rank correction.
- `methods/s4/results/answer.md:74` - Rewrote the kernel forward pass to match canonical `SSMKernelDPLR`: half FFT nodes, bilinear node transform, `dt`-scaled `A` and Cauchy weights, `Q=P.conj()`, Woodbury correction, and `irfft`.
- `methods/s4/results/answer.md:120` - Kept the convolution non-circular by padding to `2L` and added the recurrent-step convention comment with `Abar=A1@A0`, `Bbar=2A1B`.
- `methods/s4/results/answer.md:139` - Made the model container mechanically valid by materializing `ModuleList` inputs as lists.
- `methods/s4/results/answer.md:154` - Removed the hindsight-style "later refinement" sentence and replaced it with a code-faithful paper/public-kernel convention note.

## Notes and Review Artifacts

- `methods/s4/notes/synthesis.md:38` - Fixed the notes-side NPLR sign from `+1/2 I + S` to `-1/2 I + S`.
- `methods/s4/notes/synthesis.md:74` - Fixed the notes-side DPLR recurrence convention and `(2/Delta)` placement.
- `methods/s4/notes/synthesis.md:77` - Replaced the inaccurate `5N`/`P=Q` note with the paper/public-code convention split.
- `methods/s4/notes/synthesis.md:93` - Clarified that `A+PP^*` is `-1/2 I` plus a skew-symmetric part, not purely skew-symmetric.
- `methods/s4/notes/source_matrix.md:1` - Added the strict source matrix with primary paper/source, ancestors, explainers/self-accounts, and official code evidence.
- `methods/s4/notes/discovery_synthesis.md:1` - Added a source-grounded audit synthesis covering math, code faithfulness, leak/frame checks, scaffold purity, and residual risks.
- `methods/s4/notes/strict_check_output.txt:1` - Recorded that no independent strict checker exists in this workspace.
- `methods/s4/results/.codex_review.json:4` - Updated the review metadata from stale `errored` to `not_run` with the missing-checker reason and manual audit evidence.

## Verification

- `context.md` scaffold shape: 5 `##` sections.
- `reasoning.md` scaffold purity: 0 markdown headings.
- Python snippets in `answer.md` and `reasoning.md`: syntax check passed for 2 snippets.
- Answer code smoke test: `S4Layer(4, N=8)` on input `(2,4,16)` returned `(2,4,16)`, `torch.float32`, all finite.
- Strict checker: not run because no `scripts/` directory or repository-wide `*strict*.py` checker exists.
