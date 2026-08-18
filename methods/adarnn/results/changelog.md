# AdaRNN changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, after the hand-traced alpha ratchet: added the constraint on the grow-mask
  threshold, grounded in the first author's own issue reply (`notes/sources.md`) — the test is
  `d_new > d_old + eps`, and `eps` has to be set against the scale of the distance function, not out of
  habit: `1e-5` is invisible next to an order-one adversarial discrepancy but larger than a genuine
  epoch-to-epoch increase in a cosine distance, in which case the mask is empty on every epoch and
  `alpha` stays at its uniform initialization for the whole run. The passage also records why this is
  dangerous rather than merely wrong: it fails silently, with both losses still decreasing.
- FLAGGED, not changed: `code/qlib_pytorch_adarnn.py` is a vendored snapshot and ships exactly that
  broken combination — `loss_type="cosine"` (line 47) with `epsilon = 1e-5` (line 534). It is left as
  the snapshot it is; the constraint is now stated in the trace. `results/answer.md` and
  `results/train_answer.md` do not reproduce `update_weight_Boosting`, so no inconsistency was
  introduced there.
- Confirmed unchanged: the TDC max-not-min derivation. The AAAI 2024 tutorial by a co-author states
  the same worst-case rationale ("the most distinct periods ... represents the worst case of temporal
  covariate shift"), and the trace's own toy-stream check (maximizing cut at the true boundary, 4.09,
  versus the minimizing cut, 2.29) already goes beyond it. No factual errors found elsewhere.
