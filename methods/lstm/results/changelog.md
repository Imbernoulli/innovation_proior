# Changelog

## 2026-08-18 — svfix(W3_ancestors_only) verification pass
- Quality-gate review of the decisive step (CEC linear self-loop `f'·w=1`
  wrapped in multiplicative input/output gates, later a forget gate, with
  truncated backprop preserving unit-gain error flow; reasoning.md lines
  15-77): confirmed genuinely derived (the trace works the vanishing-
  gradient bound, the CEC-forcing ODE, the input/output weight conflicts,
  a hand-worked 5-step truncation example, and the forget-gate saturation
  computation, all on the page) and fully backed by the primary source
  already on disk (`refs/lstm1997_full.pdf`, Section 3.2 and Appendix
  A.1/Conclusion — quote-matched line for line in notes/sources.md).
  TRIAGE (class D) flagged `graves_phd.pdf` as unused; read it in full —
  it is a textbook-style restatement of the same 1997/2000 results in
  modern vectorized notation, never a self-account that would force a
  different step, so it was deliberately left uncited rather than grafted.
  The forget-gate motivation (unbounded state growth, tanh saturation) is
  the trace's own worked numeric computation, not sourced material.
  No rewrite; no factual errors found in the decisive step or its
  surrounding derivation.
