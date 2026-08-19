# Changelog

## 2026-08-18 — svfix(W3_ancestors_only) verification pass
- Quality-gate review of the decisive step (locality + weight-tying -> convolution;
  shared-weight backprop, reasoning.md paras 9-29): confirmed genuinely derived
  (two on-page worked checks: hand-traced shift example, finite-difference
  gradient check) and fully backed by the primary source (LeCun et al. 1998,
  pp.6 and 9). No source grafted — left as-is. See notes/sources.md.
- Fixed a factual error in reasoning.md para 35: fan-in weight-initialization
  scaling was stated as 1/sqrt(fan-in) (the independent-inputs case); the primary
  source (LeCun et al. 1998, Appendix A, p.41) explicitly chooses the
  correlated-inputs case (std scales linearly with fan-in -> init range
  ~1/fan-in, i.e. U(-2.4/F_i, 2.4/F_i)) because units in a shared kernel receive
  highly correlated inputs from overlapping receptive fields. Rewrote the passage
  to derive the linear-vs-sqrt choice from that correlation instead of asserting
  the wrong exponent. OCR'd refs/lecun1998_lenet.pdf (custom Type3 font defeats
  pdftotext/pdfplumber/PyMuPDF text extraction) to refs/lecun1998_lenet_ocr.txt to
  make the source quotable/verifiable.
