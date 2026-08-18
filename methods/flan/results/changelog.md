# Changelog - flan

## 2026-08-18 — svfix repair pass (D_candidate)
- `results/reasoning.md:34` kept the arXiv-v1-grounded decisive step ("For these five, I drop
  the suffix and fall back to plain rank classification") but the landed method had not been
  updated to match it — the first svfix pass added the reasoning claim without touching the
  code, so the trace contradicted its own landing. Fixed by making the carve-out real: added
  `NO_OPTIONS_DATASETS = {"hellaswag", "piqa", "record", "wsc273", "winogrande"}` and had
  `format_example` withhold the OPTIONS suffix for those five dataset names, in
  `results/reasoning.md`, `results/answer.md`, and `results/train_answer.md` (identical patch
  in all three, verbatim-consistent with `train_answer.md`'s contract of copying code from
  `answer.md`).
- `results/answer.md` and `results/train_answer.md` prose (classification/OPTIONS paragraphs)
  updated to state the same carve-out in words, so the landing text and the landing code agree.
- Source: `notes/sources.md` — arXiv:2109.01652v1 (pre-review preprint), Section "Commonsense
  Reasoning & Coreference Resolution": "we note a further limitation with Flan for these five
  language modeling tasks, including options actually hurts performance, and so the reported
  results are for rank classification without options." This sentence is absent from the final
  camera-ready already on disk (`src/iclr2022_conference_final.tex`).
