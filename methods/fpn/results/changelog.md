# Changelog

- `results/context.md:7` softened the research-question framing to avoid hindsight-y "deliver the accuracy" wording while preserving the pre-method scale-coverage requirement.
- `results/context.md:23` corrected the RPN positive-label case from `>=0.7` to IoU `>0.7` and added the separate best-IoU positive-anchor case.
- `results/context.md:60` and `results/context.md:71` added missing `super().__init__()` calls so the PyTorch scaffold registers submodules correctly.
- `results/reasoning.md:19` replaced an unsupported claim about 3x3 lateral convolutions with the source-grounded reason for 1x1 lateral projection and separate 3x3 output smoothing.
- `results/reasoning.md:43` added the RoI box-coordinate convention case: Detectron's legacy `x2 - x1 + 1` area versus continuous-coordinate `x2 - x1`, with the requirement that training and testing use one convention consistently.
- `results/reasoning.md:113` through `results/reasoning.md:125` updated the RoI-level code to expose `eps=1e-6`, `legacy_plus_one=True`, positive-size validation, and clamping after the paper's floor/log2 formula.
- `results/answer.md:28` made every RPN anchor-label case explicit: highest-IoU positive, IoU `>0.7` positive, and IoU `<0.3` negative.
- `results/answer.md:36` added nearest-log-space assignment for intermediate mask scales.
- `results/answer.md:86` through `results/answer.md:111` updated `assign_roi_to_level` for the canonical Detectron convention, empty input handling, positive-size checks, and Detectron's `1e-6` epsilon.
- `notes/source_matrix.md:7` through `notes/source_matrix.md:15` added the strict source matrix covering primary paper artifacts, ancestors, explainers, official code, and self-account search.
- `notes/source_matrix.md:18` through `notes/source_matrix.md:20` recorded the math/case and canonical-code checks for RoI assignment and RPN anchors.
- `notes/discovery_synthesis.md:19` through `notes/discovery_synthesis.md:28` documented the math and code-faithfulness fixes.
- `notes/discovery_synthesis.md:30` through `notes/discovery_synthesis.md:33` documented the posterior-leak, scaffold-purity, and in-frame-voice review.
- `refs/self_accounts/search_log.md:3` through `refs/self_accounts/search_log.md:15` recorded the self-account search and the absence of a usable author first-person technical retrospective.
- `results/.codex_review.json:3` through `results/.codex_review.json:10` replaced the previous uncertain marker with this completed review record.
- `notes/strict_check_output.txt:1` through `notes/strict_check_output.txt:3` recorded the strict checker command and passing result.

Validation:
- `python /srv/home/bohanlyu/.codex/skills/paper-to-reasoning-strict/scripts/check_strict_method.py methods/fpn` -> `STRICT CHECK PASSED`.
- Answer Python code blocks AST-parse; RoI smoke cases returned `legacy_levels=4,3,5,2,5`.
- Feature builder smoke test returned P2-P6 shapes `1x256x64x64;1x256x32x32;1x256x16x16;1x256x8x8;1x256x4x4`.
- Focused greps found no target-name/paper leakage in `results/context.md` and no markdown-header lines in `results/reasoning.md`.
