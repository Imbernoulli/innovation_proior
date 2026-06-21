# Innovation-Prior Expansion — Wave: Math / Physics / Chemistry / CS (best-paper sweep)

Candidate lists for ~400 new discovery-reasoning traces, produced by one web-research scout per
domain. Each candidate has a verified title/author/venue/year/arxiv and a one-line **derivable
conceptual move**. All slugs are deduplicated against the existing 1261 methods (zero collisions)
and across files (one cross-file dup, noted below).

**Status: awaiting sign-off.** Nothing has been generated yet — these are selection lists only.
After sign-off, each row becomes a full 4-file method (`context.md` / `reasoning.md` / `answer.md`
/ `train_answer.md`) via the `paper-to-reasoning` skill, generated in batches.

## Counts

| Domain | File | Candidates | Split |
| --- | --- | --- | --- |
| Mathematics | [math_candidates.md](math_candidates.md) | 114 | ~59 recent (2010–25) / ~55 classic |
| Physics (theory only) | [physics_candidates.md](physics_candidates.md) | 104 | ~38 recent / ~66 classic |
| Chemistry (theory/method) | [chemistry_candidates.md](chemistry_candidates.md) | 42 | classic + recent mix |
| CS — empirical venues | [cs_empirical_candidates.md](cs_empirical_candidates.md) | 94 | CVPR/ACL/NeurIPS/ICML/ICLR best papers 2015–25 |
| CS — theory venues | [cs_theory_candidates.md](cs_theory_candidates.md) | 72 | STOC/FOCS/COLT best papers 2010–25 |
| **Total** | | **426 unique** | |

## Selection discipline applied

- **Theory-leaning, verifiable on paper.** Math/Physics/Chemistry land on theorem+proof / formula
  / mechanism — no experiments. Physics: **experimental physics excluded by construction** (theory
  papers kept in place of detections, e.g. inflation theory instead of CMB measurement).
- **Every candidate has a re-derivable conceptual move** — a key lemma, ansatz, construction,
  reformulation, architecture, or training objective. Pure technical-verification results were skipped.
- **CS excludes evaluation / benchmark / probing / analysis / dataset / survey / position papers** —
  they have no method to reconstruct. The empirical scout dropped ~38 such papers (e.g. the ACL
  humor-evaluation "Do Androids Laugh at Electric Sheep?", CheckList, "Are Emergent Abilities a
  Mirage?", LAION-5B, all "Position:" papers).
- **CS deduped hard against the large existing ML/CV/NLP/RL collection** — ~28 already-covered
  best papers excluded (ResNet, StyleGAN, Neural ODE, GPT-3, DPO, S4, Chinchilla, lottery ticket, …).

## To resolve before generation

- **Cross-file duplicate:** `sunflower-lemma-improved` appears in both `math_candidates.md` and
  `cs_theory_candidates.md` (improved sunflower lemma, Alweiss–Lovett–Wu–Zhang). Keep in CS-theory,
  drop from Math. (Net unique already accounts for this: 426.)
- A few very-recent (2024–25) rows have a placeholder arXiv id (`xxxxx`) to re-confirm at
  trace-writing time — flagged in the CS-theory file.

## Next step

On sign-off: generate in batches by domain (each method = full `paper-to-reasoning` run with
grounding + Codex review), update `methods.json` centrally per batch. Suggest starting with one
small calibration batch (~5 across domains) to confirm trace quality before scaling.
