# contriever changelog

## 2026-08-18 — epistemic correction (svfix)
- `results/reasoning.md` (cropping-vs-ICT section, pre-fine-tuning ablation paragraph): the prior
  svfix pass (`fa1dadf56`, citing Izacard's PhD thesis Table 5.7) had the narrator claim to have
  actually pre-trained two small-scale versions — one on ICT-style complements, one on independent
  crops — and read off their pre-fine-tuning nDCG@10 on BEIR ("Averaged across the seven sets
  cropping wins, 32.2 against 25.9... On Quora, though, the gap is enormous: cropping scores 75.4,
  ICT scores 27.6..."). Real numbers or not, at this point in the frame the method's own
  experiments have not happened yet, so reporting an observed outcome is out of scope for a
  single-turn proposal.
- Fix: removed the claimed pre-training runs and every per-dataset/averaged number (32.2 vs 25.9;
  DBPedia 21.0 vs 21.3; NaturalQuestions 17.7 vs 19.4; Quora 75.4 vs 27.6). Kept everything else the
  prior svfix pass added or that already existed: the matched-conditions experiment design
  (identical architecture and MoCo setup, 200k steps on Wikipedia, batch size 2,048, read out
  pre-fine-tuning so the supervised MS MARCO stage can't paper over a weak signal), the
  hypothetical crop-overlap statistic (60% vs 0%) that motivates trying cropping at all, and a new
  explicit per-dataset prediction/decision rule: if the overlap story is right, cropping's edge
  should concentrate on high-lexical-overlap sets (Quora's near-duplicate questions are the sharp
  case) and shrink toward nothing on low-overlap sets, and whichever variant's pre-fine-tuning BEIR
  numbers actually hold up under that per-dataset breakdown is the construction that gets carried
  into the full recipe.
- The landing (choosing independent cropping over ICT) now proceeds straight into implementation
  without a resolved observation backing it — that is expected per the epistemic-fix rule, not a
  gap to paper over. This unit needs a trajectory-observation turn to actually run the small-scale
  ICT-vs-crop pre-training and supply the real per-dataset BEIR numbers.
- No changes to `answer.md` or `train_answer.md` were needed — the svfix diff (`fa1dadf56`) touched
  only `reasoning.md`, and neither of those files carries the removed numbers.
