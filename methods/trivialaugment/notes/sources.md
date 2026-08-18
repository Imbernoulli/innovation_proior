# Sources — svfix pass (B_selfaccount)

Target decisive step: "One op per image + strength resampled per image (Identity replaces drop
prob); primary states the anew-per-image contrast with RA and the weak+strong mixture."
(`results/reasoning.md`, the magnitude paragraph ending "...I sample a fresh strength bin for each
image. This gives weak and strong perturbations in the same training distribution...").

## Self-account (blog)
- `refs/self_accounts/automl_blog_trivialaugment.html` — AutoML.org Freiburg-Hannover-Tübingen lab
  blog post "TrivialAugment: You don't need to tune your augmentations for image classification"
  (posted 2021-08-18, byline Samuel Müller, one of the two paper authors). The earlier extraction
  in refs/ only grabbed a partial crude strip; re-extracted the full page with Python
  `BeautifulSoup` (script/style/nav stripped) to
  `refs/self_accounts/automl_blog_trivialaugment.txt` (5,962 chars after stripping nav
  boilerplate — the page's actual post body is short, roughly four paragraphs; the earlier ~22KB
  figure was mostly site navigation/cookie-banner HTML, not article text).
  - Load-bearing quote: "We sample the strength from a uniform distribution, as is often done in
    standard augmentation pipelines, but we also sample the augmentation itself (a fresh sample of
    both augmentation and strength for each image) and only apply a single augmentation per
    image." — confirms, in the author's own words, that BOTH the operation and the strength are
    resampled per image (not just the operation), and that only one operation is applied per
    image.
  - Secondary quote: "At some point we implemented the most trivial baseline we could come up with
    as a sanity check, and it outperformed all other methods we considered." — narrates that TA
    began as a sanity-check baseline, not a deliberately engineered final method; not used in the
    rewrite (reasoning.md's existing derivation-by-elimination framing is not a "sanity check"
    narrative and is not being changed to one — no invented anecdote).
  - Checked for, and did NOT find: any fuller stated argument for *why* RandAugment's fixed global
    magnitude is a weakness, or an explicit "why Identity replaces the drop probability" argument.
    The blog post is short and mostly narrates the empirical surprise + points to the paper; it
    does not carry additional derivation depth beyond what the primary paper's Algorithm 1/Sec. 3
    already states. Confirmed by full-text read, not just the earlier partial grep.

## Primary paper
- `refs/primary/trivialaugment_2103.10158.txt`, Section 3 (page numbers embedded), the sentence
  immediately after Algorithm 1: "We emphasize that TA is not a special case of RandAugment (RA),
  since RA uses a fixed optimized strength for all images while TA samples this strength anew for
  each image." — this exact RA-contrast is primary-derivable (not blog-unique); already reflected
  in `reasoning.md`'s existing "I want to be careful that this is not just RandAugment with N = 1"
  paragraph (lines 99-104), left unchanged.
- Section 5/UA description: "Unlike RA, it fixes the number of augmentations to N = 2 and drops
  each augmentation with a fixed probability of 0.5." — already used in reasoning.md's existing UA
  paragraph.
- Section 4.2.3/Table 7 (TA's own strength-subset ablation, "mixture of strong and weak
  augmentations") is a POST-method, proposed-method-evaluation result (per
  `notes/source_matrix.md`'s existing in-frame discipline: "Proposed-method evaluation numbers
  (TA's wins) are OUT of context.md and reasoning.md"). NOT used as forward-derivation grounding
  for the decisive step — using TA's own ablation numbers to justify the design that produced them
  would be circular/hindsight. The rewrite instead grounds the weak+strong-mixture claim in
  pre-existing RandAugment evidence already cited earlier in reasoning.md (below).

## Ancestor (RandAugment, direct predecessor being contrasted)
- `refs/ancestors/randaugment_1909.13719.txt`: "the optimal distortion magnitude is larger for
  models that are trained on larger datasets" and "Figure 3d demonstrates that the optimal
  distortion magnitude increases monotonically with training set size" — RandAugment's own
  diagnostic that a single global M is not a stable, transferable quantity; a shared M is a
  compromise fitted to one dataset/model scale. Already cited in reasoning.md's first paragraph
  ("the best global distortion magnitude rises with model width and with training-set size");
  used in the rewrite to make explicit *why* a single tuned M cannot double as a source of both
  weak and strong per-image perturbations — it is pinned to one scale, not to per-image variation.

## Verdict
Self-account confirms the per-image dual-resampling mechanic in the author's own words but adds no
deeper theoretical argument beyond the primary paper. The primary paper's own RA-contrast sentence
is legitimately primary-derivable. The actual fix here is tightening the obstacle -> resolution
logic inside reasoning.md's existing magnitude paragraph: making explicit that a single tuned M
(RandAugment's move) is a compromise pinned to one dataset/model scale, which is why it cannot
supply both weak and strong regimes for the same image distribution, whereas per-image resampling
gets the mixture without any search. No new source needed beyond what was already cited earlier in
reasoning.md (RandAugment's own magnitude-vs-scale diagnostic) plus the self-account's confirmation
of the dual-resample mechanic. Outcome: **fixed**.
