# Faster R-CNN — source verification (svfix W3_notes_unclear)

## Quality-gate finding: decisive step is sound_as_is, left untouched

TRIAGE decisive_step = "read proposals off the shared conv feature map via anchors
instead of a separate proposer, dissolving the multi-scale problem without
image/filter pyramids" (results/reasoning.md paragraphs 19-23).

This is genuinely DERIVED on the page, and the obstacle it resolves is stated
explicitly — with the same structure and the same failure reasoning — in the
primary source, which is the authors' own extended PAMI paper
(src/rpn_pami_arxiv.tex, "Object Detection Networks Learn Region Proposals"
extended version of the Faster R-CNN paper; this is the paper's own account,
not a third party, but per the fix-prompt's framing of this method it is
"effectively primary"). Quotes:

- type: primary (authors' own extended paper)
- local file: methods/faster-rcnn/src/rpn_pami_arxiv.tex
- quote (line ~201-203): "The first way is based on image/feature pyramids...
  This way is often useful but is time-consuming... The second way is to use
  sliding windows of multiple scales... it can be thought of as a 'pyramid of
  filters'... As a comparison, our anchor-based method is built on *a pyramid
  of anchors*, which is more cost-efficient."
- quote (line 109): "we introduce novel 'anchor' boxes that serve as references
  at multiple scales and aspect ratios... a pyramid of regression references...
  which avoids enumerating images or filters of multiple scales or aspect
  ratios."

reasoning.md paragraph 21 restates this exact two-branch failure (image pyramid
= "multiplies the convolution cost"; filter pyramid = "pays linearly per scale
and per aspect ratio in filters") and paragraph 23 lands on the same resolution
("a pyramid of regression references, not of images or filters") — a faithful,
mechanistic elaboration of the primary's own comparison, not an asserted/
hindsight claim. Per the fix-prompt's quality gate this is LEGITIMATE
single-source grounding (condition a: obvious moves fail for a concrete,
checkable computational-cost reason; condition b: that justification is in the
primary). No rewrite performed — grafting an external citation onto this
passage would be unneeded/damaging per the fix-prompt's explicit warning.

## Separate finding: closed the "notes cite ungrounded code" gap

notes/synthesis.md's "Canonical code (py-faster-rcnn, rbgirshick) — grounded
files in code/" section referenced five implementation files
(generate_anchors.py, bbox_transform.py, anchor_target_layer.py,
proposal_layer.py, proposal_target_layer.py) as if grounded, but no code/ dir
and no such files existed anywhere under methods/faster-rcnn/ (confirmed:
refs/ absent, src/ had only the LaTeX template + rpn_pami_arxiv.tex before this
pass). This is the defect the TRIAGE note flagged. Fetched the real files from
the paper's own released reference implementation and saved them locally to
close the gap:

- type: ancestor code (official reference implementation released by the
  paper's authors — Ross Girshick, a Faster R-CNN co-author, is the repo's
  primary author/maintainer)
- URL: https://github.com/rbgirshick/py-faster-rcnn (raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/{rpn,fast_rcnn}/*.py)
- local files: methods/faster-rcnn/src/py-faster-rcnn/{generate_anchors.py,
  bbox_transform.py, anchor_target_layer.py, proposal_layer.py,
  proposal_target_layer.py}

Line-by-line comparison against reasoning.md's four embedded code blocks
(generate_anchors/_ratio_enum/_scale_enum, bbox_transform/bbox_transform_inv/
clip_boxes, the anchor-labeling loop, the proposal/NMS loop):

- generate_anchors.py: reasoning.md's function is the same algorithm, same
  variable roles, same `_ratio_enum`/`_scale_enum`/`_mkanchors`/`_whctrs` split
  as the real file. Real file's own verification comment gives the ground-truth
  9-anchor matrix for base_size=16:
  quote: "anchors =\n\n   -83   -39   100    56\n  -175   -87   192   104\n  -359  -183   376   200\n   -55   -55    72    72..."
  (local file: methods/faster-rcnn/src/py-faster-rcnn/generate_anchors.py, lines 10-20)
  Converting those corner coords to (w,h) reproduces exactly the widths/heights
  reasoning.md's paragraph 118 computes by hand: wide-anchor widths
  {184,368,736}, square {128,256,512}, tall {88,176,352}, and sqrt-areas
  {133,266,532}/{128,256,512}/{125,249,498} — every one of the nine numbers
  matches the real repo's own regression-test matrix. No factual error found.
- bbox_transform.py: reasoning.md's `bbox_transform`/`bbox_transform_inv`/
  `clip_boxes` match the real file's formulas exactly (dx=(gcx-ecx)/ew etc.,
  same +1 inclusive-width convention, same center/size inversion). This also
  confirms the "1-px corner artifact from the inclusive-width convention" that
  reasoning.md's paragraph 162 derives by hand-checking the encode/decode
  round trip is real, not an invented quirk — it follows from the real file's
  `+ 1.0` convention on widths/heights (quote: "ex_widths = ex_rois[:, 2] -
  ex_rois[:, 0] + 1.0", local file bbox_transform.py line 11).
  quote_file for this comparison: methods/faster-rcnn/src/py-faster-rcnn/bbox_transform.py
- anchor_target_layer.py: same labeling order as reasoning.md's
  `anchor_targets` (negatives by <0.3 first, then gt-argmax fallback positives,
  then >=0.7 threshold positives, matching the real file's non-CLOBBER default
  branch), same fg/bg subsampling to RPN_BATCHSIZE=256 at up-to-1:1. No error.
- proposal_layer.py: same pipeline as reasoning.md's `generate_proposals`
  (bg/fg score split, transpose/reshape to (H*W*A,4), bbox_transform_inv,
  clip, min-size filter, pre_nms_topN sort, NMS, post_nms_topN). No error.

Net result: the code embedded in reasoning.md was already an accurate,
non-fabricated implementation — verified against the authors' own released
code rather than assumed. Nothing in reasoning.md needed to change. The only
concrete defect (synthesis.md's dangling "grounded files in code/" claim
pointing at nothing) is fixed by this note plus the files now actually present
at src/py-faster-rcnn/.

## Search log
- grep -ril across refs/ (absent), src/, notes/ for anchor/pyramid/RPN terms.
- Confirmed src/ had only cvpr.sty, cvpr_eso.sty, eso-pic.sty, ieee.bst,
  IEEEtran.cls, rpn_pami_arxiv.tex/.bbl before this pass (no refs/ dir).
- Fetched https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/rpn/generate_anchors.py ,
  .../lib/fast_rcnn/bbox_transform.py , .../lib/rpn/anchor_target_layer.py ,
  .../lib/rpn/proposal_layer.py , .../lib/rpn/proposal_target_layer.py
  (all HTTP 200, all >2KB) and diffed against reasoning.md's code blocks.
