# changelog — pifold

## 2026-08-18 (epistemic correction pass)
Prior fix commit `dd625eb52` ("svfix(W3_primary_plus_ancestors): pifold — primary Table 4 AT
ablation, one-shot-decoder bet now resolved with observed numbers") replaced the unresolved
one-shot-decoder bet ("I cannot settle on paper... I would want to compare recovery against an
autoregressive head on a held-out CATH split before believing the sequential step is redundant")
with an in-line run of the encoder/decoder layer-budget slide and its reported outcome: recovery
and CATH test-set wall time at four autoregressive/one-shot splits (49.30%/527s, 50.41%/347s,
50.96%/30s, 51.66%/36s), the claim that recovery never dips down to zero autoregressive layers,
and the conclusion that the bet is confirmed. This is a single-turn PROPOSAL frame — at this
point in the frame PiFold's own experiments have not happened yet, so reporting the method's own
ablation numbers as something already observed is out of frame regardless of whether the numbers
are real (they match the paper's Table 4 AT ablation).

Fixed by rewriting the passage in `reasoning.md` to keep: the bet/hypothesis (final node
embeddings already carry the structural context each marginal needs), the discriminating-
experiment DESIGN (matched encoder-plus-decoder layer budget, slide the split between
autoregressive and one-shot PiGNN layers, train each split to convergence on the same CATH fold
split, read off recovery and wall time), the PREDICTION under each hypothesis (indispensable
sequential conditioning costs recovery somewhere along the slide; a sufficient encoder instead
holds recovery flat/climbing all the way to zero autoregressive layers with wall time falling for
free), and an explicit decision rule (a dip anywhere on the slide means keeping enough
autoregressive layers to sit above it; no dip means shipping the fully one-shot decoder) — while
removing the claimed observations (the four numeric recovery/wall-time readings, the "running the
slide, it does not [drop]" result statement, and the "that is the discriminating result... exactly
as the bet assumed" confirmation). The "what it buys if it holds" close (linear readout, per-
residue cross-entropy, one parallel forward pass) is kept but reframed as conditional on the
one-shot branch shipping, not as an already-settled fact. This unit now needs a trajectory-
observation turn to supply the actual ablation result. No changes to answer.md or train_answer.md
were needed (svfix diff touched only reasoning.md).
