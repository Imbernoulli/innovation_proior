# Sources

This file is superseded by the strict evidence bundle:

- `notes/source_matrix.md`
- `notes/discovery_synthesis.md`
- `refs/primary/`
- `refs/ancestors/`
- `refs/explainers/`
- `refs/self_accounts/`
- `refs/final_artifact/`

## svfix pass (track B_selfaccount) — quote used and where

Self-account already on file: `refs/self_accounts/fagin-1993-finite-model-theory-personal-perspective.txt` (Ronald Fagin, "Finite-model theory - a personal perspective," 1993). Already catalogued in `notes/source_matrix.md`; this entry records the specific load-bearing passage used to ground the decisive step in a rewrite of `results/reasoning.md` paragraphs 2-4.

Quote (lines ~380-393 of the extracted text):

> "Just as a spectrum is decidable, it is easy to see by a very similar argument that the set of (encodings of) structures in a generalized spectrum is decidable... I began wracking my brains to try to understand what made these sets special. All of a sudden it occurred to me that they were not just decidable, but they were very decidable. I had heard rumblings about the exciting new field of complexity theory from my roommate John Gill... and I decided that complexity theory must be able to make precise this notion of very decidable. I immediately began studying complexity theory, and I soon realized that, in a precise sense, the class of generalized spectra are exactly the class NP... while spectra are the class NE."

And the size-mismatch numbers a few lines later: "it takes around n^2 bits to encode a graph on n points, whereas it takes only around log_2 n bits to encode the number n."

Used to replace the prior forward-derivation framing (which read as if the shift from spectra to generalized spectra was already motivated by wanting polynomial time) with the real mechanism: spectra and generalized spectra were both already known to be decidable by an identical brute-force argument that fails to distinguish them from pathological decidable sets; the "very decidable" intuition — that something stronger than decidability was going on but logic alone couldn't name it — is what sent the reasoning outside logic to nascent complexity theory, which supplied the precise NP/NE split via the log n vs n^2 encoding-size gap.
