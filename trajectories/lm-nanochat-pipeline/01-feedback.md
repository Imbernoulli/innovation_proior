The tokenizer stage produces no report-card accuracy of its own — it is the substrate the rest of the pipeline rides on. Its effect is felt indirectly, through compression (characters per token) and through the validation **bits-per-byte** metric it makes possible (vocab-invariant: cross-entropy ÷ bytes, lower is better), which is the smooth number tracked for the base model in the next rung.

What this stage commits, as built in the repo:

| Setting | Value |
|---|---|
| Vocabulary size | 32768 (2^15) |
| Algorithm | byte-level BPE, GPT-4-style regex pre-split |
| Number grouping | `\p{N}{1,2}` (2-digit max; verified best at 32K vocab: `{1,1}` worse, `{1,3}` worse still) |
| Special tokens reserved | 9 (`<|bos|>` + user/assistant/python/output start/end) |
| Training corpus | ~2B characters, each document capped at 10K chars |
| Sanity check | encode→decode round-trips exactly (numbers, contractions, punctuation, Unicode 你好世界 🌍) |

Provenance note: the `{1,2}` number-grouping choice is validated in the repo's experiment log (`dev/LOG.md`, 2026-01-13) at depth-12, vocab-32K: val_bpb 0.969 for `{1,1}`, **0.965** for `{1,2}`, 0.972 for `{1,3}` — the only stage-level number committed for this rung. The compression rate is computed by `tok_eval.py` at run time but no fixed value is committed in the repo, so none is asserted here.
