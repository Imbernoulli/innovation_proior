# Blind approach-labeling protocol (stage 2, v2)

You get problem folders under `blind/<bench>/<pair>/<problem>/`: `statement.md` and exactly two sample files named by a
random letter (`A.txt`, `C.txt`, ...). The first line of each sample file is `[score=... complete=...]`; then a
`=== REASONING (excerpt) ===` block (head and tail of the model's thinking; the middle may be omitted) and a
`=== FINAL ANSWER (full) ===` block (the code that was judged). You do NOT know which model produced which letter,
and some pairs are two runs of the SAME model (controls). Do not try to guess; judge only what is on the page.

For EACH problem folder, read statement + both samples, then append one JSON object (one line) to the output file you
were given, with exactly these keys:

- `folder`: the folder path relative to `blind/`
- `approach`: {letter: one-sentence description of the core algorithmic idea, in English}
- `standard_approach`: one sentence: what a competent competitive programmer / ML engineer would do by default for this problem (the textbook or intended solution). If the statement itself suggests the method, say so.
- `same_core_idea`: true/false — do the two samples use the same core idea?
- `novel`: {letter: one of "textbook" (the standard/intended approach or a named classic algorithm applied as-is), "variant" (standard approach plus a non-trivial twist or a knob change), "different" (a mechanism that is not the standard approach for this problem and not a trivial variant), "none" (no real attempt, degenerate or truncated output)}
- `cross_domain`: {letter: true/false — does it import a technique from another field (physics, information theory, algebra, control, economics ...) that is NOT the standard toolkit of this problem's field?}
- `why_higher_scores`: one of "different_method", "same_method_fewer_bugs", "same_method_better_tuning", "other_side_crashed_or_empty", "unclear" — why the higher-scoring letter scored higher.
- `quotes`: {letter: one verbatim substring (<=200 chars, copy exactly, from the FINAL ANSWER or the reasoning) that shows its core idea}
- `notes`: <= 2 sentences, optional.

Rules: verbatim quotes only (they are grep-checked; a non-matching quote voids the record). Judge the idea, not the
writing quality. "different" must be a different mechanism, not a different data structure or constant for the same
idea. Do not skip folders. Process folders in the order given. Never edit files inside `blind/`.
