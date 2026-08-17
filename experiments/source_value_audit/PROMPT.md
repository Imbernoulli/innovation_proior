You are auditing training-data quality in the repo /srv/home/bohanlyu/innovation_proior (read-only task: do NOT edit any data files; only append to your results file).

Each method under methods/<slug>/ is a "reconstructed discovery trace": results/reasoning.md is a first-person re-derivation of a landmark method, built from multiple sources (primary paper, author self-accounts, retrospective surveys, third-party explainers). The corpus claim we are stress-testing: **the decisive step of the method is grounded in non-primary sources — the finished paper alone would not let you reconstruct the reasoning.** Classify how true that is, per method.

Your batch: slugs (one per line) in experiments/source_value_audit/batch_N_slugs.txt. Results file: experiments/source_value_audit/batch_N.jsonl. FIRST read the results file if it exists and SKIP any slug already present; audit all remaining slugs.

For each slug:
1. Read claimed sources: methods/<slug>/notes/sources.md; if missing, notes/source_matrix.md; else notes/synthesis.md; if none exist, record notes="none".
2. Read methods/<slug>/results/reasoning.md in full.
3. Identify the decisive step(s): the move that makes the final construction/algorithm/proof work (a counterexample that kills the obvious approach, a demand that forces the design, a trick that closes a hole).
4. Judge which sources those steps depend on. If it matters, spot-check methods/<slug>/refs/ and methods/<slug>/src/ (grep .txt files) to confirm the source really contains the load-bearing reasoning.
5. Classify exactly one:
   - "A" = route-forcing: at least one decisive step depends on a non-primary source — from the primary alone you could not see WHY this step is the right one.
   - "B" = corroborating: non-primary sources cross-check formulas/numbers/definitions, but decisive steps are derivable from the primary.
   - "C" = decorative: sources are cited/quoted but their use is anecdotal/atmospheric — the reasoning would stand unchanged without them.
   - "D" = single-source paraphrase: effectively a rewrite of the primary; no non-primary source shapes the reasoning.
6. Append ONE JSON line: {"slug": "...", "class": "A|B|C|D", "notes": "sources.md|source_matrix|synthesis|none", "key_step": "<≤160 chars>", "evidence": "<≤200 chars verbatim quote from the trace>", "sources_checked": <int>}

Rules: append incrementally after each method; one valid JSON line per method; no commentary in the file. Be strict: A vs B — "would a reader of ONLY the primary see why this step is forced?" if yes → B. B vs C — "if I deleted every mention of the non-primary source, would any reasoning step lose its justification?" if no → C.

Final message: just the tally (A/B/C/D) plus 2-3 most interesting cases (best A, worst C/D) with one-line reasons. Under 15 lines.
