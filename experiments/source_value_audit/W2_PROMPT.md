# Wave-2 source-value audit + fix (combined) — per-method instructions

Repo: /srv/home/bohanlyu/innovation_proior (branch `main`; commit after EVERY method you change).
Units: experiments/source_value_audit/w2_batch_N.json. Log: experiments/source_value_audit/w2_log_N.jsonl (append one JSON line per unit as you finish it; if the log already has the slug, skip it).

Read experiments/source_value_audit/FIX_PROMPT.md FIRST — every rule there applies (what a method is, tracks, goal, procedure steps 1–7, hard rules). This file only adds the wave-2 differences:

## Difference 1: you audit first, then fix
Wave-1 units were pre-audited. Wave-2 units are only CANDIDATES:
- `D_candidate`: methods/<slug>/notes and refs are empty → suspected single-source paraphrase.
- `B_selfaccount_candidate`: an old heuristic label says a self-account exists → suspected "material present but decisive step not grounded in it".
So for each unit, first classify per the audit rubric (A route-forcing / B corroborating / C decorative / D single-source; strict criteria: A vs B — would a reader of ONLY the primary see why the decisive step is forced? if yes → B; B vs C — delete every non-primary mention, does any step lose justification? if no → C). Record `audit_class` in the log.

## Difference 2: no self-declared "already_ok"
Wave-1 showed fixers were lenient (13/17 "already_ok" were overturned on recheck). Therefore your allowed outcomes are:
- `fixed` — you grounded the decisive step in verified non-primary evidence (or corrected errors), committed.
- `no_source_found` — you did a REAL search (refs/src/notes → SELF_ACCOUNT_SOURCES.md → web: Nobel/Turing/award lectures, PhD theses of first authors, author blogs, OpenReview replies (try HuggingFace's openreview mirror if openreview.net is bot-walled), talk transcripts/podcasts, GitHub issues by authors, competition reports, co-author surveys) and found nothing beyond the primary; write the search log into notes/sources.md.
- `audit_A_confirmed` — ONLY if your audit finds the trace is ALREADY class A (decisive step already runs through verified non-primary evidence in refs). Quote the passage + the source file path in the log's `evidence` field. Expect this to be rare.
There is no `already_ok`. If the trace is B/C/D and a source exists, you fix it.

## Difference 3: synthetic/competition units are excluded already
(imo-/ioi-/ahc/circle-/hadamard-… were removed from the list.) If you nevertheless meet a unit that is a self-generated optimization/competition run with no external paper, log `outcome:"not_applicable"` with a one-line reason and move on.

## Log schema
{"slug":..., "track":..., "audit_class":"A|B|C|D", "outcome":"fixed|no_source_found|audit_A_confirmed|not_applicable", "sources_added":[...], "step_rewritten":"<≤120 chars>", "errors_corrected":[...], "evidence":"<≤200 chars>", "commit":"<sha or ''>"}

## Reminders of the hard rules (from FIX_PROMPT.md)
- Surgical rewrite of the decisive passage; no whole-file rewrites; no length targets; no injected "Wait/Alternatively/Hmm"; no deleting correct innovation content; a failed first attempt must fail for a REAL checkable reason.
- No fabricated sources/anecdotes; only content you can point to in a saved refs/ file may shape the trace.
- In-frame: no "the paper", "the authors", arXiv ids, "et al.", hindsight. Run `python3 tools/lint_inframe.py | grep "methods/<slug>/"` before committing.
- Fix factual errors found on the way; propagate to answer.md / train_answer.md / code; log in results/changelog.md.
- `git add methods/<slug>` only (never -A); one commit per method: `svfix(w2,<track>): <slug> — <source>, <step>`.
- refs/ notes/ src/ are gitignored (local provenance) — save material there anyway.

Final message: tally by outcome + audit_class; the 2 most substantive rewrites; unresolved issues. Under 15 lines.
