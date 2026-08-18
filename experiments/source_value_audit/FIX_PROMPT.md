# Source-value fix — per-method instructions

Repo: /srv/home/bohanlyu/innovation_proior (git; work on branch `main`, commit after EVERY method).
Your unit list: experiments/source_value_audit/fix_batch_N.json (N given in your task). Each unit has `slug`, `track`, and the auditor's `key_step`/`evidence`.

## What each method is
`methods/<slug>/results/{context.md, reasoning.md, answer.md, train_answer.md}` is a first-person reconstruction of how a landmark method was discovered. `reasoning.md` is the main artifact: the scientist's live derivation — pain of the problem, prior art, attempts that fail for a real reason, the move that unlocks it, the landing on the method + code. `notes/` records the sources; `refs/` and `src/` hold source material (.pdf/.txt).

## The defect you are fixing (per track)
- **B_selfaccount**: an author self-account (Nobel/Turing lecture, retrospective, interview, notebook, OpenReview reply, blog) is already in `refs/` or named in `notes/`, but the trace's DECISIVE STEP is still written as if derived from the finished paper. The material that explains WHY that step was forced (what was tried first and failed, which counterexample/obstacle forced the design) is sitting unused.
- **D_single_source**: `notes/` and/or `refs/` are missing or contain only the primary paper. The trace is a forward paraphrase of the paper.
- **C_decorative**: non-primary sources are cited but only as history/atmosphere; no reasoning step depends on them.
- **A_fake_thin**: notes admit no original text was read ("standard knowledge"); trace is a textbook proof.

## Goal (same for all tracks)
Make the decisive step(s) of `reasoning.md` genuinely grounded in non-primary evidence of how the discovery actually went: the real first attempt and why it failed, the real obstacle, the real move that resolved it — as documented by the author (preferred) or by an authoritative third-party account (survey by a co-author, official code, authors' OpenReview/blog reply, oral history). Present it as the scientist's own live thinking, in-frame (first person, present tense, no "the paper says", no "the authors", no arXiv ids, no hindsight).

## Procedure
1. `git log --oneline -3 -- methods/<slug>` and read all four results files + notes + list refs/ src/.
2. **Find the material.**
   - Check `refs/*.txt`, `src/*.txt`, `notes/*.md` first (grep for the key step's terms).
   - Then check `SELF_ACCOUNT_SOURCES.md` at repo root for this method.
   - Then search the web (WebSearch/WebFetch): "<method> Nobel lecture / Turing lecture / interview / oral history / retrospective / 'how we came up with' / OpenReview reply / author blog / official repo README / talk transcript". For recent ML papers, OpenReview discussions, authors' blog posts, GitHub issues, and talk transcripts are the usual gold; for math/physics, award lectures, AMS Notices, survey chapters by the authors.
   - Save what you find into `methods/<slug>/refs/` (pdf + extracted .txt via pdftotext, or .txt for html) and record it in `notes/sources.md` (create if absent) with type = self-account / explainer / ancestor / primary, URL, and 1-2 lines on what it supplies.
   - If after a real search nothing non-primary exists, say so in notes/sources.md ("searched: … ; none found") and go to step 4 with whatever load-bearing reasoning the primary + ancestors give — but do NOT invent an anecdote.
3. **Verify the material actually says it.** Quote the load-bearing passage into notes/sources.md (≤3 sentences per source). Only content you can point to in a saved file may shape the trace.
4. **Rewrite the decisive part of reasoning.md.** Surgical, not wholesale:
   - Insert/replace the passage where the decisive step happens so it now runs through the documented obstacle → the documented resolution. The first attempt that failed must fail for the REAL, checkable reason (a counterexample worked on the page, a quantity that doesn't close, a sign that comes out wrong), not "this feels wrong."
   - Correct any factual errors you find in the trace on the way (wrong constant, wrong sign, wrong attribution, wrong formula) — check them against the sources; if you change a formula/number also check answer.md / train_answer.md / code for the same error and fix consistently. Log every correction in `results/changelog.md` (append a dated entry).
   - Keep everything else. Do NOT rewrite the whole file. Do NOT pad; do NOT trim to a length target. Do NOT sprinkle "Wait"/"Alternatively"/"Hmm" — a genuine dead end is written as the actual computation that failed, not as a hedge word. Do NOT delete existing correct innovation content.
   - Keep the landing (final method + code) unchanged unless it is wrong.
5. **In-frame check.** From repo root: `python3 tools/lint_inframe.py | grep "methods/<slug>/"` must show no new hits (E_paperref / A_paren / B_meta / C_rsn_header) in files you edited. Fix if any. Also grep your edits for "the paper", "the authors", "arXiv", "et al." inside reasoning.md — none allowed.
6. **Commit** exactly one commit per method:
   `git add methods/<slug> && git commit -q -m "svfix(<track>): <slug> — <one line: which source, which step now grounded>"`
   Do NOT `git add -A`. Do NOT touch other methods' files.
7. Append one JSON line to `experiments/source_value_audit/fix_log_N.jsonl`:
   `{"slug":..., "track":..., "outcome":"fixed|no_source_found|already_ok", "sources_added":[...], "step_rewritten":"<≤120 chars>", "errors_corrected":[...], "commit":"<sha>"}`

## Hard rules
- One method at a time; commit before starting the next.
- Never fabricate a source or an anecdote. If you cannot point to a saved file, it doesn't go in.
- Never leak provenance ("the paper", "authors", "arXiv", "later work showed").
- Never inject backtracking filler; never change length for its own sake; never delete correct innovation content.
- If a units' `refs/` contains a broken download (HTML error page saved as .pdf, <2KB), delete it and try to re-fetch; note in sources.md.

Final message: tally of outcomes + the 2 most substantive rewrites (slug, what changed) + any unresolved problems. Under 15 lines.
