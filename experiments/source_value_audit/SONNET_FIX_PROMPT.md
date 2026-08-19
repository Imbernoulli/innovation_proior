# svfix — single-method fix instructions (Sonnet edition, lessons from wave 1 baked in)

You fix ONE method: `methods/<SLUG>/` in /srv/home/bohanlyu/innovation_proior (git, branch main). You will be given SLUG, TRACK, and (maybe) an auditor's KEY_STEP/EVIDENCE hint. Work only inside methods/<SLUG>/. Another independent agent will verify your work afterwards against the checklist at the bottom — assume it will catch anything sloppy.

## What the artifact is
`results/reasoning.md` = a first-person, present-tense reconstruction of how a landmark method was actually discovered: the pain of the problem, the obvious first move and WHY it fails (a checkable reason), the real obstacle, the move that resolves it, landing on the method + code. `results/answer.md`/`train_answer.md` = the distilled landing. `notes/` = source records; `refs/`, `src/` = source files (.pdf/.txt). notes/refs/src are gitignored (local provenance) — still write there.

## The defect
The trace's DECISIVE STEP (the move that makes the construction work) reads as a clean forward derivation from the finished paper. But finished papers edit the struggle out. What we want is the documented route: what the author actually tried first and why it failed, which counterexample/obstacle forced the design — evidence that lives OUTSIDE the primary paper: author self-accounts (Nobel/Turing/award lectures, PhD theses of first authors, retrospectives, interviews/podcasts/talk transcripts, blogs, OpenReview replies, GitHub issue replies, competition reports), co-author surveys, authoritative third-party re-derivations.
Tracks: `B_selfaccount` (material is already in refs/notes but unused) · `D_single_source`/`D_candidate` (only the primary is present) · `C_decorative` (sources cited as flavor only) · `A_fake_thin` (notes admit "standard knowledge", nothing read) · `W3_*` (wave-3: a triage agent classified it B/C/D; its notes are passed to you as TRIAGE).

## Quality gate — decide FIRST whether this trace needs fixing at all
The goal is training-data quality, not source count. A single-source trace is LEGITIMATE and must be left alone when BOTH hold:
(a) the decisive step is genuinely DERIVED on the page — the obvious first move fails for a concrete, checkable reason (a worked counterexample, a quantity that doesn't close, an explicit ablation) and the resolution follows from that failure; AND
(b) that obstacle/justification really is in the primary (modern ML papers often carry their own ablations and failed variants) or is the trace's own honest computation.
In that case: outcome = "sound_as_is", and you MUST quote (≤200 chars each) (i) the trace passage that does the deriving and (ii) the primary/source passage or the on-page computation that backs it. Do NOT graft a source onto a sound trace — a bolted-on citation that the reasoning doesn't need is damage, not improvement (a wave-2 fixer grafted a blog stat that was just the primary's own number restated, and overclaimed independence; the verifier killed it).
Fix ONLY when at least one of these defects is present:
- the decisive step is ASSERTED, not derived (hindsight tone, "this confirms", design stated with no forcing reason);
- a failed attempt is staged/vague ("this feels wrong") instead of failing for a real reason;
- a factual error (wrong constant/sign/attribution/formula);
- genuinely valuable non-primary material is ALREADY on disk (refs/notes) and unused at the decisive step;
- the trace contradicts the sources on record.
An adversarial verifier will adjudicate "sound_as_is" claims with the same rigor as fixes — thin justifications will be bounced back to you.

## Procedure (do all steps, in order)
1. Read results/reasoning.md fully; read notes/*.md; `ls refs/ src/`. Identify the decisive step yourself (don't just trust the hint). Apply the quality gate above; only continue to step 2 if the gate says fix.
2. FIND MATERIAL — search in this order and do not stop at the first empty layer:
   a. `grep -ril "<key terms>" methods/<SLUG>/refs methods/<SLUG>/src methods/<SLUG>/notes` — material is often already on disk and unused (wave-1: em-algorithm had a Laird interview, deltanet had a "Failed Attempt" blog section, isolation-forest's notes skipped pages 13-29 of the source — re-extract when a saved .txt looks truncated).
   b. `grep -i "<slug or method name>" SELF_ACCOUNT_SOURCES.md` at repo root.
   c. Web (WebSearch + WebFetch, or curl): "<method> <first author> thesis", "<first author> interview|podcast|talk transcript <method>", "<method> OpenReview" (openreview.net and `api.openreview.net/notes?forum=` are challenge-walled, but `https://api.openreview.net/notes/search?term=<title words>&content=all&group=all&source=all&limit=100` is NOT — it returns full reviews/rebuttals/decisions; filter notes by `forum == <paper id>`; it 429s after ~4 rapid calls, so space requests. Also `huggingface.co/papers/<arxiv id>`.) Other unwalled venues: arXiv e-print source diffs between versions (`arxiv.org/e-print/<id>v1`) show what authors revised; official code repos often keep commented-out earlier settings (a de-facto lab notebook); nobelprize.org lecture PDFs via `uploads/…-lecture.pdf`; NBER working-paper PDFs; OpenAlex/Semantic Scholar APIs for locating OA copies. WebSearch quota may be exhausted — use curl/WebFetch directly., "<method> Nobel|Turing|Abel lecture", "<method> 'how we came up' | retrospective | 'lessons learned'", "<method> github issue <author> why", "<method> ML Retrospectives". Wave-1 misses to avoid: fixers said "no self-account" for fastsac/mifgsm/progressive-gan while the first authors' PhD theses / Karras talks / the OpenReview thread existed. Spend real effort here; a real search is ≥5 distinct queries across ≥3 of the above venues.
   d. Save what you find: pdf → `pdftotext -layout x.pdf x.txt`; html → save the text as .txt in refs/. Broken downloads (HTML error pages saved as .pdf, files < 2 KB) → delete and refetch.
3. VERIFY THE MATERIAL SAYS IT. Quote the load-bearing passage (≤3 sentences) into notes/sources.md with type (self-account / explainer / ancestor / primary), URL, and the local file path. Only content you can point to in a saved file may shape the trace. Never invent an anecdote.
4. REWRITE — surgical, at the decisive step only:
   - The passage now runs: obvious move → fails for the REAL, checkable reason from the source (a counterexample worked on the page, a quantity that doesn't close, a sign that comes out wrong, a training run that diverged with the stated symptom) → the documented resolution. Mechanism, not story: "I spent months on X and it was intractable to train because Y" is good; "over coffee I suddenly saw it" is useless.
   - Present as the scientist's own live thinking: first person, present tense. No "the paper", "the authors", "arXiv", "et al.", "later work showed", no hindsight ("as we now know").
   - Do NOT: rewrite the whole file; pad or trim to a length target; add "Wait"/"Alternatively"/"Hmm" filler (a genuine dead end is the actual computation that failed, not a hedge word); delete existing correct innovation content; change the landing (final method + code) unless it is wrong.
   - Fix factual errors you meet (wrong constant/sign/attribution/formula) after checking the source; propagate to answer.md / train_answer.md / code consistently; append a dated entry to results/changelog.md.
5. LINT: from repo root `python3 tools/lint_inframe.py | grep "methods/<SLUG>/"` → no hits in files you edited (categories A_paren/B_meta/E_paperref/C_rsn_header). Also `grep -n -i "the paper\|the authors\|arxiv\|et al\." methods/<SLUG>/results/reasoning.md` → nothing new.
6. COMMIT immediately, this method only (if git says `index.lock` exists, another agent is committing — sleep 3-10 s and retry, up to 5 times): `git add methods/<SLUG> && git commit -q -m "svfix(<TRACK>): <SLUG> — <source>, <which step now grounded>"`. NEVER `git add -A` / `git add .` (concurrent agents share the tree; wave-1 swept others' half-edits into a commit). Then `git log -1 --format=%h` and put the sha in your output.
7. If after the full search in step 2 there is truly nothing beyond the primary: write the search log (queries + venues) into notes/sources.md, do NOT rewrite, outcome = no_source_found. This should be uncommon; if the trace additionally has factual errors, still fix those and commit.

## Output (your final message is parsed — return exactly this JSON and nothing else)
{"slug":"...", "track":"...", "outcome":"fixed|sound_as_is|no_source_found|not_applicable", "sources_added":["<type>: <title/URL> -> <local path>"], "quote":"<≤200 chars verbatim from the source file that grounds the step>", "quote_file":"<local path>", "step_rewritten":"<≤160 chars: what the decisive passage now runs through>", "errors_corrected":["..."], "commit":"<sha or ''>", "search_log":"<≤200 chars: queries/venues tried, if outcome == no_source_found>", "sound_evidence":{"trace_quote":"<≤200 chars>", "backing_quote":"<≤200 chars>", "backing_file":"<path>"}}

## What the verifier will check (so do it right the first time)
- quote_file exists, is >2 KB, and contains the quote (fuzzy match ok).
- The rewritten passage in reasoning.md actually depends on that content (deleting it would remove the justification of the step) — not a name-drop.
- The failing first attempt fails for a concrete, checkable reason.
- No provenance leaks; lint clean; git diff limited to methods/<SLUG>/; commit exists.
- No "Wait/Alternatively" injections; length not padded; landing unchanged unless an error was logged.
