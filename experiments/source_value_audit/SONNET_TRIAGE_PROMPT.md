# svfix — triage (read-only, cheap): is this method's decisive step already grounded in non-primary evidence?

Repo /srv/home/bohanlyu/innovation_proior. You get SLUG. Read methods/<SLUG>/results/reasoning.md fully, methods/<SLUG>/notes/*.md, and `ls -la methods/<SLUG>/refs methods/<SLUG>/src` (sizes matter: files < 2 KB are broken downloads). Do NOT edit anything.

Definitions (strict — wave-1 auditors applied these):
- The DECISIVE STEP = the move that makes the final construction/algorithm/proof work (which obvious approach fails, on what concrete obstacle, and which demand/insight forces the final design).
- A = route-forcing: the decisive step DEPENDS on a non-primary source (author self-account, co-author survey/thesis, official code with rationale, OpenReview reply, third-party re-derivation): a reader of ONLY the primary paper could not see why the step is forced, and the trace visibly runs through that documented obstacle→resolution. To claim A you must point to (i) the passage in reasoning.md and (ii) the local source file + a ≤200-char verbatim quote from it that the passage depends on.
- B = corroborating: non-primary sources are only used to cross-check formulas/numbers/definitions; the decisive step is derivable from the primary alone.
- C = decorative: sources cited as history/atmosphere; delete them and no reasoning step loses its justification.
- D = single-source: effectively a rewrite of the primary; no non-primary material shapes the reasoning (typical when notes/ and refs/ are empty).
Tie-breaks: A vs B — "would a reader of only the primary see why this step is forced?" yes → B. B vs C — "delete every non-primary mention; does any step lose justification?" no → C.
Also flag `material_on_disk`: does refs/ or src/ already contain a non-primary self-account/explainer file (>2 KB) — even if unused? (This decides whether the fixer needs a web hunt.)

Return exactly this JSON and nothing else:
{"slug":"...", "class":"A|B|C|D", "decisive_step":"<≤160 chars>", "trace_passage":"<≤200 chars verbatim from reasoning.md>", "quote":"<≤200 chars verbatim from the source file, only if class A>", "quote_file":"<local path, only if class A>", "material_on_disk":true|false, "notes":"<≤160 chars: what a fixer should look for / what is missing>"}
