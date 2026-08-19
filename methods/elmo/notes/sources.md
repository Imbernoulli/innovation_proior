# Sources — decisive-step sourcing check (svfix, W3_primary_plus_ancestors)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md paras ~15-18, 38-42: replace the fixed top-layer LM/MT feature
convention (CoVe, TagLM) with a learned, task-specific softmax-weighted
mixture of ALL L+1 biLM layers plus a scalar scale gamma —
`ELMo_k^task = gamma^task * sum_j softmax(w^task)_j * h_{k,j}^LM`.

## Search performed
- `grep -ril` over `methods/elmo/refs methods/elmo/notes` for layer/probe/POS/WSD
  terms → primary + the ancestors already in `notes/source_matrix.md`
  (Belinkov 2017 NMT-morphology, Melamud 2016 context2vec, McCann 2017 CoVe,
  Peters 2017 TagLM).
- `grep -i elmo SELF_ACCOUNT_SOURCES.md` (repo root) → no entry.
- Re-examined the two self-account candidates already on disk per
  `refs/self_accounts/search_log.md` (ACL Vimeo talk page, NLP Highlights
  ep.56 mention on wammar.github.io): re-fetched both via WebFetch this
  session — Vimeo returns a bot-check page, SoundCloud returns a
  browser-compat error page. No transcript text exists in either fetch;
  confirms the prior search_log's conclusion, not a new result.
- New venue tried (not previously searched): GitHub issue replies on the
  canonical implementation repo, `allenai/bilm-tf` (an explicitly-listed
  self-account venue in the fix instructions). `api.github.com/search/issues`
  + per-issue comment fetch, ~20 issues scanned. Found first-author
  (matt-peters) comments in issues #48, #95, #101, #139, #159, #190. Issue
  #95 ("Linear Combination of Embeddings") is the load-bearing one:

  > "We didn't start the ELMo project planning to use the linear
  > combination of layers, that was something we discovered along the way
  > though [sic] experimentation. As a result, the biLM architecture
  > evolved independently of the linear combination use case (and e.g.
  > using just the top layer is a perfectly sensible thing to do)."
  > — github.com/allenai/bilm-tf issue #95, comment by matt-peters
  > (fetched via `api.github.com/repos/allenai/bilm-tf/issues/95/comments`)

## Correct backing IS already in the primary, stated directly
`refs/primary/deep_contextualized_word_representations.tex`, lines 183-189
(Related Work), states the exact prior-work facts the trace's paragraphs
15-16 draw on:

> "In an RNN-based encoder-decoder machine translation system,
> \citet{Belinkov2017WhatDN} showed that the representations learned at the
> first layer in a 2-layer LSTM encoder are better at predicting POS tags
> then second layer. Finally, the top layer of an LSTM for encoding word
> context~\citep{Melamud2016context2vecLG} has been shown to learn
> representations of word sense."
> — refs/primary/deep_contextualized_word_representations.tex, lines 186-189

This is exactly the "MT encoder first-layer POS / context-biLSTM top-layer
word-sense" claim the trace cites (para 15-16) as the reason to distrust the
top-layer-only convention — a real, checkable prior-work fact already on
file (ancestors Belinkov 2017, Melamud 2016), not an invented analogy.

## Quality-gate verdict: sound_as_is — the new self-account is deliberately NOT grafted

Both prongs of the gate already hold in the current text:
(a) genuinely derived on the page: the obvious first move (top-layer-only,
    what CoVe and TagLM do) is set against the concrete, checkable
    Belinkov/Melamud layer-specialization facts above; the trace then
    explicitly refuses to extend that analogy into a claim about *its own*
    unbuilt biLM ("that's a chain of other people's results about different
    networks... not something verified on the biLM I'm about to build"),
    designs the falsifying probe experiment it *would* run, and — because
    it cannot run it on the page — makes the decision-rule move instead of
    faking the outcome: don't hard-code a layer, let the task learn the
    mixture. This is precisely the hypothesis -> experiment-design ->
    decision-rule pattern the empirical-step gate asks for.
(b) the justification is in the primary (quoted above), not invented.

The matt-peters GitHub comment (issue #95) is a genuine, on-topic
self-account, and it happens to corroborate the trace's own epistemic
honesty: the real ELMo project did NOT begin with a settled
"lower=syntax/upper=semantics therefore mix" derivation — the layer
combination was "discovered along the way though experimentation," exactly
matching why the current reasoning.md paragraph 17-18 stops short of
claiming certainty and instead hedges into a decision rule. Two things this
material could tempt a rewrite toward, both rejected:
  - Rewriting the passage into "I try the top layer, then I try mixing them,
    and the mixture wins" would be a SELF-SUPPLIED OBSERVATION — the
    narrator reporting the outcome of an experiment run mid-reasoning. The
    empirical-step gate forbids this regardless of whether it is
    historically accurate.
  - Citing the GitHub thread by name ("as Peters later said...") would be
    hindsight voice, explicitly forbidden in this first-person, present-tense
    format.
There is no way to incorporate this self-account that both (i) uses its
actual content and (ii) doesn't either decorate a step that doesn't need it
or violate the empirical-step / hindsight-voice rules. Per the gate's
explicit warning against bolting a citation onto a sound trace, it is left
out. No rewrite of reasoning.md was performed; only this file was added.
