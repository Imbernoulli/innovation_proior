# Sources — gauge-principle-yang-mills (svfix repair, track W3_notes_unclear)

## Gap this closes
`notes/synthesis.md` and `notes/grounding_gaps.md` already asserted that the
"repeated failed attempts / give up three or four times" framing and the
"a simple quadratic term ... miraculously cancel ... hit a gold mine" language
that shapes reasoning.md's decisive-step passage (the pivot from "make the
naive-curl residues cancel" to "counteract them with a term built into F_μν
from the start") come from Yang's own first-person retrospective account. But
that account was captured only as quotes inside a notes/ working file — it had
never actually been fetched and archived as a citable source file, so nothing
on disk could be pointed to for it (grounding_gaps.md named "PMC8288855" but no
such file existed under refs/ or notes/). Fixed by fetching and archiving the
actual article below and re-verifying every quote against it verbatim.

## Source (self-account) — newly archived this pass
- **Type**: self-account (author retrospective interview)
- **Title**: "Conversation with Chen-Ning Yang: reminiscence and reflection"
- **Authors/interviewers**: Mu-ming Poo & Alexander Wu Chao
- **Venue**: National Science Review 7(1):233–236 (issue date 2020 Jan; published online 2019 Aug 7)
- **DOI**: 10.1093/nsr/nwz113 — **PMCID**: PMC8288855 — **PMID**: 34692035
- **URL**: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8288855/ (Open Access, CC BY 4.0)
- **Local file**: `refs/yang-nsr-interview-2019.txt` (21,788 bytes; fetched and archived this pass)
- **Interview date**: 21 March 2019, Tsinghua University, Beijing

### Load-bearing quote (verbatim, verified present in the archived file)
> "But it was not all smooth sailing because U1 symmetry is a commuting
> symmetry and SU2 symmetry is a non-commuting symmetry: while the first
> steps to a non-commuting theory were mathematically easy, the next steps
> led to formulae that became more and more complicated, and I had to give
> up. ... Between 1947 and 1954 I must have repeated this unsuccessful
> attempt three or four times. ... we observed that the undesired
> complicated terms were quadratic and cubic. Could they be cancelled if we
> introduce quadratic and/or cubic terms at the beginning? It turns out that
> a simple quadratic term introduced at the beginning did miraculously
> cancel all the undesired complicated terms! The cancellation was so
> beautiful we knew we had hit a gold mine."

### What in reasoning.md this backs
This is the actual source of the decisive REFRAME in reasoning.md (not the
pure [D_μ,D_ν] algebra at lines 59–85, which is self-contained computation
and correctly cites nothing): the "wall I keep hitting ... I've come back to
this several times and each time the formulae proliferate and I give up"
passage (the repeated-failure framing) and the pivot "So the question flips
... could I cancel them by adding a term to F_μν at the start?" (the
quadratic/cubic-terms-at-the-beginning reframe) and "miraculously kills all
the undesired complicated terms ... This is the gold I kept walking past"
(the payoff). Deleting this source would leave that narrative structure
unsupported by anything on disk — it is not something the algebra alone
forces (the algebra only *confirms* the reframe works once tried; it doesn't
motivate trying it). reasoning.md is deliberately rendered in-frame per the
project's discipline (first person, present tense, no "Yang", no "the
interview", no dates) — the grounding is structural/narrative dependence on
the source's content, not a citation, so no wording in reasoning.md itself
needed to change; what was missing was the archived, verifiable file this
content depends on.

## Other sources already on disk (unchanged, for completeness)
- PRIMARY: `refs/yang-mills-1954.pdf` — Yang & Mills, Phys. Rev. 96, 191 (1954).
- ANCESTOR: `refs/jackson-okun-gauge-history.pdf`, `refs/oraifeartaigh-gauge-history.pdf`, `refs/weyl-higgs-gauge.pdf` — Weyl abelian gauge principle / EM covariant-derivative history.
- EXPLAINER: `refs/yangmills-redux.pdf` (`notes/redux.txt`) — Marateck, commutator derivation of F_μν and a secondhand trial-and-error account (contains the phrase "by trial and error" at line 98 of notes/redux.txt) — present on disk and real, but NOT the source that reasoning.md's decisive-step narrative depends on; it is a third-party explainer of the derivation, not Yang's own account, and the [D_μ,D_ν] algebra at reasoning.md lines 59–85 is self-contained and does not depend on it. Left as an explainer reference only; not cited as backing the decisive step.
- ANCESTOR (secondhand, corroborating): `notes/chyla.txt` (`refs/chyla-weyl-to-yang.pdf`) — Chýla, "From Hermann Weyl to Yang and Mills to QCD" — historical survey confirming dates/context (Oct 1, 1954 submission, Yang age 32) but does not contain the "gold mine"/"miraculously" quotes itself.
