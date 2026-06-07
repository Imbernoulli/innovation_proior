# Author self-account — Aho–Corasick

Source: *Alfred V. Aho Oral History* (2023), as summarized/cited on the Wikipedia
"Aho–Corasick algorithm" article (Origin/History section); corroborated by the
primary paper's own §1, §5, §7.

Genesis (the real path, to be lived out in reasoning.md as the inventor's own
thinking — NOT cited in-frame):

- Margaret J. Corasick (information scientist, PhD Lehigh) was building a
  bibliographic-search tool at Bell Laboratories. The data was machine-readable
  citation tapes — the cumulated data behind *Current Technical Papers*, a
  fortnightly internal citation bulletin. By summer 1973 there were three years
  of cumulated data: ~150,000 citations, total length ~10^7 characters (these
  exact figures are in §7 of the primary paper).
- A bibliographer queries for all titles satisfying a Boolean function of
  keywords/phrases (e.g. titles containing both "ion" and "bombardment"), with
  options for embedding (full/left/right/none) so "ions" can match "ion" but
  "motion" need not.
- Her first program did straightforward string matching: take each keyword in
  turn and scan it against each title. Cost ~ (#keywords) × (text length). With
  many keywords this scaled badly — one bibliographer hit the **$600** machine
  usage limit before a single search finished.
- Aho gave a seminar on algorithm design; afterwards they discussed her scaling
  problem. Aho suggested the finite-state-machine-over-a-trie approach (combine
  KMP's failure-function idea with a trie of all keywords). The redesigned
  program ran in a single pass over the text, cost roughly independent of the
  number of keywords; that bibliographer's search dropped from over **$600 to
  ~$25**.
- Primary-paper corroboration of motivation:
  - §1: "Our approach combines the ideas in the Knuth-Morris-Pratt algorithm
    with those of finite state machines."
  - §5: "In fact it was the time complexity of the straightforward algorithm
    that prompted the development of Algorithm 1."
  - §7: old vs new running times on a Honeywell 6070: 15 keywords .79h→.18h,
    24 keywords 1.27h→.21h (CPU hours). "the cost of a search is roughly
    independent of the number of keywords."
- Later this algorithm became the basis of the Unix `fgrep` command.

This is a true author self-account (oral history) with a genuine anecdote AND the
omitted motivating reasoning. The reasoning.md backbone is: many keywords at once;
re-scanning per keyword is wasteful; build a trie of all keywords; KMP's
single-pattern failure function generalizes to failure LINKS on the trie; output
links so all keywords ending at a node are reported; one linear pass over the text.
