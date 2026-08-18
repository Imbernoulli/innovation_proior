# Sources — chain-of-thought

## Searches run
- SELF_ACCOUNT_SOURCES.md at repo root: grepped for "chain of thought", "chain-of-thought",
  "Jason Wei", "Denny Zhou" — only hit is Turán's unrelated 1977 use of the phrase "chain of
  thought" in a different method's file (methods/turan-symmetrization). Nothing for this method.
- WebSearch: "Jason Wei blog chain of thought prompting origin jasonwei.net"
- WebSearch: "Denny Zhou chain of thought prompting discovery interview"
- WebSearch: "\"Language models perform reasoning via chain of thought\" Google AI Blog Wei Zhou"
- WebFetch: https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/
  (Google Research blog by Wei & Zhou) — checked, but it is purely a recap of the paper's own
  content (same motivation, same ablation claims), not independent source pressure. Not used.
- WebFetch: https://www.jasonwei.net/blog/research-i-enjoy — Wei's retrospective on why he
  values the CoT paper (generality, scale, no finetuning, community impact) — appreciation only,
  no design-rationale content. Not used.
- WebSearch/WebFetch: OpenReview forum for NeurIPS 2022 (_VjQlMeSB_J) and both
  api.openreview.net / api2.openreview.net endpoints — all blocked by OpenReview's bot
  "ChallengeRequiredError" (403), could not retrieve reviewer/rebuttal discussion.
- WebFetch: https://dennyzhou.github.io/Teach-Language-Models-to-Reason.pdf — PDF text
  extraction came back corrupted/unreadable; no usable content recovered.
- WebFetch: https://www.antoinebuteau.com/lessons-from-denny-zhou/ and a Medium recap of a Denny
  Zhou talk — neither contains genesis/motivation material for the 2022 CoT paper specifically
  (Medium page also 403'd).
- WebFetch: https://www.thecrimson.com/article/2023/9/18/denny-zhou-ai-talk/ — mentions CoT only
  as one bullet in a four-part framework, no design rationale.
- WebFetch: https://www.jasonwei.net/blog/some-intuitions-about-large-language-models — FOUND,
  used (see below).

## Source used

**Jason Wei — "Six intuitions about large language models"**
https://www.jasonwei.net/blog/some-intuitions-about-large-language-models
Saved: `refs/self_accounts/jasonwei-six-intuitions-about-llms.txt`

Summary: personal blog post by the CoT paper's first author, independent of the paper itself
and independent of the Google Research blog recap. "Intuition 3" gives his own retrospective
framing for why chain-of-thought works: tokens differ in *information density* — some are easy
to guess, some are hard to guess, and some are neither but are expensive to *compute* (e.g. the
result of an arithmetic expression) — and a model forced to answer immediately has no room to
spend the compute a hard token needs, so the fix is to let it reason in natural language first
and buy itself more forward passes before it has to commit.

Load-bearing quote: "Some tokens can also be very hard to compute... the next token requires a
lot of work (evaluating that expression). You can imagine that if you're ChatGPT, and as soon
as you have to see the prompt you have to immediately start typing, it would be pretty hard to
get that question right. The solution to this is to give language models more compute by
allowing them to perform natural language reasoning before giving the final answer."

Used to deepen: the "fixed compute budget" mechanical argument in `results/reasoning.md`
(paragraph beginning "Then there's something I can reason about mechanically rather than just
hope for" — the extra-computation hypothesis that later becomes ablation control Two,
variable-compute-only). Previously this argument was presented as the scientist's own unaided
mechanical reasoning with no external pressure (per the audit finding, functionally just the
paper's own ablation restated as a prediction). Rewrote it to run through the
information-density / hard-to-compute-token distinction Wei's post makes, reusing the
already-established cars exemplar so no new invented scenario is introduced. Kept the
equation-only / variable-compute / reasoning-after-answer three-control structure exactly as
written — that part is the primary paper's own ablation design and is left alone (it's already
correctly in-frame as the scientist's own controls to test the hypothesis).

## Not used / ruled out
- The three named ablations (equation-only, dots/variable-compute, reasoning-after-answer) are
  the primary paper's own Table content (`src/fables/appendix-ablation.tex`,
  `src/main_fables/ablation-bar.tex`) — confirmed by reading the .tex, matches reasoning.md's
  three controls exactly. No independent source found or needed for these; they are correctly
  primary-derived and left as-is.
