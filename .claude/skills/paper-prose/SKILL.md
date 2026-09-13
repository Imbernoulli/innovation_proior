---
name: paper-prose
description: >-
  Plain-language rules for every sentence of the Innovation Prior paper (and any
  other research writing in this repo). Load before drafting or editing any .tex
  prose, abstract, caption, or rebuttal. Merges Wikipedia's "Signs of AI writing"
  (via blader/humanizer), the Orwell/Gowers plain-English rules, the academic
  humanizer's keep-list, Paul Graham's "write like you talk", and McEnerney's
  "value for the reader" test. Also run as a self-audit before returning prose.
---

# Paper prose: say it like a person, to a reviewer

The reader is an ICLR/NeurIPS reviewer with twenty minutes. They know the field.
They do not need to be impressed; they need to be able to repeat our claim to a
colleague after one read. Every rule below serves that.

## 0. Three tests that beat every rule

1. **Value test (McEnerney).** Before writing a paragraph, name the thing the
   reader believes now and how this paragraph changes it. If nothing changes,
   cut the paragraph. Open sections with the instability (what is wrong, what
   costs what), never with background for its own sake.
2. **Colleague test (Graham).** Read the sentence aloud. If you would not say it
   that way to a colleague at a whiteboard, rewrite it in the words you would
   say. "Delving into the intricate landscape" fails; "we looked at" passes.
3. **Evidence test (MLS-Bench style).** A citation is evidence for a claim we
   make, never a survey item. Write the claim, then cite what shows it. Never
   write "X et al. propose ...; Y et al. propose ..." lists.

## 1. Hard bans (act on one sighting)

- **Em/en dashes and `---`/`--`.** Zero in the final text. Replace with a
  comma, a period, a colon, or parentheses. Dashes are the loudest AI tell and
  the old draft was full of them.
- **Not X but Y** and its cousins: not only/just/merely X but Y; X rather than
  Y; "This does not mean X. It means Y." State Y. Keep the contrast only when
  the reader actually believes X.
- **One-line closers** that restate the paragraph ("That is the real win.").
- **Sayings that sound deep:** the real question is, at its core, fundamentally,
  the heart of the matter, X is the Y of Z.
- **Staged run-ups:** In this section we..., Let us now turn to..., It is worth
  noting that..., Importantly, Notably, Interestingly (allowed at most once per
  section, and only when what follows is genuinely surprising).
- **Arguing with no one:** One might think..., A tempting approach would be...,
  To be clear, This is not to say. Cut unless the objection is one a reviewer
  will raise; then answer it in full.
- **AI vocabulary:** delve, tapestry, landscape (abstract), leverage, robust
  (figurative), pivotal, crucial, key (adj.), underscore, highlight (verb),
  showcase, foster, intricate, holistic, multifaceted, nuanced, paramount,
  seamless, groundbreaking, testament, vibrant, enduring, garner, bolster,
  interplay, align with, deep dive, meticulous, comprehensive (abstract).
- **Copula avoidance:** serves as, stands as, represents, marks, functions as,
  boasts, features. Use is / are / has.
- **Inflated significance:** marks a pivotal moment, plays a key role, sets the
  stage, reflects a broader trend, paves the way, opens the door.
- **-ing riders** bolted on for depth: highlighting, underscoring, ensuring,
  reflecting, showcasing, demonstrating the importance of.
- **Borrowed authority:** "it is widely recognized", "experts agree",
  "a growing body of work". Cite the work or cut.
- **Vague association:** associated with, linked to, tied to, connected to,
  when a specific relation is known. Say what the relation is.
- **Forced triads.** Two items are two; four are four. Never pad to three.
- **Synonym rotation.** One name per thing for the whole paper (corpus, not
  corpus/dataset/data; trace, not trace/trajectory/narrative). Vary sentences,
  never terminology.
- **Bold as decoration, Title Case headings, emoji, arrows in prose.**
- **Meta commentary:** "as discussed above", "as we will see", section
  previews and section summaries.

## 2. Plain-English rules (Orwell/Gowers, apply on every sentence)

1. Cut every word that adds nothing. If the sentence survives without it, it
   goes. "In order to" is "to"; "due to the fact that" is "because"; "prior
   to" is "before"; "utilize" is "use"; "facilitate" is "help".
2. Active voice with a named agent wherever the agent matters ("we train",
   "the verifier rejects"). Passive is fine in methods where the agent is
   obvious ("models are evaluated with five samples").
3. Concrete subject. A sentence should start with a thing or a person, not an
   abstraction ("The realization of gains ..." becomes "The model gains ...").
4. Short word over long word. Saxon over Latinate.
5. One idea per sentence. A sentence has one verb doing the work.
6. Condition before command in instructions ("If X fails, do Y").
7. Break any rule rather than write something barbarous.

## 3. What to keep (academic keep-list, do not over-trim)

- Standard logical connectives: however, although, thus, because, in contrast,
  in practice. At most one sentence-initial adverb per paragraph.
- One calibrated hedge per claim: "may", "suggests", "we observe". Never stack
  ("may potentially suggest that it is possible").
- Repetition of a technical term. Repeating "verifier" five times is correct;
  cycling to "checker" and "oracle" is a bug.
- Citations after claims. Dense is good.
- Specific numbers, but in tables or on their own line, not embedded in prose
  three at a time.
- Rhetorical questions that frame a research question ("Can the model itself
  innovate?") once per section at most.

## 4. Rhythm (highest-impact fix for AI cadence)

Vary sentence length within every paragraph: some under 12 words, some over 30,
standard deviation above 8. Vary openings: not three consecutive sentences that
start with the same subject. Move a qualifying phrase to the front sometimes
("On MLS-Bench, the gain is ..."). Never fix cadence with staccato fragments.

## 5. Paper-specific conventions for this repo

- `\paragraph{...}` heads are **claims**, not topics: "The prior pays off before
  RL." not "Pre-RL results."
- Each paragraph: claim first, evidence second, caveat last. No paragraph ends
  on a restatement.
- Numbers live in tables. Prose says direction and rough size ("about 2.7 times
  higher"), the table says 0.038 versus 0.101.
- Expand an acronym the first time it appears in the main text and again in
  the appendix. FCS, ALE, MLS are expanded once each.
- Caption = one sentence saying what the reader should see, then the legend.
- Do not narrate the writing process or earlier drafts ("an earlier version of
  the corpus"), unless the paper is explicitly reporting a data iteration as a
  finding, and then say what changed and what it did to the numbers.
- Related work follows the three-move logic: what is evaluated, what is done
  (harness, RL), what nobody asks (can the model itself innovate). Every
  citation there is evidence for one of those three moves.

## 6. Procedure

1. Draft fast in the words you would say aloud.
2. Run the tell scan on the file:

```bash
grep -nE -- '---|--|—|–|not (only|just|merely)|rather than|serves as|stands as|represents a|marks a|pivotal|crucial|leverag|robust|landscape|underscor|highlight|showcas|delv|tapestr|foster|intricate|holistic|nuanced|paramount|seamless|groundbreaking|testament|notably|importantly|interestingly|it is worth noting|in this section|as discussed|as we will see|a growing body|widely recognized|associated with|linked to' FILE.tex
```

   Every hit is either rewritten or gets a one-word reason to stay (rhythm,
   quotation, technical term).
3. Read the paragraph aloud. Fix anything you would not say.
4. Check rhythm: at least one short and one long sentence per paragraph.
5. Check that no fact, number, citation, or claim was added or lost in the
   rewrite. Numbers come from the repo's result files, never from memory.

## References (loaded on demand)

- `references/humanizer.md`: blader/humanizer v3, 25 patterns with before/after.
- `references/plain_english.md`: Orwell/Gowers rules plus AI-detox checklist.
- `references/humanizer_academic.md`: the academic keep-list and the rhythm
  evidence (pattern 34).
- `references/clearly_concisely.md`: Strunk's composition rules.
- Sources: Wikipedia "Signs of AI writing"; Paul Graham "Write Like You Talk"
  and "Write Simply"; Larry McEnerney "The Craft of Writing Effectively".
