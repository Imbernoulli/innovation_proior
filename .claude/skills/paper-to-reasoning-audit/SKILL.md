---
name: paper-to-reasoning-audit
description: Audit and fix already-written paper-to-reasoning deliverables (context.md / reasoning.md / answer.md) for in-frame-style violations. Two layers — a deterministic regex lint for the enumerable tic classes, then a full-read per-method Agent review for what rules can't catch (voice, pre-announcements, overclaims, state-then-justify ordering, hindsight, and math correctness). Use when asked to "audit / lint / check / clean up the reasoning traces", "find problems in the deliverables", "why didn't the audit catch X", or before publishing a batch to the website.
version: 1.0.0
---

# Paper → Reasoning: Audit & Fix

Companion to the `paper-to-reasoning` skill. That skill *generates* the three deliverables; this one *audits and repairs* them. It exists because the generator's own 2.4 revision pass and the Codex gate are **judgment-based**, and judgment reliably misses two things: (a) cosmetic, enumerable tics that a model glosses over while focused on math, and (b) — conversely — subtle voice/derivation problems that no regex can see. The fix is to run **both** a deterministic lint and a full-read review, not to trust one pass.

## Root cause this skill addresses

1. **Enumerable tics guarded by judgment instead of a regex.** A finite class like `(faithful to the implementation)` or `(known)` should be a hard gate, never "something the reviewer should notice."
2. **A checklist blind spot.** The generator's 2.4 hunt-list enumerates the parenthetical tics for `context.md` but not for `answer.md`; since *referencing* the canonical implementation is allowed in `answer.md`, a parenthetical compliance annotation there (`## Code (faithful to the canonical implementation)`) slips through the gray zone.
3. **Systematic, templated habits.** Because all methods share a generator and prompt template, the *same* tic recurs across dozens of methods — so a per-method judgment audit catches it inconsistently. Detect it once, globally, deterministically.

## Layer 1 — deterministic lint (run first, fix mechanically)

Run `scripts/lint_inframe.py` (report mode), then `--fix` for the mechanical classes. It scans every `<root>/<slug>/results/*.md`. Categories:

- **A_paren — parenthetical self-compliance annotations (banned anywhere; auto-fixable).** `(faithful …)`, `(known)`, `(known recipes/primitives …)`, `(pre-method)`, `(settings only …)`, `(no outcomes/results …)`. Fix = strip the parenthetical. **Exception:** when the parenthetical carries real content (e.g. `(faithful to the canonical implementation: linear interpolation, target x1−x0, L2)`), do **not** blind-strip — rewrite to keep the content and drop only the compliance framing.
- **B_meta — meta / self-narration prose (review, hand-rewrite).** "No outcomes are stated", "settings only", "strictly pre-method", "Known primitives only", "no naming of the method", "no results are reported here", "No official repository accompanies this analysis", "reference implementation" / "official repo" used as meta. The document must *be* pre-method, not *announce* that it is.
- **C_rsn_header — markdown headers inside `reasoning.md` prose (outside code fences).** `reasoning.md` is one continuous monologue: ZERO `#`/`##` headers except inside the final code fence. (Code comments `#` inside ``` fences are fine — the linter tracks fences.)
- **D_cjk — stray non-English** (CJK etc.). Deliverables are English by default.
- **E_paperref — source-paper self-reference (review).** `this paper` / `the paper` / `the authors`. **Known false positive:** datasets whose *nodes are academic papers* (citation graphs — GraphSAGE, GCN), where "the paper abstract" describes the data, not the target work. Review before deleting; prior-art ancestor citations by author/year are allowed and must stay.

Mechanical fixes for a known hit list can be scripted (see `scripts/fix_known_tics.py` as the template — exact, idempotent old→new string replacements with not-found assertions). Re-run the lint until only reviewed false positives remain.

## Layer 2 — full-read per-method Agent review (what rules can't catch)

For **each** method, dispatch an agent that **reads all three files in full** (especially `reasoning.md` and `answer.md`) against the parent rubric `~/.claude/skills/paper-to-reasoning/SKILL.md`, and **fixes in place**.

**PRIMARY LENS (the single most important check — apply before all others).** Does `reasoning.md` read as a *natural first-person derivation by the person who is inventing the method*, discovering it for the first time? It must never betray that a finished paper exists. Concretely, the trace fails this lens if it:
- **references or cites the source paper as an external artifact** — "this paper", "the authors", "as shown in §4", "the original work", a venue/arXiv/year, or any phrasing that treats the method as already-published rather than being figured out right now;
- **quotes or paraphrases the paper's own framing** — importing its theorem-first / result-first narration, its section structure, its named contributions, or its abstract's summary sentences, instead of arriving at those things as the *output* of the reasoning;
- **reads as exposition of a known result** rather than discovery — i.e. it explains the method as established fact ("the method works by…") instead of reasoning *toward* it from the problem ("I need X; the obvious move fails because…; so what if I…").
The narrator IS the inventor. If a sentence could only have been written by someone who already read the finished paper, it is a violation — rewrite it into the discovery voice (motivation → attempt → wall → fix), or cut it.
**This lens applies to `answer.md` too.** `answer.md` *may* name the method and reference its canonical implementation, but it must still not present the method as a cited external paper (no citation line, no authors/venue/arXiv, no "as the paper shows"). It is the inventor writing up *their own* result, not a literature summary of someone else's.

**WEB-GROUNDING (verify facts against retrieved sources, never against memory).** Every factual specific in all three files — equations, constants, exponents, signs, hyperparameters, loss coefficients, the exact form of each update/architecture, baseline numbers, historical/lineage claims — must be checked against an **authoritative source the reviewer actually retrieves this run** (the primary paper / arXiv source, the canonical implementation, reputable explainers via `WebSearch`/`WebFetch`). Do not "confirm" a constant or formula from memory: look it up and read the passage. A confident-looking but unsourced specific is a defect; correct it to the sourced value, or if no source can be found, flag it. This is the same grounding rule the generator must follow (`~/.claude/skills/paper-to-reasoning/SKILL.md` → "Grounding").

The lint cannot see any of the above; an agent reading end-to-end can. After the primary lens, also check:

- **Pre-announcing / labeling the rhetorical move** — "Here's the argument that turns X into Y", "now for the key insight", "the trick is going to be…". → cut the framing, walk into the argument.
- **Empty contrastive filler** — "…, not from a method", any "not from X" where X is vague. → cut or make X concrete.
- **Overclaiming the stakes** — "provably absurd", "an outright contradiction", "impossible", "paradox" for something merely surprising. → name precisely what's surprising.
- **Essay scaffolding** — "First… Second… Finally", "To summarize the above", "Having established X, I now turn to Y". → let the prose flow.
- **Gesturing instead of deriving** — "the proof is straightforward", "one can show that". → actually do the steps.
- **State-then-justify (textbook order)** — a paragraph that names the method's piece in its first sentence then justifies it. → flip to discovery order (motivation leads, the piece drops out as the conclusion).
- **Hindsight / posterior leaks** — later-work comparisons; the *proposed* method's own benchmark/ablation results (motivating/diagnostic findings about *existing* systems are fine).
- **Math / derivation correctness** — re-check every sign, factor, constant, and each case of a case-analysis. A present-but-wrong derivation (flipped inequality, reversed clip case) is a blocker, not a pass. This is the highest-value thing a full read adds.
- **Scaffold ↔ code correspondence** and **context scaffold purity** (no method-specific names / "reference implementation" wording in the `context.md` Code-framework stubs).
- **PROMPT-SIDE CONCEPTUAL LEAKAGE (SFT trainability — neither the lint nor the Codex gate sees this).** The deliverables are training data: `context.md` is the *prompt*, `reasoning.md`→`answer.md` the *target*. The lint catches *lexical* leaks (the method's name, "this paper"); this check catches *conceptual* leaks — `context.md` giving away the **move/insight/object that `reasoning.md` is supposed to discover**, even without the name. A 6-method spot audit found this in 6/6 (all scored 2/5 on leakage while the targets were strong), so treat it as the default failure mode, not an edge case. `scripts/detect_leakage.py` is a regex **suspect-flagger** for the two common channels (prescriptive baseline gaps; non-neutral scaffold TODOs) — run it first to localize, but it both over- and under-fires, so the agent is the authority. Read `context.md` **alone** and judge:
  1. **Leakage (the decisive axis):** could a reader reconstruct the method's central move from `context.md` ALONE? Pin the leaking sentences. Two channels: (a) a **baseline gap written as a prescription** — "one needs the analogue over Z_q", "the quantity that would replace N", "the missing piece: combine the current policy with the candidate" — rewrite as an *observed limitation* (where prior art stalls), never *what to build*; (b) a **scaffold TODO that pre-names the target component** (`# TODO: the per-slice constant`, `# TODO: provable(x)=∃y proof_of`) or **pre-locates the contribution** (`if gradient_is_small: <slot>`) — neutralize to one generic empty slot.
  2. **Sufficiency:** is `context.md` still a self-contained, solvable situation after de-leaking (enough tools that the reasoning *could* be derived)? Don't strip so hard the problem becomes unsolvable.
  3. **Discovery vs recitation:** does `reasoning.md` actually figure it out (motivation→wall→fix) rather than execute a plan the prompt already drew?
  Fix on the `context.md` side only (soften the prescriptive gaps, neutralize the TODOs); do **not** touch the high-quality `reasoning.md`/`answer.md` derivation. The goal: a reader of the prompt should know the *problem and the tools*, never the *answer*.

The agent makes **surgical** edits (not wholesale rewrites), preserves the in-frame discovery voice and the math, and returns a `file:line` changelog (or "clean"). Run agents in parallel batches (e.g. 5 methods per agent) for a large corpus.

## Layer 3 — independent second-reviewer (Codex) gate

Layers 1–2 are same-family (Claude). The parent skill's Phase 2.6 Codex gate (a *different* model, math-first) remains the final independent pass and catches what one model family cannot see in its own output. Run it per the parent skill when the runtime is available; it composes with this audit rather than replacing it.

## Order of operations

1. Lint (report) → mechanical `--fix` + hand-fix B/E → re-lint to a clean baseline.
2. Full-read Agent review (batched), fixing in place.
3. Re-run the lint (the review may introduce a stray header/tic).
4. Independent Codex gate.
5. Only then publish / commit.
