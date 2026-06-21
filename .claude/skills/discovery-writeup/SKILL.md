---
name: discovery-writeup
description: Write the fourth deliverable, results/train_answer.md, for a method that already has context.md / reasoning.md / answer.md. It is the scientist's own final write-up of their discovery — the polished account they would present to peers after the long, reflective reasoning is done: a short framing of the analysis, then the proposed method named and explained in concrete detail, then the code. Flowing prose (not a structured Markdown document), grounded entirely in the three existing files. Use when asked to "generate the train_answer / fourth file", "write the presented short answer", "add the scientist's write-up", or to backfill methods that lack train_answer.md.
version: 1.0.0
---

# Discovery write-up — `results/train_answer.md`

Each method folder under `methods/<slug>/results/` holds three files that already exist and are
already reviewed. (Note: a number of `train_answer.md` files already exist, written earlier by
someone else. They fix the *format and style* shown below, but they are **not** a quality ceiling —
some are thin or imprecise. Do not imitate their weaknesses: write a genuinely better account —
more faithful to the source files, more complete on the method's actual mechanism, and clearer in
its reasoning. Match the format; exceed the depth.)

- `context.md` — the situation: the problem, the prior art, the tools on the table (pre-method).
- `reasoning.md` — the long, first-person, reflective derivation: the scientist thinking it
  through, hitting walls, self-correcting, and finally landing on the method and its code.
- `answer.md` — the distilled, structured account of the finished method, with the canonical code.

This skill produces the **fourth** file, `results/train_answer.md`: the scientist's **own final
write-up of what they discovered** — the version they would hand to a colleague once the thinking is
finished. A real scientist first reasons at length and with doubt (that is `reasoning.md`); only then
do they present a clean, self-contained account of the result. That presented account is what you are
writing. It is not "short" in the trivial sense — it is as long as it needs to be to convey the
analysis, the method, its details, and the code — but it reads as one continuous spoken-or-written
explanation a person gives, **not** as a reference document carved into sections.

## What it must contain — three movements, in this order

Write it as **continuous prose** (no Markdown headers, no bullet outlines, no tables describing the
method), followed by a single code block at the end. The prose moves through three movements without
announcing them:

1. **The analysis, summarized.** Open by laying out the problem and why the existing options are not
   enough — concisely, in the scientist's voice. State the goal, then walk the prior approaches and
   the specific way each falls short, so the need for something new is felt. This is a compression of
   `context.md` + the opening of `reasoning.md`, a paragraph or two, not a literature survey.
2. **The proposed method, named and explained in real detail.** State plainly what you propose and
   **name it** (e.g. "I propose ResNet", "The method is Adam", "The solution is DCGAN"). Then —
   this is the core — **explain the method's actual mechanism in concrete detail**: the defining
   equation(s) or update rule, each component and what it is for, the key design choices and *why
   each one is made and why the obvious alternative is worse*, and the derivation steps that make the
   method work (bias correction, the scale-invariance argument, the identity-mapping reparameterization,
   whatever the load-bearing ideas are). A reader must come away understanding not just the method's
   name but **how it works and why it is built the way it is**. Do not hand-wave; carry over the real
   substance from `reasoning.md`/`answer.md`. This is usually several paragraphs.
3. **The code.** End with the working implementation (for a computational method) or the precise
   final artifact — the clean theorem statement and proof, the final formula, the protocol — for a
   non-computational discovery, exactly as the field would present it.

## Hard rules

- **The code is COPIED, not rewritten.** Take the method's final implementation **verbatim from
  `answer.md`** (it is already the canonical, reviewed code). Do not re-implement it, do not "clean it
  up", do not change names, signatures, constants, or logic. The code in `train_answer.md` must be
  **identical to the code in `answer.md`/`reasoning.md`** — byte-for-byte for the primary
  implementation. This is non-negotiable: it guarantees the write-up never diverges from the other
  deliverables and never introduces a bug. If `answer.md` contains several code blocks (a main method
  plus a sibling variant, e.g. Adam + AdaMax), carry over the **primary** method's code; include a
  secondary variant only if it is genuinely part of the method, and then also verbatim. Before
  finishing, diff the code you wrote against `answer.md` and confirm it matches.
- **Grounded entirely in the three existing files — invent nothing.** Every fact, equation, constant,
  hyperparameter, and design rationale must already be present in `context.md` / `reasoning.md` /
  `answer.md`. You are reformatting and distilling material that is already there and already verified,
  **not** doing new research and **not** adding claims. No web lookups, no new numbers, no new method
  pieces. If the three files disagree on a detail, trust `answer.md` (the reviewed final form) and
  `reasoning.md`'s derivation over a stray phrasing.
- **Read all three source files in full first.** Do not write from the file names or a skim. Read
  `context.md`, `reasoning.md`, and `answer.md` end to end so the analysis summary is faithful and the
  method details are complete and correct.
- **Math in LaTeX/Markdown.** Write equations with LaTeX math: inline as `$y = F(x) + x$`,
  `$\beta_2 = 0.999$`, and set off the load-bearing equations as display math with `$$ ... $$`
  (e.g. `$$\theta_t = \theta_{t-1} - \alpha\,\hat m_t / (\sqrt{\hat v_t} + \epsilon)$$`). Use proper
  symbols (`\alpha`, `\beta`, `\hat`, `\sqrt`, subscripts, `\nabla`, `\sum`). This is the one place
  Markdown is expected — the surrounding text is still continuous prose with **no section headers**.
  (Many existing `train_answer.md` files render math in plain ASCII; that is the older style — prefer
  LaTeX for new files.)
- **Continuous prose, no document scaffolding.** No `#`/`##` headers, no section titles, no bulleted
  feature lists *describing the method*, no "Problem / Key idea / Algorithm" structure. (One code
  block at the end is expected; comments inside the code are fine.) The three movements flow into each
  other as paragraphs.
- **English.**
- **In-frame — present it as your own discovered result, never as a cited paper.** Name the method
  freely. But do **not** include a citation line, the source paper's title/authors/venue/arXiv id, or
  phrases like "this paper" / "the authors" / "as shown in the paper". You are the scientist writing up
  *their own* finding, not summarizing someone else's publication. Citing genuine prior-art ancestors
  by author/year where they already appear in the source files is fine.
- **No meta-commentary about format or process.** Do not write "in this write-up", "to summarize the
  above", "as the reasoning showed", or any narration of the document's own structure or purpose.
  Just give the analysis, the method, and the code.

## Voice

First-person scientist presenting a finished result: confident, direct, explanatory. It is the calm
write-up *after* the messy thinking — so unlike `reasoning.md` it does **not** hit walls or
self-correct in real time; it states the analysis and the resolved method. But it keeps the
scientist's "I": "We need…", "The failure is therefore…", "I propose…", "What makes it work is…".
It is the human author's voice explaining their own discovery to a peer.

## Procedure (per method)

1. Read `methods/<slug>/results/context.md`, `reasoning.md`, and `answer.md` **in full**.
2. Identify: the problem and the prior-art gaps (movement 1), the method's name + its concrete
   mechanism/design-choices/derivation (movement 2), and the canonical code block(s) in `answer.md`
   (movement 3).
3. Write `methods/<slug>/results/train_answer.md`: analysis summary → named method explained in
   detail → code copied verbatim from `answer.md`. Continuous prose, plain math, English, in-frame.
4. Verify before finishing: the code matches `answer.md` exactly; no headers; no `this paper`/citation;
   no invented facts; the method's details are actually explained, not just named.
