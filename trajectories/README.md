# Innovation Trajectories

Where `methods/` holds **atomic** reasoning traces (how one method was derived), `trajectories/`
holds **iterative** ones: how a researcher, working a single MLS-Bench research question, climbs the
baseline ladder from weak to strong and finally lands one improvement past the strongest baseline.
Each trajectory is one MLS-Bench task.

A trajectory is a **playlist of the full `methods/` traces**, in weak→strong order, glued together
with the connective tissue of real research: it opens on the weakest baseline's full context, plays
each baseline's complete reasoning and answer, drops in the **measured numbers** after each, and
embeds in each next baseline's reasoning a **reflection** diagnosing why the previous one landed
where it did. It reuses, never duplicates — the heavy content lives once in `methods/`.

## The two artifacts per task

### 1. Per-baseline standalone traces — live in `methods/<slug>/`

Every baseline of the task gets the ordinary **paper-to-reasoning** triplet
(`results/{context,reasoning,answer}.md`), produced exactly per
`.claude/skills/paper-to-reasoning/SKILL.md` — in-frame, fully grounded in retrieved sources, and
passed through the per-method Codex review gate (`.codex_review.json`). These are the full-length
derivations (the `methods/` reasoning files run ~100–250 lines; that is the bar).

- **Reuse first.** Many baselines already exist in `methods/` (check `methods.json` and
  `methods/<slug>/`). If a trace exists, *reuse it as-is* — do not regenerate, do not re-run its Codex
  gate. Only the *missing* baselines are created.
- The **finale** (the real, known-better method the trajectory lands on) also gets a full standalone
  trace in `methods/`, created the same way, plus a `methods.json` entry.

### 2. The cumulative trajectory — lives in `trajectories/<task>/`

The **context** and **answer** are referenced from `methods/` (reused verbatim). The **reasoning** is
where the iteration lives: for `i>1` it is a *multi-round reasoning authored here*, and for `i=1` it is
the single-round `methods/<slug>` trace (nothing to reflect on yet). Files:

```
trajectories/<task>/
  meta.json                       # the playlist: initial_context slug + ordered steps
  <i>-<slug>-reasoning.md         # steps i>1: the MULTI-ROUND reasoning (see below)
  <i>-feedback.md                 # NUMBERS ONLY — the measured leaderboard result of step i
```

`meta.json` schema:

```json
{
  "task": "...", "title": "...", "domain": "...",
  "metrics": ["..."], "metric_columns": ["eval_return", "auc", "..."],
  "initial_context": "<weakest-baseline-slug>",
  "steps": [
    { "n": 1, "slug": "<weakest>", "method": "<display name>", "feedback": "01-feedback.md" },
    { "n": 2, "slug": "<next>", "method": "...", "reasoning": "02-<next>-reasoning.md", "feedback": "02-feedback.md" },
    { "n": 3, "slug": "<finale>", "method": "...", "reasoning": "03-<finale>-reasoning.md", "feedback": "03-feedback.md", "finale": true }
  ]
}
```

The website renders, in order: the **initial context** (`methods/<initial_context>/results/context.md`,
full), then for each step its **reasoning** (the step's `reasoning` file if present, else
`methods/<slug>/results/reasoning.md`) → **answer** (`methods/<slug>/results/answer.md`, full) →
**feedback** (numbers).

What you author per step:
- **`<i>-<slug>-reasoning.md`** (steps `i>1`): the **multi-round reasoning**. It is *based on* the
  single-round `methods/<slug>/results/reasoning.md` but is **not** a copy — the reflection on the
  previous baseline is **embedded throughout**, not bolted on as a preface. It **opens** by reading
  the previous `feedback` and diagnosing *why* the previous baseline landed where it did (per-seed,
  per-metric); then it derives this step's method in full (same length and depth as the single-round
  trace — that is the bar), but grounded in *this task* and threaded with back-references to the prior
  baseline's measured failure ("this is exactly the Private Eye decay I saw," "the −1000 seed was
  this"); it **closes** by stating the falsifiable expectations against the prior numbers. Reuse the
  single-round code block verbatim (the method is identical). First-person present tense; the narrator
  knows the measured results of baselines already tried (that is the genre) but never refers to any
  method's *paper* as a published artifact. **`methods/<slug>` is left untouched** — the single-round
  trace is the reference, not the edit target.
- **`<i>-feedback.md`**: **numbers only.** The real leaderboard rows for that baseline — per seed and
  mean, across every metric — as Markdown tables, with a single factual lead line naming the source
  row (`baseline:<slug>`, `is_final,true`). **No "reading the dynamics" prose, no interpretation.**
  (Interpretation belongs in the *next* reflection, not the feedback.) If the finale has no leaderboard
  row, its `feedback.md` states that in one line and shows the strongest baseline's numbers as the bar
  — still no invented numbers.

## Context scaling — weaker baseline sees less

There is exactly **one** context, the Initial Context, and it is the **weakest** baseline's full
`context.md`. From there context accumulates: the effective ground entering step *i* is
`initial context + every earlier reasoning + answer + feedback`. So later (stronger) steps carry
strictly more information than earlier (weaker) ones, and the weakest baseline reasons from the
thinnest ground. This falls out of the cumulative structure; nothing per-step is hand-trimmed.

## The trajectory frame (rules)

The atomic `methods/` traces are strictly in-frame (the narrator knows nothing of the future) — that
is unchanged, they are reused verbatim. The only *new* authored prose is the multi-round reasonings (which embed the reflection), and they
follow the same voice with one relaxation:

- **Relaxed:** the multi-round reasoning *does* know the measured results of the baselines already tried — that is
  the whole point. "My last attempt scored X on seed Y; the metric says Z, so next I'll…" is in-frame
  here, and the next method must emerge from that diagnosis (discovery order, never state-then-justify).
- **Still in force:** first-person present tense; no reference to any method's *paper* as a published
  artifact (prior-art ancestors cited by author/year are fine); derive, don't gesture; no fabricated
  numbers.

## Finale

The last step lands on a **real, known method that is genuinely stronger** than the best baseline
(grounded, verifiable — not a speculative invention). It is motivated in its multi-round reasoning's opening as the natural
next move from the strongest baseline's failure mode, and gets its own full standalone `methods/`
trace. If the task's leaderboard has a real row implementing it, its `feedback.md` quotes those real
numbers; otherwise the feedback shows the strongest baseline's numbers as the bar, with no invented
numbers.

## Website

`trajectories.json` (root) indexes the set (`[{task, title, domain, finale}]`). The site has a
**Methods | Trajectories** mode switch; trajectory mode reads each task's `meta.json` and composes the
reading view from the referenced `methods/` files + the local reasoning/feedback files. Add one
`trajectories.json` entry per completed task.

## Producing one trajectory (the per-task sub-agent recipe)

One sub-agent owns one task, end to end:

1. Read this file, `methods/adam/results/*` (exemplar of the atomic-trace bar), and
   `.claude/skills/paper-to-reasoning/SKILL.md`.
2. Read `~/MLS-Bench/tasks/<task>/{task_description.md,config.json,leaderboard.csv}`.
3. Rank the baselines weak→strong from the `is_final,true` `baseline:*` rows.
4. For each baseline: reuse `methods/<slug>/` if it exists; else create it via the skill (grounded +
   Codex-reviewed) and add it to `methods.json`.
5. Choose + create the finale method's standalone `methods/` trace the same way.
6. Author only `trajectories/<task>/`: `meta.json`, the per-step multi-round `reasoning` files (steps
   i>1), and the numbers-only `feedback` files. Add the task to `trajectories.json`.
7. `git add` + commit immediately (this repo is shared across sessions).
