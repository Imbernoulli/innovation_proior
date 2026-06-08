# Innovation Trajectories

Where `methods/` holds **atomic** reasoning traces (how one method was derived), `trajectories/`
holds **iterative** ones: how a researcher, working a single MLS-Bench research question, climbs the
baseline ladder from weak to strong and finally lands one improvement past the strongest baseline.
Each trajectory is one MLS-Bench task.

A trajectory is a **playlist of the full `methods/` traces**, in weak→strong order, glued together
with the connective tissue of real research: it opens on the weakest baseline's full context, plays
each baseline's complete reasoning and answer, drops in the **measured numbers** after each, and
prefaces each next baseline with a short **reflection** that diagnoses why the previous one landed
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

Only the connective tissue is authored here; the reasoning/answer/context are **referenced** from
`methods/` and composed by the website. Files:

```
trajectories/<task>/
  meta.json                       # the playlist: initial_context slug + ordered steps
  <i>-<slug>-reflection.md        # short: diagnose step (i-1)'s feedback, motivate moving to step i
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
    { "n": 2, "slug": "<next>", "method": "...", "reflection": "02-<next>-reflection.md", "feedback": "02-feedback.md" },
    { "n": 3, "slug": "<finale>", "method": "...", "reflection": "03-<finale>-reflection.md", "feedback": "03-feedback.md", "finale": true }
  ]
}
```

The website renders, in order: the **initial context** (`methods/<initial_context>/results/context.md`,
full), then for each step its **reflection** (if any) → **reasoning** (`methods/<slug>/results/reasoning.md`,
full) → **answer** (`methods/<slug>/results/answer.md`, full) → **feedback** (numbers).

What you author per step is small:
- **`<i>-<slug>-reflection.md`** (steps `i>1` only): one to three paragraphs, first-person present
  tense. Read the previous `feedback`, diagnose *why* the previous baseline landed where it did
  (per-seed, per-metric), and let the choice of the next method fall out of that diagnosis. End right
  as the next method is about to be derived — do **not** re-derive it here (the full reasoning follows
  inline). No prose section headers; this is a thought, not an essay.
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
is unchanged, they are reused verbatim. The only *new* authored prose is the reflections, and they
follow the same voice with one relaxation:

- **Relaxed:** a reflection *does* know the measured results of the baselines already tried — that is
  the whole point. "My last attempt scored X on seed Y; the metric says Z, so next I'll…" is in-frame
  here, and the next method must emerge from that diagnosis (discovery order, never state-then-justify).
- **Still in force:** first-person present tense; no reference to any method's *paper* as a published
  artifact (prior-art ancestors cited by author/year are fine); derive, don't gesture; no fabricated
  numbers.

## Finale

The last step lands on a **real, known method that is genuinely stronger** than the best baseline
(grounded, verifiable — not a speculative invention). It is motivated in its reflection as the natural
next move from the strongest baseline's failure mode, and gets its own full standalone `methods/`
trace. If the task's leaderboard has a real row implementing it, its `feedback.md` quotes those real
numbers; otherwise the feedback shows the strongest baseline's numbers as the bar, with no invented
numbers.

## Website

`trajectories.json` (root) indexes the set (`[{task, title, domain, finale}]`). The site has a
**Methods | Trajectories** mode switch; trajectory mode reads each task's `meta.json` and composes the
reading view from the referenced `methods/` files + the local reflection/feedback files. Add one
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
6. Author only `trajectories/<task>/`: `meta.json`, the per-step `reflection` files (steps i>1), and
   the numbers-only `feedback` files. Add the task to `trajectories.json`.
7. `git add` + commit immediately (this repo is shared across sessions).
