# Innovation Trajectories

Where `methods/` holds **atomic** reasoning traces (how one method was derived), `trajectories/`
holds **iterative** ones: how a researcher, working a single MLS-Bench research question, climbs
the baseline ladder from weak to strong and finally lands one improvement past the strongest
baseline. Each trajectory is one MLS-Bench task.

This is the chain-of-thought of *doing research over time*, not of deriving one result once.

## The two artifacts per task

A trajectory composes, never duplicates, the atomic traces in `methods/`.

### 1. Per-baseline standalone traces — live in `methods/<slug>/`

Every baseline of the task gets the ordinary **paper-to-reasoning** triplet
(`results/{context,reasoning,answer}.md`), produced exactly per
`.claude/skills/paper-to-reasoning/SKILL.md` — in-frame, fully grounded in retrieved sources,
and passed through the per-method Codex review gate (`.codex_review.json`).

- **Reuse first.** Many baselines already exist in `methods/` (check `methods.json` and
  `methods/<slug>/`). If a trace exists, *reuse it* — do not regenerate. Only the *missing*
  baselines are created.
- **Each context is genuinely distinct.** A task's baselines must not share a recycled
  `context.md`; each method's lineage and pain points are its own. (The paper-to-reasoning skill
  already guarantees this, because each method is grounded independently.)
- The **finale** (the real, known-better method the trajectory lands on, §finale) also gets a
  full standalone trace in `methods/`, created the same way.

### 2. The cumulative trajectory — lives in `trajectories/<task>/`

One running, time-ordered research narrative. Files, in order:

```
trajectories/<task>/
  meta.json                       # task ref, ordered baselines (weak→strong), metric labels, method slugs
  00-initial-context.md           # the Initial Context (leanest — the simplest baseline's setting)
  01-<slug>-reasoning.md          # baseline 1: arriving at the weakest baseline
  01-<slug>-answer.md             # baseline 1: distilled method + code (brief; links methods/<slug>)
  01-feedback.md                  # MY feedback after baseline 1 — real leaderboard metrics + read of the dynamics
  02-<slug>-reasoning.md          # baseline 2: opens by reflecting on baseline 1 + its feedback, then derives baseline 2
  02-<slug>-answer.md
  02-feedback.md
  ...                             # one (reasoning, answer, feedback) block per baseline, weak→strong
  NN-<newslug>-reasoning.md       # the finale: reflect on the strongest baseline, derive a real known-better method
  NN-<newslug>-answer.md
  NN-feedback.md                  # the finale's measured result (from a leaderboard agent row if available; else omit numbers)
```

`NN` = (#baselines + 1).

## Context scaling — weaker baseline sees less

The Initial Context (`00-initial-context.md`) is the **leanest** framing: the research question
plus only what is needed to motivate the *weakest* baseline. It is roughly the weakest baseline's
`context.md`, trimmed to the starting point — no foreknowledge of what later baselines will need.

From there context **accumulates**: the effective context entering step *i* is
`initial + every earlier baseline + every earlier feedback`. So later (stronger) steps carry
strictly more information than earlier (weaker) ones — the weakest baseline reasons from the
thinnest ground, the finale from the richest. This is the scaling the task requires, and it falls
out of the cumulative structure rather than being hand-tuned.

## The trajectory frame (rules — these differ from paper-to-reasoning)

The atomic traces are strictly in-frame (the narrator knows nothing of the future). The trajectory
is a **different genre**: the narrator is a researcher iterating *across* attempts, so the rules
relax in exactly one way and hold everywhere else.

- **Relaxed:** the narrator *does* know the measured results of methods they have already tried —
  the whole point is to reflect on them. "My last attempt scored X; the metric tells me Y, so next
  I'll…" is in-frame here. Each `<i>-reasoning.md` for `i>1` **opens** by reading the previous
  feedback, diagnosing *why* the previous baseline fell where it did, and letting the next method
  emerge from that diagnosis (discovery order, never state-then-justify).
- **Still in force (inherited from paper-to-reasoning):** first-person present tense; no reference
  to any method's *paper* as a published artifact (methods are things the researcher proposes/tries;
  prior-art *ancestors* are still cited by author/year and elaborated); derive, don't gesture;
  no fabricated numbers.
- **Feedback is grounded.** `<i>-feedback.md` reports the **real** numbers from the task's
  `leaderboard.csv` (the `is_final,true` baseline rows, per seed and mean, across every metric
  label in `config.json`). The *interpretation* of those numbers (overfitting, instability across
  seeds, a metric that didn't move, compute cost) is reasoning clearly framed as inference from the
  given metrics — never new measurements.
- **No duplication.** `<i>-reasoning.md` carries only the *iteration* (the reflection + the delta to
  the next method); for the full self-contained derivation it links to `methods/<slug>/`. For `i=1`
  the reasoning is necessarily close to the standalone trace, so keep it short and link.

## Finale

The last step lands on a **real, known method that is genuinely stronger** than the best baseline
(per the Q-decision: grounded, verifiable — not a speculative LLM invention). It is derived in-frame
as the natural next move from the strongest baseline's failure mode, gets its own standalone
`methods/` trace, and — if the task's leaderboard contains a real agent/strong row implementing it —
its `feedback.md` quotes those real numbers; otherwise the finale feedback states what one would
want to validate, with no invented numbers.

## Conventions settled on the pilot (`rl-intrinsic-exploration`)

- **Feedback files are slug-less and numbered:** `01-feedback.md` … `NN-feedback.md` (the finale gets one too).
- **Finale feedback with no leaderboard row:** still write the numbered `NN-feedback.md`, but it states
  the *bar to beat* (the strongest baseline's real numbers) + what one would want to validate — **zero
  invented numbers**. Only quote real numbers if the task's `leaderboard.csv` actually has an agent/strong
  row implementing the finale.
- **A created standalone trace's full workspace lives under `methods/<slug>/`** (not `~/paper2reasoning`):
  `methods/<slug>/{src,notes,results}/`, matching existing `methods/` entries. Commit `results/*.md`, the
  `.codex_review.json`, `notes/*.md`, and the `.tex/.bbl/.bib`; leave raw `src.tar.gz`/`.png` untracked.
- **Reused baselines are taken as-is.** Do not regenerate or re-run the Codex gate on a pre-existing
  `methods/<slug>/` even if its `.codex_review.json` is `false`/missing — re-auditing the existing library
  is a separate task. (Note any such case in your report.)

## Website

`trajectories/` gets a sibling view to `methods/` (a `trajectories.json` index + a reading mode that
walks the ordered files of a task). Wired only after the format is validated on the pilot batch.

## Producing one trajectory (the per-task sub-agent recipe)

One sub-agent owns one task, end to end:

1. Read this file, `methods/adam/results/*` (exemplar), and `.claude/skills/paper-to-reasoning/SKILL.md`.
2. Read the task's `tasks/<task>/{task_description.md,config.json,leaderboard.csv}` in `~/MLS-Bench`.
3. Rank the baselines weak→strong from the final leaderboard rows.
4. For each baseline: if `methods/<slug>/results/` exists, reuse; else create it via the skill
   (grounded + Codex-reviewed) and add it to `methods.json`.
5. Choose + create the finale method's standalone trace the same way.
6. Write `trajectories/<task>/` per the layout above.
7. `git add` + commit (this repo is shared across sessions — commit immediately).
