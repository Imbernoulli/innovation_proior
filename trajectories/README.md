# Innovation Trajectories

Where `methods/` holds **atomic** reasoning traces (how one method was derived), `trajectories/`
holds **iterative** ones: how a researcher, working a single MLS-Bench research question, climbs the
baseline ladder from weak to strong and finally lands one improvement past the strongest baseline.
Each trajectory is one MLS-Bench task.

A trajectory is the **iterative climb itself**, in weak→strong order, grounded in the task's real edit
surface: it opens on the task scaffold, derives each baseline's method as a fill of that scaffold,
drops in the **measured numbers** after each, and embeds in each next baseline's reasoning a
**reflection** diagnosing why the previous one landed where it did. The `methods/<slug>` single-round
traces are the reference for each derivation (and stay untouched for Methods mode); the trajectory
re-expresses each in this task's scaffold and threads the reflections through.

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

Everything the trajectory *shows* is authored here and grounded in the **task's real edit surface**
(the MLS-Bench scaffold); the `methods/<slug>` traces are the reference for the *derivation*, not the
code. Files:

```
trajectories/<task>/
  meta.json                       # the playlist: initial-context file + ordered steps
  00-initial-context.md           # the task scaffold, authored here (see below)
  <i>-<slug>-reasoning.md         # every step's reasoning (multi-round for i>1)
  <i>-feedback.md                 # NUMBERS ONLY — the measured leaderboard result of step i
```

`meta.json` schema:

```json
{
  "task": "...", "title": "...", "domain": "...",
  "metrics": ["..."], "metric_columns": ["eval_return", "auc", "..."],
  "initial_context_file": "00-initial-context.md",
  "steps": [
    { "n": 1, "slug": "<weakest>", "method": "<display name>", "reasoning": "01-<weakest>-reasoning.md", "feedback": "01-feedback.md" },
    { "n": 2, "slug": "<next>", "method": "...", "reasoning": "02-<next>-reasoning.md", "feedback": "02-feedback.md" },
    { "n": 3, "slug": "<finale>", "method": "...", "reasoning": "03-<finale>-reasoning.md", "finale": true }
  ]
}
```

The website renders, in order: the **initial context** (`initial_context_file`), then for each step
its **reasoning** → **answer** → **feedback** (numbers; baselines only). (`initial_context: "<slug>"` is still supported as a legacy
fallback that reuses `methods/<slug>/results/context.md`, but a MLS-Bench trajectory uses
`initial_context_file`.)

**Scaffold-grounded code (MLS-Bench trajectories).** Because the trajectory *is* the MLS-Bench task,
the **initial context and every step's code are the task's real edit surface** — the exact region the
agent edits (e.g. `custom_*.py`: the editable class + functions), in the scaffold's vocabulary, with
the loop's provided helpers. Not the methods' generic paper implementations. Read the task's
`edits/*.edit.py` (the per-baseline scaffold fills) and `edits/<scaffold>.py` (the template); the
initial context shows the **default** scaffold fill, and each step's reasoning lands the **literal edit
you'd make** for that baseline. The single-round `methods/<slug>` code (often 42×42 / a paper harness)
is *not* what the step shows — only the derivation is borrowed; the code is re-expressed in the
scaffold.

**A same-named baseline can be a very different method here — derive against THIS task, not the paper.**
A baseline named `rnd`/`icm`/`gat`/`lion`/… in a task can differ *substantially* from the same-named
`methods/<slug>` trace, because the task adapts it to its own research question and harness. The
authority is always the task's `edits/<baseline>.edit.py`, read line by line. Both the **code and the
reasoning** must match *that* implementation, not the paper's. Concrete example from this pilot: the
paper RND's signature move is a *non-episodic* intrinsic return; but this task's loop masks the
intrinsic stream with the same done-mask as the extrinsic one (`int_nextnonterminal = ext_nextnonterminal`,
bonus × `(1−next_done)`), so the intrinsic stream is **episodic** here and differs only in discount —
the reasoning must *not* import the non-episodic story. Likewise this task's ICM bonus is `0.5·mean`
MSE with loss `L_I + 0.2·L_F` on 84×84 frames in a two-value-head PPO — not the paper's η/2·sum, A3C,
42×42, or LSTM. Before writing a step: diff the task baseline against the `methods/` trace, list what
the task *doesn't* expose (e.g. NGU's UVFA/recurrent heads), and reconstruct the reasoning so it lands
exactly the task's implementation — explicitly noting any paper machinery the harness omits.

**Length / completeness bar (HARD — the most common failure).** Each step's reasoning is a **complete
derivation**, matching the depth of that baseline's single-round `methods/<slug>/results/reasoning.md`:
it walks *every* load-bearing derivation step and justifies *every* non-obvious design choice, with the
reflection on the prior result woven through. A short stub is a failure even if it's "correct."
Concrete floor: **no step reasoning is shorter than ~1500 words** (code excluded), and most land
**1800–2500+**; the reference rl-intrinsic-exploration steps run 1485–2231 words — match that. If a
single-round trace covers material the task harness removes, drop only that; otherwise the trajectory
step is *as long and complete* as the single-round derivation (plus the embedded reflection). When in
doubt, longer and more complete. A 600-word step is wrong — expand it.

**Reasoning vs answer (code lives in the answer).** Each step has a **reasoning** (the full derivation,
prose) and an **answer** (the distilled "成品式" summary — problem / key idea / why / hyperparameters —
plus the scaffold code). The scaffold code block lives in the **answer only**, not the reasoning, so the
stacked trajectory view shows it once: the reasoning derives the method and hands off to the answer
("the full scaffold module is in the answer"), the answer lands it. Authored locally
(`<i>-<slug>-answer.md`); the `methods/<slug>` answers stay as-is for Methods mode (they show the
non-scaffold paper code and are not reused here).

**Finale has no feedback.** The last step is the endpoint; it carries no `feedback` file. (Its bar /
"what I'd validate" lives at the close of its reasoning, against the strongest baseline's real numbers.)

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
  (Interpretation belongs in the *next* step's reasoning, not the feedback.) The **finale carries no
  feedback file** (see above); the bar it must clear lives at the close of its own reasoning.
- **`01-<weakest>-reasoning.md`** (step 1): there is no prior result to reflect on, so it just
  establishes the baseline within the scaffold (why start here, the default fill, what to watch). Short.

## Context scaling — weaker baseline sees less

There is exactly **one** authored Initial Context — the task scaffold (`00-initial-context.md`). From
there context accumulates: the effective ground entering step *i* is
`initial context + every earlier reasoning + feedback`. So later (stronger) steps carry
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

## Finale (OPTIONAL)

The finale is **optional — it may not exist, and you do not have to add one.** The default trajectory is
just the task's existing baselines weak→strong, ending at the strongest baseline (with its feedback).
*Only if* a genuinely stronger, real, published method that the task did not originally include exists
and fits the edit surface, you may add it as the endpoint. **Never invent one to fill the slot** — if
nothing clearly stronger is published, stop at the strongest baseline (no finale, no `endpoint` field).

When you do add a finale, three hard requirements, non-negotiable:

1. **Find the reference.** The method must be a genuinely published technique with a real citation
   (paper + arXiv/venue). No speculative or "designed-here" methods. It gets its own full standalone
   `methods/` trace grounded in the retrieved primary source.
2. **Carefully check the code implementation.** The finale's scaffold answer code (the fill of the
   task's edit surface) must be a *correct, faithful* implementation, verified line-by-line against the
   method's **canonical reference implementation** (official repo / the paper's equations). Re-express
   it in the task scaffold without breaking the algorithm; confirm it runs and matches the reference's
   math.
3. **Codex review.** Run the write-enabled Codex review gate on the finale (and re-verify its
   changelog), with explicit focus on code-vs-reference correctness; write `.codex_review.json`.

It is motivated in its reasoning's opening as the natural next move from the strongest baseline's
failure mode. It carries **no feedback file** — it is the endpoint; the bar it must clear (the
strongest baseline's real numbers) and what one would validate live at the close of its own reasoning,
with no invented numbers.

## Website

`trajectories.json` (root) indexes the set (`[{task, title, domain, finale}]`). The site has a
**Methods | Trajectories** mode switch; trajectory mode reads each task's `meta.json` and composes the
reading view from the referenced `methods/` files + the local reasoning/feedback files. Add one
`trajectories.json` entry per completed task.

## Producing one trajectory (the per-task sub-agent recipe)

One sub-agent owns one task, end to end:

1. Read this file, `methods/adam/results/*` (exemplar of the atomic-trace bar), and
   `.claude/skills/paper-to-reasoning/SKILL.md`.
2. Read `~/MLS-Bench/tasks/<task>/{task_description.md,config.json,leaderboard.csv}` **and the task's
   edit surface** — `edits/*.edit.py` (per-baseline scaffold fills), the scaffold template, and the
   editable file/line range in `config.json`. This is the code every step must show.
3. Rank the baselines weak→strong from the `is_final,true` `baseline:*` rows.
4. For each baseline: reuse `methods/<slug>/` as the *derivation reference* if it exists; else create it
   via the skill (grounded + Codex-reviewed) and add it to `methods.json`. Either way, **diff the task's
   `edits/<baseline>.edit.py` against the `methods/` trace first** — a same-named baseline can be a very
   different method here. The trajectory's code *and* reasoning must match the task's implementation, not
   the paper's; note any paper machinery the harness omits.
5. Choose + create the finale method's standalone `methods/` trace the same way.
6. Author `trajectories/<task>/`: `00-initial-context.md` (the task scaffold, default fill), every
   step's `<i>-<slug>-reasoning.md` (multi-round for i>1, code = the literal scaffold edit), the
   numbers-only `<i>-feedback.md` for the baselines (not the finale), and `meta.json`. Add the task to
   `trajectories.json`.
7. `git add` + commit immediately (this repo is shared across sessions).
