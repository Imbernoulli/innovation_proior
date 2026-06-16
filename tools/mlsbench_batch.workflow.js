export const meta = {
  name: 'mlsbench-batch',
  description: 'Create missing MLS-Bench baseline method traces, then build trajectories (existing baselines weak->strong, optional published-stronger finale)',
  phases: [
    { title: 'Methods', detail: 'one agent per missing baseline: paper-to-reasoning + Codex review' },
    { title: 'Trajectories', detail: 'one agent per task: compose baselines into a trajectory + optional finale' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const MLS = '/srv/home/bohanlyu/MLS-Bench'

const METHOD_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    slug: { type: 'string' },
    ok: { type: 'boolean' },
    title: { type: 'string' },
    arxiv: { type: 'string', description: 'arXiv id, or "" if none / pre-arXiv' },
    codex_outcome: { type: 'string', enum: ['completed', 'failed', 'not_run'] },
    notes: { type: 'string' },
  },
  required: ['slug', 'ok', 'title', 'arxiv', 'codex_outcome'],
}

const TRAJ_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    task: { type: 'string' },
    ok: { type: 'boolean' },
    title: { type: 'string', description: 'trajectory display title (baseline-iteration framing)' },
    steps: { type: 'array', items: { type: 'string' }, description: 'baseline slugs weak->strong' },
    endpoint: { type: 'string', description: 'display name of the final step (strongest baseline, or finale)' },
    finale_slug: { type: 'string', description: 'slug of the optional published-stronger finale, or "" if none' },
    finale_method: {
      type: 'object', additionalProperties: false,
      properties: { slug: { type: 'string' }, title: { type: 'string' }, arxiv: { type: 'string' }, codex_outcome: { type: 'string' } },
      description: 'present only if a finale was added',
    },
    notes: { type: 'string' },
  },
  required: ['task', 'ok', 'steps', 'endpoint', 'finale_slug'],
}

function methodPrompt(m) {
  return `Create ONE single-round atomic method trace at \`${REPO}/methods/${m.slug}/\` for the MLS-Bench baseline "${m.baseline}" (used in task \`${m.task}\`). This is the same kind of artifact as the existing \`methods/<slug>\` traces (e.g. \`methods/adam\`).

READ FIRST:
- \`${REPO}/.claude/skills/paper-to-reasoning/SKILL.md\` — follow it fully (this is the contract).
- \`${REPO}/methods/adam/results/{context,reasoning,answer}.md\` — the quality/voice bar.
- To IDENTIFY exactly which published method "${m.baseline}" is: read \`${MLS}/tasks/${m.task}/task_description.md\` and \`${MLS}/tasks/${m.task}/edits/${m.baseline}.edit.py\` (the authoritative definition). Then find its real published reference.

DO (per the skill):
1. Grounded retrieval — actually fetch + read the primary paper, its load-bearing ancestors, and a third-party explainer; capture into \`methods/${m.slug}/{src,refs,notes}/\`. NEVER write from memory; verify every equation/constant against a retrieved source.
2. Write the three deliverables to \`methods/${m.slug}/results/\`: \`context.md\` (5 sections, in-frame, no target-method name as a paper), \`reasoning.md\` (first-person present, continuous prose, NO section headers, derive don't gesture, full appendix-level depth, lands on real code/the field-appropriate final form), \`answer.md\` (distilled + faithful code). English.
3. **Run the write-enabled Codex review gate YOURSELF** (you have Bash): resolve \`ls ~/.claude/plugins/cache/*/codex/*/scripts/codex-companion.mjs\`, run \`node "<path>" task "Review AND FIX in place the paper-to-reasoning deliverables for ${m.slug} at ${REPO}/methods/${m.slug}/results/. Prioritize math/derivation correctness, then code faithfulness, then posterior/hindsight leaks, then scaffold purity. Output a file:line changelog." --write --model gpt-5.5 --effort xhigh\`. Re-verify its changelog yourself, then write \`methods/${m.slug}/results/.codex_review.json\` = {"method":"${m.slug}","codex_reviewed":true,"outcome":"completed","reviewed_at":"<UTC ISO8601>","reviewer":"gpt-5.5","effort":"xhigh","evidence":"this-run"}. If the runtime was unavailable, write outcome "not_run"/"failed" and say so (do not fake it).

HARD CONSTRAINTS:
- This is the canonical PAPER version of the method (not the task scaffold). Identify the method correctly from the edit.py, but the trace is the normal paper-to-reasoning derivation.
- Do NOT edit \`methods.json\`, \`trajectories.json\`, any other method, or run \`git\`. Only write under \`methods/${m.slug}/\`.

RETURN the schema: slug="${m.slug}", ok (did all 3 files + codex marker get written), title (the method's real published title), arxiv (id or ""), codex_outcome, notes (the reference you used + anything notable).`
}

function trajPrompt(t) {
  return `Build ONE MLS-Bench innovation trajectory for task \`${t.task}\` at \`${REPO}/trajectories/${t.task}/\`. Domain = "${t.domain}".

READ FIRST (obey exactly):
- \`${REPO}/trajectories/README.md\` — THE SPEC. Every rule is mandatory.
- \`${REPO}/trajectories/rl-intrinsic-exploration/\` — the reference conformant example (meta.json, 00-initial-context.md, 01-ppo-reasoning.md, 01-ppo-answer.md, 01-feedback.md, a later step).
- \`${MLS}/tasks/${t.task}/{task_description.md,config.json,leaderboard.csv}\` AND the edit surface \`${MLS}/tasks/${t.task}/edits/*.edit.py\` + the scaffold template/editable file in config.json.

STEPS:
1. Rank the task's baselines weak->strong from the \`is_final,true\` \`baseline:*\` leaderboard rows (state basis in meta.json; mind metric direction).
   **If \`leaderboard.csv\` has no/insufficient final baseline rows:** first look harder for real numbers — check the MLS-Bench public repo and this repo's git history/log for the task's results; if you find real measured numbers use them. If genuinely none exist, rank weak->strong by **published consensus** (which method is well-established as stronger — verify with a quick web search of the methods' papers/benchmarks; "which method is good/bad is generally agreed and a lookup settles it"). In that case each \`<i>-feedback.md\` must say plainly "No leaderboard result for this task yet; ordering by published consensus (see reasoning)" and give the consensus basis — NEVER fabricate numeric results. Still build the full ladder + optional finale.
2. Every baseline already has a single-round trace at \`methods/<slug>\` (created in the methods phase) — REUSE as the derivation reference; do not regenerate.
3. Author \`trajectories/${t.task}/\`:
   - \`00-initial-context.md\`: scaffold-based, lean (matched to the weakest baseline). Sections: Research question; Prior art before the first rung (the lineage the first baseline reacts to, concise, each with its gap); The fixed substrate; The editable interface (the contract + the DEFAULT scaffold code block); Evaluation settings (settings only, NO outcomes). **NO top-level H1 title.** Include the real scaffold code framework.
   - per baseline \`<i>-<slug>-reasoning.md\`: FULL-LENGTH derivation (>=1500 words code-excluded, matching the single-round methods/<slug> depth). For i>1 the reflection on the previous baseline's measured numbers is EMBEDDED THROUGHOUT (open diagnosing the prior result, derive in discovery order, close on falsifiable expectations vs the prior numbers). NO meta-narration opening. **NO code block in the reasoning.**
   - per baseline \`<i>-<slug>-answer.md\`: distilled (problem/key idea/why/hyperparameters) + the scaffold code block. **NO H1 title.**
   - per baseline \`<i>-feedback.md\`: NUMBERS ONLY (real per-seed+mean leaderboard tables, one factual lead line, no interpretation).
   - \`meta.json\`: like the reference — task,title,domain,metrics,metric_columns,initial_context_file,endpoint(strongest step display name),steps[{n,slug,method,reasoning,answer,feedback}],ranking_basis,notes.
4. **Same-named baseline != paper (CRITICAL).** The trajectory's CODE and REASONING must match the task's actual \`edits/<baseline>.edit.py\` implementation, NOT the paper's generic version. Diff each baseline's edit.py vs its methods/<slug> trace; note what the harness exposes/omits. Baseline answer code = the literal scaffold edit.
5. **Finale is OPTIONAL.** Only if a genuinely stronger, REAL PUBLISHED method (not in the task) exists and fits the edit surface: add it as a 4th step {finale:true, reasoning+answer, NO feedback}, create its methods/<finale-slug> trace (grounded + Codex), set endpoint to it. Carefully verify the finale's scaffold code line-by-line against the method's canonical reference impl, and run Codex on it. If nothing clearly stronger is published, STOP at the strongest baseline (no finale). Never invent one.

HARD CONSTRAINTS:
- First-person present; never call any method's paper a published artifact (prior-art ancestors by author/year are fine); no fabricated numbers; code lives in the ANSWER only (reasonings code-free); finale (if any) carries no feedback.
- Do NOT edit \`methods.json\`, \`trajectories.json\`, the reused \`methods/<slug>\` baseline traces, or run \`git\`. Write only under \`trajectories/${t.task}/\` and (if a finale) \`methods/<finale-slug>/\`.
- Validate before returning: every answer python block ast-parses; no \`\`\`python in any reasoning; all meta.json refs exist; ≥1500 words per reasoning.

RETURN the schema: task, ok, title, steps (baseline slugs weak->strong), endpoint, finale_slug ("" if none), finale_method (only if added), notes (the same-named-vs-paper diffs you found; any baseline whose methods/<slug> was missing).`
}

const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const METHODS = A.methods || []
const TASKS = A.tasks || []
const CHUNK = A.chunk || 5          // low concurrency to stay under the server throttle
const ATTEMPTS = A.attempts || 4    // in-run retry: transient throttles self-heal across passes
log(`Batch: ${METHODS.length} methods, ${TASKS.length} trajectories (chunk=${CHUNK}, attempts=${ATTEMPTS})`)
if (!METHODS.length && !TASKS.length) throw new Error('empty args — got: ' + JSON.stringify(args).slice(0, 200))

// Run items in small sequential chunks (caps concurrency), retrying nulls across passes.
// The wall-clock of intervening chunks gives a transient rate-limit time to clear.
async function runChunked(items, phaseName, mk) {
  const results = new Array(items.length).fill(null)
  let pending = items.map((_, i) => i)
  for (let a = 0; a < ATTEMPTS && pending.length; a++) {
    const todo = pending; pending = []
    for (let i = 0; i < todo.length; i += CHUNK) {
      const grp = todo.slice(i, i + CHUNK)
      const r = await parallel(grp.map((j) => () => mk(items[j])))
      grp.forEach((j, k) => { results[j] = r[k]; if (!r[k]) pending.push(j) })
    }
    log(`${phaseName} pass ${a + 1}: ${results.filter(Boolean).length}/${items.length} done, ${pending.length} retrying`)
  }
  return results
}

phase('Methods')
const methodResults = (await runChunked(METHODS, 'Methods', (m) =>
  agent(methodPrompt(m), { label: `method:${m.slug}`, phase: 'Methods', schema: METHOD_SCHEMA, agentType: 'general-purpose' })
)).filter(Boolean)
log(`Methods phase done: ${methodResults.filter((r) => r.ok).length}/${METHODS.length} ok`)

phase('Trajectories')
const trajResults = (await runChunked(TASKS, 'Trajectories', (t) =>
  agent(trajPrompt(t), { label: `traj:${t.task}`, phase: 'Trajectories', schema: TRAJ_SCHEMA, agentType: 'general-purpose' })
)).filter(Boolean)
log(`Trajectories phase done: ${trajResults.filter((r) => r.ok).length}/${TASKS.length} ok`)

return { methods: methodResults, trajectories: trajResults }
