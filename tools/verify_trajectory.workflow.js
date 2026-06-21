export const meta = {
  name: 'verify-trajectory',
  description: 'Authoritative per-trajectory ablation-heavy check: read the actual ladder (meta.json + each rung) and decide keep vs cut',
  phases: [{ title: 'Verify', detail: 'one agent per trajectory: read the ladder, grounded keep/cut verdict' }],
}
const REPO = '/srv/home/bohanlyu/innovation_proior'
const MLS = '/srv/home/bohanlyu/MLS-Bench/tasks'
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const TASKS = A.tasks || []
const CHUNK = A.chunk || 6
const ATTEMPTS = A.attempts || 4
const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    task: { type: 'string' },
    keep: { type: 'boolean', description: 'true if the ladder is a genuine weak->strong progression of >=2 DISTINCT published methods; false if it is an ablation grid / config-or-loss-or-encoder sweep of essentially one method' },
    distinct_published: { type: 'integer', description: 'count of rungs that are GENUINELY DISTINCT published methods (not config/loss/encoder variants of one underlying method, not generic controls)' },
    published_rungs: { type: 'array', items: { type: 'string' }, description: 'the rung slugs that are genuine distinct published methods, each nameable to a paper' },
    reason: { type: 'string', description: 'one-line grounded justification' },
  },
  required: ['task', 'keep', 'distinct_published', 'reason'],
}
function prompt(t) {
  return `Decide whether the innovation trajectory \`${REPO}/trajectories/${t}/\` should be KEPT or CUT, under one rule: a trajectory is valid ONLY if its ladder is a genuine weak->strong progression of at least TWO DISTINCT PUBLISHED METHODS (each rung nameable to a specific paper that introduced that method). CUT it if the ladder is really an ABLATION GRID — config/loss/encoder/normalization/optimizer-knob variants of essentially ONE underlying method, or generic textbook controls — even if each rung borrows a paper citation.

READ (authoritative):
- \`${REPO}/trajectories/${t}/meta.json\` (the steps array: each rung's slug + method name + endpoint)
- the actual rung answer files \`${REPO}/trajectories/${t}/0*-answer.md\` (what each rung really IS)
- \`${MLS}/${t}/task_description.md\` and the baseline edit surfaces \`${MLS}/${t}/edits/*.edit.py\` to see what each baseline actually changes

DECIDE (be strict, judge the LADDER not the slug names):
- keep=true ONLY if >=2 rungs are GENUINELY DISTINCT published methods. Example KEEP: a ladder GCN-dot -> VGAE -> SEAL -> BUDDY for link prediction (four distinct published link predictors). Example KEEP: LightGBM -> LSTM -> Transformer (three distinct published model families).
- keep=false if the rungs are variants of one method: e.g. recurrent-encoder/mlp-encoder/attention-encoder (three ENCODER choices for the same meta-RL context encoder), sgd/adam/adam2 (optimizer-knob swaps, adam2 a duplicate), cosine/mse/smooth_l1 (loss toggles), none/scale-opa (a regularizer on/off), full-attn/no-attn/mid (an attention-placement sweep), confidence-greedy/dfs-ranked (decoding-search variants of one base). A borrowed paper citation on an ablation rung does NOT make it a distinct method.
- A duplicate slug of a method already on the ladder (e.g. adam AND adam2) counts ONCE.
- When unsure whether two rungs are genuinely distinct published methods vs variants of one, do ONE quick web check.

Do NOT edit or delete any files, do NOT run git. Read-only. Return the schema: task="${t}", keep, distinct_published, published_rungs, reason.`
}
async function runChunked(items, mk) {
  const out = new Array(items.length).fill(null)
  let pend = items.map((_, i) => i)
  for (let a = 0; a < ATTEMPTS && pend.length; a++) {
    const todo = pend; pend = []
    for (let i = 0; i < todo.length; i += CHUNK) {
      const g = todo.slice(i, i + CHUNK)
      const r = await parallel(g.map((j) => () => mk(items[j])))
      g.forEach((j, k) => { out[j] = r[k]; if (!r[k]) pend.push(j) })
    }
    log(`Verify pass ${a + 1}: ${out.filter(Boolean).length}/${items.length} done, ${pend.length} retrying`)
  }
  return out
}
phase('Verify')
log(`Verifying ${TASKS.length} trajectories (chunk=${CHUNK})`)
const res = (await runChunked(TASKS, (t) =>
  agent(prompt(t), { label: `vtraj:${t}`, phase: 'Verify', schema: SCHEMA, agentType: 'general-purpose' })
)).filter(Boolean)
log(`Verified ${res.length}/${TASKS.length}: keep=${res.filter((r) => r.keep).length} cut=${res.filter((r) => !r.keep).length}`)
return { verified: res }
