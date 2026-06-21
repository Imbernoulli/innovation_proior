export const meta = {
  name: 'verify-ablation',
  description: 'Verify removal candidates: read the actual methods/<slug> trace + edit.py, decide keep (published method) vs remove (ablation/variant)',
  phases: [{ title: 'Verify', detail: 'one agent per candidate: read the trace, grounded keep/remove verdict' }],
}
const REPO = '/srv/home/bohanlyu/innovation_proior'
const MLS = '/srv/home/bohanlyu/MLS-Bench/tasks'
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const ITEMS = A.items || []
const CHUNK = A.chunk || 8
const ATTEMPTS = A.attempts || 3
const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    slug: { type: 'string' },
    keep: { type: 'boolean', description: 'true if the trace IS a genuine published method (keep it); false if it is an ablation/variant/generic control (remove it)' },
    paper_title: { type: 'string' },
    arxiv: { type: 'string' },
    reason: { type: 'string' },
  },
  required: ['slug', 'keep', 'reason'],
}
function prompt(it) {
  const locs = it.tasks.map((t) => `${MLS}/${t.task}/edits/${t.baseline}.edit.py`).join(', ')
  return `Decide whether the existing method trace \`${REPO}/methods/${it.slug}/results/\` should be KEPT or REMOVED, under one rule: \`methods/\` holds ONLY genuine published methods (a specific published paper introduced exactly this method). Ablations / config variants / generic textbook controls must be removed.

READ:
- \`${REPO}/methods/${it.slug}/results/context.md\` and \`${REPO}/methods/${it.slug}/results/answer.md\` — what method does this trace actually derive?
- the task edit surface(s) that use this slug as a baseline: ${locs}

DECIDE:
- keep=true ONLY if the trace is a faithful derivation of a REAL PUBLISHED METHOD that you can name (title + arXiv id / venue) — e.g. the trace for \`adam\` is Kingma & Ba's Adam (KEEP); \`seal\` is Zhang & Chen's SEAL link-prediction (KEEP); a real algorithm with a paper.
- keep=false if the trace is for an ABLATION / CONFIG VARIANT / generic control with no dedicated paper — e.g. "no weight decay", "vanilla", "identity", "default", "naive", "multi epoch", a loss/granularity/normalization toggle (k1/k2/k3, outcome-only/group-std/batch-std, first-k-tokens, mse/smooth-l1 loss choice), a plain CNN/MLP/ridge/BC/SFT control, an ad-hoc non-paper-faithful baseline. If the trace's own answer/context admits it is "inspired by"/"a family"/"not paper-faithful"/a generic baseline, that is REMOVE.
- Judge the TRACE + the method's real status, NOT merely how one task uses it. A famous published method used as a control in some task is still KEEP. When unsure whether a dedicated paper exists, do ONE quick web check; if you cannot name the paper that introduced exactly this method, keep=false.

Do NOT edit or delete any files, do NOT run git. Read-only. Return the schema: slug="${it.slug}", keep, paper_title, arxiv, reason (one line: the paper if keep, else why it is an ablation).`
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
log(`Verifying ${ITEMS.length} removal candidates (chunk=${CHUNK})`)
const res = (await runChunked(ITEMS, (it) =>
  agent(prompt(it), { label: `verify:${it.slug}`, phase: 'Verify', schema: SCHEMA, agentType: 'general-purpose' })
)).filter(Boolean)
log(`Verified ${res.length}/${ITEMS.length}: keep=${res.filter((r) => r.keep).length} remove=${res.filter((r) => !r.keep).length}`)
return { verified: res }
