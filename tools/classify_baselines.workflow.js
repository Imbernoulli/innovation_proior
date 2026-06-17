export const meta = {
  name: 'classify-baselines',
  description: 'Grounded published-paper vs ablation/variant classification of MLS-Bench task baselines',
  phases: [{ title: 'Classify', detail: 'one agent per task: read each edit.py + quick web check, label each baseline' }],
}
const MLS = '/srv/home/bohanlyu/MLS-Bench/tasks'
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const ITEMS = A.items || []
const CHUNK = A.chunk || 8
const ATTEMPTS = A.attempts || 3
const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    task: { type: 'string' },
    baselines: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          slug: { type: 'string', description: 'kebab-case slug (underscores->hyphens)' },
          published: { type: 'boolean', description: 'true ONLY if a specific published paper introduced exactly this method' },
          title: { type: 'string', description: 'the published paper title if published, else ""' },
          arxiv: { type: 'string', description: 'arXiv id or venue if published, else ""' },
          reason: { type: 'string', description: 'one-line justification (what the edit.py does; why paper or why ablation/variant)' },
        },
        required: ['slug', 'published', 'reason'],
      },
    },
    published_count: { type: 'integer' },
    ablation_heavy: { type: 'boolean', description: 'true if fewer than 2 baselines are genuine published methods (ladder is mostly ablations/variants)' },
  },
  required: ['task', 'baselines', 'published_count', 'ablation_heavy'],
}
function prompt(it) {
  return `Classify the baselines of MLS-Bench task \`${it.task}\` as PUBLISHED METHOD vs ABLATION/VARIANT. This decides whether each gets a standalone paper-to-reasoning method trace (published only) and whether the task deserves an innovation trajectory (skip if ablation-heavy).

READ (authoritative):
- \`${MLS}/${it.task}/task_description.md\`
- the edit surface for EACH baseline: \`${MLS}/${it.task}/edits/<baseline>.edit.py\`
Baselines to classify (config keys): ${JSON.stringify(it.baselines)}

RULE — a baseline is PUBLISHED only if a SPECIFIC published paper introduced EXACTLY this method (name the paper title + arXiv id / venue). It is ABLATION/VARIANT if it is:
- a config/knob toggle (e.g. "no weight decay", "identity", "l2norm", "vanilla adam", "frozen bias", "nope/no positional encoding", "multi epoch", "default"),
- a generic textbook control with no dedicated paper (plain SGD/Adam/BC/ERM/cross-entropy used as a baseline, random forest/GBDT as a generic baseline UNLESS the task is specifically about that algorithm),
- a loss-coefficient / granularity / normalization variant of one underlying method (e.g. token-level vs sequence-level IS; k1/k2/k3 KL estimators; outcome-only vs group-std vs batch-std reward normalization; weighted-nll; huber-pinball),
- an architectural sub-component toggle that is not itself a paper.
When unsure whether a dedicated paper exists, do ONE quick web search; if you cannot point to a specific paper that introduced exactly this method, mark published=false. Be strict: the bar is "a paper introduced this method", not "this resembles ideas from some paper".

For each baseline return {slug (underscores->hyphens), published, title, arxiv, reason}. Set published_count and ablation_heavy (true if <2 baselines are genuine published methods).

Do NOT write any files, do NOT edit methods.json/trajectories.json, do NOT run git. Read-only classification. Return the schema.`
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
    log(`Classify pass ${a + 1}: ${out.filter(Boolean).length}/${items.length} done, ${pend.length} retrying`)
  }
  return out
}
phase('Classify')
log(`Classifying ${ITEMS.length} tasks' baselines (chunk=${CHUNK})`)
const res = (await runChunked(ITEMS, (it) =>
  agent(prompt(it), { label: `classify:${it.task}`, phase: 'Classify', schema: SCHEMA, agentType: 'general-purpose' })
)).filter(Boolean)
log(`Classified ${res.length}/${ITEMS.length} tasks`)
return { classified: res }
