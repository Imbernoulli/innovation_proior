export const meta = {
  name: 'frontiersmith-generate',
  description: 'Generate deterministically-scored open-ended problems across formats A/B/C/D/E; each agent authors one problem and self-validates with the matching harness',
  phases: [
    { title: 'Generate', detail: 'one agent per spec: author files + self-validate to PASS' },
    { title: 'Repair', detail: 'second pass only for problems that failed the harness' },
  ],
}

const SYNTH = '/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/synth'
const SEEDS = `${SYNTH}/seeds/seed_list.jsonl`
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
A = A || {}
// specs is a COMPACT routing list: [{id, format}, ...] (full spec read from file by each agent)
const specs = Array.isArray(A) ? A : (A.specs || A.routes || [])
const MODEL = A.model
const EFFORT = A.effort || 'high'
if (!specs.length) { log(`no specs. typeof args=${typeof args}`); return { error: 'no specs' } }
log(`generating ${specs.length} problems (model=${MODEL || 'inherit'}, effort=${EFFORT})`)

const RESULT_SCHEMA = {
  type: 'object', additionalProperties: true,
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    title: { type: 'string' }, format: { type: 'string' }, family: { type: 'string' },
    metrics: { type: 'object', additionalProperties: true },
    rounds: { type: 'integer' }, notes: { type: 'string' },
  },
  required: ['id', 'verdict', 'notes'],
}

// route each format to (brief, validator)
function route(fmt) {
  if (fmt === 'A') return { brief: 'AGENT_BRIEF.md', val: 'harness/validate_problem.py' }
  if (fmt === 'B') return { brief: 'AGENT_BRIEF_PY_PROGRAM.md', val: 'harness/validate_pyproblem.py' }
  return { brief: 'AGENT_BRIEF_PY_STDOUT.md', val: 'harness/validate_problem.py' } // C/D/E
}

function genPrompt(s) {
  const r = route(s.format)
  const dir = `${SYNTH}/problems/${s.id}`
  return `You are a problem setter. Author ONE novel, DETERMINISTICALLY-scored open-ended problem
(format ${s.format}), then self-validate it with the harness until it PASSES.

STEP 0 — read YOUR full spec (family, theme, scale, variant, anchor idea, source frameworks):
  python3 -c "import json;print([l for l in open('${SEEDS}') if json.loads(l)['id']=='${s.id}'][0])"
STEP 1 — read the full authoring contract (do not skip): ${SYNTH}/${r.brief}

Output directory (create it): ${dir}
Harness (ground truth): python3 ${SYNTH}/${r.val} ${dir}${s.format === 'B' ? '' : ' --keep-testdata'}

Rules:
- Deterministic scoring ONLY (seed all randomness; never wall-time/GPU). Make it genuinely open-ended:
  graded objective, no easy optimum, multiple viable strategies.
- Instantiate the spec faithfully (family/scale/variant/theme/objective/anchor from STEP 0). Do NOT clone
  an existing benchmark problem verbatim; make a fresh instance/variant. If the literal anchor can't
  clear the discrimination gate, stay in the SAME family but pick a variant that can.
- Write every required file INCLUDING the 4-tier solution ladder (trivial/greedy/strong/invalid) so the
  harness sees trivial≈0.1, strong>trivial, invalid=0, and divergent per-test scores.
- Run the harness. If verdict != PASS, read gates/errors + validation.json, fix the failing gate, re-run.
  Iterate up to 6 rounds until "verdict": "PASS" (or genuinely exhausted).
- Report the harness's FINAL verdict truthfully.`
}

function repairPrompt(s) {
  const r = route(s.format)
  const dir = `${SYNTH}/problems/${s.id}`
  return `Problem ${s.id} (format ${s.format}) did NOT pass the harness. Read its spec:
  python3 -c "import json;print([l for l in open('${SEEDS}') if json.loads(l)['id']=='${s.id}'][0])"
Read ${SYNTH}/${r.brief}, inspect ${dir}/ and its validation.json, diagnose the failing gate(s) and FIX.
Re-run: python3 ${SYNTH}/${r.val} ${dir}${s.format === 'B' ? '' : ' --keep-testdata'}
Iterate up to 6 rounds until "verdict":"PASS". Return the result JSON with the final (truthful) verdict.`
}

const results = await pipeline(
  specs,
  (s) => agent(genPrompt(s), { label: `gen:${s.id}`, phase: 'Generate', schema: RESULT_SCHEMA, model: MODEL, effort: EFFORT }),
  (res, s) => {
    if (res && res.verdict === 'PASS') return res
    return agent(repairPrompt(s), { label: `fix:${s.id}`, phase: 'Repair', schema: RESULT_SCHEMA, model: MODEL, effort: EFFORT })
  },
)

const clean = results.filter(Boolean)
const passed = clean.filter(r => r.verdict === 'PASS')
log(`done: ${passed.length}/${specs.length} PASS (self-reported; re-verify with the harness)`)
return { total: specs.length, passed: passed.length, results: clean }
