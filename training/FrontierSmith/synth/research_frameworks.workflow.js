export const meta = {
  name: 'research-frameworks',
  description: 'Research each evolutionary-search / discovery framework (web + local clone) to map its task types and evaluation forms, then synthesize a deterministic-scoring problem taxonomy',
  phases: [
    { title: 'Research', detail: 'one agent per framework: what it is, its tasks, and each task\'s evaluation form' },
    { title: 'Synthesize', detail: 'merge into an expanded taxonomy of DETERMINISTICALLY-scorable problem archetypes' },
  ],
}

let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
const works = Array.isArray(A) ? A : (A && A.works) || []
const SYNTH = '/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/synth'
if (!works.length) { log('no works provided'); return { error: 'no works' } }
log(`researching ${works.length} frameworks`)

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    name: { type: 'string' },
    what_it_is: { type: 'string' },
    method: { type: 'string' },
    task_categories: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          category: { type: 'string' },
          description: { type: 'string' },
          eval_form: { type: 'string', description: 'one of: correctness | quality-metric | flops-or-opcount | wall-time | gpu-latency | llm-judge | other' },
          reproducible_offline: { type: 'boolean', description: 'deterministic & runnable here with no GPU/wall-time sandbox' },
          suitable_for_us: { type: 'boolean' },
          examples: { type: 'array', items: { type: 'string' } },
        },
        required: ['category', 'eval_form', 'reproducible_offline', 'suitable_for_us'],
      },
    },
    adopt_archetypes: { type: 'array', items: { type: 'string' },
      description: 'deterministic-scoring problem archetypes worth generating that would help a model GENERALIZE' },
    avoid: { type: 'array', items: { type: 'string' } },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['name', 'what_it_is', 'method', 'task_categories', 'adopt_archetypes'],
}

function prompt(w) {
  const local = w.local ? `\nA local clone exists at: ${w.local}\nInspect it FIRST (READMEs, task/benchmark/example dirs, evaluators) — it is authoritative for what tasks actually run. Then use the web for the paper-level description.` : '\nResearch via the web (search + fetch the paper/repo/blog).'
  return `Research the work "${w.name}"${w.hint ? ` (${w.hint})` : ''}.${local}

Determine, concretely:
1. What it is and its method (one paragraph each).
2. The CONCRETE task categories / benchmark domains it uses — list them with real example task names.
3. For EACH task category, its evaluation form: correctness | quality-metric | flops-or-opcount | wall-time | gpu-latency | llm-judge | other. Mark reproducible_offline=true ONLY if the score is deterministic and computable here WITHOUT a GPU/wall-time sandbox (FLOPs/op-count counts as deterministic; wall-time/GPU-latency does NOT). Mark suitable_for_us accordingly.
4. Which deterministic-scoring problem ARCHETYPES from this work are worth batch-generating so a trained model GENERALIZES (not overfits) — and which to AVOID (e.g. kernel wall-time).

Be specific and grounded (cite files you read or URLs). Return the structured object.`
}

const findings = await parallel(works.map(w => () =>
  agent(prompt(w), { label: `research:${w.name}`, phase: 'Research', schema: RESEARCH_SCHEMA, effort: 'high' })
))
const clean = findings.filter(Boolean)
log(`research done: ${clean.length}/${works.length}`)

const synth = await agent(
  `You are designing a DIVERSE, DETERMINISTICALLY-SCORABLE synthetic problem taxonomy to train a model
that GENERALIZES across the whole "LLM writes code to optimize a scored objective" space — spanning
competitive optimization (FrontierCS/ALE), evolutionary program search (FunSearch/AlphaEvolve/OpenEvolve/
ThetaEvolve/TTT-Discover), engineering optimization (Frontier-Eng), ML engineering (MLS-Bench), and
scientific discovery (Evaluation-driven Scaling).

Here are structured findings on each framework:
${JSON.stringify(clean, null, 2)}

Also read our existing design + Format-A contract:
- ${SYNTH}/DESIGN.md
- ${SYNTH}/AGENT_BRIEF.md
- ${SYNTH}/seeds/build_seed_list.py

Produce a JSON taxonomy proposal with:
- "formats": the problem FORMATS we should support (e.g. A=testlib instance-based combinatorial opt;
  B=python evaluator "evolve a program maximizing a deterministic score"; and any others). For each:
  when to use it, the file contract sketch, and how it is scored deterministically.
- "tiers": importance-ranked tiers (档) with families spanning ALL the paradigms above. For EACH family:
  {tier, family, format (A/B/...), eval_form (correctness|quality-metric|flops), source_frameworks:[...],
   why_it_generalizes, one example problem idea}. Cover graph/combinatorial, geometry/packing, sequences,
   numbers, math-discovery (cap-set/bin-packing-heuristic/matrix-mult style), symbolic-regression/
   scientific-law discovery, algorithm-design/heuristic-evolution, engineering-opt, ML-pipeline-opt (only
   the deterministic-metric parts), and FLOPs-form "kernel-like" op-count problems.
- "exclude": archetypes to explicitly avoid (wall-time/GPU-latency/sandbox-dependent) and why.
- "counts": a suggested per-tier problem count that sums to ~200 (and note how to scale each tier to 200).
Return ONLY the JSON. Write it also to ${SYNTH}/reports/taxonomy_proposal.json using your tools.`,
  { label: 'synthesize-taxonomy', phase: 'Synthesize', effort: 'high' }
)

return { researched: clean.length, findings: clean, taxonomy: synth }
