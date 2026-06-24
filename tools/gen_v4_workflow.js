export const meta = {
  name: 'gen-v4-cp-data',
  description: 'Generate >=100 verified competition-deliverable SFT traces (one subagent per datapoint) with debug+self-verify spine',
  whenToUse: 'Build the V4 batch of new FrontierCS-style training data',
  phases: [
    { title: 'Generate', detail: 'one subagent per problem: invent + write single-file C++ + self-verify (compile+brute oracle) + long organized reasoning' },
    { title: 'Verify', detail: 'independent subagent re-compiles + re-runs the brute oracle on each written solution' },
  ],
}

// 22 algorithm tags x 5 pitfall-twists = 110 datapoints, one subagent each.
const TAGS = [
  'prefix-sum', 'two-pointer', 'binary-search-answer', 'greedy-exchange', 'dp-1d',
  'dp-knapsack', 'dp-interval', 'dp-bitmask', 'graph-bfs', 'graph-dfs',
  'dijkstra', 'dsu', 'number-theory', 'combinatorics-mod', 'strings-hash',
  'strings-kmp', 'monotonic-stack', 'sorting-sweep', 'segment-tree', 'fenwick',
  'math-adhoc', 'geometry-basic',
]
// Each twist forces divergence AND seeds a genuine pitfall for the debug episodes.
const TWISTS = [
  { key: 'overflow',  hint: 'values/answers large enough that a 32-bit int silently overflows (use long long); the debug episode should catch an int-overflow.' },
  { key: 'negzero',   hint: 'inputs include negatives and zeros and an all-negative / empty corner; the debug episode should catch a wrong base case or sign handling.' },
  { key: 'boundary',  hint: 'answer is sensitive to off-by-one / inclusive-exclusive boundaries; the debug episode should catch an off-by-one via a traced small case.' },
  { key: 'greedytrap',hint: 'an obvious greedy is tempting but wrong; the trace must construct a counterexample that disproves greedy before deriving the correct method.' },
  { key: 'count',     hint: 'a counting/constructive variant where the index/modulus/dedup is easy to get subtly wrong; the debug episode catches a double-count or off-by-one in counting.' },
]

const specs = []
for (const tag of TAGS) for (const t of TWISTS)
  specs.push({ slug: `cpv4-${tag}-${t.key}`, tag, twist: t.key, hint: t.hint })

log(`V4 generation: ${specs.length} datapoints, one subagent each (22 tags x 5 twists)`)

const GEN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['slug', 'title', 'ok', 'reasoning_chars', 'oracle_cases', 'oracle_mismatches', 'note'],
  properties: {
    slug: { type: 'string' },
    title: { type: 'string' },
    ok: { type: 'boolean', description: 'true only if code compiles AND oracle_mismatches==0 AND all 3 files written' },
    reasoning_chars: { type: 'integer' },
    oracle_cases: { type: 'integer' },
    oracle_mismatches: { type: 'integer' },
    note: { type: 'string' },
  },
}
const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['slug', 'compiles', 'oracle_cases', 'oracle_mismatches', 'verdict', 'note'],
  properties: {
    slug: { type: 'string' },
    compiles: { type: 'boolean' },
    oracle_cases: { type: 'integer' },
    oracle_mismatches: { type: 'integer' },
    verdict: { type: 'string', enum: ['pass', 'fail'] },
    note: { type: 'string' },
  },
}

const REPO = '/srv/home/bohanlyu/innovation_proior'

function genPrompt(s) {
  return `You are generating ONE high-quality competitive-programming training datapoint for a FrontierCS-style benchmark (exact judge, single-shot: the model is given a problem, thinks, and emits ONE C++ solution that reads stdin / writes stdout and is judged on hidden tests). Working dir: ${REPO}.

FIRST, read the gold exemplar to match its FORMAT, ORGANIZATION, and DEPTH (do not copy its problem):
  Read ${REPO}/data_v4/cp-noadj-commit/context.md , reasoning.md , train_answer.md

YOUR DATAPOINT:
  slug: ${s.slug}
  algorithm area: ${s.tag}
  required pitfall/twist: ${s.hint}

Invent a DISTINCT, self-consistent problem in the "${s.tag}" area that naturally contains that pitfall. Avoid the single most cliche textbook phrasing; make it concrete with a clear story, exact I/O contract, constraints (choose bounds so the pitfall is real), a time limit, and at least one worked sample. It must be solvable by a single-file C++17 program reading stdin and writing stdout, judged exactly.

HARD VERIFICATION REQUIREMENT (this is the whole point — do it with Bash, do not skip):
  1. Write ${REPO}/data_v4/${s.slug}/verify/sol.cpp      -- your intended single-file solution (reads stdin).
  2. Write ${REPO}/data_v4/${s.slug}/verify/brute.py     -- an INDEPENDENT brute force that solves the SAME stated problem by exhaustive/obvious means (must be obviously correct, different method from sol).
  3. Write ${REPO}/data_v4/${s.slug}/verify/gen.py       -- a random SMALL-case generator (respecting the constraints but tiny) parameterized by an int seed arg.
  4. Compile: g++ -O2 -std=c++17 -o /tmp/${s.slug}_sol ${REPO}/data_v4/${s.slug}/verify/sol.cpp
  5. Run >=300 random small cases comparing sol vs brute; iterate until ZERO mismatches. If you cannot make them agree, SIMPLIFY the problem (smaller scope) until sol is provably correct and agrees. NEVER emit an unverified solution. Also confirm sol prints the documented sample output.

THEN write THREE deliverable files (English):
  ${REPO}/data_v4/${s.slug}/context.md      -- structured, with these headers: "# <title>", "## Research question", "## Input / output contract", "## Background", "## Evaluation settings", "## Code framework" (a pre-method C++ scaffold with a // TODO where the algorithm goes). Mirror the exemplar's context.md.
  ${REPO}/data_v4/${s.slug}/reasoning.md    -- the HEART. First-person, present-tense working-it-out, but ORGANIZED (not a loose ramble): use short bold stage labels like **Reading the problem ...**, **Candidate approaches.**, **Deriving ...**, **First implementation and a trace.**, **The bug.**, **Fix and re-verification.**, **Edge cases.**, **Final solution.**, **Causal recap.**. It MUST contain: (a) at least TWO genuine debug/self-verify episodes where you TRACE the code on a concrete input, find a REAL bug, and fix it (e.g. the overflow / off-by-one / base-case / greedy counterexample matching your twist); (b) explicit edge-case checks (empty, min, max, overflow); (c) a sanity-check of the derivation itself on the sample. End with the final, verified code (the exact contents of sol.cpp) in a \`\`\`cpp block, then a one-paragraph causal recap. LENGTH: at least 12000 characters of substantive content (the existing data is criticised as too short/shallow — go deep, but every sentence must carry real reasoning, no padding).
  ${REPO}/data_v4/${s.slug}/train_answer.md -- a STRUCTURED editorial: bold labels "Problem.", "Key idea.", correctness, "Pitfalls.", "Edge cases.", "Complexity.", "Code." ending with the SAME \`\`\`cpp block. Mirror the exemplar's train_answer.md.

The code in reasoning.md and train_answer.md MUST be character-identical to the verified verify/sol.cpp.

Return the structured result. Set ok=true ONLY if: compiled cleanly, oracle_mismatches==0 over oracle_cases>=300, all three .md files written, and reasoning.md >= 12000 chars.`
}

function verifyPrompt(slug) {
  return `Independently verify the generated datapoint at ${REPO}/data_v4/${slug}/ (do NOT trust the generator). With Bash:
  1. Compile: g++ -O2 -std=c++17 -o /tmp/${slug}_v ${REPO}/data_v4/${slug}/verify/sol.cpp  (report compiles=false if it fails).
  2. If brute.py and gen.py exist, run >=300 fresh random small cases comparing /tmp/${slug}_v against brute.py; count mismatches.
  3. Confirm data_v4/${slug}/context.md, reasoning.md, train_answer.md all exist and the \`\`\`cpp block in train_answer.md matches verify/sol.cpp.
  4. Confirm reasoning.md is >= 12000 chars and contains at least two debug/trace episodes (grep for 'trace'/'bug'/edge mentions).
verdict='pass' only if it compiles, oracle_mismatches==0, all files present, and the length/episode bar is met; else 'fail' with a short reason in note.`
}

const results = await pipeline(
  specs,
  (s) => agent(genPrompt(s), { label: `gen:${s.slug}`, phase: 'Generate', schema: GEN_SCHEMA, agentType: 'general-purpose' }),
  (genRes, s) => genRes
    ? agent(verifyPrompt(s.slug), { label: `verify:${s.slug}`, phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'general-purpose' })
        .then((v) => ({ gen: genRes, verify: v }))
    : { gen: null, verify: null },
)

const clean = results.filter(Boolean)
const passed = clean.filter((r) => r.verify && r.verify.verdict === 'pass')
const failed = clean.filter((r) => !r.verify || r.verify.verdict !== 'pass')
log(`V4 generation done: ${passed.length}/${specs.length} verified pass; ${failed.length} failed/needs-fix`)
return {
  total: specs.length,
  passed: passed.map((r) => r.verify.slug),
  failed: failed.map((r) => (r.verify && r.verify.slug) || (r.gen && r.gen.slug) || 'unknown'),
  failed_notes: failed.map((r) => ({ slug: (r.verify && r.verify.slug) || (r.gen && r.gen.slug), note: (r.verify && r.verify.note) || (r.gen && r.gen.note) })),
}
