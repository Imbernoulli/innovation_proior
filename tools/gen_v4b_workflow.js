export const meta = {
  name: 'gen-v4b-cp-data',
  description: 'Generate ~200 more verified competition traces, twists grounded in the REAL SFT failure modes (gen-only; verified by tools/verify_v4.py)',
  whenToUse: 'Second wave of V4 data, targeting observed model failures (fake algebra, tiny-n-only verification, etc.)',
  phases: [{ title: 'Generate', detail: 'one subagent per datapoint: invent + single-file C++ + self-verify (compile+brute oracle) + long organized reasoning' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const TAGS = [
  'prefix-sum', 'two-pointer', 'binary-search-answer', 'greedy-exchange', 'dp-1d',
  'dp-knapsack', 'dp-interval', 'dp-bitmask', 'graph-bfs', 'graph-dfs',
  'dijkstra', 'dsu', 'number-theory', 'combinatorics-mod', 'strings-hash',
  'strings-kmp', 'monotonic-stack', 'sorting-sweep', 'segment-tree', 'fenwick',
  'math-adhoc', 'geometry-basic',
]
// 9 twists. The first 5 repeat (a NEW distinct problem per cell -- variant "b"); the last 4 are
// grounded in the real SFT failures read from experiments/data_feedback (xor_sidon etc.):
//   - the model invented terminology and asserted FALSE algebra ("(2m-2) XOR (2m) = 2(m-2)^2+..."),
//   - it "verified" a construction only on n<=10 then shipped it for n<=1e7 and scored 0.
const TWISTS = [
  { key: 'overflow',   hint: 'a 32-bit int silently overflows; the debug episode catches the overflow by tracing a large case.' },
  { key: 'negzero',    hint: 'negatives/zeros and an all-negative/empty corner; the debug episode catches a wrong base case or sign handling.' },
  { key: 'boundary',   hint: 'off-by-one / inclusive-exclusive boundary decides correctness; caught by tracing a small case.' },
  { key: 'greedytrap', hint: 'an obvious greedy is wrong; the trace constructs an explicit counterexample disproving greedy before deriving the correct method.' },
  { key: 'count',      hint: 'a counting/constructive variant where index/modulus/dedup is easy to get subtly wrong; the debug episode catches a double-count.' },
  { key: 'fakeproof',  hint: 'the solution rests on a bound / closed-form / counting or bit identity that is tempting to ASSERT but easy to get wrong. The trace MUST derive it AND then CHECK it numerically on a concrete small case, catching a plausible-but-FALSE algebra step before relying on it. NEVER invent terminology or assert an unproven formula -- this directly targets a real failure where the model wrote a confidently-wrong XOR identity and never checked it.' },
  { key: 'construct-verify', hint: 'the task is to OUTPUT a structure (set/sequence/assignment) satisfying a property. The tempting failure is to test only tiny inputs where a wrong construction works by luck, then ship it. The trace MUST verify the property AT THE REQUIRED SCALE (or prove it in general), catch a construction that holds for n=4 but breaks larger, and fall to a provably-correct construction -- this targets a real failure on a Sidon-set problem (passed n<=10, shipped for n<=1e7, scored 0).' },
  { key: 'wrongbaseline', hint: 'a well-known "standard" algorithm for this area is subtly wrong / inapplicable for THIS exact variant. The trace applies it, VERIFIES on a case, finds it fails, and adjusts to the correct method (no appeal to "the standard solution" without checking).' },
  { key: 'precision',  hint: 'integer overflow or exact-arithmetic precision decides correctness (compare products instead of dividing; use __int128 or exact integer geometry). The debug episode catches a precision/overflow bug on an adversarial case.' },
]

const specs = []
for (const tag of TAGS) for (const t of TWISTS)
  specs.push({ slug: `cpv4b-${tag}-${t.key}`, tag, twist: t.key, hint: t.hint })
log(`V4b generation: ${specs.length} datapoints (22 tags x 9 twists), one subagent each, gen-only`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'title', 'ok', 'reasoning_chars', 'oracle_cases', 'oracle_mismatches', 'note'],
  properties: {
    slug: { type: 'string' }, title: { type: 'string' }, ok: { type: 'boolean' },
    reasoning_chars: { type: 'integer' }, oracle_cases: { type: 'integer' },
    oracle_mismatches: { type: 'integer' }, note: { type: 'string' },
  },
}

function genPrompt(s) {
  return `You are generating ONE high-quality competitive-programming training datapoint for a FrontierCS-style benchmark (exact judge, single-shot: problem in, think, emit ONE C++ solution reading stdin / writing stdout, judged on hidden tests). Working dir: ${REPO}.

FIRST read the gold exemplar for FORMAT, ORGANIZATION, DEPTH (do not copy its problem):
  Read ${REPO}/data_v4/cp-noadj-commit/context.md , reasoning.md , train_answer.md

YOUR DATAPOINT:  slug: ${s.slug}   algorithm area: ${s.tag}   required emphasis: ${s.hint}

Invent a DISTINCT, self-consistent problem in the "${s.tag}" area embodying that emphasis. Avoid the most cliche phrasing; concrete story, exact I/O contract, constraints (so the pitfall is real), a time limit, a worked sample. Single-file C++17 reading stdin / writing stdout, judged exactly.

HARD VERIFICATION (with Bash; do not skip):
  1. Write ${REPO}/data_v4/${s.slug}/verify/sol.cpp  (single-file solution, reads stdin).
  2. Write ${REPO}/data_v4/${s.slug}/verify/brute.py  (INDEPENDENT, obviously-correct brute force for the SAME stated problem).
  3. Write ${REPO}/data_v4/${s.slug}/verify/gen.py  (random SMALL-case generator: 'python3 gen.py <seed>').
  4. g++ -O2 -std=c++17 -o /tmp/${s.slug}_sol ${REPO}/data_v4/${s.slug}/verify/sol.cpp
  5. Run >=300 random small cases sol vs brute; iterate to ZERO mismatches; if you cannot, SIMPLIFY until provably correct. NEVER emit an unverified solution. Confirm the documented sample.

THEN write THREE files (English), mirroring the exemplar:
  ${REPO}/data_v4/${s.slug}/context.md      -- headers "# <title>", "## Research question", "## Input / output contract", "## Background", "## Evaluation settings", "## Code framework" (pre-method C++ scaffold, one // TODO).
  ${REPO}/data_v4/${s.slug}/reasoning.md    -- first-person present-tense, ORGANIZED with short bold stage labels (**Reading the problem...**, **Candidate approaches.**, **Deriving...**, **First implementation and a trace.**, **The bug.**, **Fix and re-verification.**, **Edge cases.**, **Final solution.**, **Causal recap.**). MUST contain >=2 genuine debug/self-verify episodes that TRACE the code on a concrete input, find a REAL bug, and fix it; explicit edge-case checks; AND a numeric self-check of any derived formula/bound/claim on a concrete case (never assert an unproven identity). End with the final verified code (identical to sol.cpp) in a \`\`\`cpp block, then a one-paragraph causal recap. LENGTH >= 13000 characters of substantive content (no padding).
  ${REPO}/data_v4/${s.slug}/train_answer.md -- STRUCTURED editorial (bold "Problem.", "Key idea.", "Pitfalls.", "Edge cases.", "Complexity.", "Code.") ending with the SAME \`\`\`cpp block.

Code in reasoning.md and train_answer.md MUST be character-identical to verify/sol.cpp.
Return the structured result; ok=true only if compiled, oracle_mismatches==0 over oracle_cases>=300, three .md files written, reasoning.md >= 13000 chars.`
}

const results = await parallel(specs.map((s) => () =>
  agent(genPrompt(s), { label: `gen:${s.slug}`, phase: 'Generate', schema: SCHEMA, agentType: 'general-purpose' })))
const ok = results.filter((r) => r && r.ok)
log(`V4b generated ${ok.length}/${specs.length} self-reported ok; run tools/verify_v4.py to confirm`)
return { generated: ok.map((r) => r.slug), failed: specs.map((s) => s.slug).filter((sl) => !ok.find((r) => r.slug === sl)) }
