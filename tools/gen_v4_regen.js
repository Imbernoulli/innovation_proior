export const meta = {
  name: 'gen-v4-regen',
  description: 'Re-generate the V4 datapoints that the first run lost to rate-limiting (gen-only; verified separately by tools/verify_v4.py)',
  whenToUse: 'Fill the gap after gen-v4-cp-data was throttled',
  phases: [{ title: 'Regenerate', detail: 'one subagent per missing slug: invent + write single-file C++ + self-verify (compile+brute oracle) + long organized reasoning' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const TWIST_HINT = {
  overflow:  'values/answers large enough that a 32-bit int silently overflows (use long long); the debug episode should catch an int-overflow.',
  negzero:   'inputs include negatives and zeros and an all-negative / empty corner; the debug episode should catch a wrong base case or sign handling.',
  boundary:  'answer is sensitive to off-by-one / inclusive-exclusive boundaries; the debug episode should catch an off-by-one via a traced small case.',
  greedytrap:'an obvious greedy is tempting but wrong; the trace must construct a counterexample that disproves greedy before deriving the correct method.',
  count:     'a counting/constructive variant where the index/modulus/dedup is easy to get subtly wrong; the debug episode catches a double-count or off-by-one in counting.',
}

let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
let slugs = []
if (Array.isArray(A)) slugs = A
else if (A && Array.isArray(A.slugs)) slugs = A.slugs
if (!slugs.length) { log('No args.slugs provided.'); return { regenerated: [] } }

function parse(slug) {
  // slug = cpv4-<tag...>-<twist>
  const rest = slug.replace(/^cpv4-/, '')
  const twist = Object.keys(TWIST_HINT).find((t) => rest.endsWith('-' + t))
  const tag = rest.slice(0, rest.length - twist.length - 1)
  return { tag, twist, hint: TWIST_HINT[twist] }
}

log(`Re-generating ${slugs.length} missing V4 datapoints (gen-only; verified by tools/verify_v4.py)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'title', 'ok', 'reasoning_chars', 'oracle_cases', 'oracle_mismatches', 'note'],
  properties: {
    slug: { type: 'string' }, title: { type: 'string' }, ok: { type: 'boolean' },
    reasoning_chars: { type: 'integer' }, oracle_cases: { type: 'integer' },
    oracle_mismatches: { type: 'integer' }, note: { type: 'string' },
  },
}

function genPrompt(slug, s) {
  return `You are generating ONE high-quality competitive-programming training datapoint for a FrontierCS-style benchmark (exact judge, single-shot: the model is given a problem, thinks, and emits ONE C++ solution reading stdin / writing stdout, judged on hidden tests). Working dir: ${REPO}.

FIRST read the gold exemplar to match its FORMAT, ORGANIZATION, DEPTH (do not copy its problem):
  Read ${REPO}/data_v4/cp-noadj-commit/context.md , reasoning.md , train_answer.md

YOUR DATAPOINT:  slug: ${slug}   algorithm area: ${s.tag}   required pitfall/twist: ${s.hint}

Invent a DISTINCT, self-consistent problem in the "${s.tag}" area that naturally contains that pitfall. Concrete story, exact I/O contract, constraints (so the pitfall is real), a time limit, at least one worked sample. Single-file C++17 reading stdin / writing stdout, judged exactly.

HARD VERIFICATION (do it with Bash, do not skip):
  1. Write ${REPO}/data_v4/${slug}/verify/sol.cpp   (your single-file solution, reads stdin).
  2. Write ${REPO}/data_v4/${slug}/verify/brute.py  (an INDEPENDENT, obviously-correct brute force for the SAME stated problem).
  3. Write ${REPO}/data_v4/${slug}/verify/gen.py    (random SMALL-case generator parameterized by an int seed arg: 'python3 gen.py <seed>').
  4. g++ -O2 -std=c++17 -o /tmp/${slug}_sol ${REPO}/data_v4/${slug}/verify/sol.cpp
  5. Run >=300 random small cases comparing sol vs brute; iterate until ZERO mismatches. If you cannot, SIMPLIFY the problem until sol is provably correct and agrees. NEVER emit an unverified solution. Also confirm sol prints the documented sample.

THEN write THREE files (English), mirroring the exemplar's structure:
  ${REPO}/data_v4/${slug}/context.md      -- headers: "# <title>", "## Research question", "## Input / output contract", "## Background", "## Evaluation settings", "## Code framework" (pre-method C++ scaffold with one // TODO).
  ${REPO}/data_v4/${slug}/reasoning.md    -- first-person present-tense, ORGANIZED with short bold stage labels (**Reading the problem...**, **Candidate approaches.**, **Deriving...**, **First implementation and a trace.**, **The bug.**, **Fix and re-verification.**, **Edge cases.**, **Final solution.**, **Causal recap.**). MUST contain >=2 genuine debug/self-verify episodes that TRACE the code on a concrete input, find a REAL bug, and fix it; explicit edge-case checks; a derivation sanity-check. End with the final verified code (identical to sol.cpp) in a \`\`\`cpp block, then a one-paragraph causal recap. LENGTH >= 12000 characters of substantive content (no padding).
  ${REPO}/data_v4/${slug}/train_answer.md -- STRUCTURED editorial (bold "Problem.", "Key idea.", "Pitfalls.", "Edge cases.", "Complexity.", "Code.") ending with the SAME \`\`\`cpp block.

Code in reasoning.md and train_answer.md MUST be character-identical to verify/sol.cpp.
Return the structured result; ok=true only if compiled, oracle_mismatches==0 over oracle_cases>=300, three .md files written, reasoning.md >= 12000 chars.`
}

const results = await parallel(slugs.map((slug) => () => {
  const s = parse(slug)
  return agent(genPrompt(slug, s), { label: `regen:${slug}`, phase: 'Regenerate', schema: SCHEMA, agentType: 'general-purpose' })
}))
const ok = results.filter((r) => r && r.ok)
log(`Re-generated ${ok.length}/${slugs.length} (self-reported ok); run tools/verify_v4.py to confirm`)
return { regenerated: ok.map((r) => r.slug), all: results.filter(Boolean).map((r) => ({ slug: r.slug, ok: r.ok, note: (r.note || '').slice(0, 200) })) }
