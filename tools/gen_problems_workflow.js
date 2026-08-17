export const meta = {
  name: 'gen-frontiercs-problems',
  description: 'Generate new FrontierCS problems from the candidate list; each subagent self-verifies AND calls Codex to independently review',
  whenToUse: 'Step 2: flesh PROBLEM_CANDIDATES.md FCS rows into verified single-file C++ datapoints',
  phases: [{ title: 'Generate', detail: 'one subagent per candidate: build problem + C++ solution + oracle, self-test, then invoke Codex to review/fix' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const CODEX = '/home/bohanlyu/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const ids = (A && Array.isArray(A.ids) && A.ids.length) ? A.ids : []
if (!ids.length) { log('No args.ids (candidate ids).'); return { generated: [] } }
log(`Generating ${ids.length} FrontierCS datapoints (subagent reads its candidate, writes + Codex reviews)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['id', 'ok', 'compiles', 'oracle_cases', 'oracle_mismatches', 'codex_verdict', 'reasoning_chars', 'notes'],
  properties: {
    id: { type: 'string' }, ok: { type: 'boolean' },
    compiles: { type: 'boolean' }, oracle_cases: { type: 'integer' }, oracle_mismatches: { type: 'integer' },
    codex_verdict: { type: 'string', enum: ['pass', 'fixed-pass', 'fail', 'not-run'] },
    reasoning_chars: { type: 'integer' }, notes: { type: 'string' },
  },
}

function prompt(id) {
  return `Generate ONE FrontierCS training datapoint from candidate "${id}", then have CODEX independently verify it. Working dir: ${REPO}.

YOUR CANDIDATE: read ${REPO}/data_v4/PROBLEM_CANDIDATES.md and find the table row whose first column is exactly "${id}". That row gives: title | area | THE INSIGHT (the non-obvious idea the solution hinges on -- the "innovation" the trace must discover) | I/O sketch | constraints/difficulty | oracle approach (to verify). Use those fields. (The cell text may contain '|'/'->' visually -- read the whole row carefully.)

FIRST read the gold trace for format + depth: ${REPO}/data_v4/cp-noadj-commit/{context.md,reasoning.md,train_answer.md}.

THE ONE THING TO OBSESS OVER -- STRONGEST / SOTA ALGORITHM: the solution MUST use the CURRENT BEST-KNOWN / state-of-the-art algorithm for this problem at these constraints (the candidate's INSIGHT names the technique; if an even stronger standard approach exists, use that). Not merely a correct solution -- the canonical SOTA one, with the right asymptotic complexity to pass the stated limits. If you are unsure what the strongest known approach is, RESEARCH it (your knowledge / web) before implementing. (Problem originality / no-leakage was already handled when the candidate list was authored; you do NOT need to worry about it -- just implement the assigned problem faithfully with the strongest algorithm.)

BUILD (FrontierCS = exact judge, single-file C++17 read from stdin, write stdout):
1. Flesh the candidate into a concrete, self-contained problem: a clear statement, EXACT stdin/stdout format, constraints (matching the sketch), time limit, and >=1 worked sample.
2. Write a single-file C++17 solution that implements THE INSIGHT with the STRONGEST (SOTA) algorithm. Write it to ${REPO}/data_v4/${id}/verify/sol.cpp.
3. Write an INDEPENDENT brute force ${REPO}/data_v4/${id}/verify/brute.py (the oracle approach above -- a slow-but-obviously-correct method) and a random small-case generator ${REPO}/data_v4/${id}/verify/gen.py (param: int seed).
4. SELF-VERIFY (with Bash): g++ -O2 -std=c++17 -o /tmp/${id}_x verify/sol.cpp ; run >=500 random small cases + explicit edge cases through sol vs brute; iterate until ZERO mismatches. Also confirm the documented sample. If you cannot make sol correct, SIMPLIFY the problem until provably correct -- NEVER emit an unverified solution.

THEN — CODEX REVIEW (mandatory, this is the second independent check). Run:
  node "${CODEX}" task "Independently verify the FrontierCS solution at ${REPO}/data_v4/${id}/. Extract the C++ from verify/sol.cpp, read the problem in context.md, write your OWN independent brute-force oracle (do not reuse verify/brute.py), and differential-test the compiled C++ on >=300 random small cases AND adversarial edge cases. If you find ANY mismatch, the solution has a bug: FIX verify/sol.cpp AND keep the cpp block in train_answer.md and reasoning.md byte-identical to it. Report PASS or the bug you fixed." --write --model gpt-5.5 --effort high
  Capture Codex's verdict. After Codex returns, RE-RUN your own oracle to confirm 0 mismatches (Codex can err too).

WRITE the deliverables (English):
  ${REPO}/data_v4/${id}/context.md      -- structured: "# <title>", "## Research question", "## Input / output contract", "## Background", "## Evaluation settings", "## Code framework" (a PRE-METHOD C++ scaffold: int main reading stdin, neutral // TODO body). Mirror the gold trace; C++ scaffold (not Python).
  ${REPO}/data_v4/${id}/reasoning.md    -- first-person, organized with bold stage labels. The SPINE: discover THE INSIGHT (present the obvious approach, show on a concrete case why it is too slow / wrong, then derive the insight as the resolution -- the innovation must feel earned, not announced); then implement; then a REAL debug+self-verify episode (trace the code, find+fix a bug, check edges); end on the final verified C++ (identical to sol.cpp). >= 13000 chars of substantive content, no padding.
  ${REPO}/data_v4/${id}/train_answer.md -- structured editorial (Problem / Key idea = the insight / Pitfalls / Edge cases / Complexity / Code) ending with the SAME cpp block.

Code in reasoning.md and train_answer.md MUST equal verify/sol.cpp byte-for-byte.
Return the result: ok=true only if it compiles, your oracle has 0 mismatches over >=500 cases, Codex verdict is pass/fixed-pass, the 3 .md files are written, and reasoning.md >= 13000 chars.`
}

const results = await parallel(ids.map((id) => () =>
  agent(prompt(id), { label: `gen:${id}`, phase: 'Generate', schema: SCHEMA, agentType: 'general-purpose' })))
const ok = results.filter((r) => r && r.ok)
const codexPass = ok.filter((r) => r.codex_verdict === 'pass' || r.codex_verdict === 'fixed-pass')
log(`Generated ${ok.length}/${ids.length} (Codex-passed ${codexPass.length}); run tools/verify_v4.py to re-confirm`)
return { generated: ok.map((r) => r.id), codex_passed: codexPass.map((r) => r.id), failed: ids.filter((id) => !ok.find((r) => r.id === id)) }
