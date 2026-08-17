export const meta = {
  name: 'gen-p2p3-data',
  description: 'DATA_FIX_FCS_LANDING P2 (ship-the-baseline after verifying the clever idea fails) + P3 (anti-hardcoding) data points',
  whenToUse: 'Colleague-doc P2/P3: inject traces that verify a tempting approach is wrong and ship the simpler provable solution / resist hardcoding',
  phases: [{ title: 'Generate', detail: 'one subagent per theme: build problem + correct C++ + the verify-then-ship-baseline / anti-hardcoding trace; self-verify + Codex review' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const CODEX = '/home/bohanlyu/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const specs = (A && Array.isArray(A.specs) && A.specs.length) ? A.specs : []
if (!specs.length) { log('No args.specs.'); return { generated: [] } }
log(`Generating ${specs.length} P2/P3 data points (verify-then-ship-baseline / anti-hardcoding)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['id', 'ok', 'compiles', 'oracle_cases', 'oracle_mismatches', 'codex_verdict', 'reasoning_chars', 'notes'],
  properties: {
    id: { type: 'string' }, ok: { type: 'boolean' }, compiles: { type: 'boolean' },
    oracle_cases: { type: 'integer' }, oracle_mismatches: { type: 'integer' },
    codex_verdict: { type: 'string', enum: ['pass', 'fixed-pass', 'fail', 'not-run'] },
    reasoning_chars: { type: 'integer' }, notes: { type: 'string' },
  },
}

function prompt(s) {
  const p2 = s.track === 'p2'
  return `Generate ONE training data point (DATA_FIX_FCS_LANDING ${p2 ? 'P2 ship-the-baseline' : 'P3 anti-hardcoding'}) from this theme, then have CODEX verify it. Working dir: ${REPO}. id: ${s.id}.

THEME: ${s.theme}

Read the gold trace shape: ${REPO}/data_v4/cp-noadj-commit/{context.md,reasoning.md,train_answer.md}. FrontierCS format: single-file C++17 reading stdin, exact judge.

${p2 ? `THE POINT (P2): build a problem where a TEMPTING clever/greedy/fancy approach is WRONG or risky-to-get-right-in-budget, and the CORRECT answer is a SIMPLER, PROVABLE method (a clean DP / exhaustive / standard algorithm that is fast enough at the chosen constraints). The reasoning's spine: reach for the clever idea -> CONSTRUCT a concrete counterexample (or argue it is error-prone in the budget) that shows it is wrong -> then DERIVE and SHIP the simpler correct method "Z" that you can prove and have traced. The destination is the SIMPLE CORRECT solution, reached BY the verification killing the clever idea -- NOT a predetermined fancy method. Set the constraints so the simple correct method comfortably passes.`
  : `THE POINT (P3): build a problem that TEMPTS hardcoding small-n constants (the samples / small cases have a tidy closed pattern), but the HIDDEN tests go to large n so hardcoding fails. The reasoning's spine: notice the small cases look hardcodable, explicitly say "I could hardcode n<=K, but the constraints go to N=<large> so the hidden tests will break that" -> then DERIVE the general recurrence/algorithm and ship it. The shipped C++ is the general solution.`}

BUILD: a concrete problem (statement, EXACT stdin/stdout, constraints, time limit, >=1 sample). Write the CORRECT single-file C++17 to ${REPO}/data_v4/${s.id}/verify/sol.cpp. Write an INDEPENDENT brute oracle verify/brute.py + generator verify/gen.py.
SELF-VERIFY (Bash): compile; run >=500 random + edge cases sol vs brute, ZERO mismatches, iterate till correct. NEVER emit an unverified solution.
CODEX REVIEW (mandatory): node "${CODEX}" task "Independently verify ${REPO}/data_v4/${s.id}/: extract verify/sol.cpp, write your OWN brute oracle for the problem in context.md, differential-test >=300 random+edge cases; fix verify/sol.cpp (+ keep train_answer.md/reasoning.md cpp blocks identical) on any mismatch. Report PASS or the bug fixed." --write --model gpt-5.5 --effort high
Then re-run your own oracle to confirm 0 mismatches.

WRITE (English): ${REPO}/data_v4/${s.id}/context.md (structured, C++ stdin scaffold with neutral // TODO), reasoning.md (first-person, organized; the ${p2 ? 'verify-then-ship-baseline' : 'resist-hardcoding'} spine above, with a REAL debug+self-verify episode and the explicit counterexample/over-large-n argument; end on the final verified C++ = sol.cpp; >=13000 chars), train_answer.md (structured editorial ending with the SAME cpp block).
Code in reasoning.md/train_answer.md MUST equal verify/sol.cpp byte-for-byte.
ok=true only if compiles, 0 oracle mismatches over >=500, Codex pass/fixed-pass, 3 .md written, reasoning >=13000 chars.`
}

const results = await parallel(specs.map((s) => () =>
  agent(prompt(s), { label: `${s.track}:${s.id}`, phase: 'Generate', schema: SCHEMA, agentType: 'general-purpose' })))
const ok = results.filter((r) => r && r.ok)
log(`Generated ${ok.length}/${specs.length} P2/P3 (Codex-passed ${ok.filter(r=>['pass','fixed-pass'].includes(r.codex_verdict)).length})`)
return { generated: ok.map((r) => r.id), failed: specs.map((s) => s.id).filter((id) => !ok.find((r) => r.id === id)) }
