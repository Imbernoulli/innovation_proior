export const meta = {
  name: 'verify-cpp-codex',
  description: 'Differential-test each P0.1 C++ conversion for CORRECTNESS via Codex (vs original Python / brute oracle); fix on mismatch',
  whenToUse: 'After cpp-landing-conversion: confirm the C++ is not just compiling but algorithmically correct',
  phases: [{ title: 'Verify', detail: 'one Codex agent per method: extract C++, build oracle, differential-test, fix if wrong' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const PRECONV = 'b994a519'  // commit with the original (pre-conversion) Python landings
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const slugs = (A && Array.isArray(A.slugs) && A.slugs.length) ? A.slugs : []
if (!slugs.length) { log('No args.slugs.'); return { verified: [] } }
log(`Codex-verifying ${slugs.length} C++ conversions (differential test vs oracle)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'verdict', 'oracle_source', 'cases', 'mismatches', 'fixed', 'notes'],
  properties: {
    slug: { type: 'string' },
    verdict: { type: 'string', enum: ['pass', 'fixed-pass', 'fail', 'no-oracle'] },
    oracle_source: { type: 'string', description: 'code/brute.py | code/<ref>.py | git-preconversion-python | self-written-brute | none' },
    cases: { type: 'integer' },
    mismatches: { type: 'integer', description: 'mismatches AFTER any fix (0 for pass/fixed-pass)' },
    fixed: { type: 'boolean', description: 'did you edit the .md C++ to fix a real bug introduced by the conversion' },
    notes: { type: 'string' },
  },
}

function prompt(slug) {
  return `Verify the C++ LANDING conversion for competition method "${slug}" is CORRECT (not merely compiling). Working dir: ${REPO}. Use Bash freely (compile, run, diff).

CONTEXT: methods/${slug}/results/{context.md, reasoning.md, answer.md, train_answer.md} were just converted so the landing is a single-file C++17 program reading stdin (was a Python class/library). Confirm the algorithm survived the translation.

STEPS:
1. Extract the final \`\`\`cpp block from methods/${slug}/results/train_answer.md to /tmp/${slug}.cpp; compile: g++ -O2 -std=c++17 -o /tmp/${slug}_x /tmp/${slug}.cpp (it should compile; if not, that's a fail to fix).
2. Get an ORACLE for the SAME problem, in this preference order:
   a. methods/${slug}/code/brute.py (a brute force), or another reference in methods/${slug}/code/*.py;
   b. the PRE-CONVERSION Python landing: \`git show ${PRECONV}:methods/${slug}/results/train_answer.md\` (extract its \`\`\`python block) -- adapt it to read the SAME stdin format as the C++ if needed;
   c. if neither exists, write your own small correct brute force from the problem statement in context.md / refs/.
3. Determine the exact stdin/stdout I/O contract from context.md (and methods/${slug}/refs/ if present). Write a random SMALL-input generator respecting the constraints (and an exhaustive/edge generator for corners).
4. Run >= 200 random cases through the compiled C++ and the oracle; compare stdout EXACTLY. Also run any sample I/O from refs/.
5. If outputs MATCH on all cases -> verdict "pass". If they DIVERGE, the conversion introduced a bug: FIX the C++ in methods/${slug}/results/train_answer.md AND answer.md AND the reasoning.md code block (keep all three byte-identical) until it matches the oracle on >=200 cases, then verdict "fixed-pass". If you genuinely cannot build any trustworthy oracle (e.g. it is a heuristic/optimizer with no exact answer), verdict "no-oracle" and just confirm it compiles + runs on a sample without crashing.

Report the structured result honestly: oracle_source, cases run, mismatches AFTER fixing (must be 0 unless verdict=fail), whether you fixed anything, and a one-line note on what (if anything) was wrong.`
}

const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `cxverify:${slug}`, phase: 'Verify', schema: SCHEMA, agentType: 'codex:codex-rescue' })))
const r = results.filter(Boolean)
const pass = r.filter((x) => x.verdict === 'pass' || x.verdict === 'fixed-pass')
const fixed = r.filter((x) => x.fixed)
const fail = r.filter((x) => x.verdict === 'fail')
log(`Codex verify: ${pass.length}/${slugs.length} pass (${fixed.length} needed a fix); ${fail.length} fail; ${r.filter(x=>x.verdict==='no-oracle').length} no-oracle`)
return { pass: pass.map(x => x.slug), fixed: fixed.map(x => x.slug), fail: fail.map(x => ({ slug: x.slug, notes: x.notes })), no_oracle: r.filter(x => x.verdict === 'no-oracle').map(x => x.slug) }
