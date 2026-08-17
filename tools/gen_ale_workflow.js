export const meta = {
  name: 'gen-ale-problems',
  description: 'Generate new ALE-Bench (AtCoder-Heuristic) datapoints from the candidate list; subagent builds solver+scorer, self-verifies, Codex reviews',
  whenToUse: 'Step 2 (ALE track): flesh PROBLEM_CANDIDATES.md ale-* rows into verified single-file heuristic-solver datapoints',
  phases: [{ title: 'Generate', detail: 'one subagent per candidate: build problem + C++ heuristic solver + deterministic local scorer, run/score, then Codex reviews' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const CODEX = '/home/bohanlyu/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const ids = (A && Array.isArray(A.ids) && A.ids.length) ? A.ids : []
if (!ids.length) { log('No args.ids.'); return { generated: [] } }
log(`Generating ${ids.length} ALE-Bench datapoints (solver + local scorer; subagent self-verifies + Codex reviews)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['id', 'ok', 'compiles', 'seeds_scored', 'all_feasible', 'beats_baseline', 'codex_verdict', 'reasoning_chars', 'notes'],
  properties: {
    id: { type: 'string' }, ok: { type: 'boolean' }, compiles: { type: 'boolean' },
    seeds_scored: { type: 'integer' }, all_feasible: { type: 'boolean' },
    beats_baseline: { type: 'boolean', description: 'solver score beats the trivial baseline on the seed set' },
    codex_verdict: { type: 'string', enum: ['pass', 'fixed-pass', 'fail', 'not-run'] },
    reasoning_chars: { type: 'integer' }, notes: { type: 'string' },
  },
}

function prompt(id) {
  return `Generate ONE ALE-Bench (AtCoder-Heuristic-style) training datapoint from candidate "${id}", then have CODEX independently review it. Working dir: ${REPO}.

YOUR CANDIDATE: read ${REPO}/data_v4/PROBLEM_CANDIDATES.md and find the table row whose first column is exactly "${id}". That row gives: title | objective (maximize/minimize WHAT) | I/O sketch | the heuristic-design INNOVATION (the non-obvious lever) | local scoring rule. Use those fields. (Cells may contain '|'/'->' visually -- read the whole row.)

ALE-Bench = heuristic OPTIMIZATION: NP-hard / no exact answer, judged by a CONTINUOUS score; an invalid/infeasible output floors the score to 0. The solution is a single-file C++17 program reading the instance from stdin and writing a feasible solution to stdout. Read the gold ALE trajectory for shape: ${REPO}/trajectories/ale-atcoder-ahc039/ , and the gold single-file trace ${REPO}/data_v4/cp-noadj-commit/.

THE ONE THING TO OBSESS OVER -- STRONGEST HEURISTIC: use the current BEST-KNOWN heuristic family for this problem type (the candidate's INNOVATION names the lever; if a stronger standard approach exists -- a better neighborhood, a known relaxation, an established metaheuristic for this structure -- use it). The solver must be a genuinely strong heuristic that scores well, not a toy greedy. If unsure what the strongest known approach is, RESEARCH it before implementing. (Problem originality / no-leakage was already handled when the candidate list was authored; you do NOT need to worry about it -- just implement the assigned problem faithfully with the strongest heuristic.)

BUILD:
1. Flesh the candidate into a concrete problem: statement, EXACT stdin format (instance), EXACT stdout format (solution), constraints, the scoring function precisely (with the feasibility->0 floor), a time budget, and how instances are generated.
2. Write a single-file C++17 heuristic SOLVER implementing the INNOVATION (e.g. the incremental-eval SA / LNS / greedy construction). Write to ${REPO}/data_v4/${id}/verify/sol.cpp. It must ALWAYS output a FEASIBLE solution (never crash, never invalid) within a small time budget.
3. Write a deterministic local SCORER ${REPO}/data_v4/${id}/verify/score.py (reads instance + solution, returns the score per the rule, 0 if infeasible) and an instance generator ${REPO}/data_v4/${id}/verify/gen.py (param: int seed).
4. SELF-VERIFY (with Bash): compile sol; generate a fixed SEED SET (e.g. seeds 1..20); for each, run sol, score it, and ALSO score a trivial BASELINE (e.g. empty/greedy/identity). Confirm: every output is FEASIBLE (score > 0, parses), and the solver's mean score strictly BEATS the trivial baseline. Iterate on the solver until both hold. NEVER emit a solver that produces infeasible output or fails to beat the baseline.

THEN — CODEX REVIEW (mandatory). Run:
  node "${CODEX}" task "Independently review the ALE-Bench datapoint at ${REPO}/data_v4/${id}/. (a) Verify verify/score.py correctly implements the scoring rule stated in context.md (including the feasibility->0 floor) -- write your own check; (b) compile verify/sol.cpp and run it on >=15 generated seeds, confirming EVERY output is feasible (the scorer does not 0 it) and the solver's score beats a trivial baseline; (c) if the solver ever produces infeasible output or fails to beat baseline, FIX verify/sol.cpp (and keep the cpp block in train_answer.md/reasoning.md identical). Report PASS or what you fixed." --write --model gpt-5.5 --effort high
  After Codex returns, re-run your own seed-set scoring to confirm.

WRITE the deliverables (English):
  ${REPO}/data_v4/${id}/context.md      -- "# <title>", "## Research question", "## Input / output contract" (instance in, solution out), "## Background", "## Evaluation settings" (the scoring rule + feasibility floor + how instances are made), "## Code framework" (a pre-method C++ scaffold: int main reading the instance, // TODO heuristic, print a feasible solution).
  ${REPO}/data_v4/${id}/reasoning.md    -- first-person, organized. SPINE: understand the objective; reach a feasible BASELINE first (always have a valid solution); then discover the INNOVATION (why the obvious local search is too slow / weak, then the incremental-eval / neighborhood / relaxation idea); implement; a REAL debug+self-verify episode (run it, find a feasibility/score bug, fix it; confirm it beats baseline on seeds). End on the final solver C++ (identical to sol.cpp). >= 13000 chars, no padding.
  ${REPO}/data_v4/${id}/train_answer.md -- structured editorial (Problem / Objective+scoring / Baseline / Key idea = the heuristic innovation / Feasibility & pitfalls / Complexity-per-step / Code) ending with the SAME cpp block.

Code in reasoning.md / train_answer.md MUST equal verify/sol.cpp byte-for-byte.
Return: ok=true only if it compiles, all seeds feasible, beats baseline, Codex verdict pass/fixed-pass, 3 .md files written, reasoning.md >= 13000 chars.`
}

const results = await parallel(ids.map((id) => () =>
  agent(prompt(id), { label: `genale:${id}`, phase: 'Generate', schema: SCHEMA, agentType: 'general-purpose' })))
const ok = results.filter((r) => r && r.ok)
const codexPass = ok.filter((r) => r.codex_verdict === 'pass' || r.codex_verdict === 'fixed-pass')
log(`Generated ${ok.length}/${ids.length} ALE (Codex-passed ${codexPass.length})`)
return { generated: ok.map((r) => r.id), codex_passed: codexPass.map((r) => r.id), failed: ids.filter((id) => !ok.find((r) => r.id === id)) }
