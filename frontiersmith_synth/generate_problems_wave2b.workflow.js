export const meta = {
  name: 'frontiersmith-generate-wave2b',
  description: 'Author wave-2b problems: one agent per spec, harness-validated + Codex (gpt-5.6-terra xhigh) independent review, with innovation-headroom acceptance',
  phases: [
    { title: 'Generate', detail: 'one agent per spec: author + self-validate to PASS + Codex review + innovation metrics' },
    { title: 'Repair', detail: 'second pass only for problems that failed harness, Codex review, or innovation acceptance' },
  ],
}

const SYNTH = '/scratch/gpfs/CHIJ/bohan/fs/innovation_prior/frontiersmith_synth'
const SEEDS = `${SYNTH}/seeds/seed_list.jsonl`
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
A = A || {}
const DEFAULT_SPECS = [{"id":"fsx_S_1054","format":"C"},{"id":"fsx_B_1079","format":"C"},{"id":"fsx_S_1083","format":"C"},{"id":"fsx_B_1086","format":"D"},{"id":"fsx_B_1087","format":"D"},{"id":"fsx_A_1088","format":"D"},{"id":"fsx_A_1089","format":"D"},{"id":"fsx_B_1090","format":"C"},{"id":"fsx_A_1091","format":"C"},{"id":"fsx_A_1092","format":"C"},{"id":"fsx_S_1093","format":"D"},{"id":"fsx_A_1094","format":"C"},{"id":"fsx_A_1095","format":"C"},{"id":"fsx_S_1096","format":"D"},{"id":"fsx_B_1097","format":"C"},{"id":"fsx_S_1098","format":"D"},{"id":"fsx_B_1099","format":"C"},{"id":"fsx_A_1100","format":"E"},{"id":"fsx_A_1101","format":"C"},{"id":"fsx_A_1102","format":"C"},{"id":"fsx_B_1103","format":"D"},{"id":"fsx_C_1104","format":"C"},{"id":"fsx_A_1105","format":"C"},{"id":"fsx_B_1106","format":"C"},{"id":"fsx_A_1107","format":"D"},{"id":"fsx_B_1108","format":"C"},{"id":"fsx_A_1109","format":"D"},{"id":"fsx_B_1110","format":"E"},{"id":"fsx_A_1111","format":"E"},{"id":"fsx_B_1112","format":"D"},{"id":"fsx_B_1113","format":"C"},{"id":"fsx_B_1114","format":"B"},{"id":"fsx_B_1115","format":"C"},{"id":"fsx_B_1116","format":"B"},{"id":"fsx_B_1117","format":"E"},{"id":"fsx_A_1118","format":"B"},{"id":"fsx_A_1119","format":"C"},{"id":"fsx_A_1120","format":"B"},{"id":"fsx_B_1121","format":"C"},{"id":"fsx_N_1122","format":"D"},{"id":"fsx_A_1123","format":"B"},{"id":"fsx_A_1124","format":"B"},{"id":"fsx_S_1125","format":"E"},{"id":"fsx_A_1126","format":"B"},{"id":"fsx_B_1127","format":"E"},{"id":"fsx_B_1128","format":"C"},{"id":"fsx_B_1129","format":"C"},{"id":"fsx_A_1130","format":"C"},{"id":"fsx_A_1131","format":"E"},{"id":"fsx_B_1132","format":"C"},{"id":"fsx_A_1133","format":"B"},{"id":"fsx_B_1134","format":"C"},{"id":"fsx_B_1135","format":"C"},{"id":"fsx_B_1136","format":"C"},{"id":"fsx_B_1137","format":"B"},{"id":"fsx_S_1138","format":"D"},{"id":"fsx_A_1139","format":"D"},{"id":"fsx_G_1140","format":"E"},{"id":"fsx_A_1141","format":"E"},{"id":"fsx_B_1142","format":"C"},{"id":"fsx_A_1143","format":"B"},{"id":"fsx_A_1144","format":"D"},{"id":"fsx_B_1145","format":"B"},{"id":"fsx_S_1146","format":"D"},{"id":"fsx_B_1147","format":"C"},{"id":"fsx_B_1148","format":"C"},{"id":"fsx_C_1149","format":"C"},{"id":"fsx_A_1150","format":"C"},{"id":"fsx_B_1151","format":"E"},{"id":"fsx_B_1152","format":"C"},{"id":"fsx_A_1153","format":"C"},{"id":"fsx_A_1154","format":"C"},{"id":"fsx_A_1155","format":"C"},{"id":"fsx_A_1156","format":"E"},{"id":"fsx_A_1157","format":"C"},{"id":"fsx_A_1158","format":"C"},{"id":"fsx_B_1159","format":"C"},{"id":"fsx_A_1160","format":"B"},{"id":"fsx_B_1161","format":"C"},{"id":"fsx_S_1162","format":"E"},{"id":"fsx_S_1163","format":"C"},{"id":"fsx_B_1164","format":"C"},{"id":"fsx_A_1165","format":"D"}]
let specs = Array.isArray(A) ? A : (A.specs || A.routes || [])
if (!specs.length) { specs = DEFAULT_SPECS; log(`args had no specs — using DEFAULT_SPECS (${specs.length})`) }
const MODEL = A.model || 'opus'
const EFFORT = A.effort || 'high'
if (!specs.length) { log(`no specs. typeof args=${typeof args}`); return { error: 'no specs' } }
log(`generating ${specs.length} problems (model=${MODEL}, effort=${EFFORT})`)

// innovation-headroom acceptance (wave-2b addition; see AGENT_BRIEF_INNOVATION_ADDENDUM.md)
const MIN_STRONG_MINUS_GREEDY = 0.06
const MAX_STRONG = 0.92
const MIN_GREEDY_MINUS_TRIVIAL = 0.03

const RESULT_SCHEMA = {
  type: 'object', additionalProperties: true,
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    title: { type: 'string' }, format: { type: 'string' }, family: { type: 'string' },
    metrics: {
      type: 'object', additionalProperties: true,
      properties: {
        trivial: { type: 'number' }, greedy: { type: 'number' },
        strong: { type: 'number' }, invalid: { type: 'number' },
        divergence: { type: 'number' },
      },
    },
    rounds: { type: 'integer' }, notes: { type: 'string' },
    codex_verdict: { type: 'string', enum: ['LGTM', 'DEFECTS', 'UNAVAILABLE'] },
  },
  required: ['id', 'verdict', 'metrics', 'notes', 'codex_verdict'],
}

function route(fmt) {
  if (fmt === 'A') return { brief: 'AGENT_BRIEF.md', val: 'harness/validate_problem.py' }
  if (fmt === 'B') return { brief: 'AGENT_BRIEF_PY_PROGRAM.md', val: 'harness/validate_pyproblem.py' }
  return { brief: 'AGENT_BRIEF_PY_STDOUT.md', val: 'harness/validate_problem.py' } // C/D/E
}

function accepted(res) {
  if (!res || res.verdict !== 'PASS' || !res.metrics) return false
  if (res.codex_verdict === 'DEFECTS') return false // unresolved independent-review defects
  const m = res.metrics
  const g = Number(m.greedy), s = Number(m.strong), t = Number(m.trivial)
  if (!isFinite(g) || !isFinite(s) || !isFinite(t)) return false
  return (s - g) >= MIN_STRONG_MINUS_GREEDY && s <= MAX_STRONG && (g - t) >= MIN_GREEDY_MINUS_TRIVIAL
}

const CODEX_REVIEW = (dir) => `timeout 1800 codex exec -m gpt-5.6-terra -c model_reasoning_effort='"xhigh"' --sandbox read-only --cd ${dir} "You are reviewing one deterministically-scored optimization problem for an RL training corpus. Read statement.md, config.yaml, the generator, the checker/verifier, and every file under solutions/. Hunt ONLY for real defects: (1) scoring bugs or metric loopholes a submitted program could exploit for unearned score; (2) nondeterminism (unseeded randomness, wall-time dependence, dict/set iteration order affecting scores); (3) statement claims that mismatch what the code implements; (4) a strong solution that is really just greedy plus parameter tuning instead of exploiting the stated innovation hook; (5) trap test cases that do not actually punish the obvious greedy. Be concrete: cite file and line. End with exactly one line: CODEX VERDICT: LGTM  or  CODEX VERDICT: DEFECTS followed by a numbered list."`

function genPrompt(s) {
  const r = route(s.format)
  const dir = `${SYNTH}/problems/${s.id}`
  return `You are a problem setter. Author ONE novel, DETERMINISTICALLY-scored open-ended problem
(format ${s.format}), then self-validate it with the harness until it PASSES *and* meets the
innovation-headroom acceptance numbers.

STEP 0 — read YOUR full spec (family, mechanisms, theme, innovation_hook, trap, objective):
  python3 -c "import json;print([l for l in open('${SEEDS}') if json.loads(l)['id']=='${s.id}'][0])"
STEP 1 — read the full authoring contract (do not skip): ${SYNTH}/${r.brief}
STEP 2 — read the innovation addendum (do not skip): ${SYNTH}/AGENT_BRIEF_INNOVATION_ADDENDUM.md

Output directory (create it): ${dir}
Harness (ground truth): python3 ${SYNTH}/${r.val} ${dir}${s.format === 'B' ? '' : ' --keep-testdata'}

Rules:
- Deterministic scoring ONLY (seed all randomness; never wall-time/GPU). Genuinely open-ended:
  graded objective, no easy optimum, multiple viable strategies.
- Instantiate the spec faithfully: the spec's "mechanisms" MUST all shape the objective, the spec's
  "innovation_hook" MUST be what your strong solution exploits, and the generator MUST include the
  spec's trap cases (the obvious greedy approach lands far from strong on >=3 of the 10 cases).
- Write every required file INCLUDING the 4-tier ladder (trivial/greedy/strong/invalid).
- ACCEPTANCE NUMBERS (validation.json means, on top of harness PASS):
    strong - greedy >= ${MIN_STRONG_MINUS_GREEDY}   (the insight visibly beats the recipe)
    strong <= ${MAX_STRONG}                (leave score headroom above the reference; rescale the
                                    checker baseline if strong saturates)
    greedy - trivial >= ${MIN_GREEDY_MINUS_TRIVIAL}  (the ladder is sane)
- Run the harness; if verdict != PASS or the numbers miss, fix the failing gate and re-run.
  Iterate up to 6 rounds. Report the FINAL verdict and metrics truthfully — they are re-checked.
- Keep the statement <= ~700 words, time limit 2-5s, each .in <= 5 MB.

STEP FINAL — independent Codex review (REQUIRED, after harness PASS + acceptance numbers):
  ${CODEX_REVIEW(dir)}
If it ends with CODEX VERDICT: DEFECTS — fix every REAL defect it cites, re-run the harness, and
run the review once more (max 2 review rounds total). If a cited defect is wrong, rebut it in your
notes instead of changing code. Set codex_verdict in your result JSON to the FINAL verdict
('LGTM' or 'DEFECTS'). If the codex CLI itself errors (auth/network/timeout), set
codex_verdict='UNAVAILABLE', note the error, and finish normally — do not block on it.`
}

function repairPrompt(s, prev) {
  const r = route(s.format)
  const dir = `${SYNTH}/problems/${s.id}`
  const why = prev && prev.metrics
    ? `Previous attempt: verdict=${prev.verdict}, metrics=${JSON.stringify(prev.metrics)} — ` +
      `failed harness PASS and/or the innovation acceptance (need strong-greedy>=${MIN_STRONG_MINUS_GREEDY}, strong<=${MAX_STRONG}, greedy-trivial>=${MIN_GREEDY_MINUS_TRIVIAL}).`
    : `Previous attempt failed or returned nothing.`
  return `Problem ${s.id} (format ${s.format}) needs repair. ${why}
Read its spec:
  python3 -c "import json;print([l for l in open('${SEEDS}') if json.loads(l)['id']=='${s.id}'][0])"
Read ${SYNTH}/${r.brief} AND ${SYNTH}/AGENT_BRIEF_INNOVATION_ADDENDUM.md, inspect ${dir}/ and its
validation.json, diagnose, and FIX. Typical fixes: make strong exploit the innovation_hook for real
(not greedy+iterations); add planted/trap generator cases; rescale the checker baseline so strong
lands in [greedy+0.06, 0.92]. Re-run:
  python3 ${SYNTH}/${r.val} ${dir}${s.format === 'B' ? '' : ' --keep-testdata'}
Iterate up to 6 rounds.${prev && prev.codex_verdict === 'DEFECTS' ? `
The previous attempt FAILED the independent Codex review — its cited defects are in the previous
notes: ${JSON.stringify(String(prev.notes || '').slice(0, 1500))}. Fix those first.` : ''}
After the harness PASSES with the acceptance numbers, run the independent Codex review (REQUIRED):
  ${CODEX_REVIEW(dir)}
Fix real cited defects and re-review (max 2 rounds); set codex_verdict to the final verdict, or
'UNAVAILABLE' if the codex CLI errors. Return the result JSON with FINAL truthful verdict+metrics.`
}

const results = await pipeline(
  specs,
  (s) => agent(genPrompt(s), { label: `gen:${s.id}`, phase: 'Generate', schema: RESULT_SCHEMA, model: MODEL, effort: EFFORT }),
  (res, s) => {
    if (accepted(res)) return res
    return agent(repairPrompt(s, res), { label: `fix:${s.id}`, phase: 'Repair', schema: RESULT_SCHEMA, model: MODEL, effort: EFFORT })
  },
)

const clean = results.filter(Boolean)
const ok = clean.filter(accepted)
const passedOnly = clean.filter(r => r.verdict === 'PASS' && !accepted(r))
const cx = { lgtm: 0, defects: 0, unavailable: 0 }
for (const r of clean) {
  if (r.codex_verdict === 'LGTM') cx.lgtm++
  else if (r.codex_verdict === 'DEFECTS') cx.defects++
  else cx.unavailable++
}
log(`done: ${ok.length}/${specs.length} accepted; codex reviews LGTM=${cx.lgtm} DEFECTS=${cx.defects} UNAVAILABLE=${cx.unavailable}`)
return {
  total: specs.length,
  accepted: ok.map(r => r.id),
  codex_reviews: cx,
  pass_low_headroom: passedOnly.map(r => ({ id: r.id, metrics: r.metrics })),
  failed: clean.filter(r => !accepted(r) && r.verdict !== 'PASS').map(r => ({ id: r.id, notes: r.notes })),
  unresolved_codex_defects: clean.filter(r => r.verdict === 'PASS' && r.codex_verdict === 'DEFECTS').map(r => r.id),
  results: clean,
}