export const meta = {
  name: 'audit-edit-reasoning',
  description: 'Batch audit + in-place edit of existing reasoning traces: strip quota-padding / ritual self-checks / hindsight, fix real defects, AND deepen under-explained method insights (grounded in src/ or web sources — never invented). Sonnet xhigh editor self-reviews via codex (gpt-5.6-terra); orchestrator commits each unit to git as it lands so nothing is ever lost.',
  whenToUse: 'When methods/ or trajectories/ or data_v4/ reasoning carries useless filler and must be tightened WITHOUT a length target — content-driven editing, not lengthening.',
  phases: [
    { title: 'Edit', detail: 'one Sonnet(xhigh) agent per unit: classify every paragraph, cut filler, fix real defects (answers frozen), then self-review via codex exec (gpt-5.6-terra) and fix what it finds; return the FINAL verdict', model: 'sonnet' },
    { title: 'Commit', detail: 'orchestrator stages + commits each accepted unit immediately (per-unit durability; codex failures are git-restored)' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
A = A || {}

// ---------------- unit selection ----------------
const PRIORITY_PATH = A.unitsFile || `${REPO}/tmp/audit_priority.json`
const scope = A.scope || 'both'
const pilot = A.pilot !== false
const limitT = A.limitTraj ?? (pilot ? 4 : 400)
const limitM = A.limitMethods ?? (pilot ? 6 : 1400)
const limitV = A.limitV4 ?? (pilot ? 3 : 400)
const COMMIT_EVERY = A.commitEvery ?? 1   // commit after each accepted unit (max durability)

const loader = await agent(
  `Read ${PRIORITY_PATH} and return its JSON verbatim as your structured output (top ${limitT} trajectories, top ${limitM} methods, top ${limitV} v4 by 'priority' desc; if the file or an array is missing, return empty arrays for it).`,
  { label: 'load-priority', phase: 'Edit', effort: 'low', schema: {
      type: 'object', additionalProperties: false, required: ['trajectories', 'methods', 'v4'],
      properties: {
        trajectories: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['task'], properties: { task: { type: 'string' }, rungs: { type: 'integer' }, priority: { type: 'integer' } } } },
        methods: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['slug'], properties: { slug: { type: 'string' }, words: { type: 'integer' }, priority: { type: 'integer' } } } },
        v4: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['slug'], properties: { slug: { type: 'string' }, priority: { type: 'integer' } } } },
      } } })

let trajUnits = (A.tasks && A.tasks.length) ? A.tasks.map(t => ({ task: t })) : (loader ? loader.trajectories : [])
let methodUnits = (A.slugs && A.slugs.length) ? A.slugs.map(s => ({ slug: s })) : (loader ? loader.methods : [])
let v4Units = (A.v4slugs && A.v4slugs.length) ? A.v4slugs.map(s => ({ slug: s })) : (loader ? (loader.v4 || []) : [])
if (scope === 'methods') { trajUnits = []; v4Units = [] }
if (scope === 'trajectories') { methodUnits = []; v4Units = [] }
if (scope === 'v4') { trajUnits = []; methodUnits = [] }
trajUnits = trajUnits.slice(0, limitT)
methodUnits = methodUnits.slice(0, limitM)
v4Units = v4Units.slice(0, limitV)
log(`audit-edit: ${trajUnits.length} traj + ${methodUnits.length} methods + ${v4Units.length} v4 (pilot=${pilot}, commitEvery=${COMMIT_EVERY})`)

// ---------------- shared rubric ----------------
// Evidence base (2026-07-16..19 investigation, experiments/DATA_REASONING_BLOAT_AUDIT_zh.md):
//  - traj deepen (9819ba7a) filled EVERY rung to one length band (growth-vs-orig corr -0.74):
//    length set by quota, not content; ~30-40% of the added text is genuinely valuable.
//  - "Let me verify/check/trace" density x3.5; ~78-83% of those checks NEVER find anything.
//  - CONFIRMED hindsight: deepen rewrote "predictions" to numerically match a rung's own feedback.
//  - answer code pasted verbatim into reasoning tail: 24 traj + 244 method files.
//  - v4: 83% share the IDENTICAL opening sentence, 47% the same two-act bug play.
//  - padding is paraphrastic (12-gram dup ~0%) -> judge semantically, per paragraph.
const CUT_RULES = `
WHAT TO CUT (judge every paragraph):
  1. RITUAL SELF-CHECKS: "Let me verify/check/trace/make sure", "Before I commit/move on", blocks
     whose outcome carries NO risk (algebra that cannot fail, a trace that only re-states the
     construction, a check ending in bare confirmation). KEEP a check only if it catches an error
     and redirects, pins a nontrivial quantity USED later, or supports an otherwise-bare claim;
     then strip the announcement and keep only the computation.
  2. TEMPLATE FINGERPRINTS: "rather than assert/quote/guess", "before I trust", "with my own eyes",
     "worth pricing/sizing", "kill it with arithmetic", "keeps me honest", "limit check", and
     paragraph-ending self-congratulation ("That is the entire prize", "Good; the bookkeeping checks out").
  3. MOOT ALTERNATIVE MENUS: a walk through "my options" where the choice is predetermined by the
     scaffold/protocol/frozen answer and the menu never overturns anything — compress to one clause.
  4. DILUTION & RESTATED BACKGROUND: a point restated in different words; empty meta-narration; and
     (traj) re-deriving what an earlier rung already established (reference in one clause, never re-derive).
  5. ANSWER CODE PASTED INTO REASONING: a tail code block duplicating the frozen answer verbatim —
     cut it (a one-line pointer is fine); keep only fragments the reasoning works through line-by-line.
  6. FAKE STAKES / STAGED WALLS: manufactured drama around steps never in doubt.
  6b. TAIL RECAPS & STOCK TICS: methods' "causal chain" recap (~40% of files), traj "falsifiable
     signature" closer (59% of rungs). Keep at most ONE honest a-priori prediction sentence; cut the
     ceremony. Vary/cut stock tics ("what actually hurts", "There it is", bare "Wall.").
  6c. WITHIN-TRACE ARGUMENT REPLAY: the same point argued in full more than once — keep the strongest.
  7. HINDSIGHT DRESSED AS PREDICTION (MOST HARMFUL — actively hunt): "falsifiable predictions" whose
     numbers match later-revealed results to implausible precision, and "I expect X to bite hardest
     on Y" naming the later-measured worst case. Coarsen to what is honestly computable a priori
     (order-of-magnitude, direction, mechanism) or cut. For traj rung i, CROSS-CHECK every
     quantitative prediction against that rung's own NN-feedback.md — matching it beyond a-priori
     computability is an info-boundary violation.
WHAT TO KEEP / FIX (this is an EDIT, not a blanket shrink — ~30-40% of the enhancement additions are good):
  - PROTECTED, never strip: (a) FALLBACK DISCIPLINE (weigh "not converging -> ship the simpler
    correct approach"), (b) GENUINE ERROR-CORRECTION (a real mistake caught by a computation and
    corrected), (c) genuine dead-ends, quantitative eliminations, computed-on-page checks that bite,
    feedback-table digestion. When unsure if a check is ritual or real: KEEP and tighten, don't delete.
  - Every design decision in the final answer must still be MOTIVATED (the bridge). If a cut would
    orphan a decision, tighten instead.
  - FIX real defects: numbers inconsistent with context/feedback, hindsight leaks, destination
    pre-announcement, in-frame violations ("the paper"/"the authors"), dataset-artifact frame-breaks
    ("the reference implementation", "the deliverable", paper-internal "Lemma 10.4", corpus jargon
    "rung"/"scaffold" as meta) — rewrite into the discovery voice.
LENGTH: NO quota in either direction. Inflated traj rungs usually NET SHRINK; genuinely thin
single-turn traces may NET GROW but ONLY from real content (never compensate cuts with filler).
VOICE: first-person present-tense discovery prose, in-frame, no headers, English. Never add facts
that could not be known in-frame.`

const SELF_REVIEW = `

MANDATORY SELF-REVIEW (after writing edits, before returning):
1. Mechanical: run "git -C ${REPO} diff --name-only" — it must list ONLY files you were allowed to
   edit; grep your edited files for the ritual phrases above (every survivor must be load-bearing).
2. Independent review via Codex. Write the review prompt to a UNIT-SPECIFIC temp file (parallel
   agents must not share a path — use /tmp/review_<unit>.txt with your unit name), then run from
   the repo root:
     codex exec --skip-git-repo-check -m gpt-5.6-terra -c 'model_reasoning_effort="high"' "$(cat /tmp/review_<unit>.txt)"
   The prompt must give absolute paths to every edited file and the frozen answer/feedback, tell the
   reviewer to read the edited reasoning IN FULL and run "git -C ${REPO} diff" itself, and check:
   (a) NO LOAD-BEARING LOSS; (b) NO NEW DEFECTS (fabrication, in-frame violation, numbers
   contradicting frozen files, traj info-boundary); (c) HINDSIGHT SWEEP (surviving predictions vs
   the rung's own feedback); (d) NO VERBATIM ANSWER CODE in the reasoning tail; (e) if depth was
   ADDED: every added technical specific (constant, bound, construction, lineage claim) is grounded
   in src/ or one of the retrieved-source URLs you list in the review prompt — flag anything that
   smells written-from-memory as fabrication. Tell it to END with
   exactly one line "VERDICT: pass" / "VERDICT: minor" / "VERDICT: fail" then a bullet list.
3. Fix any real problems it raises, then re-run codex once more (max 2 rounds). Report the FINAL
   verdict as codex_verdict and any unfixed problems as codex_problems. If codex errors/times out
   twice, do the checks yourself and set codex_verdict="unavailable".
Return ok=false (or codex_verdict="fail" you could not fix) ONLY if the unit is worse than the
original — the orchestrator will then git-restore it.`

// ---------------- schema ----------------
const EDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['unit', 'ok', 'files_edited', 'orig_words', 'new_words', 'rituals_cut', 'deepened', 'sources', 'defects_fixed', 'codex_verdict', 'codex_problems', 'notes'],
  properties: {
    unit: { type: 'string' }, ok: { type: 'boolean' },
    files_edited: { type: 'array', items: { type: 'string' } },
    orig_words: { type: 'integer' }, new_words: { type: 'integer' },
    rituals_cut: { type: 'integer' },
    deepened: { type: 'boolean', description: 'true only if real depth was ADDED (methods: from src/ or retrieved sources); false for cut-only edits and for traj/v4 units' },
    sources: { type: 'array', items: { type: 'string' }, description: 'URLs actually retrieved and used for added depth; empty if deepened=false or depth came only from src/' },
    defects_fixed: { type: 'array', items: { type: 'string' } },
    codex_verdict: { type: 'string', enum: ['pass', 'minor', 'fail', 'unavailable'] },
    codex_problems: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}
const DONE_SCHEMA = { type: 'object', additionalProperties: false, required: ['done'], properties: { done: { type: 'boolean' }, detail: { type: 'string' } } }

// ---------------- prompts ----------------
function trajEditPrompt(task) {
  return `Audit and EDIT IN PLACE the reasoning files of ONE trajectory ladder. Task: ${task}. Working dir: ${REPO}.

CONTEXT: commit 9819ba7a deepened every rung reasoning here to a fixed ~20-26k char quota; much of the added text is filler.

READ IN ORDER (the whole ladder): ${REPO}/trajectories/${task}/00-initial-context.md, then each rung NN: NN-*-reasoning.md, NN-*-answer.md, NN-feedback.md. Read every rung BEFORE editing — cross-rung redundancy is a target. The pre-deepen leaner baseline is at "git diff 9819ba7a^ 9819ba7a -- trajectories/${task}/" (do NOT blindly revert to it).
${CUT_RULES}
KNOWN HITS: ${REPO}/tmp/answer_code_in_reasoning.json lists rung files already detected with near-verbatim answer code — handle each.
TRAJECTORY-SPECIFIC HARD RULES:
  - EDIT ONLY the NN-*-reasoning.md files. answers / train_answers / feedback / meta.json are FROZEN.
  - Info boundary per rung i: only initial context + rungs<i answers/feedback + rung i's own answer as landing; never its own feedback or later rungs.
  - Each rung's reasoning must still land exactly on its frozen answer.
  - Rung 2+ openings digesting the PREVIOUS feedback table is GOOD content (measured cross-rung redundancy is low); do not cut those. Cut WITHIN-rung replay and the template skeleton.
  - Deepest ladders overflow the 53760-token cutoff; cut harder on lm-nanogpt-speedrun / sys-* / dl-resnet-imagenet-speedup / optimization-multi-objective / stf-traffic / cv-cifar10-speedrun.
${SELF_REVIEW}`
}

function methodEditPrompt(slug) {
  return `Audit and EDIT IN PLACE one method reasoning trace. Method: ${slug}. Working dir: ${REPO}.

READ: ${REPO}/methods/${slug}/results/context.md, reasoning.md, answer.md, train_answer.md (if present).
${CUT_RULES}
KNOWN HITS: ${REPO}/tmp/answer_code_in_reasoning.json lists method reasoning files with near-verbatim answer code — if ${slug} is listed, handle that hit.
METHOD-SPECIFIC HARD RULES:
  - PRIMARY edit target is results/reasoning.md. context.md and answer.md are FROZEN.
  - BRIDGE CHECK: every key design decision in answer.md must be reached BY the reasoning, not asserted after it. If the reasoning says A but the answer does B, repair the path to genuinely derive B (no copying answer prose backwards).
  - CLAIM-vs-DELIVERABLE BREAK (most harmful): compare what reasoning/train_answer CLAIM vs what answer.md code actually IS. Fix reasoning to land honestly on the delivered artifact; ONLY for a contradicted claim may you edit train_answer.md PROSE — its code blocks stay byte-identical.
  - DEEPEN (CO-EQUAL SECOND JOB, not an afterthought — cutting bloat and deepening under-explained insight are the two halves of this edit): after cutting, judge whether this method has a REAL insight (a why-it-works, a non-obvious design choice, a natural failed attempt) that the current reasoning states flatly without earning. "Short" is NOT the trigger and "simple method" is NOT a disqualifier — a simple, effective method can deserve deep reasoning; conversely a trace that already conveys everything worth saying stays untouched. If under-explained:
      * with src/: ADD the real depth from ${REPO}/methods/${slug}/src/ (paper TeX incl. appendix) and notes/synthesis.md — do the algebra it states, never gesture at it.
      * without src/ (or src/ too thin for the gap): RESEARCH ONLINE FIRST — load WebSearch/WebFetch via ToolSearch("select:WebSearch,WebFetch"), find AUTHORITATIVE sources (original paper, author retrospective/talk, reputable expositions), and reconstruct with precision (1) the discovery-time CONTEXT (what was known, what was stuck) and (2) the DERIVATION PATH (motivating question, natural first attempts and exactly why they fail, the idea that breaks the impasse, why the final form works). Every technical specific you add must come from src/ or a source retrieved THIS RUN — record the URLs in the "sources" output field. If the sources you find do not support an honest reconstruction, DO NOT deepen — leave the trace as cut and say so in notes.
      * never invent facts, never toward a length target. Set "deepened" true only if you added real depth.
${SELF_REVIEW}`
}

function v4EditPrompt(slug) {
  return `Audit and EDIT IN PLACE one v4 competition-programming reasoning trace. Problem: ${slug}. Working dir: ${REPO}.

READ: ${REPO}/data_v4/${slug}/context.md, reasoning.md, train_answer.md (glance at verify/).

CONTEXT: the v4 batch shares one generator: 83% of traces open with the IDENTICAL sentence "**Reading the problem and pinning the contract.**", 90% contain "deliberately", 47% stage the same two-act bug play across unrelated problems. Forensics show these tics transfer verbatim into model generations. The verified C++ landing is the value; the theatrical wrapper is the defect.
${CUT_RULES}
V4-SPECIFIC HARD RULES:
  - EDIT ONLY reasoning.md. context.md, train_answer.md and verify/ are FROZEN.
  - REWRITE THE OPENING problem-specifically (this problem's constraint sizes, trap, I/O quirk) — never the shared template sentence, and not a new shared template.
  - THE BUG PLAY: cut generic theater; keep only a REAL problem-specific pitfall, stated and dodged, with no "convinced myself" reversal ceremony.
  - The reasoning must still land exactly on train_answer's frozen code, including its I/O contract.
  - Strip evaluation-harness artifacts (getenv/ALE_BASELINE, "// ale-NN", scaffold tokens) from the PROSE; code files are frozen.
${SELF_REVIEW}`
}

// ---------------- per-unit pipeline: edit(+self-review) -> commit or restore ----------------
function pathspecFor(kind, unit) {
  return kind === 'traj' ? `trajectories/${unit}/` : kind === 'v4' ? `data_v4/${unit}/` : `methods/${unit}/results/`
}

async function runUnit(kind, unit) {
  const label = `${kind}:${unit}`
  const prompt = kind === 'traj' ? trajEditPrompt(unit) : kind === 'v4' ? v4EditPrompt(unit) : methodEditPrompt(unit)
  const pathspec = pathspecFor(kind, unit)

  const edit = await agent(prompt, { label: `edit:${label}`, phase: 'Edit', model: 'sonnet', effort: 'xhigh', schema: EDIT_SCHEMA, agentType: 'general-purpose' })
  if (!edit) return { unit, kind, status: 'edit_error' }

  if (!edit.ok || edit.codex_verdict === 'fail') {
    await agent(`Run exactly: git -C ${REPO} checkout -- ${pathspec}\nThen confirm "git -C ${REPO} diff -- ${pathspec}" prints nothing. Return done=true only if restored clean.`,
      { label: `restore:${label}`, phase: 'Commit', effort: 'low', schema: DONE_SCHEMA })
    return { unit, kind, status: 'failed_restored', edit }
  }

  // COMMIT THIS UNIT NOW so it can never be lost to a later revert / limit / crash.
  const status = edit.codex_verdict === 'minor' ? 'minor' : edit.codex_verdict === 'unavailable' ? 'pass_unreviewed' : 'pass'
  const commit = await agent(
    `Commit ONLY this unit's edited files. Working dir: ${REPO}. Steps, in order:\n` +
    `1. git -C ${REPO} add -- ${pathspec}\n` +
    `2. git -C ${REPO} diff --cached --quiet && { echo "NOTHING-STAGED"; } || git -C ${REPO} commit -q -m "audit-edit ${label}: ${status} (codex ${edit.codex_verdict})" -m ${JSON.stringify((edit.defects_fixed || []).slice(0, 6).join('; ') || 'ritual/dilution cleanup')}\n` +
    `3. Return done=true if a commit was created OR nothing needed staging; done=false only if git errored. Put the short commit hash (or NOTHING-STAGED) in detail.\n` +
    `Do NOT add or commit anything outside ${pathspec}. Do NOT use "git add -A" or "git add .".`,
    { label: `commit:${label}`, phase: 'Commit', effort: 'low', schema: DONE_SCHEMA })
  return { unit, kind, status, edit, committed: !!(commit && commit.done), commit_detail: commit && commit.detail }
}

const allUnits = [
  ...trajUnits.map(u => ({ kind: 'traj', unit: u.task })),
  ...methodUnits.map(u => ({ kind: 'method', unit: u.slug })),
  ...v4Units.map(u => ({ kind: 'v4', unit: u.slug })),
]
const results = await pipeline(allUnits, u => runUnit(u.kind, u.unit))

// ---------------- summary ----------------
const done = results.filter(Boolean)
const okOnes = done.filter(r => ['pass', 'minor', 'pass_unreviewed'].includes(r.status))
const committed = okOnes.filter(r => r.committed)
const shrunk = okOnes.filter(r => r.edit && r.edit.new_words < r.edit.orig_words)
const grew = okOnes.filter(r => r.edit && r.edit.new_words > r.edit.orig_words)
const totOrig = okOnes.reduce((a, r) => a + ((r.edit && r.edit.orig_words) || 0), 0)
const totNew = okOnes.reduce((a, r) => a + ((r.edit && r.edit.new_words) || 0), 0)
log(`audit-edit done: ${okOnes.length}/${allUnits.length} accepted (${committed.length} committed); ${shrunk.length} shrank / ${grew.length} grew; words ${totOrig} -> ${totNew} (${totOrig ? Math.round((1 - totNew / totOrig) * 100) : 0}% delta)`)
return {
  accepted: okOnes.map(r => ({ unit: r.unit, kind: r.kind, status: r.status, committed: r.committed, hash: r.commit_detail, orig: r.edit.orig_words, new: r.edit.new_words, rituals_cut: r.edit.rituals_cut, deepened: r.edit.deepened, sources: r.edit.sources, defects: r.edit.defects_fixed })),
  uncommitted_accepted: okOnes.filter(r => !r.committed).map(r => ({ unit: r.unit, kind: r.kind })),
  failed: done.filter(r => !okOnes.includes(r)).map(r => ({ unit: r.unit, kind: r.kind, status: r.status, problems: (r.edit && r.edit.codex_problems) || [] })),
}
