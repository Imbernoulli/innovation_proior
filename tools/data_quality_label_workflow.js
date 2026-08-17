export const meta = {
  name: 'data-quality-label',
  description: 'Label every SFT unit for three defect classes: (1) meta-reasoning about producing the artifact instead of doing the science, (2) inconsistency with the dataset premise (nothing is actually discovered), (3) triviality. Labels only — no edits, no commits.',
  whenToUse: '2026-07-25 full-corpus quality sweep. Read-only labelling pass over methods / v4 / trajectory units; results are aggregated by the caller, who then decides what to fix and what to delete. Nothing is deleted or edited by these agents.',
  phases: [
    { title: 'Label', detail: 'one agent per unit: read the full unit, return a structured judgement with verbatim evidence', model: 'sonnet' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
A = A || {}
const units = (A.units || []).slice(A.start ?? 0, (A.start ?? 0) + (A.limit ?? 100000))
log(`data-quality-label: ${units.length} units`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['unit', 'kind', 'meta_reasoning', 'meta_evidence', 'premise_fit', 'premise_failure',
             'premise_evidence', 'substance', 'substance_reason', 'language', 'other_defect',
             'verdict', 'verdict_reason', 'confidence'],
  properties: {
    unit: { type: 'string' },
    kind: { type: 'string', enum: ['method', 'v4', 'traj'] },
    meta_reasoning: { type: 'string', enum: ['none', 'minor', 'severe'],
      description: 'does the trained text reason about PRODUCING the artifact rather than about the science' },
    meta_evidence: { type: 'array', items: { type: 'string' }, description: 'verbatim quotes, empty if none' },
    premise_fit: { type: 'string', enum: ['fits', 'weak', 'inconsistent'] },
    premise_failure: { type: 'string', enum: ['none', 'answer_given_in_context', 'recitation_no_derivation',
      'fake_innovation', 'unearned_success', 'claim_deliverable_mismatch', 'hindsight_or_anachronism', 'other'] },
    premise_evidence: { type: 'array', items: { type: 'string' } },
    substance: { type: 'string', enum: ['substantive', 'thin', 'trivial'] },
    substance_reason: { type: 'string' },
    language: { type: 'string', enum: ['english', 'mixed', 'non_english'] },
    other_defect: { type: 'string', description: 'anything else that would actively teach the model something wrong; "none" if nothing' },
    verdict: { type: 'string', enum: ['keep', 'fix', 'delete'] },
    verdict_reason: { type: 'string', description: 'one or two sentences' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const PREMISE = `WHAT THIS DATASET IS FOR (judge against this, nothing else):
It trains a model to DO RESEARCH — to take a real situation where something is not yet known, reason its way to a new method, and deliver that method as a working artifact. One sample = an input describing the situation (what is known, what is broken, what is available), a long first-person <think> deriving the method, and an answer that presents the method and lands on the deliverable (runnable code, or for a theoretical result the theorem/formula/protocol).
The sample is written IN-FRAME: the writer is a researcher at that moment in time who does not yet know the answer. They are NOT a student summarizing a known paper, and NOT an assistant completing a writing assignment.`

const RUBRIC = `LABEL THREE THINGS. Be strict about evidence: every non-clean label needs a VERBATIM quote.

(1) meta_reasoning — reasoning about PRODUCING THE ARTIFACT instead of doing the science.
 severe: the text discusses the writing task itself — what the trace/write-up should contain, the source paper or its sections being reconstructed, the prompt/task instructions, files of the generation pipeline, the reader of the document, or narrates "now I will present/structure the answer" as its organizing move.
 minor: one or two stray phrases of that kind in an otherwise scientific trace.
 none: the whole text is the researcher thinking about the problem.
 NOT meta_reasoning: ordinary mathematical register ("we are asked to bound X", "let me verify"), a method whose SUBJECT is reasoning traces / assistants / prompts (deepseek-r1, ReAct, tool agents — technical usage), or in-frame references to the writer's own derivation ("this trace of the argument").

(2) premise_fit — does the sample actually teach discovery?
 inconsistent (give the failure code):
  - answer_given_in_context: the input already names the method or hands over the key formula/algorithm, so nothing is discovered — the trace only implements what it was told.
  - recitation_no_derivation: the think restates a known result and its properties without any real derivation: no wrong turn considered and rejected for a reason, no constraint that forces the design, no alternative weighed. A polished exposition is still recitation.
  - fake_innovation: the "new method" is a rename or a cosmetic variant of what the input already had, or the improvement is asserted rather than reasoned.
  - unearned_success: every step works the first time and every self-check confirms what was just claimed; nothing is ever measured against something that could have refuted it.
  - claim_deliverable_mismatch: the derivation argues for one thing, the delivered artifact is a different (usually weaker) thing, and the write-up still claims the first.
  - hindsight_or_anachronism: the writer uses knowledge from after their own moment (later results, later papers, the method's eventual fame, numbers they could not have).
 weak: it fits the premise but the discovery content is shallow relative to the length.
 fits: a genuine derivation.

(3) substance — is it worth a training slot?
 trivial: an expert would produce the answer immediately with no design decision to make; the derivation has no branch point; the deliverable is a one-liner or a restatement of the input. DELETE-worthy.
 thin: real but slight — a small lemma, a routine baseline, a variant that changes one constant.
 substantive: there is a genuine problem being solved.
 Judge by CONTENT, not length. A short exact argument can be substantive; a long routine one can be thin. Competitive-programming units (v4) are substantive if the solution needs a real algorithmic idea, trivial if it is a direct simulation of the statement.

Also report: language (english / mixed / non_english — the corpus is English; Chinese or other non-English prose in the input or the trained text is a defect) and other_defect (anything else that would actively teach something wrong: fabricated numbers, contradictory claims, a deliverable that cannot run, evaluation artifacts, leaked identifiers).

VERDICT: keep (usable as-is), fix (has a specific repairable defect — say what in verdict_reason), delete (teaches nothing, or teaches something wrong that cannot be repaired without rewriting the science).
Default to keep. Do not invent defects to look thorough; a clean unit labelled clean is a useful result. Do not judge prose style, formatting, or length.`

function unitPrompt(u) {
  let files, extra = ''
  if (u.kind === 'method') {
    const d = `${REPO}/methods/${u.id}/results`
    files = `${d}/context.md (the input), ${d}/reasoning.md (the trained <think>), ${d}/train_answer.md (the trained answer). ${d}/answer.md is the reviewed reference deliverable — read it to check claims, it is not itself trained.`
  } else if (u.kind === 'v4') {
    const d = `${REPO}/data_v4/${u.id}`
    files = `${d}/context.md (the problem statement given as input), ${d}/reasoning.md (the trained <think>), ${d}/train_answer.md (the trained answer, a single-file C++ solution).`
    extra = `\nThis is a competitive-programming unit: the system prompt asks for one self-contained C++ program reading stdin. Judge the DERIVATION (is there a real algorithmic idea, is the debugging genuine or a scripted two-act drama where a planted bug is found and fixed) and whether the problem itself is trivial.`
  } else {
    const d = `${REPO}/trajectories/${u.id}`
    files = `${d}/meta.json (the ladder: each rung's reasoning / answer / feedback files), ${d}/00-initial-context.md or the initial_context_file named in meta.json (the input), and for EVERY rung its *-reasoning.md (trained think), *-train_answer.md (trained answer) and *-feedback.md (the measured result fed back).`
    extra = `\nThis is a multi-rung ladder: the researcher proposes a method, gets measured feedback, and proposes the next. Judge the ladder as a whole. Specific things to check: does each rung's proposal actually follow from the previous feedback, or is the next method simply announced; does any rung claim an improvement the feedback does not show; is every rung a success (nothing ever fails or gets retracted).`
  }
  return `Label ONE unit of the SFT corpus for data quality. Unit: ${u.kind}:${u.id}.

Working dir: ${REPO}. READ IN FULL: ${files}${extra}

${PREMISE}

${RUBRIC}

Read everything before judging — do not label from the first page. You may run a quick grep/python check if you want to confirm something concrete (e.g. whether a number in the write-up matches the feedback file). Change NO files and commit NOTHING: this is a labelling pass only.

Return the structured result with unit="${u.id}", kind="${u.kind}".`
}

const results = (await pipeline(units, u =>
  agent(unitPrompt(u), { label: `${u.kind}:${u.id}`, phase: 'Label', model: 'sonnet', effort: 'high', schema: SCHEMA, agentType: 'general-purpose' })
    .then(r => r ? { ...r, unit: u.id, kind: u.kind } : { unit: u.id, kind: u.kind, verdict: 'error' })
)).filter(Boolean)

const count = (f) => results.reduce((m, r) => { const k = r[f] || 'error'; m[k] = (m[k] || 0) + 1; return m }, {})
log(`verdicts ${JSON.stringify(count('verdict'))} | meta ${JSON.stringify(count('meta_reasoning'))} | premise ${JSON.stringify(count('premise_fit'))} | substance ${JSON.stringify(count('substance'))}`)
return {
  totals: { verdict: count('verdict'), meta_reasoning: count('meta_reasoning'), premise_fit: count('premise_fit'),
            premise_failure: count('premise_failure'), substance: count('substance'), language: count('language') },
  flagged: results.filter(r => r.verdict !== 'keep' || r.meta_reasoning !== 'none' || r.premise_fit !== 'fits'
                              || r.substance !== 'substantive' || r.language !== 'english'),
  all: results.map(r => ({ unit: r.unit, kind: r.kind, verdict: r.verdict, meta: r.meta_reasoning,
                           premise: r.premise_fit, failure: r.premise_failure, substance: r.substance,
                           language: r.language, confidence: r.confidence, reason: r.verdict_reason })),
}
