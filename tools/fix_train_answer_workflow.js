export const meta = {
  name: 'fix-train-answer',
  description: 'Fix the train_answer.md code-integrity defects: (A) invented never-executed code where answer.md has none — execute-verify-keep, repair, or replace with the proper final-artifact ending; (B) silent rewrites where canonical answer.md code exists — swap back verbatim + fix prose seams; (C) missing train_answer.md — backfill per discovery-writeup skill; (D) named claim-vs-deliverable breaks. Per-unit self-commit.',
  whenToUse: 'The train_answer 编造代码 specialist track from the reasoning-bloat audit (rescanned 2026-07-23: 218 A + 330 B + 23 C + 3 D). NOTE 2026-07-23: class A is now handled at BUILD level instead — sft/build_sft.py bypasses to answer.md for slugs in sft/use_answer_for_theory.txt while their train_answer still carries a code fence (theory methods should deliver the theorem/formula, not invented code). Run agents only for B/C/D. Only train_answer.md is editable (D units excepted); context/reasoning/answer frozen.',
  phases: [
    { title: 'Fix', detail: 'one agent per unit: verify-by-execution / verbatim-swap / backfill, self-commit', model: 'sonnet' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
A = A || {}
const units = (A.units || []).slice(A.start ?? 0, (A.start ?? 0) + (A.limit ?? 100000))
log(`fix-train-answer: ${units.length} units`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'cls', 'action', 'execution_ran', 'execution_summary', 'bugs_found', 'committed', 'commit_hash', 'notes'],
  properties: {
    slug: { type: 'string' },
    cls: { type: 'string', enum: ['A', 'B', 'C', 'D'] },
    action: { type: 'string', enum: ['kept-verified', 'repaired-verified', 'code-removed-artifact-ending', 'replaced-verbatim', 'distillation-verified', 'backfilled', 'claims-reconciled', 'skipped'] },
    execution_ran: { type: 'boolean', description: 'true if you actually executed code this run' },
    execution_summary: { type: 'string', description: 'what you ran and what it showed, one or two lines; "n/a" if nothing needed running' },
    bugs_found: { type: 'array', items: { type: 'string' }, description: 'each real defect found in the original block (empty if none)' },
    committed: { type: 'boolean' },
    commit_hash: { type: 'string', description: 'short hash or "none"' },
    notes: { type: 'string' },
  },
}

const COMMON = (slug, dir) => `Working dir: ${REPO}. Unit: ${slug}.

READ IN FULL first: ${dir}/context.md, ${dir}/reasoning.md, ${dir}/answer.md (if present), ${dir}/train_answer.md (if present).

GROUND RULES (all classes):
- ONLY ${dir}/train_answer.md may be modified (class D units get their own explicit file list). context.md / reasoning.md / answer.md are FROZEN — verify with git status at the end that nothing else changed.
- train_answer.md is a scientist's final write-up (see .claude/skills/discovery-writeup/SKILL.md — read it if you change more than the code block): continuous prose, no section headers, LaTeX math, English, in-frame (no "the paper"/"the authors"/citations), no meta-commentary, first-person confident voice, three movements (analysis → named method in mechanism detail → final code or final artifact).
- PROTECT deliberately-added richness: prose that discusses alternatives, limitations, hedges, edge cases or fallback options is intentional training content — NEVER delete it just because it is "extra". Only touch text that is tied to the specific defect you are fixing (a false claim, a reference to code you changed/removed).
- Never leave a prose claim dangling: if you change or remove code, every sentence that referred to it (its name, its outputs, its complexity, "here is a compact implementation…") must be reconciled.
- Never put fabricated numbers in the file: any specific numeric output claimed in prose must either come from a run you actually did now, or already be grounded in the frozen source files.
- Executing code: write temp files under ${REPO}/tmp/ta_exec/${slug}/ (mkdir -p), run with a timeout (e.g. \`timeout 180 python3 …\`), never commit temp files, and remove that temp dir when done. numpy/scipy/sympy/torch(+CUDA) are available. Downsize demo parameters for speed ONLY in your test harness, never in the committed file.
- COMMIT this unit only (never git add -A): \`git -C ${REPO} add -- <exact file(s) you changed>\` then \`git -C ${REPO} commit -q -m "fix train_answer:${slug} — <action>" -m "<one-line evidence: what was wrong, what you verified>"\`. If commit fails on .git/index.lock (parallel agents), wait 3s and retry, up to 5 times. Put the short hash in commit_hash. If you changed nothing, do not commit.`

function promptA(u) {
  const dir = `${REPO}/methods/${u.slug}/results`
  return `Fix ONE train_answer.md with INVENTED code (class A). ${u.note}.

${COMMON(u.slug, dir)}

THE DEFECT: this method's answer.md delivers no runnable code (it is a theorem / analysis / protocol method${u.note.includes('pseudocode') ? ', its fences are pseudocode/math only' : ''}), yet train_answer.md ends with a Python block that was written from thin air at write-up time — it matches nothing in reasoning.md or answer.md, was never executed, and this class has a confirmed history of hard bugs (the audit's sample had wrong finite-difference directions). It is trained model output, so a buggy block trains fabrication.

YOUR JOB — verify-by-execution, keep what earns its place:
1. RUN the block exactly as committed (copy to temp, timeout, capture output). Record what happened.
2. JUDGE FAITHFULNESS, not just exit-code: does the code implement what the prose and the method's actual math say? Check the load-bearing semantics against answer.md/reasoning.md (signs, directions, normalizations, the inequality/quantity it claims to demonstrate). A demo can run cleanly and still compute the wrong thing.
3. DECIDE:
   - Runs clean AND faithful AND any specific numbers in prose match reality → keep it. action=kept-verified.
   - Real but repairable defect (crash, wrong sign/direction, off-by-one, misleading output) → make the MINIMAL fix, re-run until it verifies, reconcile any prose numbers. action=repaired-verified. List every defect in bugs_found.
   - Unsalvageable, wrong-headed, or not honestly verifiable (e.g. it purports to "check" something it cannot check) → DELETE the code block and rewrite the ENDING of the write-up into the precise final artifact this method actually has — the clean theorem statement, the final formula, the protocol — drawn strictly from answer.md (this is the discovery-writeup skill's sanctioned ending for non-computational discoveries). Smooth the lead-in sentence. action=code-removed-artifact-ending.
   The bar for keeping: the demo must genuinely illustrate or check the method's real content. Do NOT delete a correct, working, on-topic demonstration merely to satisfy formality — added value that is verified stays.
4. Self-check the final file against the GROUND RULES, git-status check, commit.

Return the structured result with cls="A".`
}

function promptB(u) {
  const dir = `${REPO}/methods/${u.slug}/results`
  const extraOnly = u.note.includes('EXTRA invented block')
  return `Fix ONE train_answer.md whose code silently DIVERGED from the canonical answer.md code (class B). ${u.note}.

${COMMON(u.slug, dir)}

THE DEFECT: the discovery-writeup contract says the code in train_answer.md is COPIED VERBATIM from answer.md (the reviewed canonical implementation) — this guarantees the trained write-up never diverges from the reviewed deliverable and never introduces a new bug. This file violates it: ${extraOnly ? 'its primary block IS the verbatim copy, but an additional invented block was appended' : 'the write-up author re-implemented the method from scratch instead of copying'}.

YOUR JOB:
${extraOnly ? `- The primary verbatim block stays untouched. Apply the class-A verify-by-execution procedure to the EXTRA block only: run it, judge faithfulness, keep-if-verified / minimally-repair / delete-and-reconcile-prose.` : `1. Identify the PRIMARY method implementation in answer.md — the block that defines the method itself. Skip pure driver/experiment scaffolding and output-log fences; if answer.md has one main implementation plus sibling variants, take the main one (variants only if the prose presents them as part of the method).
2. REPLACE the train_answer implementation with that canonical code VERBATIM — byte-for-byte, no cleanup, no renaming, no trimming. After writing, diff the block against answer.md's block and confirm identical.
3. FIX THE PROSE SEAMS: every reference in the surrounding prose to function/class names, signatures, hyperparameter values, printed outputs, or structure of the old rewrite must now match the canonical code. Keep the prose's explanatory substance — you are re-aiming its references, not shortening it.
4. NARROW EXCEPTION (only if the unit note says answer is a LARGE scaffold): if the canonical code is a long multi-part experiment scaffold and the existing write-up block is a focused distillation of its core method, you MAY keep the distillation instead — but then you must (a) line-by-line reconcile it against the canonical code and remove every semantic divergence (defaults, update rules, normalizations), and (b) actually execute the distilled block cleanly. If either fails or you are unsure, swap verbatim. action=distillation-verified in that case, otherwise replaced-verbatim.`}
5. Self-check the final file against the GROUND RULES (no dangling references to the old code), git-status check, commit.

Return the structured result with cls="B". List in bugs_found any real semantic defects you noticed in the old rewritten code (evidence for the audit), empty if it was a faithful re-implementation.`
}

function promptC(u) {
  const dir = `${REPO}/methods/${u.slug}/results`
  return `BACKFILL one missing train_answer.md (class C). ${u.note}.

${COMMON(u.slug, dir)}

READ ${REPO}/.claude/skills/discovery-writeup/SKILL.md IN FULL and follow it exactly: write ${dir}/train_answer.md as the scientist's own final write-up — (1) the analysis summarized, (2) the method named and its mechanism explained in real detail, (3) the final code or final artifact. Grounded ENTIRELY in context.md / reasoning.md / answer.md — invent nothing, no web research, no new numbers.

CODE RULE (this is the entire reason this track exists — do not repeat the old mistake):
- If answer.md contains a runnable primary implementation → copy it VERBATIM, byte-for-byte, and diff to confirm.
- If answer.md has NO runnable code (theorem/analysis/protocol method) → end with the precise final artifact (clean theorem statement, final formula, protocol) exactly as the field would present it. DO NOT write illustration code from memory. Never invent an implementation.

Self-check against the skill's checklist (continuous prose, no headers, LaTeX math, in-frame, no meta-commentary), git-status check, commit. Return the structured result with cls="C", action="backfilled" (or "skipped" with notes if the source files are too degenerate to support an honest write-up).`
}

const D_UNITS = [
  {
    slug: 'negative-weight-sssp', cls: 'D',
    prompt: () => `Fix a CLAIM-vs-DELIVERABLE break (class D). Unit: methods/negative-weight-sssp.

${COMMON('negative-weight-sssp', `${REPO}/methods/negative-weight-sssp/results`)}

THE DEFECT (from the 2026-07-16 audit): reasoning.md spends most of its length deriving the Bernstein–Nanongkai–Wulff-Nilsen near-linear negative-weight SSSP algorithm (O(m log^8 n)-flavored), but the code actually delivered in answer.md/train_answer.md is a much simpler algorithm with a worse worst-case bound; train_answer.md's prose still presents the near-linear method as what is being delivered. Training on this teaches "claim A, deliver B".

YOUR JOB: read all four files carefully and establish exactly what the delivered code implements and what its true worst-case complexity is. Then edit ONLY the PROSE of train_answer.md so the claims match the deliverable: the write-up may still present the near-linear derivation as the analysis/insight story, but the description OF THE DELIVERED CODE (what it implements, its complexity, its guarantees) must be true of that code. Code bytes in every file are FROZEN. Do not delete the derivation richness — re-scope the claims. If you find the claims are actually already consistent (a prior pass may have fixed it), verify thoroughly and return action="skipped" with notes. Otherwise action="claims-reconciled". cls="D".`,
  },
  {
    slug: 'cpv4-geometry-basic-boundary', cls: 'D',
    prompt: () => `Fix a delivered-code-fails-its-own-constraints defect (class D). Unit: data_v4/cpv4-geometry-basic-boundary (note: NOT under methods/; files are data_v4/cpv4-geometry-basic-boundary/{context.md,reasoning.md,train_answer.md} and a verify/ dir with gen.py, brute.py, sol.cpp).

${COMMON('cpv4-geometry-basic-boundary', `${REPO}/data_v4/cpv4-geometry-basic-boundary`)}

THE DEFECT (audit row 1306): the code delivered in train_answer.md is O(n^2 log n) and would TLE at the problem's stated constraints, while the prose self-defends with "fine for the intended scale". The verify/ dir has a real harness (gen.py generator, brute.py reference, sol.cpp).

YOUR JOB:
1. Read context.md for the actual constraints; read verify/ to understand the harness; establish empirically (build & run) whether the delivered code really violates the constraints (generate a max-size case, time it).
2. If it genuinely TLEs: implement the properly-complexity-correct solution, validate it with the harness (correctness vs brute.py on many random cases including edge cases, AND timing at max constraints), then replace the code block in train_answer.md with the verified solution and reconcile the complexity claims in the prose. If reasoning.md's final complexity claims directly contradict the new code, you may make the MINIMAL consistency edits to reasoning.md's claims (this unit only — keep its derivation content intact). For this unit train_answer.md AND (minimally) reasoning.md are editable; context.md and verify/ are frozen.
3. If it does NOT actually TLE at the real constraints: fix nothing but the dishonest framing if any, and say so in notes.
4. If you cannot produce a verified faster solution, DO NOT land an unverified one: return action="skipped" with a precise diagnosis in notes.
action="repaired-verified" (or per above). cls="D".`,
  },
  {
    slug: 'causal-observational-nonlinear-04-cam', cls: 'D',
    prompt: () => `Fix a comment-vs-code contradiction (class D). Unit: trajectories/causal-observational-nonlinear rung 04 (CAM). Files: trajectories/causal-observational-nonlinear/04-cam-answer.md, 04-cam-train_answer.md, and agentic.txt in the same dir if it repeats the code.

${COMMON('causal-observational-nonlinear-04-cam', `${REPO}/trajectories/causal-observational-nonlinear`)}

THE DEFECT (audit row 1608): a code comment says "highest marginal variance" while the implementation right under it uses argmin (picks the LOWEST). The scored/verified behavior is the CODE's — so the fix is to correct the COMMENT to describe what the code does (check CAM's actual semantics in the rung's reasoning/feedback to phrase it right), never to change code logic.

YOUR JOB: locate every occurrence of the wrong comment across 04-cam-answer.md, 04-cam-train_answer.md, and agentic.txt; fix the comment text only (code logic, whitespace, everything else byte-identical); verify with git diff that only comment lines changed. For this unit those files are editable; all other files frozen. action="repaired-verified" (or "skipped" if a prior pass already fixed it). cls="D".`,
  },
]

async function runUnit(u) {
  let p, model = 'sonnet', effort = 'xhigh'
  if (u.cls === 'A') p = promptA(u)
  else if (u.cls === 'B') { p = promptB(u); effort = 'high' }
  else if (u.cls === 'C') p = promptC(u)
  else { p = u.prompt(); model = undefined; effort = 'xhigh' }
  const opts = { label: `${u.cls}:${u.slug}`, phase: 'Fix', effort, schema: SCHEMA, agentType: 'general-purpose' }
  if (model) opts.model = model
  const r = await agent(p, opts)
  if (!r) return { slug: u.slug, cls: u.cls, action: 'error' }
  return { ...r, slug: u.slug, cls: u.cls }
}

const all = A.includeD ? [...units, ...D_UNITS] : units
const results = (await pipeline(all, u => runUnit(u))).filter(Boolean)
const by = (a) => results.filter(r => r.action === a)
log(`done: kept ${by('kept-verified').length}, repaired ${by('repaired-verified').length}, artifact-ending ${by('code-removed-artifact-ending').length}, swapped ${by('replaced-verbatim').length}, distilled ${by('distillation-verified').length}, backfilled ${by('backfilled').length}, reconciled ${by('claims-reconciled').length}, skipped ${by('skipped').length}, error ${by('error').length}`)
return {
  summary: Object.fromEntries(['kept-verified','repaired-verified','code-removed-artifact-ending','replaced-verbatim','distillation-verified','backfilled','claims-reconciled','skipped','error'].map(a => [a, by(a).length])),
  bugs: results.filter(r => (r.bugs_found || []).length).map(r => ({ slug: r.slug, bugs: r.bugs_found })),
  units: results.map(r => ({ slug: r.slug, cls: r.cls, action: r.action, hash: r.commit_hash, committed: r.committed })),
  skipped: by('skipped').map(r => ({ slug: r.slug, notes: r.notes })),
}
