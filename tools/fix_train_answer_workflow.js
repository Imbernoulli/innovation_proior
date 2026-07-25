export const meta = {
  name: 'fix-train-answer',
  description: 'Fix the train_answer.md code-integrity defects: (A) invented never-executed code where answer.md has none — execute-verify-keep, repair, or replace with the proper final-artifact ending; (B) silent rewrites where canonical answer.md code exists — swap back verbatim + fix prose seams; (C) missing train_answer.md — backfill per discovery-writeup skill; (D) named claim-vs-deliverable breaks. Per-unit self-commit.',
  whenToUse: 'The train_answer 编造代码 repair track. 2026-07-23 the build-level gate (sft/build_sft.py + tools/ta_gate_check.py) stopped these files from being TRAINED by falling back to answer.md — a stopgap that left 589/1184 method rows landing as a structured doc (only 30% end on the deliverable, vs 94% for the prose channel) and left the fabricated code in the release artifact. 2026-07-25 rescan (tmp/ta_repair_units.json): 204 A + 337 B + 11 C, after 37 near-verbatim units were restored mechanically. Success is objective: the unit must PASS tools/ta_gate_check.py afterwards, which puts its write-up back in the trained answer channel. Only train_answer.md is editable (D units excepted); context/reasoning/answer frozen.',
  phases: [
    { title: 'Fix', detail: 'one agent per unit: verbatim-swap / code-removal + artifact ending / backfill, gate-check, self-commit', model: 'sonnet' },
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
  required: ['slug', 'cls', 'action', 'gate_pass', 'execution_ran', 'execution_summary', 'bugs_found', 'committed', 'commit_hash', 'notes'],
  properties: {
    slug: { type: 'string' },
    cls: { type: 'string', enum: ['A', 'B', 'C', 'D', 'T'] },
    gate_pass: { type: 'boolean', description: 'true iff `python3 tools/ta_gate_check.py <slug>` printed PASS on the FINAL committed file' },
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

START by running \`cd ${REPO} && python3 tools/ta_gate_check.py ${slug}\` — it names the exact fence that fails and why. Then READ ${dir}/train_answer.md and ${dir}/answer.md IN FULL (these two are the whole job). Read ${dir}/reasoning.md and ${dir}/context.md as far as you need to settle any claim you touch — always read them if you are removing code or rewriting an ending.

WHY THIS MATTERS (do not skip): the answer channel of the SFT sample is train_answer.md, but ONLY if it passes the code-integrity gate; today this file fails it, so the build trains a structured answer.md document instead — losing the write-up's flowing-prose register and, in 70% of those rows, losing the "end on the deliverable" landing. Your fix is what puts this method's write-up back into training. So the unit is DONE only when \`python3 tools/ta_gate_check.py ${slug}\` prints PASS.

GROUND RULES (all classes):
- FINISH WITH THE GATE: run \`cd ${REPO} && python3 tools/ta_gate_check.py ${slug}\` on the final file and report gate_pass. Never satisfy it by gaming (splitting one block into sub-200-char fences, truncating code to a stub, switching to ~~~ or indented blocks): the only legitimate passes are "every code fence is byte-identical to an answer.md fence" or "the write-up has no large code fence at all".
- PROVENANCE BEFORE VERDICT: before calling a block invented, grep \`methods/${slug}/code/\` and \`methods/${slug}/notes/\` — a few of these blocks were drafted there and are real work, which changes how you treat their content (it does not change the gate: only answer.md counts as canonical).
- ONLY ${dir}/train_answer.md may be modified (class D units get their own explicit file list). context.md / reasoning.md / answer.md are FROZEN — verify with git status at the end that nothing else changed.
- train_answer.md is a scientist's final write-up (see .claude/skills/discovery-writeup/SKILL.md — read it if you change more than the code block): continuous prose, no section headers, LaTeX math, English, in-frame (no "the paper"/"the authors"/citations), no meta-commentary, first-person confident voice, three movements (analysis → named method in mechanism detail → final code or final artifact).
- PROTECT deliberately-added richness: prose that discusses alternatives, limitations, hedges, edge cases or fallback options is intentional training content — NEVER delete it just because it is "extra". Only touch text that is tied to the specific defect you are fixing (a false claim, a reference to code you changed/removed).
- Never leave a prose claim dangling: if you change or remove code, every sentence that referred to it (its name, its outputs, its complexity, "here is a compact implementation…") must be reconciled.
- Never put fabricated numbers in the file: any specific numeric output claimed in prose must either come from a run you actually did now, or already be grounded in the frozen source files.
- Executing code: write temp files under ${REPO}/tmp/ta_exec/${slug}/ (mkdir -p), run with a timeout (e.g. \`timeout 180 python3 …\`), never commit temp files, and remove that temp dir when done. numpy/scipy/sympy/torch(+CUDA) are available. Downsize demo parameters for speed ONLY in your test harness, never in the committed file.
- COMMIT this unit only (never git add -A): \`git -C ${REPO} add -- <exact file(s) you changed>\` then \`git -C ${REPO} commit -q -m "fix train_answer:${slug} — <action>" -m "<one-line evidence: what was wrong, what you verified>"\`. If commit fails on .git/index.lock (parallel agents), wait 3s and retry, up to 5 times. Put the short hash in commit_hash. If you changed nothing, do not commit.`

function promptA(u) {
  u = { ...u, note: u.note || 'answer.md has no canonical code block; the write-up ends on a fabricated one' }
  const dir = `${REPO}/methods/${u.slug}/results`
  return `Fix ONE train_answer.md with INVENTED code (class A). ${u.note}.

${COMMON(u.slug, dir)}

THE DEFECT: this method's answer.md delivers no runnable code (it is a theorem / analysis / protocol method${u.note.includes('THEOREM/TEXT') ? '; note the offending fence here is a theorem/text block rather than a program' : ''}), yet train_answer.md ends with a code block written from thin air at write-up time — it matches nothing in answer.md, was never executed, and this class has a confirmed history of hard bugs (wrong finite-difference directions, demos that "verify" identities that hold by construction). The deliverable of a theoretical discovery is the theorem/formula/protocol, not an illustrative script invented after the fact.

YOUR JOB — remove the un-canonical code and land the write-up on its real artifact:
1. Establish what this method's ACTUAL final artifact is, from answer.md: the clean theorem statement with its hypotheses, the final formula with its symbols defined, the protocol/algorithm as the field states it, the bound with its constants.
2. DELETE the invented code block, and rewrite the ENDING so the write-up lands on that artifact — stated precisely and completely enough to stand as the deliverable (this is the discovery-writeup skill's sanctioned ending for non-computational discoveries). Use LaTeX for the math. Smooth the lead-in sentence so nothing dangles.
3. Preserve everything else. The analysis, the mechanism explanation, the alternatives-considered and limitations prose are deliberate training content — you are replacing a fabricated ending with a real one, not shortening the file. If the deleted block carried a genuine idea the prose did not already state (e.g. "the estimator is computed by sorting then prefix-summing"), say it in prose instead of losing it.
4. If — and only if — you conclude this is really a COMPUTATIONAL method whose implementation is load-bearing and answer.md merely failed to include it, do NOT invent a landing and do NOT keep the unverified block: return action="skipped" with a precise diagnosis (what answer.md delivers, what the block does, why you think it must stay) and leave the file untouched for human review. Executing the block is only worth your time in that dispute case.
5. Self-check the final file against the GROUND RULES, run the gate check, git-status check, commit. action=code-removed-artifact-ending.

Return the structured result with cls="A". Put in bugs_found any real defect you noticed in the removed block (evidence for the audit record), empty if you did not inspect it that closely.`
}

function promptB(u) {
  u = { ...u, note: u.note || 'the write-up re-implemented the method instead of copying answer.md' }
  const dir = `${REPO}/methods/${u.slug}/results`
  const extraOnly = u.note.includes('EXTRA invented block')
  return `Fix ONE train_answer.md whose code silently DIVERGED from the canonical answer.md code (class B). ${u.note}.

${COMMON(u.slug, dir)}

THE DEFECT: the discovery-writeup contract says the code in train_answer.md is COPIED VERBATIM from answer.md (the reviewed canonical implementation) — this guarantees the trained write-up never diverges from the reviewed deliverable and never introduces a new bug. This file violates it: ${extraOnly ? 'its primary block IS the verbatim copy, but an additional invented block was appended' : 'the write-up author re-implemented the method from scratch instead of copying'}.

YOUR JOB:
${extraOnly ? `- The primary verbatim block stays untouched. Apply the class-A verify-by-execution procedure to the EXTRA block only: run it, judge faithfulness, keep-if-verified / minimally-repair / delete-and-reconcile-prose.` : `1. Identify the PRIMARY method implementation in answer.md — the block that defines the method itself. Skip pure driver/experiment scaffolding and output-log fences; if answer.md has one main implementation plus sibling variants, take the main one (variants only if the prose presents them as part of the method).
2. REPLACE the train_answer implementation with that canonical code VERBATIM — byte-for-byte, no cleanup, no renaming, no trimming. After writing, diff the block against answer.md's block and confirm identical.
3. FIX THE PROSE SEAMS: every reference in the surrounding prose to function/class names, signatures, hyperparameter values, printed outputs, or structure of the old rewrite must now match the canonical code. Keep the prose's explanatory substance — you are re-aiming its references, not shortening it.
4. IF THE CANONICAL CODE IS A LARGE MULTI-PART SCAFFOLD: do not re-summarize it in your own code. Quote the core of it — a CONTIGUOUS excerpt of a canonical block is still verbatim and still passes the gate, and two separated regions may be quoted as two fences. Choose the region(s) that define the method; drop pure driver/plotting scaffolding. action=replaced-verbatim either way.`}
5. Self-check the final file against the GROUND RULES (no dangling references to the old code), run the gate check, git-status check, commit.

Return the structured result with cls="B". List in bugs_found any real semantic defects you noticed in the old rewritten code (evidence for the audit), empty if it was a faithful re-implementation.`
}

function promptC(u) {
  u = { ...u, note: u.note || 'train_answer.md is MISSING; the build falls back to answer.md' }
  const dir = `${REPO}/methods/${u.slug}/results`
  return `BACKFILL one missing train_answer.md (class C). ${u.note}.

${COMMON(u.slug, dir)}

READ ${REPO}/.claude/skills/discovery-writeup/SKILL.md IN FULL and follow it exactly: write ${dir}/train_answer.md as the scientist's own final write-up — (1) the analysis summarized, (2) the method named and its mechanism explained in real detail, (3) the final code or final artifact. Grounded ENTIRELY in context.md / reasoning.md / answer.md — invent nothing, no web research, no new numbers.

CODE RULE (this is the entire reason this track exists — do not repeat the old mistake):
- If answer.md contains a runnable primary implementation → copy it VERBATIM, byte-for-byte, and diff to confirm.
- If answer.md has NO runnable code (theorem/analysis/protocol method) → end with the precise final artifact (clean theorem statement, final formula, protocol) exactly as the field would present it. DO NOT write illustration code from memory. Never invent an implementation.

Self-check against the skill's checklist (continuous prose, no headers, LaTeX math, in-frame, no meta-commentary), run the gate check, git-status check, commit. Return the structured result with cls="C", action="backfilled" (or "skipped" with notes if the source files are too degenerate to support an honest write-up).`
}

function promptT(u) {
  // trajectory rung: files are trajectories/<task>/<NN-name>-{answer,train_answer}.md
  const dir = `${REPO}/trajectories/${u.task}`
  const ta = `${dir}/${u.rung}`
  const ans = ta.replace('-train_answer.md', '-answer.md')
  return `Fix ONE trajectory rung whose train_answer code DIVERGED from the rung's answer (class T). Task ${u.task}, rung file ${u.rung}.

Working dir: ${REPO}.

START by reading ${ta} and ${ans} IN FULL, plus the rung's reasoning and its feedback file (same NN- prefix in ${dir}) as far as you need.

THE DEFECT: on a ladder, the rung's ANSWER is the artifact that was actually run and scored — the feedback file reports its measured numbers. The rung's train_answer.md is the write-up presented as the model's output for that rung, and its code must be the SAME code, copied verbatim. This one silently re-implemented or edited it, so the write-up ships code that was never the thing measured (the audit found rungs where the edited copy even carried out-of-frame citations inside docstrings).

YOUR JOB:
1. Replace the diverging code fence(s) in ${ta} with the corresponding fence(s) from ${ans}, byte-for-byte. A contiguous excerpt of a canonical block is still verbatim; splicing separate regions into one fence is not (quote them as separate fences instead).
2. Reconcile the surrounding prose: function/class names, hyperparameters, printed numbers, structure — every reference must match the code now present. Keep the prose's explanatory substance and any alternatives/limitations discussion; you are re-aiming references, not shortening.
3. Any measured number the write-up states must match the rung's feedback file. Never invent numbers.
4. ONLY ${ta} may be modified. The answer, reasoning, feedback and agentic.txt files are FROZEN (agentic.txt is a transcript of the real run — it is not supposed to track the write-up).
5. Verify with \`cd ${REPO} && python3 tools/ta_gate_check.py --path ${dir}\` is NOT applicable here; instead confirm by diffing that every large fence in ${ta} appears verbatim in ${ans} (e.g. a short python check), and report that as gate_pass.
6. COMMIT only that file: \`git -C ${REPO} add -- trajectories/${u.task}/${u.rung}\` then \`git -C ${REPO} commit -q -m "fix train_answer:${u.task}/${u.rung} — restore verbatim rung code" -m "<one-line evidence>"\`. Retry up to 5 times on .git/index.lock.

Return the structured result with cls="T", slug="${u.task}/${u.rung}", action="replaced-verbatim" (or "skipped" with a precise diagnosis if the answer file has no usable canonical block). List real semantic defects of the old rewrite in bugs_found.`
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
  else if (u.cls === 'T') { p = promptT(u); effort = 'high' }
  else if (u.cls === 'B') { p = promptB(u); effort = 'high' }
  else if (u.cls === 'C') { p = promptC(u); model = 'opus' }   // writing a whole write-up from scratch
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
const gated = results.filter(r => r.gate_pass).length
log(`done: gate PASS ${gated}/${results.length} — artifact-ending ${by('code-removed-artifact-ending').length}, swapped ${by('replaced-verbatim').length}, backfilled ${by('backfilled').length}, kept ${by('kept-verified').length}, repaired ${by('repaired-verified').length}, reconciled ${by('claims-reconciled').length}, skipped ${by('skipped').length}, error ${by('error').length}`)
return {
  gate_pass: gated,
  gate_fail: results.filter(r => !r.gate_pass).map(r => ({ slug: r.slug, cls: r.cls, action: r.action, notes: r.notes })),
  summary: Object.fromEntries(['kept-verified','repaired-verified','code-removed-artifact-ending','replaced-verbatim','distillation-verified','backfilled','claims-reconciled','skipped','error'].map(a => [a, by(a).length])),
  bugs: results.filter(r => (r.bugs_found || []).length).map(r => ({ slug: r.slug, bugs: r.bugs_found })),
  units: results.map(r => ({ slug: r.slug, cls: r.cls, action: r.action, hash: r.commit_hash, committed: r.committed })),
  skipped: by('skipped').map(r => ({ slug: r.slug, notes: r.notes })),
}
