export const meta = {
  name: 'data-quality-fix',
  description: 'Targeted polish pass over units that carry a concrete quality signal: non-English prose, generation-pipeline framing, year anachronism, scripted bug drama, never-audited units, and suspected-trivial competitive-programming units (judged, never deleted by the agent).',
  whenToUse: '2026-07-25 targeted sweep. NOT a full-corpus re-audit — coverage is already ~1181/1184 methods; these are the units a mechanical scan flagged. Agents fix in place and self-commit; TRIV units are judged only and reported back for a human deletion decision.',
  phases: [
    { title: 'Polish', detail: 'one agent per flagged unit: confirm the signal, repair it in place, self-commit', model: 'sonnet' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
A = A || {}
// `trim: ["slug", ...]` is shorthand for the v4 input-give-away class (same prompt for all of them).
const expanded = [...(A.units || []), ...(A.trim || []).map(id => ({ cls: 'TRIM', id, kind: 'v4', note: 'the Background declares the winning approach and/or pre-computes the trap this unit is built around' }))]
const units = expanded.slice(A.start ?? 0, (A.start ?? 0) + (A.limit ?? 100000))
log(`data-quality-fix: ${units.length} units`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['unit', 'cls', 'signal_real', 'action', 'evidence', 'files_changed', 'committed', 'commit_hash', 'notes'],
  properties: {
    unit: { type: 'string' },
    cls: { type: 'string', enum: ['LANG', 'FRAME', 'YEAR', 'DRAMA', 'COVER', 'TRIV'] },
    signal_real: { type: 'boolean', description: 'was the flagged defect actually there' },
    action: { type: 'string', enum: ['fixed', 'no-defect', 'recommend-delete', 'needs-human'] },
    evidence: { type: 'array', items: { type: 'string' }, description: 'verbatim quotes that justify signal_real / the verdict' },
    files_changed: { type: 'array', items: { type: 'string' } },
    committed: { type: 'boolean' },
    commit_hash: { type: 'string' },
    notes: { type: 'string', description: 'one or two sentences; for TRIV, the triviality judgement and why' },
  },
}

const FRAME_RULE = `IN-FRAME RULE (the whole corpus obeys it): the writer is a researcher living at that moment, working the problem for the first time. They never mention the writing task, the prompt, the source paper being reconstructed, the files of the generation pipeline, or anything they could not know yet. They write English.`

const COMMIT = (u, files) => `COMMIT only the files you changed: \`git -C ${REPO} add -- ${files}\` then \`git -C ${REPO} commit -q -m "<slug>: <what you fixed>" -m "<one-line evidence>"\`. Retry up to 5 times on .git/index.lock. If you change nothing, do not commit.`

const PRESERVE = `PRESERVE THE CONTENT. You are repairing a specific defect, not rewriting the unit. Do not shorten, do not remove the alternatives-considered / limitations / dead-end passages (they are deliberate training content), do not touch code inside fences unless the defect IS in the code (then keep the change minimal and mirror it into every file that carries the same block byte-identically — answer.md and train_answer.md must stay byte-identical in their code).`

function prompt(u) {
  const md = `${REPO}/methods/${u.id}/results`
  const vd = `${REPO}/data_v4/${u.id}`
  if (u.cls === 'LANG') return `This unit's prose is partly or wholly in Chinese; the corpus is English. Unit: methods/${u.id}.

Files: ${md}/context.md, ${md}/reasoning.md, ${md}/answer.md, ${md}/train_answer.md. ${u.note || ''}

${FRAME_RULE}

YOUR JOB: translate every non-English passage into English, in the register of the surrounding corpus — first-person, flowing, technical, in-frame. This is a translation, not a summary: every argument, hedge, dead end and caveat in the Chinese text must survive in the English. Keep LaTeX, symbols, numbers and code exactly as they are. If a file is entirely Chinese, translate the whole file. Check all four files (a unit often has Chinese in only some of them).

${PRESERVE}

Afterwards run \`cd ${REPO} && python3 tools/ta_gate_check.py ${u.id}\` and confirm it still PASSes, and grep the four files for any remaining CJK characters. ${COMMIT(u, `methods/${u.id}/results/`)}

Return cls="LANG", action="fixed" (or "no-defect" if there is genuinely no non-English text).`

  if (u.cls === 'FRAME') return `This unit's trained text refers to the GENERATION PIPELINE instead of staying in frame. Unit: ${u.kind === 'v4' ? `data_v4/${u.id}` : `methods/${u.id}`}. Flagged phrase: ${u.note}

Files: ${u.kind === 'v4' ? `${vd}/context.md, ${vd}/reasoning.md, ${vd}/train_answer.md` : `${md}/context.md, ${md}/reasoning.md, ${md}/answer.md, ${md}/train_answer.md`}.

${FRAME_RULE}

YOUR JOB: find every phrase in the trained text (reasoning + answer) that talks about the artifact being produced rather than the science — "it is in the answer file", "this trace should show", references to context.md / answer.md / the task instructions / the reader — and rewrite it so the sentence says the same technical thing from inside the frame (e.g. "…; the full module is the answer file." -> "…; that is the whole module."). Also fix any template-sounding opening that describes the writing plan rather than the problem.

${PRESERVE}

${COMMIT(u, u.kind === 'v4' ? `data_v4/${u.id}/` : `methods/${u.id}/results/`)}

Return cls="FRAME".`

  if (u.cls === 'YEAR') return `Possible ANACHRONISM. Unit: methods/${u.id}. This method's moment is ${u.year}; a scan found later year-like numbers in its trained text: ${u.note}.

Files: ${md}/context.md, ${md}/reasoning.md, ${md}/answer.md, ${md}/train_answer.md.

${FRAME_RULE}

YOUR JOB:
1. FIRST decide whether the signal is real. Most flags are false: a bare number (n = 2000, a 1024-dimensional embedding, a constant 1997), a bound, a problem size, a year the writer legitimately knows because it is in their past, or a year inside quoted data. Set signal_real accordingly and quote what you found.
2. If it IS a real anachronism — the writer cites work, results, terminology or fame from after ${u.year} — fix it: drop the forward reference, or re-express the point using only what was available then. If the whole passage depends on later knowledge, cut the passage down to the part that stands on its own. Never invent a substitute citation.

${PRESERVE}

${COMMIT(u, `methods/${u.id}/results/`)}

Return cls="YEAR", action="fixed" or "no-defect".`

  if (u.cls === 'DRAMA') return `Possible SCRIPTED BUG DRAMA. Unit: ${u.kind === 'v4' ? `data_v4/${u.id}` : `methods/${u.id}`}.

Files: ${u.kind === 'v4' ? `${vd}/context.md, ${vd}/reasoning.md, ${vd}/train_answer.md` : `${md}/context.md, ${md}/reasoning.md, ${md}/answer.md, ${md}/train_answer.md`}.

THE PATTERN WE ARE HUNTING: a generated trace that stages a fake mistake — the writer "deliberately" introduces or overlooks a bug, then catches it a paragraph later, so the trace looks self-correcting without any real uncertainty ever existing. It reads as theatre: the error is announced as intentional, the catch is immediate and clean, nothing about the surrounding derivation actually changes. Training on it teaches performing self-correction rather than doing it.

NOT the pattern (leave these alone): a genuine wrong turn the writer really took and had to undo; a real edge case discovered by testing; a considered decision that a simpler approach is deliberately chosen; any use of the word "deliberately" for an actual design choice.

YOUR JOB: read the trace and decide. If it is genuine, action="no-defect" and quote the passage that convinced you. If it is staged, rewrite that passage so the reasoning is honest: either the writer simply gets it right and says why the tempting wrong version fails, or the difficulty is stated as a real uncertainty resolved by a real check. Keep the technical content and the length; remove the theatre, not the thinking.

${PRESERVE}

${COMMIT(u, u.kind === 'v4' ? `data_v4/${u.id}/` : `methods/${u.id}/results/`)}

Return cls="DRAMA".`

  if (u.cls === 'TRIM') return `The INPUT of this competitive-programming unit gives away the solution. Unit: data_v4/${u.id}. Scan signals: ${u.note}.

Files: ${vd}/context.md (the input — this is what you edit), ${vd}/reasoning.md and ${vd}/train_answer.md (read them, edit only if something dangles after your trim).

THE PROBLEM: at evaluation the model receives a bare problem statement — story, constraints, I/O format, samples, limits — and must find the algorithm itself. Here the statement also contains a Background section that does the solver's work for it: it declares which approach works ("a single left-to-right sweep … suffices", "the canonical tool is Dijkstra"), or pre-computes the very trap the unit is built around ("a distance can reach 2*10^14, so it overflows a 32-bit int"), or the Evaluation-settings section states the answer for the tricky case. The trace then "discovers" what it was handed, so the sample teaches transcription instead of derivation, and trains the model to expect a hint that the real evaluator will not give.

YOUR JOB — trim the input back to a real problem statement:
1. DELETE from context.md: any sentence that picks the winning approach or calls it sufficient/canonical/intended; any pre-computed trap analysis (the overflow arithmetic, the precision bound, the resolution of an off-by-one or base case); any statement of "the open question is X" followed by its answer; and anything in Evaluation settings that reveals the expected output for a tricky case (keep that a plain description of how the solution is tested).
2. KEEP: the story and the exact task, all constraints, the I/O contract, the samples, the time/memory limits, and any genuinely generic field background that does NOT resolve this instance (naming a family of techniques that exist is fine; declaring which one solves this problem is not). Keep the section structure and the register — you are cutting sentences, not rewriting the document.
3. If a whole section exists only to hand over the approach, delete the section.
4. Then check ${vd}/reasoning.md and ${vd}/train_answer.md for anything that pointed AT the deleted text ("as the statement notes…", "the background already says…"). If found, rewrite that clause so the trace derives the point itself; otherwise leave both files untouched. The trace must still read as if it worked the problem out — it almost always already does.
5. Sanity: after the trim the statement must still fully specify the problem. Nothing about the required output may become ambiguous.

${COMMIT(u, `data_v4/${u.id}/`)}

Return cls="TRIM", action="fixed" (or "no-defect" if on reading it the Background really does not give anything away).`

  if (u.cls === 'TRIV') return `JUDGE ONLY — CHANGE NOTHING. Unit: data_v4/${u.id}. This competitive-programming unit is unusually thin (${u.note}); we need to know whether it is worth a training slot.

Files: ${vd}/context.md (the problem), ${vd}/reasoning.md (trained think), ${vd}/train_answer.md (trained answer, single-file C++).

The dataset trains a model to solve hard problems by reasoning. A unit earns its slot if solving the problem needs a real idea — an algorithmic insight, a non-obvious complexity argument, a correctness subtlety, an overflow/precision trap that a careful solver must see. It does NOT earn its slot if a competent programmer would type the answer straight from the statement with no decision to make, or if the "reasoning" is padding around a direct simulation of the statement.

Judge both the PROBLEM (is it trivial) and the TRACE (does it derive anything, or narrate the obvious). Quote the decisive evidence. action="recommend-delete" if trivial, "no-defect" if it earns its slot, "needs-human" if you cannot tell. Make NO edits and NO commits: committed=false, files_changed=[].

Return cls="TRIV".`

  return `AUDIT AND POLISH one unit that no previous quality pass has ever looked at. Unit: methods/${u.id}.

Files: ${md}/context.md (input), ${md}/reasoning.md (trained think), ${md}/answer.md (reviewed deliverable), ${md}/train_answer.md (trained answer). Read all four IN FULL.

${FRAME_RULE}

CHECK, in this order, and fix what you find:
1. Frame/meta: any reference to the writing task, the source paper being reconstructed, pipeline files, or the reader; any non-English prose; any citation of the method's own paper.
2. Anachronism: anything the writer could not know at their moment.
3. Honesty: claims that the delivered artifact does not support; numbers that appear nowhere in the sources; a "verification" that checks something true by construction; a derivation that argues for one method while the code implements another.
4. Code integrity: train_answer's code must be byte-identical to answer.md's (verify with \`cd ${REPO} && python3 tools/ta_gate_check.py ${u.id}\`, and it must PASS when you are done).
Then say in notes whether the unit teaches a real discovery at all, or is a recitation of a known result (do not delete it either way — report it).

${PRESERVE}

${COMMIT(u, `methods/${u.id}/results/`)}

Return cls="COVER", action="fixed" or "no-defect".`
}

const results = (await pipeline(units, u =>
  agent(prompt(u), { label: `${u.cls}:${u.id}`, phase: 'Polish', model: 'sonnet', effort: 'high', schema: SCHEMA, agentType: 'general-purpose' })
    .then(r => r ? { ...r, unit: u.id, cls: u.cls } : { unit: u.id, cls: u.cls, action: 'error' })
)).filter(Boolean)

const by = (f) => results.reduce((m, r) => { const k = `${r.cls}/${r[f]}`; m[k] = (m[k] || 0) + 1; return m }, {})
log(`actions ${JSON.stringify(by('action'))}`)
return {
  totals: by('action'),
  real_signal: results.filter(r => r.signal_real).length,
  recommend_delete: results.filter(r => r.action === 'recommend-delete').map(r => ({ unit: r.unit, notes: r.notes, evidence: r.evidence })),
  needs_human: results.filter(r => r.action === 'needs-human').map(r => ({ unit: r.unit, notes: r.notes })),
  fixed: results.filter(r => r.action === 'fixed').map(r => ({ unit: r.unit, cls: r.cls, hash: r.commit_hash, notes: r.notes })),
  no_defect: results.filter(r => r.action === 'no-defect').map(r => ({ unit: r.unit, cls: r.cls, notes: r.notes })),
}
