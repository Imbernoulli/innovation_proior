export const meta = {
  name: 'recover-from-history',
  description: 'Recover the audit-edit results that were lost to an erroneous git checkout. Each agent reads the ORIGINAL edit agent\'s full transcript (which recorded every read/Edit/Bash/codex step it took), reproduces that agent\'s FINAL edited file state faithfully onto the clean HEAD baseline, then commits the unit so it can never be lost again.',
  whenToUse: 'One-off: restore the ~519 audit-edited units whose working-tree edits were reverted before being committed. Transcript replay by program is unreliable (Bash heredocs / codex subprocess writes are not in the Write/Edit tool stream), so a model reads the whole history and reconstructs intent.',
  phases: [
    { title: 'Recover', detail: 'one Sonnet(xhigh) agent per unit: read the original transcript in full, reproduce its final edited reasoning onto clean HEAD, sanity-check, then git add+commit only this unit', model: 'sonnet' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
A = A || {}

// Units come either DIRECTLY as args.unitList, or from args.waveFile (a JSON file
// {"unitList":[...]}). For waveFile we load it via a Bash `cat` (a plain file dump does NOT
// truncate the way "read and structure this 130-entry JSON" does) and JSON.parse the raw string.
// Each entry: {unitDir, kind, unit, pathspec, transcript, newWords?, verdict?}
const limit = A.limit ?? 100000
let units = Array.isArray(A.unitList) ? A.unitList : []
if (!units.length && A.waveFile) {
  const raw = await agent(
    `Run: cat ${A.waveFile}\nReturn the file's EXACT raw contents as {json: "<contents>"} — do not summarize, truncate, reformat, or parse it; copy every byte verbatim into the json string.`,
    { label: 'load-wave', phase: 'Recover', effort: 'low', schema: { type: 'object', additionalProperties: false, required: ['json'], properties: { json: { type: 'string' } } } })
  if (raw && raw.json) {
    try { units = (JSON.parse(raw.json).unitList) || [] } catch (e) { log(`waveFile parse error: ${e}`); units = [] }
  }
}
units = units.slice(0, limit)
log(`recover-from-history: ${units.length} units`)

const REC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['unitDir', 'ok', 'files_written', 'new_words', 'matches_reported', 'committed', 'commit_detail', 'notes'],
  properties: {
    unitDir: { type: 'string' }, ok: { type: 'boolean' },
    files_written: { type: 'array', items: { type: 'string' } },
    new_words: { type: 'integer' },
    matches_reported: { type: 'boolean', description: 'true if reconstructed word count is within ~6% of the original agent\'s reported new_words' },
    committed: { type: 'boolean' }, commit_detail: { type: 'string' },
    notes: { type: 'string' },
  },
}
const DONE_SCHEMA = { type: 'object', additionalProperties: false, required: ['done'], properties: { done: { type: 'boolean' }, detail: { type: 'string' } } }

function recoverPrompt(u) {
  return `You are RECOVERING one unit of audit-edit work that was lost before it could be committed. Working dir: ${REPO}.

Unit: ${u.unitDir}  (kind=${u.kind})
Allowed pathspec (edit ONLY files under here): ${u.pathspec}
Original edit agent's FULL transcript (JSONL, one message per line): ${u.transcript}
The original agent reported new_words ≈ ${u.newWords ?? 'unknown'} after its edit.

WHAT HAPPENED: an earlier agent edited this unit's reasoning (cutting ritual filler / hindsight /
pasted answer code, fixing real defects) and it PASSED review, but the working-tree changes were
reverted by a mistaken \`git checkout\` before being committed. The current files under ${u.pathspec}
are back at the clean HEAD baseline (the PRE-edit state). Your job: reproduce the original agent's
FINAL edited state and commit it.

HOW TO RECOVER (the transcript is your source of truth — it is large, read it carefully):
1. Read ${u.transcript}. It is the original agent's message log. Walk it IN ORDER and extract every
   mutation it made to files under ${u.pathspec}:
   - "tool_use" entries named "Write" carry {file_path, content} — a full-file overwrite.
   - "tool_use" entries named "Edit" carry {file_path, old_string, new_string, replace_all}.
   - "tool_use" entries named "Bash" may ALSO write files (heredocs like \`cat > f <<'EOF'\`,
     redirects \`... > f\`, \`sed -i\`, \`mv /tmp/x f\`). READ these commands and account for their
     writes too — this is exactly what a naive replay misses.
   - A "codex exec" Bash call may have caused the codex subprocess to edit files; if so, the agent's
     LATER steps (its git diff dumps, its final Read of the file, its grep output) reveal the
     resulting text. Use those later observations to pin the true final content.
   The transcript often contains, near its end, the agent re-reading the file or dumping
   \`git diff\` / \`wc -w\` — use that as ground truth for the final state.
2. Reconstruct each edited file's FINAL text and Write it in place under ${u.pathspec}. Start from
   the current HEAD baseline (read the file as it is now) and apply the reconstructed final state.
   For a trajectory ladder this may be several NN-*-reasoning.md files; for a method it is
   results/reasoning.md (and possibly results/train_answer.md if the transcript shows a claim fix);
   for v4 it is reasoning.md.
3. FROZEN INVARIANTS (must hold in your result, regardless of what the transcript shows):
   - traj: edit ONLY NN-*-reasoning.md; answers/train_answers/feedback/meta are frozen.
   - method: reasoning.md always; train_answer.md ONLY if the original changed its PROSE for a
     claim-vs-deliverable fix, and its code blocks stay byte-identical; context.md/answer.md frozen.
   - v4: reasoning.md only; context.md/train_answer.md/verify frozen.
   If the transcript appears to have touched a frozen file, do NOT reproduce that; keep the frozen
   file at HEAD and note it.
4. SANITY CHECK before committing: run \`wc -w\` on your reconstructed reasoning file(s) and compare
   the total to the reported new_words (${u.newWords ?? 'unknown'}). Set matches_reported=true if
   within ~6%. If it is wildly off (e.g. you reproduced far more or far less), you probably missed a
   Bash-write or a codex edit — re-read the transcript tail and fix before committing. Also confirm
   \`git -C ${REPO} diff --name-only\` lists ONLY files under ${u.pathspec}.
5. COMMIT this unit and nothing else. MANY AGENTS RUN IN PARALLEL against the same git index, so
   you MUST serialize with flock, and you MUST scope the commit to your pathspec (otherwise you can
   sweep a sibling agent's staged files into your commit):
     flock /tmp/recover_git.lock bash -c 'cd ${REPO} && git add -- ${u.pathspec} && (git diff --cached --quiet -- ${u.pathspec} || git commit -q -o -- ${u.pathspec} -m "recover audit-edit ${u.kind}:${u.unit}")'
   The "-o/--only <pathspec>" form commits ONLY your paths even if other files are staged.
   NEVER use "git add -A" / "git add ." / a pathspec-less "git commit". Put the commit hash (or
   NOTHING-STAGED) in commit_detail. Afterwards verify with:
     git -C ${REPO} status --porcelain -- ${u.pathspec}   (must be empty)

Return the structured result. If you genuinely cannot reconstruct the final state with confidence,
set ok=false, do NOT write garbage, restore with \`git -C ${REPO} checkout -- ${u.pathspec}\`, and
explain in notes what was ambiguous (that unit will be redone from scratch instead).`
}

async function recoverUnit(u) {
  const label = `${u.kind}:${u.unit}`
  const rec = await agent(recoverPrompt(u), { label: `recover:${label}`, phase: 'Recover', model: 'sonnet', effort: 'xhigh', schema: REC_SCHEMA, agentType: 'general-purpose' })
  if (!rec) return { unitDir: u.unitDir, kind: u.kind, status: 'agent_error' }
  if (!rec.ok || !rec.committed) {
    // ensure no half-written garbage is left staged/dirty for this unit
    await agent(`Run: git -C ${REPO} reset -q -- ${u.pathspec} ; git -C ${REPO} checkout -- ${u.pathspec}\nConfirm "git -C ${REPO} diff -- ${u.pathspec}" and "git -C ${REPO} diff --cached -- ${u.pathspec}" are both empty. Return done=true only if clean.`,
      { label: `clean:${label}`, phase: 'Recover', effort: 'low', schema: DONE_SCHEMA })
    return { unitDir: u.unitDir, kind: u.kind, status: rec.ok ? 'not_committed' : 'reconstruct_failed', rec }
  }
  return { unitDir: u.unitDir, kind: u.kind, status: rec.matches_reported ? 'recovered' : 'recovered_wordcount_off', rec }
}

const results = await pipeline(units, u => recoverUnit(u))
const done = results.filter(Boolean)
const ok = done.filter(r => r.status === 'recovered' || r.status === 'recovered_wordcount_off')
const clean = ok.filter(r => r.status === 'recovered')
const off = ok.filter(r => r.status === 'recovered_wordcount_off')
log(`recover done: ${ok.length}/${units.length} committed (${clean.length} word-count-clean, ${off.length} word-count-off); ${done.length - ok.length} failed/redo`)
return {
  recovered: ok.map(r => ({ unit: r.unitDir, status: r.status, hash: r.rec.commit_detail, new_words: r.rec.new_words })),
  wordcount_off: off.map(r => ({ unit: r.unitDir, new_words: r.rec.new_words, notes: r.rec.notes })),
  needs_redo: done.filter(r => !ok.includes(r)).map(r => ({ unit: r.unitDir, status: r.status, notes: (r.rec && r.rec.notes) || '' })),
}
