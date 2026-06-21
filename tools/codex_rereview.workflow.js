export const meta = {
  name: 'codex-rereview',
  description: 'Re-run the write-enabled Codex review gate on methods whose prior review failed/was skipped (Codex logout window)',
  phases: [{ title: 'Re-review', detail: 'one agent per method: genuine Codex review-and-fix + verify + marker' }],
}
const REPO = '/srv/home/bohanlyu/innovation_proior'
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const SLUGS = A.slugs || []
const CHUNK = A.chunk || 4
const ATTEMPTS = A.attempts || 4
const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { slug: { type: 'string' }, ok: { type: 'boolean' }, outcome: { type: 'string' }, notes: { type: 'string' } },
  required: ['slug', 'ok', 'outcome'],
}
function prompt(slug) {
  return `Run a GENUINE write-enabled Codex review on the existing method trace at \`${REPO}/methods/${slug}/results/\` (context.md, reasoning.md, answer.md). Its prior Codex review did not complete (the Codex CLI was logged out); Codex is logged in again now.

1. Resolve the companion: \`ls ~/.claude/plugins/cache/*/codex/*/scripts/codex-companion.mjs\`.
2. Run: \`node "<path>" task "Review AND FIX in place the paper-to-reasoning deliverables for ${slug} at ${REPO}/methods/${slug}/results/. Verify math/derivation correctness (signs, constants, every case), then code faithfulness to the canonical reference implementation, then posterior/hindsight leaks, then scaffold purity and in-frame voice. Output a file:line changelog." --write --model gpt-5.5 --effort xhigh\`
3. **Confirm Codex actually ran** (read its stdout/changelog; if it errored with auth/usage/logout, RETRY once, then report ok:false outcome:"not_run"/"failed" — do NOT fake it).
4. Re-verify its changelog yourself (diff the files; confirm the math fixes are correct and it didn't reintroduce markdown headers in reasoning.md, CJK, or target-paper references; confirm the answer python block still ast-parses).
5. Write \`${REPO}/methods/${slug}/results/.codex_review.json\` = {"method":"${slug}","codex_reviewed":true,"outcome":"completed","reviewed_at":"<UTC ISO8601>","reviewer":"gpt-5.5","effort":"xhigh","evidence":"this-run-rereview"} ONLY if the review genuinely completed; otherwise write outcome:"failed"/"not_run" honestly.

Do NOT edit methods.json/trajectories.json/other methods/git. Only touch \`methods/${slug}/\`. Return the schema: slug, ok (review genuinely completed), outcome, notes (what Codex fixed).`
}
async function runChunked(items, mk) {
  const out = new Array(items.length).fill(null)
  let pend = items.map((_, i) => i)
  for (let a = 0; a < ATTEMPTS && pend.length; a++) {
    const todo = pend; pend = []
    for (let i = 0; i < todo.length; i += CHUNK) {
      const g = todo.slice(i, i + CHUNK)
      const r = await parallel(g.map((j) => () => mk(items[j])))
      g.forEach((j, k) => { out[j] = r[k]; if (!r[k]) pend.push(j) })
    }
    log(`Re-review pass ${a + 1}: ${out.filter(Boolean).length}/${items.length} returned, ${pend.length} retrying`)
  }
  return out
}
phase('Re-review')
log(`Re-reviewing ${SLUGS.length} methods (chunk=${CHUNK})`)
const res = (await runChunked(SLUGS, (s) =>
  agent(prompt(s), { label: `recodex:${s}`, phase: 'Re-review', schema: SCHEMA, agentType: 'general-purpose' })
)).filter(Boolean)
log(`Done: ${res.filter((r) => r.ok).length}/${SLUGS.length} genuinely re-reviewed`)
return { rereviewed: res }
