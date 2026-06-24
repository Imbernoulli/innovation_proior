export const meta = {
  name: 'augment-verify-existing',
  description: 'Polish existing reasoning traces by appending a genuine debug + self-verify pass (one subagent per method)',
  whenToUse: 'Run track-A2 of the data remediation: make existing traces longer/deeper by adding real verification of reasoning and code',
  phases: [
    { title: 'Augment', detail: 'one subagent per method: read its reasoning+code, append an organized debug+self-verify continuation' },
  ],
}

// Default scope: the existing Combinatorial & Competitive Algorithms methods, whose code is concrete
// enough to actually trace. `args` may override with an explicit list of slugs.
const REPO = '/srv/home/bohanlyu/innovation_proior'

// Build the slug list from args (workflow scripts have no fs; the subagents read files themselves).
// Expect args = { slugs: [...] } — e.g. the competitive-algorithm methods.
const slugs = (args && Array.isArray(args.slugs) && args.slugs.length) ? args.slugs : []
if (!slugs.length) {
  log('No slugs passed via args.slugs — pass {slugs:[...]} (e.g. the competitive-algorithm methods). Nothing to do.')
  return { augmented: [], note: 'pass args.slugs' }
}
log(`Augmenting ${slugs.length} existing methods with a debug+self-verify pass (one subagent each)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'ok', 'added_chars', 'found_issue', 'note'],
  properties: {
    slug: { type: 'string' },
    ok: { type: 'boolean' },
    added_chars: { type: 'integer' },
    found_issue: { type: 'boolean', description: 'true if the verification pass found a real bug/edge issue in the existing code' },
    note: { type: 'string' },
  },
}

function prompt(slug) {
  return `Polish ONE existing training trace by adding a genuine debug + self-verify pass. Method: ${slug}. Working dir: ${REPO}.

1. Read ${REPO}/methods/${slug}/results/context.md , reasoning.md , answer.md . Understand the problem and the FINAL code in answer.md (or the code block at the end of reasoning.md).
2. The existing reasoning derives the method and writes code but (this is the known defect) almost never VERIFIES the code afterward. Your job: append an organized verification+debugging continuation to the END of reasoning.md that does what a careful author actually does before shipping:
   - **Tracing the code.** Pick a concrete small input and trace the final code step by step, confirming the output.
   - **Edge cases.** Enumerate the real corners for THIS problem (empty / size-1 / extremes / numerical overflow or precision / degenerate inputs) and state how the code handles each.
   - **Hunting a bug.** Genuinely look for a defect (off-by-one, wrong init, overflow, a mishandled edge, an unstated assumption). If you find one, state it precisely and give the corrected code; if after a real check you find none, say what you checked and why it's correct — do not invent a fake bug.
   - **Sanity-checking the derivation.** Re-verify one load-bearing step of the reasoning on a small case (a formula, an invariant, a complexity claim).
   Write it first-person, present-tense, ORGANIZED with short bold stage labels (e.g. **Tracing the final code.**, **Edge cases.**, **A bug I missed.**, **Re-checking the derivation.**), matching the file's English voice. Keep it grounded in the ACTUAL code — if the code is research Python (a class/library), trace a representative call rather than pretending it reads stdin.
3. If you found and fixed a real bug, also update the code in answer.md to the corrected version (and keep reasoning.md's final code consistent).
4. Append at least 1500 characters of substantive verification (no padding). Do NOT rewrite the existing reasoning; only append (and, if a bug was fixed, correct the code blocks).

Return the structured result (added_chars = characters you appended; found_issue = whether a real bug/edge issue was found).`
}

const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `aug:${slug}`, phase: 'Augment', schema: SCHEMA, agentType: 'general-purpose' })))

const ok = results.filter((r) => r && r.ok)
const withBug = ok.filter((r) => r.found_issue)
log(`Augmented ${ok.length}/${slugs.length}; ${withBug.length} had a real bug/edge issue found`)
return { augmented: ok.map((r) => r.slug), found_issues: withBug.map((r) => r.slug) }
