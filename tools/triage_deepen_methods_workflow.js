export const meta = {
  name: 'triage-deepen-methods',
  description: 'Triage thin single-turn method reasoning traces into fine / deepen / trivial, then act: DEEPEN the genuinely-under-reasoned ones from real source material (never invent), and DELETE the trivial/placeholder ones (git rm, one commit each, reason in message). Fine ones are left untouched. Per-unit commit so nothing is lost.',
  whenToUse: 'After the audit-edit de-bloating pass: the "make thin good methods deeper" half. A short trace is NOT automatically a problem — only a short trace that fails to convey the real insight of a method that HAS one is a deepen target; a genuinely placeholder/trivial method is a delete target; a short trace that already conveys everything worth saying is left alone.',
  phases: [
    { title: 'Triage+Act', detail: 'one Sonnet(xhigh) agent per method: classify fine/deepen/trivial, then deepen-from-source or git-rm accordingly, commit', model: 'sonnet' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
A = A || {}

const EMBEDDED = A.units || null // [{slug, words, has_src}]
const limit = A.limit ?? 100000
let units = (EMBEDDED || []).slice(A.start ?? 0, (A.start ?? 0) + limit)
const dryRun = A.dryRun === true // if true: only classify, do not edit/delete/commit
log(`triage-deepen: ${units.length} thin methods (dryRun=${dryRun})`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'verdict', 'reason', 'insight', 'sources', 'acted', 'orig_words', 'new_words', 'committed', 'commit_detail'],
  properties: {
    slug: { type: 'string' },
    verdict: { type: 'string', enum: ['fine', 'deepen', 'trivial'] },
    reason: { type: 'string', description: 'one-sentence justification for the verdict' },
    insight: { type: 'string', description: 'the key insight/why-it-works of this method in one line, or "none" if genuinely trivial' },
    sources: { type: 'array', items: { type: 'string' }, description: 'for deepen: the URLs actually retrieved and used to reconstruct the context/derivation (empty for fine/trivial)' },
    acted: { type: 'boolean', description: 'true if you edited (deepen) or deleted (trivial); false for fine or dryRun' },
    orig_words: { type: 'integer' }, new_words: { type: 'integer', description: 'reasoning word count after deepen; same as orig for fine; 0 for deleted' },
    committed: { type: 'boolean' }, commit_detail: { type: 'string' },
  },
}

function prompt(u) {
  const d = `${REPO}/methods/${u.slug}`
  return `Triage and then act on ONE thin single-turn method reasoning trace. Method: ${u.slug} (currently ${u.words} words of reasoning, has_src=${u.has_src}). Working dir: ${REPO}.

READ, in full: ${d}/results/context.md, ${d}/results/reasoning.md, ${d}/results/answer.md, ${d}/results/train_answer.md (if present)${u.has_src ? `, and the paper source under ${d}/src/ (list it, read the relevant derivation/appendix sections)` : ' (there is NO src/ — you have no paper material to deepen from, so be conservative: see rule below)'}${''}. Also glance at ${d}/notes/synthesis.md if present.

CLASSIFY into exactly one verdict. The guiding question is NOT "is this short?" and NOT "is the method simple?" — a simple, elegant method can have a deep discovery reasoning, and a short trace that already conveys everything worth saying is FINE. The question is: **does this method have a real insight (a why-it-works, a non-obvious design choice, a natural failed attempt), and does the current reasoning FAIL to convey it?**

  - fine  — the reasoning already conveys the method's real insight (or the method is elementary and there is genuinely nothing more worth saying, and it is a legitimate method worth keeping). Short is not a defect by itself. DEFAULT to 'fine' when unsure. Do NOT edit.
  - deepen — the method HAS a real insight (why this form and not the obvious alternatives; why it works; a genuine dead-end the derivation hits) but the current reasoning states the method flatly without earning it. This is the only edit target.
  - trivial — the method itself is a placeholder / near-empty / degenerate entry with no insight worth any reasoning at all (e.g. a stub, a duplicate, a one-line "we use X" with nothing to discover). This is a DELETE target.

THEN ACT${dryRun ? ' (THIS IS A DRY RUN: do NOT edit, delete, stage, or commit anything — only classify and return your verdict; leave every file exactly as you found it)' : ''}:
  - If verdict=fine: do nothing. Return acted=false.
  - If verdict=deepen: FIRST GATHER REAL SOURCES, then rewrite. Do NOT write from memory.
    ${u.has_src ? `SOURCE (a): read the paper material under ${d}/src/ — do the algebra it states, never gesture. AND ALSO web-search to corroborate the historical context.` : `You have NO src/, so you MUST research online.`}
    RESEARCH STEP (mandatory for deepen): use WebSearch / WebFetch (load their schemas via ToolSearch("select:WebSearch,WebFetch") if not already available) to find AUTHORITATIVE sources for THIS specific method: the original paper (arXiv / journal / author's page), the author's own retrospective or talk if one exists, and reputable third-party expositions (surveys, lecture notes, well-known blog explainers by domain experts). From these, reconstruct with PRECISION:
      (1) the CONTEXT at the time of discovery — what was already known, what the standard tools were, what specific problem or obstruction was open;
      (2) the DERIVATION PATH — how the author, starting from that context, actually arrived at this method: the motivating question, the natural first attempts and exactly why they fail, the key idea that breaks the impasse, and why the final form is the one that works.
    Ground every technical specific (a bound, a constant, a construction, a lemma it relies on, a date/lineage claim) in a source you actually retrieved this run — not memory. If the sources you can find do not let you honestly reconstruct the derivation with confidence, DOWNGRADE to fine and do not edit. NEVER invent derivations, constants, history, or a discovery story to hit a length.
    THEN rewrite ${d}/results/reasoning.md as genuine first-person discovery reasoning built on that researched context: the derivation-time WHY of the key choices, the natural failed attempt(s) walked out and refuted, the load-bearing algebra inline. FROZEN: context.md, answer.md, train_answer.md unchanged. NO length target — length is whatever the real derivation earns. Voice: first-person present-tense discovery prose, in-frame (NO "the paper"/"the authors"/citations-as-external-artifact — you are the person figuring it out; prior-art ancestor citations by author/year are allowed), English, no section headers. (This mirrors the repo's paper-to-reasoning skill: reconstruct how it was derived, not a summary of the finished result.)
    Record the sources you actually used (URLs) in the returned "sources" field, so the grounding can be audited.
  - If verdict=trivial: DELETE the whole method directory: run \`git -C ${REPO} rm -r -- methods/${u.slug}\`. (This method is being removed from the corpus.)

COMMIT this unit only (never \`git add -A\`):
  - deepen: \`git -C ${REPO} add -- methods/${u.slug}/results/reasoning.md\` then \`git -C ${REPO} commit -q -m "deepen thin method:${u.slug}" -m "<one line: the insight you surfaced>"\`
  - trivial: the \`git rm\` already staged it; \`git -C ${REPO} commit -q -m "drop trivial method:${u.slug}" -m "<one line: why it is a placeholder with no insight>"\`
  - fine: no commit.
Confirm \`git -C ${REPO} status --porcelain\` shows only your intended change (or nothing for fine). Put the short hash (or "none") in commit_detail.

Return the structured result. Be conservative: the corpus is more harmed by a fabricated deepen or a wrong deletion than by leaving a short-but-fine trace alone.`
}

async function runUnit(u) {
  const r = await agent(prompt(u), { label: `triage:${u.slug}`, phase: 'Triage+Act', model: 'sonnet', effort: 'xhigh', schema: SCHEMA, agentType: 'general-purpose' })
  if (!r) return { slug: u.slug, verdict: 'error' }
  // DRY-RUN safety: if dryRun and the agent claims it acted or committed, flag it — dry run must never mutate.
  if (dryRun && (r.acted || r.committed)) return { slug: u.slug, ...r, verdict: 'dryrun_violation' }
  return { slug: u.slug, ...r }
}

const results = (await pipeline(units, u => runUnit(u))).filter(Boolean)
const by = (v) => results.filter(r => r.verdict === v)
log(`triage done: fine ${by('fine').length}, deepen ${by('deepen').length}, trivial ${by('trivial').length}, error ${by('error').length}`)
return {
  fine: by('fine').map(r => r.slug),
  deepened: by('deepen').map(r => ({ slug: r.slug, orig: r.orig_words, new: r.new_words, insight: r.insight, sources: r.sources, hash: r.commit_detail, committed: r.committed })),
  deleted: by('trivial').map(r => ({ slug: r.slug, reason: r.reason, hash: r.commit_detail, committed: r.committed })),
  errors: by('error').map(r => r.slug),
}
