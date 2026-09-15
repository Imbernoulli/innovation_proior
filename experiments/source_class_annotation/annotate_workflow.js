export const meta = {
  name: 'source-class-annotation-v2',
  description: 'Per-unit source-class labels for methods/<slug> using the restored bundle, the notes recovered from transcripts, and the August audit-agent transcript digests',
  phases: [
    { title: 'List', detail: 'units from methods.json minus decontam drops' },
    { title: 'Annotate', detail: 'one Sonnet reader per unit' },
    { title: 'Recheck', detail: 'second reader for low-confidence units' },
  ],
}
const LIST_SCHEMA = { type: 'object', properties: { slugs: { type: 'array', items: { type: 'string' } } }, required: ['slugs'] }
const CLS = { type: 'object', properties: { present: { type: 'boolean' }, used: { type: 'boolean' }, evidence: { type: 'string' } }, required: ['present', 'used', 'evidence'] }
const LABEL_SCHEMA = { type: 'object', properties: {
  slug: { type: 'string' }, primary_captured: { type: 'boolean' },
  ancestor: CLS, explainer: CLS, self_account: CLS, code: CLS, other: CLS, other_kind: { type: 'string' },
  n_records: { type: 'integer' },
  basis: { type: 'array', items: { type: 'string', enum: ['notes', 'refs', 'src', 'code', 'digest', 'results_only'] } },
  confidence: { type: 'string', enum: ['high', 'medium', 'low'] }, note: { type: 'string' },
}, required: ['slug', 'primary_captured', 'ancestor', 'explainer', 'self_account', 'code', 'other', 'other_kind', 'n_records', 'basis', 'confidence', 'note'] }

const RULES = `Repository: /srv/home/bohanlyu/innovation_proior (cd there first; the methods/ working tree is checked out). For unit methods/<slug> you have up to three kinds of material:
(1) The bundle on disk: methods/<slug>/results/{context,reasoning,answer,train_answer}.md, and if present methods/<slug>/notes/*, refs/*, src/*, code/*. Some notes/refs files were recovered from transcripts after a deletion; treat them as normal.
(2) A digest of the August audit agents' transcripts for this unit at /tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/cbc9e24b-cb14-40d1-996b-d3b8209e16b4/scratchpad/digests/<slug>.txt (may not exist). It contains those agents' shell commands and their outputs, i.e. verbatim listings and contents of notes/refs/src files that existed then, plus the agents' own findings about which sources the trace rests on. It can be long: use grep -n for "TOOL", "ls ", "notes/", "refs/", "self", "explainer", "ancestor", "source" and read the relevant windows; do not cat the whole file if it exceeds ~60k characters.
Never read PDFs or archives.

Classes of record:
- primary: the paper/text that introduced the method.
- ancestor: prior work the method builds on or reacts to.
- explainer: material about the method by people who are NOT its authors: blogs, lecture notes, course pages, textbooks, annotated implementations, surveys, retrospective analyses.
- self_account: the method's own authors outside the primary paper: award lectures, memoirs, interviews, notebooks, working blogs, theses, talks, the authors' own commit history, issue or OpenReview replies, commented-out drafts in the paper source.
- code: a canonical implementation (official repo or standard reimplementation) grounding the final code.
- other: anything else (third-party GitHub issues, reviews by non-authors, datasets, leaderboards); name it in other_kind.

present = material of that class exists in the bundle or is shown in the digest as having existed (a listing, a cat, a notes entry describing it). used = results/reasoning.md demonstrably draws on it: a specific fact, number, formula, design rationale, failed route, or ordering of steps that comes from that record rather than from the primary paper; for ancestors, reasoning about a named prior method's mechanism or limitation counts. A bare name-drop, or a record only listed in notes, is present but not used. One concrete evidence sentence per used class. n_records = distinct records across all classes including the primary. basis = which materials you actually had (notes / refs / src / code / digest / results_only). confidence reflects how much material you had. note = one line on anything odd.`

phase('List')
const list = await agent(`Read /srv/home/bohanlyu/innovation_proior/methods.json (a list or {methods:[...]} of objects with a 'slug' field) and /srv/home/bohanlyu/innovation_proior/decontam/decontam_rules.json ('drop_method_slugs'). Return every slug in methods.json that is NOT in drop_method_slugs, in file order. Use a python3 one-liner; do not summarize.`, { schema: LIST_SCHEMA, model: 'sonnet', effort: 'low', label: 'list units' })
const all = list.slugs
const start = (args && args.start) || 0, end = (args && args.end) || all.length
const slugs = all.slice(start, end)
log(`${all.length} units; annotating ${slugs.length} (${start}..${end})`)

const out = await pipeline(slugs,
  (slug) => agent(`${RULES}

Unit: methods/${slug}. Procedure: (1) ls -R methods/${slug} (skip results-only listing noise); (2) read every notes/*.md and any refs/*.md|*.txt|*.tex you find (skim src/ and code/ file names); (3) if the digest exists, grep it as described and read the windows that list or print notes/refs/src files and the agents' source findings; (4) read results/reasoning.md in full; (5) label each class. Return the structured label for slug "${slug}".`,
    { schema: LABEL_SCHEMA, model: 'sonnet', effort: 'medium', phase: 'Annotate', label: `annotate:${slug}` }),
  async (lab, slug) => {
    if (!lab) return null
    if (lab.confidence !== 'low') return lab
    const re = await agent(`${RULES}

A first reader labeled methods/${slug} with LOW confidence: ${JSON.stringify(lab)}. Re-read the unit independently (same procedure) and return your own label; where you disagree, your evidence must cite the specific passage.`,
      { schema: LABEL_SCHEMA, model: 'sonnet', effort: 'medium', phase: 'Recheck', label: `recheck:${slug}` })
    return re ? { ...re, rechecked: true, first: lab } : lab
  })
const labels = out.filter(Boolean)
const cnt = (k, f) => labels.filter(l => l[k] && l[k][f]).length
log(`done ${labels.length}/${slugs.length}; present A/E/S/C/O = ${cnt('ancestor','present')}/${cnt('explainer','present')}/${cnt('self_account','present')}/${cnt('code','present')}/${cnt('other','present')}; used = ${cnt('ancestor','used')}/${cnt('explainer','used')}/${cnt('self_account','used')}/${cnt('code','used')}/${cnt('other','used')}`)
return { total: all.length, range: [start, end], labels }