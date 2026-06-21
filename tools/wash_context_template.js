export const meta = {
  name: 'wash-context-CHUNKTAG',
  description: 'Condense + de-leak a context file into a plain contemporaneous snapshot (strip gap analysis, no hindsight, no answer leakage); one subagent per file',
  phases: [{ title: 'Clean', detail: 'one subagent per context file' }],
}

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    path: { type: 'string' },
    gaps_removed: { type: 'integer', description: 'count of Gap/weakness/limitation clauses deleted' },
    hindsight_removed: { type: 'integer', description: 'count of hindsight phrases removed' },
    leak_status: { type: 'string', description: 'did the original name/telegraph the invented method? what you fixed (or "none")' },
    old_words: { type: 'integer' },
    new_words: { type: 'integer' },
    note: { type: 'string', description: 'one-line summary' },
  },
  required: ['path', 'gaps_removed', 'old_words', 'new_words', 'leak_status', 'note'],
}

const LIST = __LIST__

log(`Washing ${LIST.length} context files`)

function prompt(path) {
  return `You are cleaning ONE context file for an LLM-training dataset. This file is the INPUT shown to the model: a plain, contemporaneous snapshot of the research situation at the time, BEFORE the method-to-be-invented exists. It must NOT do the reasoning's job of finding problems.

TARGET FILE (edit in place): ${path}

LEAK-CHECK REFERENCE — first read the answer of the method this context leads into, ONLY to learn its NAME and MECHANISM so you can guarantee the context never names, describes, or telegraphs it. NEVER copy any answer material into the context.
- If TARGET is methods/<slug>/results/context.md  -> read methods/<slug>/results/answer.md (and train_answer.md if it exists).
- If TARGET is trajectories/<slug>/00-initial-context.md -> read the FIRST rung's answer: the file matching trajectories/<slug>/01-*-answer.md (use ls or glob on trajectories/<slug>/ to find it). That first method is what this initial context leads into.

STEPS
1. Read the TARGET FILE in FULL.
2. Read the leak-check reference above (only to know what to avoid).
3. Rewrite the TARGET in place (use the Write or Edit tool) following the rules below. Preserve the existing section structure and Markdown (headings, bullet lists, code/equation blocks).

REWRITE RULES
(A) Contemporaneous viewpoint ONLY — describe what exists, is known, or is practiced at that era, as a scientist standing in that moment would see it. Remove ALL hindsight: "would later", "this anticipates", "turns out", "in retrospect", "the key insight that...", "this paved the way", or any reference to how things developed after this point.

(B) **Do NOT point out problems, gaps, weaknesses, or what is missing. This is the single most important rule.** Delete EVERY "Gap:", "Gap it leaves:", "Weakness:", "Limitation:", and every clause of the form "but it fails / stalls / cannot / is fatal / is too expensive / doesn't scale / leaves open / the problem is...". Describe each existing method ONLY by what it IS and what it does. Also strip any "pain points" / "the problem with X is..." editorializing from the Background and Research-question sections. State the setting plainly and naively — finding the shortcomings is the job of the separate reasoning trace, not this file.

(C) Never name or telegraph the invented method (you learned it from the answer). The Research-question section should pose the broad question / setting only — NOT an answer-shaped wishlist of desiderata that the new method happens to satisfy, and NOT "none of the existing methods achieves all of this".

(D) Condense: remove the leaky/editorial material and obvious verbosity (this typically lands ~15-25% shorter). BUT keep every contemporaneous technical fact — equations, each existing method's mechanism and formula, dataset/benchmark/metric details, and the code framework. Do NOT invent facts, numbers, or citations. Do NOT cut real technical content just to hit a length target.

(E) Markdown is fine. Keep section headings exactly as they are (e.g. "## Baselines" stays "## Baselines" — just strip the gap analysis from inside it). Keep code blocks and equation blocks.

GOLD STANDARD (methods/adam, already done): the Research question's answer-shaped 6-point wishlist ("...none achieves all six; closing that gap is the problem") became a plain "how to set the per-parameter step size from a stream of noisy first-order gradients"; every "**Gap:**" clause on SGD/AdaGrad/RMSProp/AdaDelta/SFO was deleted, leaving only each method's neutral description and formula; Background's "pain points" became plain statements of the setting; all equations, framings, datasets, and the code framework were kept.

If the TARGET is already clean by these rules, make only the edits that are still needed (do not pad it back up). Then return the structured summary (gaps_removed = count of Gap/weakness clauses deleted; old_words/new_words = wc -w before and after; leak_status = what the original telegraphed and how you fixed it, or "none").`
}

phase('Clean')
const results = await parallel(LIST.map(p => () =>
  agent(prompt(p), { label: `clean:${p.split('/').slice(0,2).join('/')}`, phase: 'Clean', schema: SCHEMA, agentType: 'general-purpose', model: 'sonnet' })
))

return results.filter(Boolean)
