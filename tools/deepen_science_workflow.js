export const meta = {
  name: 'deepen-science-reasoning',
  description: 'Deepen existing method-discovery reasoning traces: work the paper appendix in + add genuine derivation depth (one subagent per method)',
  whenToUse: 'When the science/method-discovery reasoning reads like surface storytelling and must incorporate appendix proofs + deeper derivation logic',
  phases: [
    { title: 'Deepen', detail: 'one subagent per method: read existing trace + local src/ appendix + notes, rewrite reasoning.md deeper (appendix proofs worked inline, design choices justified, real dead-ends)' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'

// Default validation batch: diverse domains, all confirmed to have local src/ with appendix/supplementary.
// Override with args.slugs = [...] to scale (948 methods have src/).
const DEFAULT = ['a3c', 'adam', 'resnet', 'trpo', 'adiprasito-huh-katz-matroid-hodge', 'gae', 'batchnorm']
const slugs = (args && Array.isArray(args.slugs) && args.slugs.length) ? args.slugs : DEFAULT
log(`Deepening ${slugs.length} science reasoning traces (one subagent each): ${slugs.join(', ')}`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'ok', 'orig_chars', 'new_chars', 'appendix_items_added', 'note'],
  properties: {
    slug: { type: 'string' },
    ok: { type: 'boolean', description: 'true only if reasoning.md was deepened in place, strictly supersedes the original, stays in-frame' },
    orig_chars: { type: 'integer' },
    new_chars: { type: 'integer' },
    appendix_items_added: { type: 'integer', description: 'count of distinct appendix proofs/derivations/impl-details actually worked into the trace' },
    note: { type: 'string' },
  },
}

function prompt(slug) {
  return `Deepen ONE existing method-discovery reasoning trace. Method: ${slug}. Working dir: ${REPO}.
The complaint about this corpus: the reasoning reads like one person telling a story / narrating the surface of a finished paper, instead of the REAL derivation logic — and it skips the paper's appendix. Your job is to make it genuinely deeper, grounded in the actual source.

READ (all of these):
  - ${REPO}/methods/${slug}/results/context.md , reasoning.md , answer.md   (the existing trace)
  - ${REPO}/methods/${slug}/notes/synthesis.md  (if present: the prior research synthesis)
  - ${REPO}/methods/${slug}/src/*.tex  — list them, then READ IN FULL the APPENDIX / SUPPLEMENTARY / proof sections (files/sections named supplementary, appendix, proofs, or the high-numbered sections). This is where the load-bearing derivations live. If there is no src/, fall back to refs/ and notes/.
  - ${REPO}/.claude/skills/paper-to-reasoning/SKILL.md  — sections "2.2 reasoning.md" and "2.4 revision pass" define the depth/voice bar. Follow them.

THEN rewrite ${REPO}/methods/${slug}/results/reasoning.md so it is strictly DEEPER, preserving the existing content and first-person present-tense voice but adding real substance:
  1. **Work in the appendix.** Every appendix/supplementary proof, lemma, bound, derivation, and implementation/parameterization detail that the current trace skips or merely gestures at ("one can show that", "the proof is straightforward", "证明骨架如下") must be DERIVED INLINE — actually do the algebra, every step, every case of a case-analysis, the telescoping, the matrix-vector-product / efficiency argument, the constant/normalization. A trace that gestures at an appendix proof is a failure.
  2. **Deepen every non-obvious design choice.** For each constant, knob, normalization, architectural choice: reconstruct the derivation-time WHY and why the obvious alternatives are worse (the mechanism), not just the what. Surface the small insights the paper omits.
  3. **Add genuine dead-ends.** Real derivations hit walls. Where the method has a natural failed first attempt (the estimator that blows up, the bound that is too loose, the parameterization that is unstable), live it out and show the correction — do not make every step land smoothly. This is the difference between reasoning and storytelling.
  4. **Self-verify the derivation.** At least once, check a load-bearing step on a concrete small case or by a sanity limit (units, a special case reducing to a known result), as the narrator's own check.
  5. **Stay in-frame.** First-person present-tense continuous prose; NO "the paper" / "the authors" / "the appendix" / citations to the target method-as-paper; reconstruct it as live discovery. Prior-art ancestor citations (e.g. "Sutskever et al. 2013") stay. No section headers describing the reasoning (keep it as flowing thought; the existing file's style). English.

Write the deepened reasoning.md in place. If the appendix reveals a more precise final form/constant, also fix answer.md to match (keep them consistent). Do NOT shorten or delete correct existing content — only deepen.

Return the structured result (orig_chars / new_chars of reasoning.md; appendix_items_added = how many distinct appendix derivations you worked in).`
}

const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `deepen:${slug}`, phase: 'Deepen', schema: SCHEMA, agentType: 'general-purpose' })))

const ok = results.filter((r) => r && r.ok)
const grew = ok.filter((r) => r.new_chars > r.orig_chars)
const totalAppendix = ok.reduce((a, r) => a + (r.appendix_items_added || 0), 0)
log(`Deepened ${ok.length}/${slugs.length}; ${grew.length} grew; ${totalAppendix} appendix derivations worked in total`)
return {
  deepened: ok.map((r) => ({ slug: r.slug, orig: r.orig_chars, new: r.new_chars, appendix: r.appendix_items_added })),
  failed: results.filter((r) => !r || !r.ok).map((r, i) => (r && r.slug) || slugs[i]),
}
