export const meta = {
  name: 'derewrite-reasoning',
  description: 'Fix the structural reverse-engineering stance: inject genuine verification + a real wall/abandonment into reasoning traces (also lengthens them)',
  whenToUse: 'Remove the "always-lands confident discovery" stance that causes assert-without-verify + the empty-think refusal artifact',
  phases: [{ title: 'De-rewrite', detail: 'one subagent per method: rewrite reasoning.md with real checks whose outcome can redirect, and one genuine failed-and-abandoned approach' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const slugs = (A && Array.isArray(A.slugs) && A.slugs.length) ? A.slugs
  : ['gale-shapley', 'vae', 'dino', 'ppo', 'dqn', 'word2vec', 'simsiam', 'dropout']
log(`De-rewriting ${slugs.length} reasoning traces (one subagent each): ${slugs.join(', ')}`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'ok', 'orig_chars', 'new_chars', 'added_verification', 'added_abandonment', 'notes'],
  properties: {
    slug: { type: 'string' }, ok: { type: 'boolean' },
    orig_chars: { type: 'integer' }, new_chars: { type: 'integer' },
    added_verification: { type: 'boolean', description: 'added a real check whose outcome is computed on the page (could redirect)' },
    added_abandonment: { type: 'boolean', description: 'added a genuine wall: an approach tried, verified to fail, and abandoned' },
    notes: { type: 'string' },
  },
}

function prompt(slug) {
  return `Rewrite ONE method-discovery reasoning trace to remove its structural "reverse-engineering" stance and add genuine reasoning. Method: ${slug}. Working dir: ${REPO}.

READ: ${REPO}/methods/${slug}/results/context.md , reasoning.md , answer.md (the final method + code).

THE DEFECT to fix (narrow and specific — do NOT overcorrect): the trace too often ANNOUNCES its destination and then "confirms" claims it never actually computes ("this confirms the construction holds", "guaranteed correct", "the key insight is..."). That reads as fabricated and teaches the model to assert without checking. The goal is to make the trace read like a real person reasoning carefully — a reader should NOT be able to tell it was reverse-engineered. The PRIMARY fix is real verification; this is just what careful reasoning naturally contains.

CRITICAL GUARDRAIL: do not trade one tell for another. A STAGED/theatrical failure ("let me try X... oh no it fails... so Y") looks just as fabricated as a staged success. Only include a wall where the real derivation genuinely had one. Never manufacture a dramatic dead-end for effect. Naturalness over drama.

REWRITE ${REPO}/methods/${slug}/results/reasoning.md IN PLACE (preserve the first-person present-tense voice, in-frame, NO "the paper"/"the authors"):
1. **Add at least one REAL verification whose outcome is actually computed on the page, not asserted** — work a concrete small example, a numeric check of a derived bound/identity, a limiting/special case that reduces to a known result, or (for a computational method) trace the code on an input. Show the computation and the result. Replace every "this confirms X" / "guaranteed" that is currently a bare assertion with an actual check (or, if it genuinely can't be checked here, soften it to honest uncertainty — "I expect X; I'd want to verify it on..."). This is the main lever.
2. **De-announce the destination**: cut sentences that name the answer before it is derived ("the key insight is", "it is provably X", "this is exactly the method"); let the method emerge as the conclusion of the reasoning. The narrator should reason toward it, not pre-state it.
3. **A genuine wall is OPTIONAL and only-if-natural**: if the real derivation honestly had a wrong turn or a tempting-but-worse alternative, walk it and let a concrete check rule it out. If there is no natural one, DO NOT invent one — a forced failure is worse than none.
4. **Preserve the correct final method and code exactly** — the destination stays right and unchanged; you are improving the PATH's honesty, not the answer.
5. The added real verification naturally LENGTHENS the trace. Do not pad; every added sentence must carry a real computation or a real decision. The test: nothing you add should look inserted — it should read as how this person would actually have checked their own work.

Write the rewritten reasoning.md. Return counts; added_verification = you added a real computed check; added_abandonment = you walked a genuine (not staged) wall.`
}

const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `derw:${slug}`, phase: 'De-rewrite', schema: SCHEMA, agentType: 'general-purpose' })))
const ok = results.filter((r) => r && r.ok)
const both = ok.filter((r) => r.added_verification && r.added_abandonment)
const grew = ok.filter((r) => r.new_chars > r.orig_chars)
log(`De-rewrote ${ok.length}/${slugs.length}; ${both.length} added BOTH real verification + abandonment; ${grew.length} got longer`)
return { done: ok.map((r) => ({ slug: r.slug, orig: r.orig_chars, new: r.new_chars, verify: r.added_verification, abandon: r.added_abandonment })) }
