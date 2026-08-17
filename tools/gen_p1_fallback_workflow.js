export const meta = {
  name: 'p1-fallback-discipline',
  description: 'DATA_FIX_FCS_LANDING P1: append a grounded fallback-decision (commit discipline) to competition-method reasonings',
  whenToUse: 'Add the "if I am not sure of the clever approach in budget, ship the simpler correct one" decision to the C++ competition methods',
  phases: [{ title: 'Fallback', detail: 'one subagent per method: add one natural, method-specific fallback-decision sentence near the landing' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const slugs = (A && Array.isArray(A.slugs) && A.slugs.length) ? A.slugs : []
if (!slugs.length) { log('No args.slugs.'); return { done: 0 } }
log(`Adding P1 fallback discipline to ${slugs.length} competition methods`)

const SCHEMA = { type: 'object', additionalProperties: false, required: ['slug', 'ok', 'added', 'notes'],
  properties: { slug: { type: 'string' }, ok: { type: 'boolean' }, added: { type: 'boolean' }, notes: { type: 'string' } } }

function prompt(slug) {
  return `Add ONE grounded "fallback decision" (commit discipline) to the reasoning of competition method "${slug}". Working dir: ${REPO}.

Read ${REPO}/methods/${slug}/results/reasoning.md and train_answer.md (the final C++ solution and its approach).

Per DATA_FIX_FCS_LANDING P1: competition reasoning should model the decision to ship the simpler provable solution when not confident the clever approach is right within budget. Near the END of reasoning.md -- AFTER the verification/trace, just before the final code block -- insert ONE short, natural, FIRST-PERSON sentence (2-4 lines max) that names THIS method's actual approach and states the fallback discipline, e.g.: "The <the actual clever step in this solution> is the part I'd most easily get wrong under time pressure; if I weren't confident I could implement it correctly in the budget, I'd fall back to <the simpler/standard correct variant for this problem> that I've already traced as correct and ship that -- a plain correct submission beats an ambitious broken one."

Make it SPECIFIC to this method (use the real algorithm names / the real risky step), grounded, and natural -- NOT a generic template, NOT staged drama. If the solution is already dead-simple with no risky step, instead add a one-line note that you deliberately kept it simple and provable rather than over-engineering. Do NOT change the code, the final answer, or anything else; only insert this one sentence. Keep it in-frame (no "the paper").

Return {slug, ok, added, notes}.`
}

const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `p1:${slug}`, phase: 'Fallback', schema: SCHEMA, agentType: 'codex:codex-rescue' })))
const ok = results.filter((r) => r && r.ok && r.added)
log(`Added fallback to ${ok.length}/${slugs.length}`)
return { done: ok.map((r) => r.slug), failed: slugs.filter((s) => !ok.find((r) => r.slug === s)) }
