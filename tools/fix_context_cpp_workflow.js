export const meta = {
  name: 'fix-context-cpp-scaffold',
  description: 'Make context.md scaffold C++ to match the now-C++ answer (scaffold<->answer correspondence) for the P0.1-converted methods',
  phases: [{ title: 'FixContext', detail: 'one subagent per method: convert the context.md Code-framework Python scaffold to a pre-method C++ skeleton + state C++/stdin contract' }],
}
const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const slugs = (A && Array.isArray(A.slugs) && A.slugs.length) ? A.slugs : []
if (!slugs.length) { log('No args.slugs.'); return { fixed: 0 } }
log(`Converting ${slugs.length} context.md scaffolds Python -> C++ to match the answers`)
const SCHEMA = { type:'object', additionalProperties:false, required:['slug','ok','scaffold_now_cpp','notes'],
  properties:{ slug:{type:'string'}, ok:{type:'boolean'}, scaffold_now_cpp:{type:'boolean'}, notes:{type:'string'} } }
function prompt(slug) {
  return `Fix ONE mismatch: methods/${slug}/results/ was converted so reasoning.md / answer.md / train_answer.md land on a single-file C++ program reading stdin, but context.md still describes a PYTHON scaffold. The context.md "## Code framework" scaffold is the PROMPT and must correspond to the C++ answer (paper-to-reasoning rule). Working dir: ${REPO}.

1. Read methods/${slug}/results/context.md and methods/${slug}/results/train_answer.md (the final C++).
2. In context.md, replace the Python scaffold (its \`\`\`python block: solve()/class/def main) with a PRE-METHOD C++17 skeleton that corresponds to the final C++: '#include <bits/stdc++.h>', 'using namespace std;', 'int main(){ ... read input from stdin per the contract ... // TODO: <neutral, do NOT name the method> ... print to stdout ... }'. Same I/O and entry point as the final code, but the ALGORITHM BODY hollowed to a neutral '// TODO:' (it is pre-method -- must NOT give away the method's idea). Use a C++ (\`\`\`cpp) fence.
3. Make the prose match C++: in the Research question / Input-output contract, state plainly the deliverable is a single self-contained C++ program reading from stdin and writing to stdout (remove any 'top-level solve() returning a list' / Python-class / library phrasing). Fix the surrounding sentences that describe the Python scaffold.
4. Keep everything else (Background, Baselines, Evaluation settings) intact. Do NOT touch reasoning/answer/train_answer.
Return {slug, ok, scaffold_now_cpp, notes}.`
}
const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `ctxcpp:${slug}`, phase: 'FixContext', schema: SCHEMA, agentType: 'codex:codex-rescue' })))
const ok = results.filter((r) => r && r.ok && r.scaffold_now_cpp)
log(`Fixed ${ok.length}/${slugs.length} context scaffolds to C++`)
return { fixed: ok.map((r) => r.slug), failed: results.filter((r) => !r || !r.scaffold_now_cpp).map((r,i)=> (r&&r.slug)||slugs[i]) }
