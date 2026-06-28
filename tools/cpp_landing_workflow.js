export const meta = {
  name: 'cpp-landing-conversion',
  description: 'P0.1 main fix: rewrite competition-method landing from Python class/library -> single-file C++ reading stdin (reasoning code blocks too)',
  whenToUse: 'After de-rewrite; convert the Combinatorial & Competitive Algorithms methods landing format so FrontierCS can score them',
  phases: [{ title: 'Convert', detail: 'one subagent per method: Python class landing -> single-file C++ stdin, in train_answer.md AND reasoning.md; compile-check' }],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
const slugs = (A && Array.isArray(A.slugs) && A.slugs.length) ? A.slugs : []
if (!slugs.length) { log('No args.slugs (competition method list) provided.'); return { converted: [] } }
log(`Converting ${slugs.length} competition-method landings to single-file C++ stdin (one subagent each)`)

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'ok', 'was_python_class', 'now_cpp_stdin', 'reasoning_blocks_converted', 'compiles', 'notes'],
  properties: {
    slug: { type: 'string' }, ok: { type: 'boolean' },
    was_python_class: { type: 'boolean' },
    now_cpp_stdin: { type: 'boolean', description: 'landing is now a single-file C++ program reading stdin' },
    reasoning_blocks_converted: { type: 'integer', description: 'how many Python code blocks in reasoning.md were converted/removed' },
    compiles: { type: 'boolean' },
    notes: { type: 'string' },
  },
}

function prompt(slug) {
  return `Convert ONE competition method's LANDING FORMAT from Python class/library to a single-file C++ program reading stdin. Method: ${slug}. Working dir: ${REPO}. This is the main FrontierCS fix (DATA_FIX_FCS_LANDING_zh.md P0.1): the algorithm/reasoning is already correct -- ONLY the deliverable language/shape is wrong (FrontierCS only scores a single-file C++ program read from stdin; a Python class scores 0).

READ: ${REPO}/methods/${slug}/results/{context.md, reasoning.md, answer.md, train_answer.md}, and ${REPO}/methods/${slug}/refs/ (if present -- often the original problem statement with the exact I/O format and samples) and ${REPO}/methods/${slug}/code/ (existing reference solution).

Decide first: is this method a concrete competitive-programming PROBLEM with a stdin/stdout I/O contract (e.g. dsu-on-tree = CF600E, a specific judged task)?
 - If YES: the landing must become a single self-contained C++17 program with \`int main()\` that reads the problem's input from stdin and writes the answer to stdout. Use the problem's real I/O format (from refs/). Add 'long long' / overflow awareness.
 - If it is a general ALGORITHM/TECHNIQUE with no single canonical I/O (e.g. a data-structure technique): still convert the deliverable to idiomatic single-file C++ (free functions / structs, NOT a Python 'class' library), wrapped in a small \`int main()\` that reads representative input from stdin and prints results -- the disposition we want is "competition single-file C++", not "Python library".

THEN make these edits IN PLACE, preserving the reasoning PROSE (complexity awareness + verification narrative -- do NOT touch that):
1. **train_answer.md**: replace the final Python landing code block with the single-file C++ program. Keep the surrounding distilled prose.
2. **answer.md**: same -- replace the Python code block with the C++ one (keep it consistent with train_answer).
3. **reasoning.md (THE GOTCHA -- do not skip)**: if reasoning.md contains Python IMPLEMENTATION code blocks (e.g. a 'class ...:' / 'def solve' / 'def main' library block), convert them to the equivalent C++ OR delete them, leaving only the prose and any genuinely-needed short snippet. The SFT target is <think>{reasoning}</think> + train_answer, so a Python class block left in reasoning.md still trains "emit a Python library first" -- it MUST go. Count how many you converted/removed.
4. **context.md (SCAFFOLD MUST MATCH ANSWER)**: context.md is the prompt; its "## Code framework" scaffold must correspond to the final answer (paper-to-reasoning rule). If it still has a Python scaffold (solve()/class), replace it with a PRE-METHOD C++ scaffold: a single-file C++17 skeleton (#include <bits/stdc++.h>, int main() that reads the input from stdin per the contract, prints to stdout, with a neutral '// TODO:' where the algorithm goes -- algorithm body hollowed out). Same I/O / entry point as the final C++, body removed. ALSO state plainly in context.md's Research question / Input-output contract that the deliverable is a single self-contained C++ program reading stdin (not a Python class/library). Keep the rest of context.md (Background / Baselines / Evaluation) intact.
5. Keep one I/O-contract sentence near the top of the landing (what it reads / prints).

VERIFY with Bash: write the final C++ to /tmp/${slug}_main.cpp and compile \`g++ -O2 -std=c++17 -o /tmp/${slug}_main /tmp/${slug}_main.cpp\`. If refs/ has sample input/output, run it and check. If it does not compile, FIX it until it does (the code must be real and compile). Make the code in reasoning.md / answer.md / train_answer.md character-identical to the compiled C++.

If the landing was ALREADY single-file C++ stdin, set was_python_class=false and leave it; do not Python-ify anything.

ALSO SKIP (set was_python_class=false, ok=true, change nothing) if this method is NOT a FrontierCS-style stdin task: a general optimization metaheuristic with no fixed problem (CMA-ES, particle swarm, NSGA/evolutionary, ant colony, simulated annealing/tabu as a generic optimizer), or a numeric "record"/search for a math-discovery task. Forcing those into a stdin C++ driver adds no value -- leave them.

Return the structured result (compiles must be true for ok=true).`
}

const results = await parallel(slugs.map((slug) => () =>
  agent(prompt(slug), { label: `cpp:${slug}`, phase: 'Convert', schema: SCHEMA, agentType: 'general-purpose' })))
const ok = results.filter((r) => r && r.ok)
const converted = ok.filter((r) => r.now_cpp_stdin && r.was_python_class)
log(`Converted ${converted.length}/${slugs.length} to C++ stdin; ${ok.filter((r) => r.compiles).length} compile`)
return { converted: converted.map((r) => r.slug), all: ok.map((r) => ({ slug: r.slug, cpp: r.now_cpp_stdin, rblocks: r.reasoning_blocks_converted, compiles: r.compiles })) }
