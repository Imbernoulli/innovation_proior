export const meta = {
  name: 'investigate-model-failures',
  description: 'Deep root-cause analysis of the trained models real outputs: characterize every failure mode, trace to a data cause, semantic leakage hunt',
  whenToUse: 'Understand WHY the SFT models regress, from the real eval dumps in experiments/raw_outputs + data_feedback',
  phases: [
    { title: 'Analyze', detail: 'parallel analyzers over real-output slices + training data' },
    { title: 'Synthesize', detail: 'one agent merges findings into a ranked root-cause -> fix report' },
  ],
}

const REPO = '/srv/home/bohanlyu/innovation_proior'
const RO = `${REPO}/experiments/raw_outputs`
const FB = `${REPO}/experiments/data_feedback/examples`

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slice', 'failure_modes'],
  properties: {
    slice: { type: 'string' },
    failure_modes: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['mode', 'evidence', 'prevalence', 'data_cause', 'fix'],
        properties: {
          mode: { type: 'string', description: 'short name of the failure mode' },
          evidence: { type: 'string', description: 'verbatim snippet(s) from a real output, with tag/problem if known' },
          prevalence: { type: 'string', description: 'how common (counts if you measured, else qualitative)' },
          data_cause: { type: 'string', description: 'the specific training-data property that most likely causes this' },
          fix: { type: 'string', description: 'concrete data/pipeline fix' },
        },
      },
    },
  },
}

const HOWTO = `The real eval dumps are at ${RO} (per benchmark/model_tag: samples.jsonl.gz with field text=full output incl <think>, metrics.score; summary.json plaintext; prompts.jsonl.gz for statements) and curated start/sft/average comparisons at ${FB} (plain markdown). model_tags include q35_inst_start (baseline), q35_a100_method_sft, q35_a100_method_soup10, q35_a100_methodtraj_sft, q35_a100_methodtraj_soup10 and q3 equivalents. Read with: zcat <file>.jsonl.gz | python3 -c "import json,sys; [print(json.loads(l)['text'][:4000]) for l in sys.stdin]" (or json.tool a single line). ALWAYS quote verbatim evidence and, where you can, COUNT prevalence (e.g. grep the gz for a pattern across samples). Compare SFT vs start to isolate what SFT changed.`

const tasks = [
  { slice: 'fcs-algorithm', prompt: `Analyze FrontierCS ALGORITHM-track failures of the SFT models vs the instruct start. Read ${FB}/fcs_tree_distance__{start,sft,average}.md and ${FB}/fcs_xor_sidon__{start,sft,average}.md in full, then sample ${RO}/frontiercs_algorithm/q35_a100_methodtraj_sft/samples.jsonl.gz and q3_a100_method_sft. Characterize the failure modes that make SFT score ~0: hollow register (invented terminology, name-dropping), CONFIDENTLY-FALSE math asserted but never checked (e.g. the xor_sidon "(2m-2) XOR (2m) = 2(m-2)^2..." line), verifying only tiny inputs then shipping a wrong construction, over-engineering vs the start's pragmatic solution, non-termination / token-budget blowups, output-format misses. Quote verbatim, count where you can.` },
  { slice: 'fcs-research', prompt: `Analyze FrontierCS RESEARCH-track (Python Solution.solve()) outputs. Sample ${RO}/frontiercs_research_gpu/{q35_inst_start,q35_a100_method_sft}/samples.jsonl.gz and the cpu set. Compare start vs SFT: does SFT degrade or help here, and how (the research track is more open-ended than the algorithm track)? Characterize concrete failure modes with verbatim evidence and any score deltas from summary.json.` },
  { slice: 'mls-bench', prompt: `Analyze MLS-Bench. Read ${RO}/mlsbench/*/summary.json (scores) and several task_logs/*.log.gz transcripts for q35_a100_method_sft vs q35_inst_start. MLS is the one benchmark where method-SFT reportedly BEATS the start. Characterize WHAT the SFT model does differently here that helps (research orientation paying off?), AND what still fails (13/20 tasks score 0 -- editing disallowed packages, never submitting, infra). Quote verbatim.` },
  { slice: 'thinking-pathologies', prompt: `Across ALL model_tags in ${RO}, characterize and COUNT thinking-level pathologies in the text field: (1) "I cannot complete this thought ... the next thinking is empty/garbled/corrupted" then closing </think> and dumping code; (2) "as an AI" / "I am an AI text model, I must provide the code"; (3) hitting the 32k token cap without closing </think>; (4) abandoning mid-derivation. For each: which model_tags show it, rough rate (grep the gz across samples), and verbatim examples. Hypothesize the cause of each (e.g. reasoning-folding empty-think conditioning, base-model RLHF resurfacing, truncated think targets) but mark it a HYPOTHESIS -- a separate research pass is studying the Qwen3 chat template.` },
  { slice: 'semantic-leakage-hunt', prompt: `Hunt for "rewriting-process" / construction-process LEAKAGE, SEMANTICALLY (not by exact string -- it may appear with small variations/paraphrases). Look in BOTH (a) the model outputs in ${RO} and (b) the TRAINING data: read a stratified SAMPLE -- ~25 random methods/*/results/reasoning.md, ~25 trajectories/*/*-reasoning.md, ~10 methods/*/results/answer.md, and skim agentic.json / agentic_messages.jsonl. Flag ANYTHING that reads like meta-commentary about the writing/discovery PROCESS rather than the math: "let me rewrite/redraft/reconsider how to present", "the previous draft", "as I reconstruct this", "discovering this for the first time", references to a finished paper, "I cannot complete this thought", "the thinking is empty/garbled", AI-assistant/refusal tells, truncation placeholders ("// ...", "[omitted]"). Report each as {file, verbatim snippet, why it is leakage, paraphrase-variant of which canonical artifact}. Crucially: estimate whether such leakage is genuinely PRESENT in the training data (with variations) or essentially absent (model-only). Be concrete and quote.` },
  { slice: 'data-cause-map', prompt: `Read ${REPO}/sft/build_sft.py, ${REPO}/.claude/skills/paper-to-reasoning/SKILL.md, ${REPO}/experiments/DATA_REMEDIATION_zh.md, and run ${REPO}/tools/data_audit.py. Produce the mapping from each known SFT-data property (landing point = research-Python library not stdin C++; ~0% post-code verification; reverse-engineered always-lands traces with no failure samples; system='good researcher'+'narrative tone'; reasoning-folding empty-think; per-rung trajectory under-length) to the specific model behavior it most plausibly causes. This is the data-side half; the output-side analyzers cover the behaviors.` },
]

const findings = await parallel(tasks.map((t) => () =>
  agent(`${t.prompt}\n\n${HOWTO}`, { label: `analyze:${t.slice}`, phase: 'Analyze', schema: FINDINGS_SCHEMA, agentType: 'general-purpose' })))

const clean = findings.filter(Boolean)
const synth = await agent(
  `You are writing the definitive root-cause report on why our Innovation-Prior SFT models regress on competition evals, from these analyzer findings (verbatim JSON):\n\n${JSON.stringify(clean, null, 1)}\n\n` +
  `Write a tight, evidence-grounded Chinese markdown report to ${REPO}/experiments/MODEL_FAILURE_ROOTCAUSE_zh.md. Structure: (0) 一句话; (1) 失败模式清单 -- each with verbatim evidence, prevalence, and the score impact; (2) 根因映射 (failure mode -> data/pipeline cause), ranked by impact; (3) leakage 结论 (is rewrite/meta leakage really in the training data with variants, or model-only? cite evidence); (4) thinking 病理 (empty-think narration / as-an-AI / non-termination) with the leading hypothesis, noting the Qwen3-template research is separate; (5) 解决方案 -- concrete, prioritized, split into (a) optimize existing data, (b) new data, (c) pipeline/build fixes, (d) what to verify next. Be honest about uncertainty and about what is measured vs hypothesized. Do not pad. Return the absolute path you wrote.`,
  { label: 'synthesize:rootcause', phase: 'Synthesize', agentType: 'general-purpose' })
return { report: synth, analyzers: clean.length }
