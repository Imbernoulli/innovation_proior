# changelog — rmsnorm-parallel

## 2026-08-19 (svfix, W3_primary_plus_ancestors)
Decisive step (parallel block adoption + tied-vs-untied norm choice) was previously justified
by an unsourced from-scratch op-sharding computation, with self-account material already on
disk (`refs/self_accounts/*`, `refs/explainers/kipply-transformer-taxonomy.txt`) sitting
unused. Fixed at three points in `results/reasoning.md`:

1. The all-reduce-count argument (why the serial block costs two collectives per block) is now
   corroborated by GPT-J's own account: its shipped code comments the tied/local-sum design "to
   minimize all reduces" (`code/gptj_layers.py`), and Komatsuzaki's release write-up names
   "decreased communication" as the reason in prose (`refs/self_accounts/komatsuzaki-gpt-j-blog.txt`).
2. The parallel-block payoff paragraph previously conflated "halved collectives" and "fusible
   projections" into one unattributed, confused magnitude guess ("low tens of percent ...
   implementation-dependent"). Rewritten to name these as two separate, differently-priced wins
   (communication-bound vs. compute-bound/arithmetic-intensity), grounded in two *separate*
   classes/comments in GPT-J's own source (`code/gptj_layers.py`: `TransformerLayerShard` "to
   minimize all reduces" vs. the later `TransformerLayerShardV2` "combines the input and output
   projection into one matmul for better efficiency") and corroborated independently by
   `refs/explainers/kipply-transformer-taxonomy.txt` ("increase arithmetic intensity for better
   performance"). No magnitude number is stated — which win dominates is left as a profiling
   question, consistent with the method's own measured number (primary `src/model-arch.tex`
   line 9, "roughly 15% faster... since the MLP and Attention input matrix multiplications can
   be fused") being the method's own experimental result and therefore excluded from this
   channel per the empirical-outcome rule.
3. The tied-vs-untied-norm parenthetical previously offered its "harmless if accidentally
   untied" claim as pure unsupported speculation ("if my expectation holds"). Grounded in
   GPT-NeoX-20B's documented account of hitting exactly this accident and finding no measurable
   quality difference in a post-hoc check (`refs/explainers/gptneox-src/bs_workshop.tex`,
   "Parallel Attention + FF Layers").

Also removed the method's own empirical outcomes (the ~15% throughput figure; the 8B/62B
quality-ablation result; "the evidence is that tied vs. untied makes no measurable difference"
stated as settled fact) from `results/answer.md` and `results/train_answer.md`, where they had
been sitting as achieved-fact claims — this is the method's own experimental result and is not
a legitimate channel for it per the fix-prompt's empirical-outcome rule. Replaced with the same
hypothesis/profiling-question framing now used in `reasoning.md`; the landing (method + code) is
unchanged.

Full grounding quotes and local file paths: `notes/sources.md`.
