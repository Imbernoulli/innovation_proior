# Innovation-Prior SFT 模型在竞赛评测上回退的根因报告

## 0. 一句话

Method/discovery 风格的 SFT 教会了模型**研究叙事的表演**（命名方法、"Key Insight" 段落、自信落地），却没教会它**可运行的交付纪律**——于是模型把代码当附录、不验证、幻觉 API、并在思考里打转直到撞上 token 上限，导致 FrontierCS algorithm 99% 得 0、research GPU 掉约 90%；唯一稳定的增益来自"提交纪律"和"在符号回归类任务上选对库"。

---

## 1. 失败模式清单

下表每条给出 verbatim 证据、普遍度、以及对分数的影响。证据均来自分析器对 `experiments/raw_outputs` 下 16 个 `samples.jsonl.gz`（约 2955 个样本）及 `mlsbench`/curated dumps 的统计。

### 1.1 代码无法编译 / 不连贯（algorithm 轨道的主导失败）

- **证据**：q35 SFT longest-cpp 编译统计：64%（129/200）编译失败，19%（38/200）根本没代码，只有 16% 能编译且其中 31/33 仍得 0。Problem 16：循环外的 `continue;`、越界的 `t[a]/d/idx`；Problem 24：对一个 int 做 `l += ans_.size()`。q3_method_sft 编译率 72%，q3_soup10 84%。
- **普遍度**：约 83% 从未运行；99% 得 0。
- **分数影响**：q35 SFT mean 0.0005（99% 得 0）；对照 q3_method_sft mean 1.365，soup10 mean 2.8–3.3。curated：tree_distance start 85.87 vs sft 0.0；xor_sidon start 43.65 vs sft 0.0。

### 1.2 思考不终止：'Wait' 螺旋烧光 token 预算

- **证据**：17/200（8.5%）撞到 32768 token 上限且无代码；被截断输出平均含 87 个 'Wait'（最多 177），其余样本仅 1 个。Problem-3 末尾循环一张真值表："Wait, AND kills 1? AND receives (1,0)->0. Yes, AND kills 1..."
- **普遍度**：8.5% 截断、无最终代码。
- **分数影响**：截断样本无法产出代码块 → 0 分。

### 1.3 自信但错误的数学断言，从不验证

- **证据**：xor_sidon SFT："maximum even-even XOR is (2m-2) XOR (2m), which equals 2(m-2)^2 + 2(m-1)"。数值上 `(2m-2)^(2m)` 对 m=3,4,5,10,100 为 2,2,14,6,14，而该闭式给出 6,14,26,146,19406（仅 m=4 偶然对上）；为一个按位 XOR 编造了二次闭式，并把它当作区间分离的"证明"。
- **普遍度**：解释了"能编译但答错"——q35 编译 OK 的 31/33 仍得 0；q3 编译率 72% 但 94% 得 0。
- **分数影响**：是 compile-OK-but-wrong 的主因。

### 1.4 只验证极小样例，交付在规模上错误的构造

- **证据**：xor_sidon SFT 只验证 n=6,8,10 然后 "confirms the construction holds for larger inputs too"；n=1e7 需要 m≈2236，却交付了所有偶数+奇数（约 n/2 个元素，大量碰撞），且计数误写为 `n - odds.size()`。34/200（17%）在小/示例 n 上验证后即提交。
- **普遍度**：17% 明确只做小输入验证。
- **分数影响**：构造在约束规模下失效 → 0 分。

### 1.5 输出格式 / 交互协议违规

- **证据**：Problem 30 描述指数搜索但循环在第 1 次迭代无条件 break；Problem 20 一个畸形 query + 非协议答案；Problem 28 错误 map 格式、硬编码 n=4,m=8000；截断样本不输出 `!` 答案行。
- **普遍度**：在 <1500-tok 的 0 分桶常见（36/198），叠加 19% 无代码与 8.5% 截断。
- **分数影响**：违反交互/IO 契约 → 0 分。

### 1.6 重复 / 退化直到 token 上限（SFT 引入，research 轨道）

- **证据**：GPU SFT flash_attn（32768 tok，0 分）无限循环 `x=tl.zeros(...) / y=tl.zeros(...) / z... / p... / q... / r...` 循环 x/y/z/p/q/r；fused_linear_jsd `BLOCK_ROWS = BLOCK_ROWS + 1 ... - 1 ... if <=0: +1` 来回振荡；flash_attn 另一样本 `out = tl.where(out<0.0,0.0,out)` 重复几十次。单行最大连续重复：start=2，SFT=1591。
- **普遍度**：GPU 11/105 SFT 样本有退化尾部（gzip 压缩比 >6）vs start 0/105；6/105 SFT 撞 32768 上限 vs start 0/105。CPU 5/215 退化 vs start 1/215。**Start 在 GPU 上从不截断。**
- **分数影响**：主导 GPU nonzero 从 10/105 崩到 2/105。

### 1.7 幻觉的库 / 框架 API（research 轨道最大回退源）

- **证据**：GPU SFT vector_addition/2_24（start 50 → SFT 0）：`torch.from_dlpack(torch.cuda.IntDlpackPtr(output_ptr))`、`add_kernel[(1,)](x_ptr,...,block_size=2**24)`（不存在的 API、传 int 指针而非 tensor、一个 16M 线程的 block）。CPU SFT vdb_pareto/balanced（start 100 → SFT 0）：`faiss.IndexHNSWFlat(self.dim, M, max_links=..., quantizer=None, metric=metric)` 与 `self.index.search(xq, k=k, distances=..., indices=...)`（faiss 无这些 kwargs）。幻觉 Triton：`tl.static_mask` / `tl.static_less_than` / `tl.static_cast`。
- **普遍度**：驱动最大 CPU 回退：vdb_pareto/* 全崩（balanced 100→0、recall95 98.7→0、low_latency 69.9→0、high_recall 27.5→0）、imagenet_pareto/*（2_5m 44→0、500k 25.7→0、5m 11.3→0）、llm_router 26.1→0。Start 用真实 faiss API 正确解出这些。
- **分数影响**：是 CPU mean@5 13.695→10.750 中可定位回退的主因。

### 1.8 结构 / 输出契约破裂（无 class Solution、无代码块、不闭合 think）

- **证据**：GPU SFT decoding_attn/gemm_optimization/transformerish/qknorm 样本以空尾 `''` 结束、从未到达代码。`has class Solution`：GPU start 93/105 → SFT 56/105。CPU SFT llm_router 样本 5（21829 tok）在 `__main__` 块之后才定义一个空类：`class Solution: def solve(...): pass`；另一个写顶层 driver 且签名错误 `solution.solve("""...""", eval_name="mbpp", candidate_models=[...])` 还带散落的 `print()`。
- **普遍度**：GPU no-codeblock 5→15、never-closes-`</think>` 6→19、has-class-Solution 93→56（start→SFT）。CPU 较轻：class Solution 154 vs 155、no-codeblock 35→39、`__main__` driver 13→27、顶层 print 0→3、body-just-pass 0→1。
- **分数影响**：不符合 `class Solution.solve()` 契约 → 直接 0 分。

### 1.9 过度工程的脚手架，返回空代码或不可用代码

- **证据**：GPU SFT vector_addition/2_24 样本 5（8795 tok）建了 autotune/benchmark 框架 `def benchmark_all(stream): ... best=min(results,...)` 然后返回 `{'code':'', 'program_path': tmpfile.name, ...}`——空 code 字符串。CPU SFT vdb_pareto 样本 1 发明 'spectral hash'，用了未定义名 `matching_indices = np.flatnonzero((query_vectors @ base_vectors.T / dim)...)`（dim/h_refined 未定义）而非可用的 faiss index。
- **普遍度**：GPU 'autotune' 提及 24→28、'benchmark' 11→16；'返回空代码' start 11 vs SFT 10（两者都常见，但 SFT 配上了精致的死框架）。
- **分数影响**：贡献于 GPU nonzero 10→2 的崩塌。

### 1.10 有用的 SFT 偏置（唯一稳定增益）：在数学拟合任务上选对工具

- **证据**：CPU SFT symbolic_regression/mixed_polyexp_4d 0→90.9 best（干净的 `PySRRegressor(niterations=80, binary_operators=[...], unary_operators=["sin","cos","exp","log"], maxsize=25, ...); model.fit(X,y, variable_names=[...])`）；mccormick +45.5、peaks +20、ripple +4.5、grammar_fuzzing/seed/sql +10.0。
- **普遍度**：PySR 使用率 symbolic_regression start 9/25 → SFT 16/25；SFT 在 10/43 CPU 问题上更好，集中在 symbolic_regression 和几个 cant_be_late 变体。**这是唯一明确受益的家族；GPU 上完全不出现。**
- **分数影响**：正向，但不足以抵消其余回退。

### 1.11 MLS-Bench：唯一的"胜"是一次救活，不是普遍提升

- **证据**：SFT mean 0.07937 vs START 0.06431。SFT 超过 START 的任务仅：ml-clustering-algorithm +0.3875（0.3875 vs 0.0）、optimization-nas +0.0318、ml-dimensionality-reduction +0.0129。14/20 是完全平手（如 optimization-evolution-strategy 0.4866=0.4866）。一个**回退**：ml-symbolic-regression −0.1308（SFT 0.0 vs START 0.131）。整个 +0.015 的均值差几乎全来自 clustering 这一次救活。在该胜出 log 里，SFT 自己的编辑全被拒（'✘ ERROR: Package custom_clustering.py is not in allowed packages' ×2），它得 0.3875 只是因为对预置 baseline 文件跑了 `test` 然后 'submit / Submitting test #1 as FINAL'；START 在同任务 'No action returned after 3 attempts, stopping' + 'recording empty finals' → 0.0。
- **普遍度**：3/20 改善、1/20 回退、14/20 完全相同；均值 delta +0.0150，约 96% 由 clustering 贡献。
- **分数影响**：所谓提升是评分机制的副产物（对通过 `test`+`submit` 的任意文件读分），非研究质量提升。

### 1.12 MLS：提交/收尾纪律才是 SFT 真正改善的东西

- **证据**：'recording empty finals'（强制 0 分）在 START 13 个任务出现 vs SFT 仅 8 个；'done: True' SFT 15/20 vs START 11/20；硬 0 分 START 13/20 vs SFT 11/20。START 反复卡住：optimization-evolution-strategy 跑满 20 步、'Test budget exhausted (3/3). You MUST call submit(n=N)' 然后 'recording empty finals' → 从未提交；SFT 提交并收尾。SFT 还更省：completion_tokens 1.12M vs 2.65M、llm_calls 239 vs 404、✘ERROR 106 vs 132、mean elapsed 951s vs 1160s。
- **普遍度**：少 5 个 empty-finals（13→8）、多 4 个 done（11→15）、少 2 个硬 0（13→11）；token/调用/时间约减半。
- **分数影响**：正向但"空心"——它收尾的是 baseline 质量的文件。

### 1.13 MLS：解码崩溃 / 乱码 token 爆发（至少一个任务灾难性）

- **证据**：optimization-nas SFT 'Step 2 edit' 输出整页退化 token：`# best_auto.py — THREADF interprens encoding NAS search for K=30 / # EDI: F SD REV U EVER,$I"$DI'n D FSTE'$DI! ... $DI$DI,rb,DP,^EI,r,D:b,$DI$DI ...`（单个 log 里 `$DI` 出现 1220 次；任何 START log 中 0 次）。模型从未恢复有效编辑；NAS 得 0.0318 只因预置 baseline 在 CIFAR-100 上有分。
- **普遍度**：集中：1/20 任务（optimization-nas）完全崩溃，1220 个 `$DI`；20 个 START log 中全部为 0。
- **分数影响**：该任务的有效编辑全部作废。

### 1.14 MLS：浮夸"新方法"命名挂在非功能/被拒代码上（研究取向的表演）

- **证据**：clustering = `"""MiMiClust: Manifold-Informed Multi-Scale Clustering.` 带 '17th-Power Local Density ... density_power=17'；NAS = `POP-TA (Portfolio of Policies with Thompson-Sampling Dimensional Annealing)`（字符串 'POP-TA' ×11）；还有 `MetaVNS`。MiMiClust 与 POP-TA 编辑都被 harness 拒（'is not in allowed packages' / 超出可编辑范围），这些"贡献"从未运行——得分的是普通预置 baseline。
- **普遍度**：命名方法 docstring 几乎遍布所有 SFT 解法；在检查的任务中命名方法被拒/乱码 2/2。
- **分数影响**：风格迁移、能力未迁移；命名不带来分数。

### 1.15 MLS：编辑协议 / 禁止包摩擦仍主导两个模型的失败（基础设施，非科学）

- **证据**：两模型都反复试图新建文件而非就地编辑：SFT 28× 'not in allowed packages' + 14× 'X and X are required for op=X'（畸形 replace，缺 start_line/end_line）+ 2× 'allow_create is false'；START 37× 'not in allowed packages' + 23× malformed-replace + 9× 'allow_create is false' + 12× 'exceed the editable range'。SFT 还浪费测试预算：26× 'Test budget exhausted (3/3)'。
- **普遍度**：硬错误总数 SFT 106、START 132；'not in allowed packages' 是两者最常见单一错误（SFT 28 / START 37）。这些基础设施/格式错误锁死了大多数 11–13 个 0 分。
- **分数影响**：是两模型多数 0 分的门槛因素（与训练配方无关）。

---

## 2. 根因映射（失败模式 → 数据/管线根因，按影响排序）

> 排序依据：对竞赛分数的可定位影响大小 × 普遍度。

**R1（影响最大）全参 SFT 覆写了代码电路。**
→ 失败模式 1.1。证据：q35 比 q3 更差；souping 恢复编译率（q3_soup10 84% vs q3_method_sft 72%）。在 prose-heavy 的 reasoning trace 上做全参微调，盖掉了基模型可编译代码的能力。

**R2 训练数据 ~0% 的交付后验证 + 逆向工程"总是落地"的 trace。**
→ 失败模式 1.3、1.4。证据：仅 2.2%（28/1245）reasoning.md 含落地/IO/测试标记；88% 含代码的 reasoning 在最后代码块之后没有验证语言。eval 中验证语言从 33%（q35 inst_start）降到 20–22%（method_sft），cpu 72/215→48/215、gpu 35/105→22/105。根因：trace 从"已完成、假定正确"的论文逆向得到，paper-to-reasoning §2.2 明确禁止运行/测量（"recall and reason; do not measure"），所以验证在每个目标里结构性缺席；语料里**没有任何失败样本**。

**R3 train_answer.md 落地是研究库（Python class），几乎没有单文件 C++/stdin 交付。**
→ 失败模式 1.7、1.8。证据：train_answer.md 98.6% 是 Python、0.2% C++（2/1233）、1.9% 读 stdin（23/1233）、58.4% 定义 class。eval 中 research_cpu/gpu 落 `class Solution` 72%/53%、读 stdin 0%；algorithm 因 methodtraj 携带单个 AHC039 stdin C++ golden seed 才有 97% cpp、87% stdin，但产出是不可编译垃圾（`int root = find((long long)n*n/n, m);` m 未定义、`while(G.empty());` 死循环）。根因：build_sft.py 把 train_answer.md verbatim 作为落地；唯一的 stdin/C++ 范例只有一条轨迹，于是被当模板**记忆**而非泛化。

**R4 method/discovery SFT 增加了"继续扩展思考"的倾向、降低 EOS/commit 概率。**
→ 失败模式 1.2、1.6。证据：截断样本平均 87 个 'Wait'；GPU 退化尾部 11/105（SFT）vs 0/105（start）；单行最大重复 start=2 → SFT=1591。research 推理 repetition_penalty 当前为 1.0。根因：method/reasoning trace 长且自我展开，微调放大了不停展开思路、在难题上落入 n-gram 循环。

**R5 system prompt + 格式说明强加"good researcher / narrative tone"register。**
→ 失败模式 1.9、1.14（及 1.11 的表演成分）。证据：build_sft.py system prompt 曾为 `"It is now year {year}. You are a good researcher."`（无交付纪律，现已增补 _DELIVERY 行 45-52）；格式说明曾为 'narrative, telling tone rather than a heavily formatted writeup'（现已重述为以可运行交付收尾，行 56-65）。eval 中 bold '**Key Insight:**' register 从 20%→41%（cpu）、11%→60%（gpu）。根因：每个样本都被字面指示研究叙事 register，与 FCS/ALE 要求的可执行交付纪律相反。

**R6 逆向工程"端点已知 + 模板化开场"→ 自信落地错误方法/证明。**
→ 失败模式 1.3、1.14。证据：top-2 开场 stem（'let me start from what actually...' 18%、'ok, let me think this through' 13%）覆盖 31% 的 1245 条 trace；eval 中 q35 algorithm methodtraj_sft 写出自信的**假证明**'I'll now prove this formally... XOR acts as addition without carries' 然后交付不可编译代码。根因：skill 的逆向哲学 + 缺少任何被放弃/纠错的轨迹，让模型学到"发现的戏剧表演"而非校准的不确定性。

**R7 per-rung 折叠把当前 rung 目标缩到约半个方法的推理长度 → 思考过短。**
→ 失败模式 1.1、1.5（思考不足导致过早出 buggy 代码）。证据：trajectory per-rung reasoning 中位数约 10,672 字符（约 2668 tok）vs method reasoning 约 21,559 字符（约 5389 tok）；eval head-to-head：methodtraj_sft 中位 1793 tok（q3 algo）vs method_sft 5035（−64%）、2946 tok（q35 algo）而其 soup10 恢复到 20,174 tok（证明是数据而非架构缩短了输出）。根因：build_sft.py Mode-2 折叠（行 164-174）把每个 rung 的推理对着被剥离的历史训练。

**R8（基础设施，非配方）MLS 编辑工具语法/禁止包从未被教。**
→ 失败模式 1.15。两个训练 regime 都没教 op + start_line/end_line + 不可新建文件 + 允许包白名单；两模型默认"写新文件"，被沙箱禁止。这门槛锁死多数 MLS 0 分，与 SFT 无关。

**R9 souping 把权重平均回退向基模型行为。**
→ 失败模式 1.6（GPU 退化）、思考病理 4.2/4.3/4.4。证据：soup10 推理长度/上限溢出/give-up 明显比对应 SFT 严重（如 'I'm stuck' q3 method_soup10 58% vs methodtraj_sft 3%）。

---

## 3. Leakage 结论：rewrite/meta 泄漏是否真在训练数据里？

**结论：真正的"构造过程泄漏"在训练数据里基本不存在——这是 model-only（具体说是 q35 基模型）的产物，而 SFT 反而把它压低约 3–4 倍。**

证据（全语料扫描 + 分层抽样）：

- "I cannot complete this thought / no thinking process to rewrite here / the next thinking is empty/garbled" 这一族：在训练数据里 **0/1245 reasoning、0/1243 answer、0/1233 train_answer、0/680 traj、0/127 agentic**。
- 但在**模型输出**里集中于 q35 家族：q35_inst_start research_gpu 54 次 / 41.0% 样本、research_cpu 40 次 / 22.8%；SFT 降到 research_gpu 14（11.4%）、research_cpu 10（3.7%）；soup10 又放大（research_gpu 47、38.1%）。**整个 q3 家族（start/sft/soup/method/methodtraj）= 0**。
- eval prompt 从不提及 'thinking'/'rewrite'，证明非 prompt 诱导。

判定：这是 Qwen-3.5 基 instruct checkpoint 内化的"改写/总结所提供 <think> trace"任务 schema——当它自己的部分生成看起来像原始代码/二进制时，它就叙述"源 thinking 为空/乱码"，闭合 `</think>` 然后倒出 buffer 里的代码形状 token。**SFT 已经把它压低 3–4 倍**，souping 把它拉回基模型行为。

误报（**不是**泄漏，无需修数据）：
- 'in the paper' / 'the paper'：均为对源方法**原始论文**作为外部对象的合法比较（如 noisy-nets、dino、ml-continual-regularization），在框架内（"我的方法 vs 已发表的先前方法"）。training 命中 3/1245 reasoning、12/1243 answer，全部 FP。
- 'let me rewrite / restructure'：均针对数学表达式或代码块的就地自我纠正（adaboost 改记号、dllm-cache、llm-pretrain-loss、security-machine-unlearning）。7/1245 reasoning，全 FP。
- '# ...' / 'omitted for brevity' 占位：'# ...' 是 q3 基模型 sketch-then-not-fill 习惯（q3_inst_start research_gpu 78/105，SFT 降到 12），训练数据里 'omitted for brevity' 仅 1/1245，且都是刻意、有说明的省略。
- refusal / 'as an AI' tells：训练数据真正泄漏 = 0；2 个命中分别是 constitutional-ai 把 refusal 作为**研究主题**的正确引用、以及 'rephr-as-ed' 单词内的子串 FP。

---

## 4. Thinking 病理（empty-think 叙述 / as-an-AI / 不终止）

> 说明：以下均为**假设**，需待单独的 Qwen3.5 chat-template 研究确认（见 §5d）。基于文本场统计（2955 样本）。

### 4.1 Empty/garbled-think "I cannot complete this thought"（reasoning-fold 产物）

- **现象**：模型叙述一个不存在的"next thinking"去改写/总结，然后闭合 `</think>` 倒代码。
- **普遍度**：依正则松紧 150–194 样本（约全体 5–7%；约 q35 样本的 13%）。**专属 q35**：q35=194 vs q3=6（q3 的 6 个是子串 FP，cannot_complete_thought 对每个 q3 tag 都是 0）。**SFT 减少**：research_gpu 37%→10%、research_cpu 16%→3%；soup10 放大。
- **两条来源（并存）**：（a）基模型 Qwen3.5 的"改写/总结所提供 think"schema（§3 已述）；（b）我们 build_sft.py 旧版 fold_think 把折叠历史的 reasoning 替换成字面空块 `<think>\n\n</think>`，2738 个折叠历史 turn 含此 → 训练让模型把"空 think"当成可评论事件。**后者已修复**：当前 build 把 `<think>...</think>` 整块从折叠历史移除，已验证 sft/innovation_sft.jsonl 中 0 个 empty-think 块；这些 dump 来自 pre-fix 模型。

### 4.2 'As an AI' RLHF 人格在 `<think>` 内复现

- **现象**：基模型 assistant/RLHF 身份（'I am an AI text model'、'AI assistant'）渗进 CoT，然后为草率倒代码找理由（'I must provide the code now'）。
- **普遍度**：31–33 样本（约 1%）。集中在 q35 algorithm soup10：methodtraj_soup10 19/240 + method_soup10 11/240 = q35 的 31 个命中里的 30 个。**q35 SFT-only 与 q35_inst_start（=0）几乎没有**——是 q35 + **权重 soup** 现象。
- **假设**：weight-soup（10 checkpoint 平均）部分恢复了单 checkpoint SFT 抑制掉的基模型 assistant-persona 方向；长 token（20k–32k）样本里它表现为放弃式的合理化。

### 4.3 撞 32768 上限却从未闭合 `</think>`（runaway reasoning）

- **现象**：文本在上限处戛然而止，或一直在叙述意图、退化重复填满上限，从不到达真正代码块。
- **普遍度**：120 样本（约 4%）；117/120 是 q35（q3=3）。主导 q35 algorithm soup10：method_soup10 49/240、methodtraj_soup10 53/240 vs methodtraj_sft 仅 14/240。**后果严重**：methodtraj_soup10 的 at-cap-no-close 样本 51/53 得 0、33/53 从未输出任何代码栅栏。≥32000 tok 的样本总数 184，几乎全 q35。
- **假设**：Qwen3.5 是 thinking-mode（config_used.yaml verbatim："spends most tokens inside <think>"）；soup 拉长/破坏推理（soup10 均值约 17k tok vs methodtraj_sft 约 3–6k），在 max_model_len 内来不及出 `</think>` → 无代码块 → 0 分。

### 4.4 中途放弃（'I'm stuck' / give-up prose）

- **现象**：q3 在散文里放弃；q35 则跑到上限。
- **普遍度**：按 'I'm stuck'：489（q3）vs 106（q35）——**主要是 q3 现象**，几乎是病理 4.1/4.3 的反面。q3 algorithm methodtraj_soup10 150/240（63%）、method_soup10 140/240（58%）、method_sft 60/240（25%）、methodtraj_sft 仅 8/240（3%）。
- **假设**：q3（非 thinking-mode）推理更短，缺乏长思考预算，在真正难的构造/NP-hard 问题（knight's tour N=666、packing N=1e4）上口头放弃而非螺旋。soup-vs-sft 巨大差距（58% vs 3%）表明 10-checkpoint 平均强烈放大了放弃倾向。

> **注意**：Qwen3.5 chat-template 是否真的注入"改写/总结 thinking"turn，是一项**独立**的待办研究（§5d）。4.1–4.4 的归因都依赖它，目前是有据假设而非已证事实。

---

## 5. 解决方案（具体、按优先级）

### (a) 优化现有数据

1. **补验证-修复段落**（最高优先，对应 R2/失败 1.3/1.4）：用 code-reading subagent 给现有 reasoning.md 追加真实第一人称验证段——在具体输入上 trace、走边界（n=1/溢出/空）、找到并修复 ≥1 个真 bug（DATA_REMEDIATION track A2 / augment_verify_workflow.js）。让"先验证再交付"出现在被训练的落地里。
2. **去模板化开场**（对应 R6）：对前 N token 近重去重或在 SFT 中下权 'let me start from what actually...' / 'ok, let me think this through'，打破"制造发现"指纹。
3. **加长/加深 per-rung 推理**（对应 R7，DATA_REMEDIATION #8）：每个 rung 目标追加信息密集（非填充）的验证段，把训练的"该思考多长"锚点拉回全问题长度；或减少折叠激进度 / 混入更多全长 method trace 重新锚定。

### (b) 新数据

4. **加单文件 C++ stdin 交付底线**（对应 R3，DATA_REMEDIATION track B）：扩展 AHC039 explore→fallback→land 模板，build_v4.py 产出读 stdin 的单文件 C++；用 sft/_sft_tags.jsonl 已有的 reads_stdin/has_cpp/defines_class 标签封顶 class-library 落地，要求 ≥15–20% 可执行交付样本（当前约 2%）。
5. **注入真实失败/放弃样本**（对应 R6，DATA_REMEDIATION B1c）：探索被验证为错/不确定后，**故意回退到更简单可验证的解**而非硬落一个花哨未验证的方案；加上"雄心但无效的编辑被修正为小而可用的 diff"的负例。
6. **库使用 trace 必须可运行**（对应 R3/失败 1.7）：每条 'use library X' 都配一个已验证、最小的真实调用（如真实 faiss/PySR API），保留 1.10 的有用偏置而不引入幻觉 API。
7. **MLS 编辑工具语法范例**（对应 R8/失败 1.15）：加正确 edit-tool 调用（op + start_line/end_line 在可编辑窗口内、不新建文件、遵守允许包白名单），并带"被拒新建 → 有效 in-range replace"的失败→纠错模式。
8. **提交-前-预算模式配真实有效编辑**（对应失败 1.11/1.12）：保留并放大 submit-before-budget 习惯，但每个 submit 前都配一个**有效、提分**的编辑，使纪律不空心。

### (c) 管线 / build 修复

9. **gate 执行**（对应 R1/失败 1.1、1.7、1.9）：SFT 代码目标过 compile-and-run gate，只保留实际跑过/得分 >0 的 trace；对返回的 `{'code':...}` 做编译+非空+无未定义名检查。
10. **降 LR / 改 LoRA / 混入纯可编译代码 SFT**（对应 R1）：避免全参覆写代码电路。
11. **重做 souping 配方**（对应 R9/病理 4.2–4.4）：排除或下权重新引入 assistant persona、拉长推理、放大放弃倾向的 checkpoint；或 soup 后做轻量 anneal（在 methodtraj_sft 集上，它已把 'I'm stuck' 压到约 3%）。
12. **system prompt / 格式说明已部分修复**（对应 R5）：build_sft.py 已追加交付纪律（_DELIVERY 行 45-52）、格式说明已重述为"以最终可运行代码收尾，而非草图"（行 61-63）。保留第一人称分析声音（MLS 增益来源），但确保每条都以可运行交付要求收尾；用 emitted tags 验证落地仍可执行。
13. **fold_think 已修复**（对应病理 4.1b）：当前 build 把折叠历史里的 think 整块移除，已验证 0 残留 empty-think 块。

### (d) 推理时护栏 + 接下来要验证的

14. **推理时**：提高 repetition_penalty（当前 1.0）+ no-repeat-ngram guard + 预留答案块 token 预算；对 q35 thinking-mode 评测提高 max_model_len 或加 token 预算处的强制 `</think>`+出代码；eval 时检测 'I cannot complete this thought / next thinking is empty' 与 at-cap-no-close 并丢弃/重试。
15. **下一步要验证（measured vs hypothesized）**：
    - **[待证]** 跑独立的 Qwen3.5 chat-template pass，确认模板是否注入"改写/总结 thinking"turn——§4 全部归因依赖此项。
    - **[待证]** 验证 gate/compile-run filter 与库-grounding 在重训后是否真把 algorithm 99%-0 和 GPU nonzero 10→2 拉回（目前 souping 恢复编译率只是侧证）。
    - **[待证]** prompt tag q35_a100_method_sft / q35_inst_start 在 raw dump 中缺失，algorithm 切片用 methodtraj_sft 代替 q35；需对齐 tag 后复核 algorithm 的 SFT-vs-start 精确对比。
    - **[已测]** train_answer 语言/stdin/class 分布、reasoning 验证语言比例、empty-think/at-cap/'I'm stuck' 计数、MLS per-task delta——这些是统计实测，可直接据以行动。

---

### 测量 vs 假设的诚实边界

- **已测**：所有计数、比例、per-task/per-problem 分数 delta、verbatim 片段、训练语料的属性分布（语言/class/stdin/验证语言/开场 stem）。
- **假设**：empty-think 叙述、as-an-AI、不终止的**因果归因**（依赖未完成的 Qwen3.5 template 研究）；"全参 SFT 覆写代码电路"由 q35-vs-q3 与 souping 恢复编译率**间接**支持，未做机制级消融；algorithm 切片的 tag 替代（methodtraj_sft 代 q35）是已知的对齐缺口。
