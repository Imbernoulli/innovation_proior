# r3 创新先验 Case Study(基于最新 r3 结果)

> 基于 **r3 代**模型(methodv4_r3 / methodtraj_v4_r3 / full_r3 + soup),在**发现类任务**上逐题对比我们的模型 vs base 的**真实生成代码**,找 out-of-the-box 的具体证据。这是对早前 r1/r2 版 case study 的更新。
> 口径:所有引用均来自评测的 `text`/`task_logs` 真实生成;高方差任务落到逐题代码差异;诚实标注 base 崩 vs base 偷懒、以及"靠交更简单代码赢"的情况。

---

## Part A — FrontierCS **Research**(64 题;methodv4_a20=10.32 / full_wd0_a20=10.29 vs base 9.08)

> 说明:22 个 GPU/Triton kernel 题在 CPU 机上三方全 0(RNG 而非能力差),已剔除;symbolic_regression 是 PySR 随机种子噪声,不算。以下 4 例都是**双方都产出完整代码**、差异在算法质量。

| 问题 | base(崩/naive) | ours(结构洞察) | Δ |
|---|---|---|---|
| **vdb_pareto**(ANN 向量索引,硬延迟门) | 幻觉出不存在的 FAISS API(`from faiss import IVFFlat,IPQC...`),不可 import → 0.0 | 正确 `IndexHNSWFlat` + 调 `efSearch`(召回/延迟的真旋钮) | **+77.3** |
| **imagenet_pareto**(200K 参数硬上限) | 猜 3 层架构、只用了 ~64% 预算,19.1 | **解析求解饱和上限的最大隐层宽度** `max_hidden=(limit-C)/(in+C+1)` | **+46.9** |
| **cant_be_late**(spot/on-demand 在线调度) | 状态读错对象(`self.env.*` 全是 AttributeError)→ 0.0 | 正确 Strategy 属性 + slack-vs-work 缓冲 + 渐进 `safety_ratio` | **+47.0** |
| **llm_sql**(列重排最大化 KV 前缀复用) | 随机搜索 + 语法错代码 → 0.0 | **按列基数排序(低基数在前→共享前缀更长)** + 贪心构造 | **+52.6** |

**诚实反证**:`llm_router` base 赢(50.4 vs 25.4);一个 `cant_be_late_multi` base 峰值 100 但方差大 `[0,0,87.8,100,87.8]`,ours 稳在 87.85(**抬地板而非冲峰值**)。

**A 部分结论**:r3 模型反复抓住"承重结构"(efSearch 旋钮、参数上限求解、slack 缓冲、低基数前缀);base 要么幻觉 API、要么用不满预算、要么读错对象、要么退化成随机搜索。

---

## Part B — MLS-Bench(20 题;methodtraj_a10=0.091 / methodv4_a10=0.082 vs base 0.038)

### 真·创新赢(想法更 out-of-the-box **且**跑通)
**B1. causal-discovery-discrete(离散贝叶斯网 CPDAG 恢复)** — traj **0.3027** vs base 0.0
- base:想用 PC+GES 混合,但导入报错、改到 BOSS 又没测就 undo,最后交了**空图**(SHD=每个网的满边数)。
- ours(traj):也先试了花哨的自适应 CI-test(崩了),但**恢复到正确的三段式**:`chisq CI → SkeletonDiscovery(stable-PC) → UCSepset 定向 collider → Meek 闭包`。真恢复出结构(Alarm SHD 46→11、AdjP 0.95)。

**B2. causal-treatment-effect(CATE 估计,有混杂)** — traj **0.2962** / method 0.2775 vs base 0.0549
- base:DR/X-learner 拼凑 + 手调 balance 权重,泛化差(PEHE_ihdp=1.744,ATE 误差 1.503)。
- ours:**理论支撑的正交化估计**(Robinson 残差化 → quasi-oracle R-learner,双残差 + GBM nuisance)→ PEHE 1.104、**ATE 误差 0.105(14× 降低)**。**两个 r3 build 都体现这种"正交化"倾向**。

### 诚实标注的"非创新赢"(重要:不夸大)
- **B3. MoE 负载均衡** — traj 0.372 vs base 0.255,但 balance 质量其实**打平**(base 甚至略好),ours 赢在**把算法向量化到 ~6ms(runtime 项)**。→ **工程 runtime 赢,非更聪明的均衡器**。
- **B4. calibration** — base 的花哨 ECE-最小化想法**崩了**;ours 也先试花哨的再**退回教科书 isotonic regression**跑通(ECE~0.012)。→ **靠交更简单能跑的代码赢**,是恢复力不是新颖性。
- **B5. evolution-strategy** — 两 build 同分 0.4866 = 都是**保留了能跑的默认模板**(改 CMA-ES 的编辑失败了),而 base 把工作模板改崩了。→ **纪律/鲁棒赢**。

---

## 跨任务总结(诚实定性)

**每个案例里双方都先冲一个野心方案**,而创新先验训练一致地帮模型:
1. **在结构化因果任务上"接住"复杂正确方法**(stable-PC 管线、doubly-robust R-learner)——真创新;
2. **在别处从失败里恢复到能跑的方案**(calibration 退回 isotonic、evolution 不改崩模板)——纪律/鲁棒;

而 base 更常**过度伸手然后崩**(calibration/evolution)、**读错对象**(cant_be_late)、或**退化成空 stub/随机搜索**(causal-discovery/llm_sql)。

**一句话**:r3 创新先验在**结构化发现任务上提升"创造性触及"**(抓承重结构、接住理论方法),在其它任务上表现为**失败恢复的工程鲁棒**;在奖励 runtime 或简单-正确代码的任务上,则体现为工程稳健而非算法新颖。这与定量结果一致——发现类(Research/MLS/Theta)上 a10/a20 低-alpha soup 稳超 base,而竞赛类(FCS)不提点。

---

复现:所有 TAG / problem_idx / sample_idx / task 目录均见正文;Research 样本在 `FrontierSmith/outputs/cc_eval_<TAG>_research_thinking_32k_vllm/shard_0/samples.jsonl`,MLS 在 `cc_eval_all_r3_<TAG>/mls/task_logs/`。姊妹文档:`INNOVATION_CAMPAIGN_REPORT_zh.md`(公平对比主报告)、`CASE_STUDY_zh.md`(r1/r2 版:FCS/MLS/Theta/ALE)。
