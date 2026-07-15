# 控制变量:SFT ckpt(α=1)vs 它的平均体 —— 创新能力到底减弱了多少?(2026-07-14)

> 姊妹文档 [AVERAGE_INNOVATION_zh.md](AVERAGE_INNOVATION_zh.md) 给了 α 聚合曲线;本文下沉到**同一 SFT run、同一题集、同 n=5、逐生成对读**的控制变量研究,把"创新能力"拆成**意图(intent)**与**兑现(success)**两层分开测,并对每个比率同时报**按字符归一**版本以排除"SFT 话多/话少"的长度混淆。
> 数据:nomaintain_wd01 / full_wd01 两家族(同 SFT 权重,仅 α∈{0,.05,.1,.2,.3,.5,1} 不同),FCS 172 题×5、Research 64 题×5、MLS 20 任务 agent 日志,全部离线在盘;newmt/wd03 直测+新网格出分后并入(job 见姊妹文档 §6)。
> 脚本:`experiments/scripts/agg/soup_intent_vs_success.py`(意图/兑现/多样性/指纹偏离)+ `soup_jaccard_alpha.py`(MLS 编辑 Jaccard)+ 人工头对头精读(§4)。

## 0. 一句话定量结论(可引用)

**α=0.1 相对 α=1(nomaintain 臂):创新意图分维度保留 = 计划体密度 0%(KI/10k 字符 0.39→0.05,回到 base 水平)/ 方案分布宽度 0%(同题 5 样本 distinct-方法簇 4.23→2.23 ≈ base 2.26)/ 方法指纹偏离 0%(FCS)–38%(MLS Jaccard,基准面依赖)/ MLS 自命名提案 >100%(a10 反而 5.9 个/MB vs SFT 1.4,反超 4 倍);而创新兑现率反向:意图样本 score>0 从 3.8% 升到 11.9%(3.1×),全样本 1.3%→15.2%(FCS,11.9×)、7.2%→20.9%(Research,2.9×)。Average 削弱的是创新的表达浓度与方案分布宽度,兑现能力反升 3–12 倍。**

**长度归一后一个预期翻转(诚实):**"SFT 意图更强"只在 KI 计划体一项成立(每字符 7.6×)。多方案枚举的**每字符密度**SFT 反而是全场最低(0.034/10k vs base 0.052、a10 0.054)——SFT 的"创新腔"是**短文本里的高浓度计划体宣言**,不是更宽的探索;它每题考虑的方案并不比 soup 多,只是说得更像论文。

## 1. 设计

- **控制**:同一 SFT 权重(`sft_q35_clean_nomaintain_wd01` / `sft_q35_clean_full_wd01`),唯一自变量 = 与 base 的插值比例 α;同题、同采样协议、同 n=5、同判分。
- **意图检测器**:(a) 自命名提案 = 缩写+Title-Case 展开(`TeeMOEA: Tangential …` 式)∪ "I'll/let's call this/it" ∪ "dubbed/termed" ∪ "novel contribution";(b) 多方案枚举 = distinct `Approach/Option/Idea/Method/Plan + 编号` 标签数(≥2 记多方案);(c) KI 计划体 = "key insight"/万字符;(d) 偏离 = 方法指纹(FCS:23 个算法关键词;Research:import+API 调用)对 base 同题指纹并集的 Jaccard,以及 MLS 编辑 Jaccard(附录 A 原协议)。
- **兑现** = score>0(FCS/Research 官方判分),分意图样本/非意图样本两组报。
- **宽度** = 同题 5 样本方法指纹的贪心聚类簇数(Jaccard≥0.4 归并)。

## 2. 意图层 vs α(nomaintain 臂;full 臂方向一致,见脚本输出)

| α | MLS 自命名(个/MB 日志) | Research 自命名(样本占比) | FCS 多方案 share(raw) | FCS 枚举密度(/10k 字符) | KI/10k | FCS 指纹 sim-to-base ↓=更偏离 | 同题 5 样本方法簇数(FCS) |
|---|---|---|---|---|---|---|---|
| 0(base) | 1.7 | 1.3% | 16.2% | 0.052 | 0.051 | (0.54*) | 2.26 |
| 0.05 | 1.9 | 0.9% | 15.7% | 0.048 | 0.047 | 0.475 | 2.33 |
| 0.10 | **5.9** | 0% | 15.7% | 0.054 | 0.050 | 0.481 | 2.23 |
| 0.20 | 2.8 | 0.9% | 14.8% | 0.054 | 0.059 | 0.451 | 2.44 |
| 0.30 | 5.1 | 1.6% | 10.1% | 0.051 | 0.115 | 0.330 | 3.17 |
| 0.50 | 5.1 | 0.3% | 3.7% | 0.056 | 0.424 | 0.108 | 4.40 |
| 1.0(SFT) | 1.4 | 1.6% | 1.3% | **0.034** | 0.389 | 0.117 | 4.23 |

\* base 行的 sim-to-base 含自身样本,系统性偏高,只作方向参考;soup/SFT 行无此偏。

**读法:**
1. **自命名提案根本不是 α=1 的强项。** MLS agent 日志里,去掉误报(IMPLEMENTATION/CONTRIBUTION 等)后 distinct 命名:base 1(GraphLASS)、a10 5(TIC-IR/ET-Learner/IA-NAS…)、a30 3、a50 4、**SFT 只有 1(ClusterGEM,且展开句已语无伦次:"Gaussian Embedding of Mixture through two-phase L2 crime")**;full 臂同形(a5/a30 5-6 个,SFT 3 个)。命名行为是**倾向×执行续航的交互效应**:SFT 有倾向但 agent 早死(日志 0.7MB vs 1.0-1.2MB),mid-α 才既想命名又活得到命名。Research 上两端都稀少(≤1.6%),无 α 梯度。
2. **多方案枚举的 raw share 随 α→1 从 16% 掉到 1.3%,但这是长度效应**:每字符枚举密度 α 无梯度(0.048-0.056),α=1 反而最低(0.034)。**多方案探索意图并没有"注入-稀释"结构——它本来就主要是 base 的长思考习惯。**(头对头佐证:imagenet_pareto 上枚举 Option 1/2/3 的是 a10,不是 SFT。)
3. **KI 计划体是唯一真正的 SFT 签名意图**(每字符 7.6×),其 α 稀释曲线 = 姊妹文档 §2.2(α≤0.2 归零)。
4. **方法指纹偏离随 α 单调**,但保留率依基准面而异:FCS 指纹 a10≈0%、a20≈6%、a30≈40%、a50≈100%;MLS 编辑 Jaccard a10=38%、a50=81%。**竞赛题上 a10 的行为几乎完全塌回 base,开放 agent 任务上偏离倾向存活得多**。
5. **方案分布宽度(5 样本簇数)是 average 真实削掉的东西**:SFT 4.23 → a10 2.23(=base),保留 0%;要到 α=0.3 才保留 46%。诚实警告:α=1 的高簇数部分来自**代码不连贯导致的指纹散射**(见 §5),4.23 是"宽度+噪声"的上界。

## 3. 兑现层 vs α(创新意图样本的 score>0 率)

| α | FCS 意图样本兑现 | FCS 非意图样本 | FCS 全样本 | Research 全样本 |
|---|---|---|---|---|
| 0(base) | 12.7% | 20.3% | 15.6% | 21.3% |
| 0.05 | 11.6% | 19.0% | 14.4% | 21.9% |
| 0.10 | **11.9%** | 20.5% | 15.2% | 20.9% |
| 0.20 | 10.5% | 22.0% | 15.4% | 20.0% |
| 0.30 | 8.9% | 12.8% | 11.3% | 16.9% |
| 0.50 | 11.0% | 4.8% | 5.5% | 13.1% |
| 1.0(SFT) | **3.8%** | 1.2% | 1.3% | 7.2% |

- **兑现率在每一档 soup 都数倍于 α=1**;SFT 的意图样本 26 个里只有 1 个得分。
- 意图样本兑现率在低 α 段**低于**非意图样本(12.7% vs 20.3%@base)——"越 novel 越容易崩"的老混淆(附录 A A.3)在受控设置下依然在:探索型生成本身就更难落地,与 α 无关。
- Research 端 conv_intent 小样本(每档意图样本仅 10-17 个)不可单独引用;全样本 score>0 的 α 梯度(21%→7%)是稳的。

## 4. 同题头对头精读(SFT 意图最强的 8 道 Research + 2 个 MLS 任务;SFT vs a10 vs a50)

| # | 题 | SFT(α=1) | a10 | a50 | 一行判定 |
|---|---|---|---|---|---|
| 1 | cant_be_late/low_tight_small | 命名("call this L")+真策略洞见("NEVER NONE→SPOT, 用 OD 追进度"),代码自比较恒假 → 0/5 | 无命名,平实 urgency-ratio 内核 → **65** | 短 think 同内核 → 65(2/5) | **掉的是洞见的表达;得分内核 soup 自产,SFT 的更深洞见从未进代码** |
| 2 | cant_be_late/mixed_tight_small | "I'll call it"+31k 探索(think 里跑 Python)→ 交纯 ON_DEMAND fallback+坏 pop → 0/5 | 平实、带状态跟踪实现 → **46** | 保留枚举(Pure Spot/Pure OD/Hybrid)+类名自命名,但引用未定义属性 → 0/5 | 意图 SFT 最强、a50 半保留;兑现只在 a10 |
| 3 | cant_be_late_multi/high_tight_small | "I call this"+23k 环境逆向工程,`while j>i: if j==i` 死代码 → 0/5 | 常规分析 → **27**(2/5) | 1.2k 短 think → 0/5 | 同 #1:想得最深的写不出 |
| 4 | cant_be_late_multi/high_loose_large | 自命名启发式 **"Conservative Spot First"**(名字+原则齐全)→ 0/5 | 无命名常规 → 19(2/5) | `greedy_spot` 简单正确 → 19(2/5) | **SFT 的命名启发式与 a50 的 greedy_spot 内核相同(spot 优先+OD fallback)——表达消失,方法等价,差别只在能不能跑** |
| 5 | cant_be_late/low_loose_large | **全组唯一真非平凡框架**:`gap-committed-forward-greedy`(按 gap 承诺+前向状态最小成本),但 `spec_path.parse_args`(对 str!)+未定义 `_forward_step` → 0/5 | 平实 greedy → **58**(1/5) | 概率化 spot 选择(小创意)→ 58/33(2/5) | **average 确实掉了这个真创新框架——但该框架在 α=1 也从未跑起来过** |
| 6 | imagenet_pareto/1m | 69k 字符预算推演(含自纠 "WAY OVER BUDGET!"),**答案为空** → 0/5 | **枚举 Option 1/2/3**,选型正确可跑 → 0(没超 80% 线) | 直接 MLP → 0 | 多方案枚举在 a10 出现——枚举非 SFT 专属;全员没过线 |
| 7 | nbody/random_10k | 枚举 Option 1-3,选对 spatial hashing+复杂度推演(1250×),代码幻觉类型 → 0/5 | 同款 spatial hashing,结构合理 → 0 | bounding-box+quadtree 混合 → 0 | **方法选择三档同族**;全员没跑通 |
| 8 | vdb_pareto/low_latency | Approach A/B+"IVF+HNSW refinement"组合设计,import 幻觉 API → 0/5 | 保守 HNSW 调参,构造签名错 → 0 | **自设计 256 级量化+批处理**(有想法)但 O(n·d·256) 必超时 → 0 | 设计野心 SFT/a50 > a10;都死在 API/性能 |
| 9 | MLS ml-clustering | 命名 **ClusterGEM**(扩展句乱码)| 无命名 | 无命名 | 三档分数完全相同(0.388)= 命名是装饰,没改变可运行方法 |
| 10 | MLS causal-treatment-effect | agent 早死,只余题面回显;分 0.261 | **提案 ET-Learner(正交化 DML+doubly robust,表述完全连贯)**但 0.031 | 无提案,0.261 | **最连贯的创新提案出现在 a10 而非 α=1**;分数与提案解耦(单种子) |

**头对头总判定:** average 掉的是 (i) KI 计划体浓度、(ii) 方案分布宽度、(iii) 偶发的真非平凡框架(#5,唯一一例,且从未兑现);**方法内核在 8/10 题三档同族**;自命名在 agent 设置里 mid-α 反而最多;兑现清一色发生在 soup 侧。"措辞变了而方法内核还在"是 8/10 题的准确描述,#5 是诚实的反例(真掉了一个框架),#10 是反向反例(a10 比 SFT 更能把创新提案说完整)。

## 5. 长度/风格混淆与检测器诚实声明

1. 所有 share 类指标同报 /10k 字符版;**翻转案例**:FCS 多方案枚举(raw 16%→1.3% 看似"soup 保留意图",归一后 α 无梯度且 α=1 最低)。"命名短语"检测器(`call this/it`)在 FCS 上主要捕获 base 的**变量命名**习惯(302/860 vs SFT 12/860),不测方法命名——FCS 的 named_share 列已从 §2 剔除,以 MLS 缩写-展开检测器为准(手工去误报)。
2. α=1 的高"方案簇数"部分是**代码不连贯的指纹散射**(编译错误率 83%),4.23 是上界;但 a50(编译率 40.6%)簇数 4.40 同样高,宽度信号不全是噪声。
3. Research 指纹是高基数 API 集合,Jaccard 绝对值小且噪;FCS 低基数算法关键词的 α 梯度(0.48→0.12)更可信。
4. MLS 单种子;命名计数已手工剔除 IMPLEMENTATION/CONTRIBUTION/EDITABLE/DR-Learner(教科书方法)等误报。
5. Research 意图样本兑现率(conv_intent)每档仅 10-17 个样本,只报方向不报倍数。

## 6. 结论

- **"Average 之后创新能力减弱了多少"的分解答案:表达浓度减 ~100%(KI,α≤0.2)、方案宽度减 ~100%(α≤0.2)、方法指纹偏离减 62–100%(基准面依赖)、自命名(agent 设置)不减反增、真非平凡框架偶发损失(10 题中 1 例);而创新兑现率增 3–12 倍。**
- 结合姊妹文档:α=1 的"创新"是**高浓度、零兑现、窄题干深挖**;α=0.1 的"创新"是**低浓度、可兑现、行为上偏离 base 有限但在开放任务上存活**。两者不是同一能力的强弱两档,而是**两种不同的表型**;soup 不是把 SFT 的创新"调淡",而是换到了"能活到评测结束"的表型——这正是 RL 能在 soup 起点放大创新(wd03+RL 三线超 base+RL)而在 α=1 起点无从谈起的原因。

## 7. 复现

```bash
python3 experiments/scripts/agg/soup_intent_vs_success.py out.json   # §2/§3 全表(fcs/research × nom/full)
python3 experiments/scripts/agg/soup_jaccard_alpha.py                # MLS 编辑 Jaccard
# 头对头样本定位:cc_eval_clean_clean_nomaintain_wd01_{sft,a10,a50}_research_*/shard_0/samples.jsonl
#   按 ground_truth + sample_idx 检索 §4 表中的题;MLS 日志在 cc_mlsbench_cpu_<tag>/task_logs/<task>.log
```
