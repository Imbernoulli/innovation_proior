# Model-Soup 能否在「原有能力」与「创新 pattern」之间取得 trade-off?

**—— 一份基于真实生成内容的中文 case study**

> 模型:`soup = α·(method-SFT) + (1−α)·START`(权重空间线性插值)。
> START = Qwen3.5 instruct 起点;method-SFT = 注入「研究/创新取向」的全参 SFT;α∈{10,20,30,50,70}%。
> 数据:`/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/cc_eval_<tag>_thinking_32k_both_vllm/shard_0/samples.jsonl`,主线为 q35 instruct 臂(`q35_a100*`),`data_source=='frontiercs'` 共 **40 题 × 5 样本 = 200 行/模型**。`</think>` 之前为推理,之后为答案/代码。score 为**连续质量分**(不是 0/1;构造/优化/交互题按提交质量给分,故均值被少数高分题主导)。

---

## 0. 摘要(先给结论)

1. **「创新腔」与「务实腔」都随 α 上升(向 SFT)而被压低**,不是「SFT 加强务实腔」那么简单 —— SFT 把推理整体压短(思考长度 52.8k→31k 字符)、两腔密度同时下降(innov/1k 2.17→1.24,prag/1k 1.21→0.79)。
2. **「中间 α 两腔皆强」这一现象是稳定的、但仅是「腔调」层面**:`min(创新腔密度, 务实腔密度)` 在 START / soup10 / soup20 / soup30 上维持高位(≈0.97–1.04),到 soup50/soup70/SFT 才骤降。读起来最像「既探索又想落地」的折中点是 **soup20**(soup30 次之)。
3. **但「兼有 pattern」≠「兼得能力」。** 我把所有 soup 样本里「soup 严格超过 START 最好成绩」和「START=0 而 soup>0」的反例全部挖出来逐一读原文比对:**没有一例是 soup 提出了一个 START 不会提的、新颖的、且真的得分的解。** 每一个 soup 增益都能归结为三种平凡机制之一:**(a) 同一想法的采样方差/更好的 tie-break/更少 bug;(b) 执行了 START 自己提出但又放弃的想法;(c) 交互题上的评分假象(截断、没出代码却拿了部分分)。**
4. **soup 的得分故事 = 「撤销 SFT 损害 + 调参/方差」,而非「落地了新创新」。** 纯 SFT 把 FCS 打到 ≈0(均分 0.015),soup10 把均分恢复到 2.92(接近 START 3.14);但**没有任何 α 让 FCS 越过 START 的前沿**。
5. **概念上**:权重平均把行为近似约束在 START↔SFT 的**线段**上;线段上的点能让文本**同时显出两种腔调**(叠加),但到不了线段之外那个「**提出新颖思路 且 写对落地**」的 off-segment 能力 —— 这是两端都不具备的第三种能力,线性插值合成不出来。
6. **MLS 双重分离**(metric 层面)与上述 pattern 层面的结论**同向印证**:SFT 在 FCS 上崩、在 ML 研究任务 MLS 上反超起点;soup 恢复 FCS 的同时把 MLS 拖回起点以下 —— soup 是在「执行纪律 ↔ 研究取向」之间滑动,而不是把两者**合成**到更高前沿。

---

## 1. 关键词清单(我用的正则)

> 说明:用**每 1000 字符的命中密度**(innov/1k、prag/1k)而非原始计数,以消除「高分题往往有 60k–100k 字符长 trace、低分题只有几 k」带来的长度混淆。原始计数会让长 trace 天然「两腔都高」,密度才公平。

**创新腔(innovation register)—— 质疑标准解 / 追问机制 / 重构 / 提出新构造:**
`wait`、`actually`、`hmm`、`reconsider`、`rethink`、`re-examine`、`what if`、`is there a better/smarter/cleverer/deeper`、`but wait/actually`、`hold on`、`interesting`、`insight`、`key idea/insight/observation`、`novel`、`clever`、`elegant`、`generaliz*`、`deeper`、`underlying`、`why does/is/would`、`not obvious/trivial/standard`、`non-trivial`、`can we do/prove/show/construct`、`perhaps`、`maybe`、`conjecture`、`alternativ*`、`another approach/idea/way`、`different approach/angle`、`optimal`、`better construction/bound/approach`、`turns out`、`surpris*`、`subtle`、`mismatch`、`doesn't make sense/work/hold`、`contradic*`、`rabbit hole`。

**务实腔(pragmatic register)—— 收尾 / 交可编译代码 / 调格式 / 兜边界:**
`implement`、`just use/do/output/print/return/submit/go with`、`let me code/write/implement/just`、`simple approach/solution/construction`、`straightforward`、`good enough`、`should work`、`this works`、`edge case`、`compile`、`output format`、`print`、`greedy`、`brute force`、`valid solution/output/answer`、`(let me) verify`、`correct(ness)`、`make sure`、`submit`、`final answer/solution/code`、`I'll write/code/implement/use/go with`、`constraints`、`safe`、`simplest`、`practical`、`for now`、`within time`、`efficient`、`O(`。

---

## 2. 多题验证:α 梯度对比表

### 2.1 全局聚合(40 题 × 5 样本)

| 模型 | 均分 | 非零样本数/200 | 创新腔/1k | 务实腔/1k | 创新∶务实 | **min(两腔)** | 思考长度(字符) | %出代码 |
|---|---|---|---|---|---|---|---|---|
| **START** (α=0) | **3.139** | 16 | 2.17 | 1.21 | 1.80 | **1.02** | 52767 | 80% |
| soup10 | 2.924 | 13 | 2.20 | 1.25 | 1.76 | **1.04** | 54319 | 86% |
| soup20 | 2.348 | **17** | 2.15 | 1.17 | 1.85 | **0.97** | 54375 | 80% |
| soup30 | 1.446 | 10 | 2.08 | 1.16 | 1.79 | **0.98** | 53719 | 80% |
| soup50 | 1.313 | 8 | 1.78 | 1.07 | 1.67 | 0.83 | 41563 | 84% |
| soup70 | 0.010 | 2 | 1.38 | 0.97 | 1.42 | 0.70 | 28680 | 89% |
| **SFT** (α=1) | 0.015 | 3 | 1.24 | 0.79 | 1.58 | 0.59 | 31003 | 88% |

**读这张表的三条要点:**

- **两腔密度都随 α 单调下行**(创新腔 2.17→1.24,务实腔 1.21→0.79)。所以 SFT 的特征**不是**「务实腔更浓」,而是「**两腔都更稀、且 trace 更短**」—— 它直接收掉了长链探索(从 52.8k 压到 31k 字符),整体话更少、更快收口。务实腔/1k 反而是 START/soup 端最高。
- **「两腔皆强」(min(两腔) 高位)从 START 一直维持到 soup30(≈0.97–1.04),到 soup50 才掉到 0.83、soup70/SFT 掉到 0.59–0.70。** 这证实了「中间 α 两腔皆强」是**稳定特征而非 p11 个例**:整个低-中 α 区间都同时保有两种腔调,高 α 才失衡。
- **关键:均分随 α 单调坍塌(3.14→0.015),而腔调平衡度在 soup50 之前几乎不变。** 也就是说**腔调是否平衡,完全预测不了得分**。「读起来既创新又务实」与「写得对」是**脱钩**的。

### 2.2 逐题 α 梯度(13 道有信号的 frontiercs 题,5 样本中的最高分;`*` = 该 α 超过 START 最好成绩)

| 题 | 题型 | START | soup10 | soup20 | soup30 | soup50 | soup70 | SFT |
|---|---|---|---|---|---|---|---|---|
| p2 | tree distances(交互) | 85.87 | **85.87***s0 | 0 | 0 | 0 | 0 | 0 |
| p3 | — | 6.25 | 0 | 0 | 0 | 0 | 0 | 0 |
| p5 | — | 31.43 | 0 | 12.53 | 0 | 0 | 0 | 0 |
| p8 | Knight's tour(Warnsdorff) | 65.90 | **90.43***s2 | **90.41***s3 | **90.41***s2,3 | **89.48***s2 | 0 | 0 |
| p9 | palindrome grid path | 30.00 | 0 | **70.00***s0 | 0 | 0 | 0 | 0 |
| p10 | digit-grid 构造 | 0.83 | 0 | **4.77***s3 | 0 | 0.44 | 0 | 0 |
| p11 | XOR 子集(Sidon-类) | 43.65 | 43.65 | 43.65 | 43.65 | 40.52 | 1.11 | 1.11 |
| p12 | 球面堆积(Tammes) | 91.86 | **97.54***s3 | 91.86 | 1.12 | 91.86 | 0 | 0.80 |
| p17 | DNA 匹配概率(容斥) | 23.81 | **28.40***s1 | **28.40***s4 | **28.56***s4 | 0 | 0 | 0 |
| p22 | 交互·找钻石(query 复杂度) | **0** | **10.00***s3 | 0 | **10.00***s0,2 | 0 | 0 | 0 |
| p23 | 交互·围困机器人 | **0** | **19.86***s0 | 0 | 0 | 0 | 0 | 0 |
| p24 | group testing | **0** | 0 | 0 | **13.33***s0 | 0 | 0 | 0 |
| p32 | — | 20.00 | 0 | 0 | 0 | 0 | 0 | 0 |

> 注:p11 的「START 43.65→soup20 0」是**单样本**现象 —— soup20 的 5 个样本里 s1 仍拿到 43.65(与 START 同分),只是 s3 那条短 trace(用户引用的「99/96 两腔皆强却 0 分」)失败。逐样本看,p11 在 START→soup30 每一档都有样本拿到 43.65;到 soup70/SFT 才坍塌到 1.11。**这说明 soup20 不是「想法变了」,而是 5 取 1 的命中率变了。**

**表里两类「soup 看似赢了」的现象,正是第 4 节要逐一证伪的对象:**
- **A 类(mid-α 分数高于 START):** p8、p9、p12、p17。
- **B 类(START=0 而 soup>0 的「反转」):** p22、p23、p24。

---

## 3. 「pattern 折中点」定位:哪个 α 读起来最像「既创新又务实」?

**答案:soup20(soup30 紧随其后)。** 它保留了 START 的长链探索与质疑密度,又叠加了明显更强的「落地、接受 good-enough」的务实层;到 soup50 长/平衡 trace 变稀,soup70/SFT 则坍缩成短链。下面给该档真实推理原文(全部经逐字校验为 `text` 子串):

**(原文 1)soup20,p9(palindrome path),sample 1(score 0.0)—— 最干净的「一口气两腔融合」:** 在它自己写的代码注释里,既说出创新腔的怀疑(构造未必真能到出口),又以务实理由直接发车:

> "We assume executing P^R from Start will reach (er, ec) due to grid connectivity properties (and verified for examples). However, strictly speaking, **we should check if S ends at (er, ec). But since we can't dynamically fix P easily without more complex search, and constraints/time allow one pass, we assume validity.**"

**(原文 2)soup30,p12(球面堆积),sample 0(score 1.12)—— 两腔压进连续三行注释,撞在一起后随手挑一个收口:**

> "// Our fib might give randomish. // **We should optimize anyway, but hardcoded is safest** for known N? // Let's rely on optimization."

**(原文 3)soup30,p2(tree distances),sample 0(score 0.0)—— 长段试探后落到「相信随机、把它包起来」,并以创新→务实的明确交接收尾:**

> "Let's trust the random tree property. The solution I wrote is the most logical one. I will wrap it. **Wait. I should include `<cstdio>` for `fflush`** ... But `cout.flush()` is fine."
> （结尾)"Hopefully random inputs are friendly. ... **But without knowledge of distribution, this is the best deterministic solution.**"

**对照 —— 两端长什么样:**

- **START(创新腔偏盛、探索最久、最晚收口):** `q35_a100` p9 s2(score 0)的 trace 末尾还在嵌套假设的算法结构("Try all pairs (u,v)... Then run DFS... Wait. DFS on grid."),迟迟不出最终程序。START 探得最狠,但常常「探到没预算了」也没落地。
- **纯 SFT(务实/简短、几乎不持续质疑):** `q35_a100_method` p11 s1(score 1.11)虽也闪过创新一击(它确实发现样例错了),但**一拍即收、立刻发车**,没有 soup 那种长时间盘旋:
  > "Oh, I see! **The sample isn't optimal.** ... The sample actually violates the constraint. My powers-of-2 construction gives 6 distinct elements instead, which is genuinely longer. **I should implement the powers-of-2 approach** since it guarantees no collisions and maximizes the count."

**判断:折中点真实存在(soup20),但它是「论辩式叠加」而非「合成出第三种声音」。** mid-α trace 的实际行为是:**把一个创新念头和一个务实念头并排放着,然后明确地化解张力 ——「该检查 X,但难做,所以假定成立」「该优化,但硬编码最稳」。** 这跟 START(只提怀疑、继续打转)和 SFT(从不长时间起这个怀疑)都不同,所以两腔是**共存且互动**的,不是孤立词频的偶合。**但它是把两种嗓音拼接、每个决策点挑一个,而不是真的长出一个统一的新嗓音。**

---

## 4.(最关键)「兼有 pattern」能不能转化成「兼得能力」?—— 反例搜索的结果

> 方法:脚本扫全 40 题,定位 (a)「soup 严格超过 START 最好成绩」与 (b)「START=0、soup>0」的全部样本,然后**逐例同时读 soup 胜出样本 + 该题 START 全部 5 个样本**的推理与最终代码,判断是「**新想法**」还是「**同想法的方差/执行差**」。

### 4.1 「soup 超过/反转 START」的完整清单与逐例裁决

| 题 | soup 胜出(tag,样本) | soup 分 | START 最好 | 裁决 | 关键依据 |
|---|---|---|---|---|---|
| p2 | soup10 s0 | 85.875 | 85.873 | 同想法 | 差 0.002,纯噪声 |
| p8 | soup10 s2(及 soup20/30/50) | 90.43 | 65.90 | **同想法·更好 tie-break** | 两端开篇都说要用 *Warnsdorff's rule*;唯一实质差异是平局选择:START 用随机 `(rng()%2)`,soup 用确定性 `a.r<b.r` |
| p9 | soup20 s0 | 70.00 | 30.00 | 同想法·更好采样 | 两端都建同一构造:DFS 生成树「前进+反向回溯」拼出回文 `A+reverse(A)`,随机化邻居序以使终点落在出口;soup 只是命中更好的一次随机 |
| p10 | soup20 s3 | 4.77 | 0.83 | 同想法 | 同一 digit-grid 构造,框架一致 |
| p12 | soup10 s3 | 97.54 | 91.86 | **执行了 START 放弃的想法** | 见 4.2,最像「新想法」的一例,但想法逐字就在 START 的被弃推理里 |
| p17 | soup30 s4(及 soup10/20) | 28.56 | 23.81 | 同想法 | 两端都用「对 m 个 pattern 子集做容斥,逐列算交集概率(冲突→0/相同→1/4/自由→1),符号 (−1)^(k−1)」;差在实现/精度 |
| p22 | soup10 s3(及 soup30 s0/s2) | 10.00 | **0** | **评分假象** | 见 4.3,胜出样本**撞 32768 token 上限、未闭合 `</think>`、零最终代码**却拿 10 分 |
| p23 | soup10 s0 | 19.86 | **0** | **评分假象** | 同上:撞 token 上限、无 `</think>`、无完整代码,却拿 19.86 |
| p24 | soup30 s0 | 13.33 | **0** | 同想法 | soup30 用「随机二进制掩码+签名匹配」;START 5 个样本独立把它认成 group testing(s2:"a variation of the group testing problem… log2(...)≈19 bits"),只是都没写对;soup 也远未真正正确(13.33 远低于满分) |
| p36/p38/p39 | soup20/30 | 1.4–2.6 | 0 | 噪声 | p38/p39 是 AHC 启发式赛题,任何合法输出给部分分;1–2.6 分是方差,START 全 trace 里框架完全一致 |

### 4.2 最像「新想法」的一例:p12 —— 但想法是 START 自己提了又丢的

soup10 s3(97.54)在初始分布上加了一个**斥力松弛/力导向**优化环。问题是:**START s0 一字不差地提出过这个想法,又因(误判的)性能顾虑亲手放弃**:

- START 提出:"this looks like it could be solved using an optimization approach with **gradient descent or particle swarm optimization**, OR I can use known patterns..."
- START 放弃:"50 billion operations... **which exceeds the 2-second limit**... I'll focus on known configurations... without heavy optimization loops",退回到静态 Fibonacci 螺旋。

**soup 只是执行了 START 拒绝的那条路。** 这恰恰证明:**新想法本就在 START 的假设空间里**,soup 没有「合成出」START 够不到的东西,它只是在「探索 ↔ 落地」的权衡上把刻度往落地挪了一格,于是这次没被自己劝退。

### 4.3 两个最大的「反转」(p22→10、p23→19.86)是评分假象 —— 诚实必须点破

逐字核验胜出样本:

| 样本 | score | 闭合 `</think>`? | completion_tokens | 末尾状态 |
|---|---|---|---|---|
| `q35_a100_method_soupa10` p22 s3 | **10.0** | **否** | **32768(撞顶)** | 截在推理半句:"...we are surrounded by" |
| `q35_a100_method_soupa10` p23 s0 | **19.86** | **否** | **32768(撞顶)** | 截在代码半行:"vector<pair<int,int>> candidates = { {rx+1," |

两条都**顶满 32768 token 被硬截、根本没闭合思考、也没有完整提交**(`text` 里出现的 ```` ``` ```` 是推理中途的开栏,不是成品),却分别拿了 10 和 19.86 的部分分。**这是评分后端对截断 trace 的部分给分,不是「解出来了」。** 而该题 START 的 5 个样本提出的核心思路与 soup 完全相同(p22:都是「query 得 (0,0)⇒钻石,据 a0/a1 是否为 0 做二分」;p23:都是「向第一象限边界堵截、封死逃逸格做围困」)。把它们当作「soup 找到了 START 找不到的解」会是误读。

> 旁证(p22 非截断样本):soup30 p22 s2(10.0,正常出代码)与 START p22 s0(0.0,正常出代码)用的是**同一个二分算法**;唯一差别是「a0>0 且 a1>0」歧义分支的启发式,soup 那版恰好通过交互判题、START 那版恰好没过 —— **执行级方差,而非新洞见**。

### 4.4 裁决

**在所有 soup 样本里,没有一例满足「提出了一个 START 不会提的、新颖的、且真的得分的解」。** 每个 soup 增益都落入:**①同想法的采样方差/更好 tie-break/更少 bug(p2/p8/p9/p10/p17/p24);②执行了 START 自己提出又放弃的想法(p12);③交互题的截断部分分假象(p22/p23)。**

同时不要忘记反方向证据:**soup 也频繁摧毁 START 本来做对的解**(p3 6.25→0、p5 31→12/0、p32 20→0、p34 21.5→0、p12 在 soup30 掉到 1.12),且一切在 soup70 全面坍塌。整幅图景与「soup 在部分撤销 SFT 损害 + 重新洗采样方差」一致,与「发现了新算法想法」**不一致**。

**因此:soup 的得分增益,全部来自「撤销 SFT 损害 / 调参 / variance」,而非「落地了新创新」。** 没有任何 α 让 FCS 越过 START 的前沿。

---

## 5. 概念分析:为什么权重平均只能在「线段」上调和腔调,合成不出「创新且落地」

**(1)插值约束在线段上。** `θ(α)=α·θ_SFT+(1−α)·θ_START` 在参数空间是 START 与 SFT 两点连线上的一点。神经网络非线性,行为不严格线性,但经验上(本数据)行为**近似落在这条线段所张的低维流形上**:腔调密度随 α 平滑单调插值(第 2.1 表),得分随 α 平滑(且坍塌)。

**(2)线段上的点 = 两种腔调的「叠加」,不是「合成」。** START 端是「创新腔强、爱质疑、晚收口」,SFT 端是「话短、快收口、质疑被削」。线段中点能让一段 trace **同时显出**两种腔调的语言标记(第 3 节原文)—— 因为权重平均把两套「输出风格的方向」按比例混进了 logits/激活,文本上就**既有 `wait/reconsider` 也有 `just implement/good enough`**。但这是**两个一维偏好的线性混合**:质疑的强度 ↓、收口的倾向 ↑,二者此消彼长地共存。

**(3)「创新且落地」是 off-segment 能力,线性插值到不了。** 「提出 START 不会提的新构造 **并且** 把它写成可提交的正确代码」需要的不是「质疑多一点 + 收口多一点」的某个混比,而是一个**新的联合策略**:在探索出非平凡想法后,还能把它**收敛到一个被验证过的、可编译的提交**。这要求模型把「探索」和「验证落地」**串联**起来(先发散、再把发散的结果收敛验证),而 START 缺「收敛验证的纪律」、SFT 缺「发散到非平凡想法的探索」。**两端都不具备这个串联能力,它们的任何线性组合也不具备** —— 它是线段之外的点。p12 是最佳例证:soup 能做的极限,只是「把刻度移到落地端,于是没把 START 提过的那个想法劝退掉」;它**没有**在 START 提不出想法的题上凭空补出想法。

**(4)soup 与「真 trade-off 模型」的区别。** soup 是在**固定的 Pareto 前沿(START–SFT 线段)上滑动**:要纪律就牺牲研究取向,要研究取向就牺牲纪律,两轴此长彼消。「真 trade-off 模型」应当**把前沿整体外推**:在保持高执行纪律的同时,还能稳定产出并落地非平凡想法 —— 即第 4.3 里那种「正常出代码、且解法是 START 全员都想不到的」样本。本数据里**这种样本一个都没有**。

**(5)要拿到真 trade-off 需要什么(可执行建议):**
- **改 SFT 数据的「落点」**:不要只注入「研究腔/质疑腔」,而要让训练轨迹**示范「探索出新想法 → 收敛到通过测试的可提交解」的完整闭环**,使创新行为本身**收敛到可提交解**,而不是停在质疑与发散。
- **用 RL 直接奖励「新颖且正确」**:以「提交可编译/通过判题」为硬闸门,在通过的前提下再奖励「与标准/样例解不同的新构造」。把「新颖」和「正确」**绑进同一个奖励**,才可能把行为推到 off-segment。
- **再用 soup/RL 在更高前沿上做平衡**:当两个端点本身都已具备「创新且落地」的部分能力后,线段插值/再 RL 才是在**更高的前沿**上平衡,而不是在「纪律 ↔ 取向」的旧前沿上拆东补西。

---

## 6. 结合 MLS 双重分离(metric 层面的旁证)

把 FCS 与 MLS-Bench(ML 研究类任务)并看,出现一个干净的**双重分离**(数值与既有 `CASE_STUDY_zh.md` 第 6 节一致):

| 模型 | FCS(奖励「提交简单正确代码」) | MLS(奖励 ML 研究任务) |
|---|---|---|
| q35-inst START | 3.139 | 0.0643(基线) |
| q35-inst **纯 method-SFT** | **0.015(最差)** | **0.0794(最好,>起点)** |
| q35-inst **method-soup10** | **2.924(恢复,接近起点)** | **0.0538(最差,<起点)** |

- 纯 SFT:**FCS 崩、MLS 升**;soup10:**FCS 恢复、MLS 反被拖到起点以下**。
- **这正是 pattern 层面结论在 metric 上的镜像**:SFT 注入的「研究取向」在 ML 研究任务上**有用**(MLS 反超),在竞赛代码上**有害**(惩罚探索、不落地);soup 恢复执行纪律(FCS↑)的代价是**稀释研究取向**(MLS↓)。**soup 在「执行纪律 ↔ 研究取向」之间滑动,而不是把两者合成到更高前沿** —— 与第 4、5 节完全同向。

**一个值得注意的例外(base 臂):** `q35_a00_method_soupa20` 在 FCS(2.082)≈ 起点(2.080)、`a00_method_soupa50`(2.203)甚至略超起点;base 线 MLS 也是 0.0764→0.0943(soup20 双轴都好)。这说明在某些起点/数据落点下,soup 能找到「不牺牲 FCS」的混比 —— 但这仍是**在既有前沿上找了个不亏的点**,而非把前沿外推;它没有在 FCS 上越过起点、也没有产出「新颖且落地」的解。

**诚实警告:** MLS 单跑、20 噪声任务,绝对差值小;但**方向**与上面所有机制一致,且双重分离本身比单点更难用噪声解释。

---

## 7. 结论与对后续的启示

1. **「中间 α 两腔皆强」是真的、稳定的,但只停在腔调(register)层面。** soup20 读起来最像「既创新又务实」,且这种平衡从 START 一直保持到 soup30。
2. **「兼有 pattern」没有转化成「兼得能力」。** 穷举所有 soup 反例后,**没有一例**是「soup 提出 START 提不出的新想法、且正常落地得分」。最像的 p12 也只是「执行了 START 自己放弃的想法」;两个最大反转 p22/p23 是**截断零代码的评分假象**。
3. **腔调平衡与得分脱钩** —— 得分随 α 单调坍塌,腔调平衡在 soup50 前几乎不变。**会说「既创新又务实」的话,不等于会写出「既创新又对」的代码。**
4. **soup 的全部得分增益 = 撤销 SFT 损害 + 调参/方差**,没有任何 α 越过 START 的 FCS 前沿。这与 MLS 双重分离(SFT↑MLS/↓FCS,soup 反之;base-soup20 例外)同向印证:soup 是**沿旧前沿滑动**,不是**外推前沿**。
5. **对后续的启示:** 要的不是「在 START–SFT 线段上找折中」,而是**把两个端点都升级到「创新且落地」**——靠 SFT 数据示范「发散→收敛验证」闭环 + RL 奖励「新颖且正确(以通过判题为硬闸门)」,把行为推到 off-segment;之后再用 soup/RL 在**更高前沿**上平衡。同时,**评测口径**必须配上能识别「新颖且正确」的指标(FCS/ALE 只测落地纪律、会把创新误判为退化;发现类评测又噪声过大),否则这个 off-segment 能力即便出现也测不出来。

---

## 8. 关键样本路径(便于复核)

数据根目录:`/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/cc_eval_<tag>_thinking_32k_both_vllm/shard_0/samples.jsonl`(`data_source=='frontiercs'`)

**腔调折中原文(第 3 节):**
- soup20 p9 s1(0.0,「assume validity」一口气两腔):`tag=q35_a100_method_soupa20, problem_idx=9, sample_idx=1`
- soup30 p12 s0(1.12,「optimize anyway, but hardcoded is safest」):`tag=q35_a100_method_soupa30, problem_idx=12, sample_idx=0`
- soup30 p2 s0(0.0,「trust random / best deterministic」):`tag=q35_a100_method_soupa30, problem_idx=2, sample_idx=0`
- 对照 SFT:`tag=q35_a100_method, problem_idx=11, sample_idx=1`(1.11);对照 START:`tag=q35_a100, problem_idx=9, sample_idx=2`(0.0)

**反例搜索关键样本(第 4 节):**
- p12「执行了 START 放弃的想法」:soup 胜出 `tag=q35_a100_method_soupa10, problem_idx=12, sample_idx=3`(97.54);START 提了又弃 `tag=q35_a100, problem_idx=12, sample_idx=0`
- p8 Warnsdorff(同想法·更好 tie-break):`tag=q35_a100_method_soupa10, problem_idx=8, sample_idx=2`(90.43)vs `tag=q35_a100, problem_idx=8, sample_idx=2`(65.90)
- p9 回文路径(同构造·更好采样):`tag=q35_a100_method_soupa20, problem_idx=9, sample_idx=0`(70.00)vs `tag=q35_a100, problem_idx=9, sample_idx=1`(30.00)
- **评分假象(必看):** `tag=q35_a100_method_soupa10, problem_idx=22, sample_idx=3`(10.0,撞顶无代码);`tag=q35_a100_method_soupa10, problem_idx=23, sample_idx=0`(19.86,撞顶无代码)
- p22 同想法对照(正常出代码):soup `tag=q35_a100_method_soupa30, problem_idx=22, sample_idx=2`(10.0)vs START `tag=q35_a100, problem_idx=22, sample_idx=0`(0.0)
- p24 group testing(START 全员认出但都没写对):`tag=q35_a100_method_soupa30, problem_idx=24, sample_idx=0`(13.33)vs `tag=q35_a100, problem_idx=24, sample_idx=*`(全 0)

**MLS 双重分离数据(第 6 节):** `cc_mlsbench_cpu_*` 目录(`cc_mlsbench_cpu_mls_q35_base_start`、`..._a100_method_soup10`、`..._a00_method_soup20` 等),数值另见 `CASE_STUDY_zh.md` 第 6 节。
