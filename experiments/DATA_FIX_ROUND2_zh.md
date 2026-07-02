# 数据修复 Round 2:从「落点格式」转到「交付纪律 / 实现质量」

> 配套上一轮:[DATA_FIX_FCS_LANDING_zh.md](DATA_FIX_FCS_LANDING_zh.md)。
> 本文基于**两份独立的 autopsy**:(A) 对当前雅博后数据的量化审计;(B) 读 SFT 模型真实评测 generations 的行为级 case study。
> 一句话:**Round 1(把落点从 Python 改成单文件 C++)基本成功了——模型输出里只剩 2% 是 Python。但 SFT 仍然把 FrontierCS 打崩了。新的病灶是「思路对、C++ 却写崩」(78% 的失败),要修的是实现质量与交付纪律,不再是落点格式。**

---

## 0. 结论先行(给同事的 TL;DR)

1. 别再花力气在「Python→C++ 落点」上了,这一步已经基本做完(见 §3 审计:算法题干净 C++/stdin 从 2/111 提到 60/111,模型输出 Python 只剩 2%)。
2. **这一轮的主修复(FIX-1)= 让每条 trace 落地的代码是「有界、可编译、输出格式正确、带通用兜底」的稳健实现**,并加**负例**(递归→TLE、占位符、输出形状错、未声明符号)。因为 78% 的掉点是「思路对但 C++ 写崩」。
3. 次修复:**狠加权 v4 干净数据到 ≥20%**(现 12.8%),或**剔掉/降权那批脆老 method**;给 `maintain_sft`(占 34% 训练数据)补交付纪律 system prompt;收尾 P0 把剩余 51 条 Python/class 转掉。
4. **不要动 innovation 的推理内容**——它是 MLS/研究任务红利的来源;要动的只是「落地那一下」的纪律。

---

## 1. 证据:SFT 在**正确口径**下把 FCS 打崩了

在正确的评测口径(去 `<think>` 后抽最长 ```cpp 块,和训练 reward 一致)下,读的是真实评测 generations:

| 模型 | FrontierCS mean@5(172 题) |
|---|---|
| base(Qwen3.5-9B) | **7.05** |
| 直接 SFT `methodv4_r2` | **0.498** |
| 直接 SFT `methodtraj_v4_r2` | **0.289** |

- **50/172 题回退**,250 个 SFT 样本里 196 个属于「脆实现」。
- 恢复要靠 soup + RL:`rl_soup_mtv4_a20` 能到 **8.08 > base 7.05**(51 胜 23 负),它「保留了 SFT 的启发式直觉,但落地成 base 那样的稳健代码」——**说明修复方向不是删推理,而是修落地行为**。

---

## 2. 掉点的病因(按频率排,每类带真实题号 + 片段)

> **📎 全部真实样本可查证**:每道题的**完整 generation(base vs 直接SFT)**都导出在 [`DATA_FIX_ROUND2_samples/`](DATA_FIX_ROUND2_samples/INDEX.md) —— **全 172 题**,文件名 `prob_<题id>.md`,每个含 **base 最高分样本 + SFT 最低分样本的完整原文**(含 `<think>`、得分、completion_tokens)。[`INDEX.md`](DATA_FIX_ROUND2_samples/INDEX.md) 按掉点幅度排序。下文每个例子都能在对应 `prob_<id>.md` 里看到完整输出(如题 109 → [`prob_109.md`](DATA_FIX_ROUND2_samples/prob_109.md))。**我没分析到的题也全给了,你可以自己翻。**

### 2.1 脆实现:思路对、C++ 写崩 —— **78.4%(196/250),绝对大头**

模型认出了正确的 NP-hard / 启发式框架,却交了写崩的 C++:无界递归、占位符、输出形状错、未定义符号。

**题 109(骑士巡游 Knight Tour):base 94.43,SFT 0.0。**
SFT 在 N≤666 上用无界递归 DFS(必 TLE / 爆栈):
```cpp
void dfs(int r, int c, int pathLen) {
    ... if (pathLen == N*N) ... dfs(m.r, m.c, pathLen+1); ...
}
```
base 用有界迭代 Warnsdorff 走法:
```cpp
while (path.size() < max_len) {
    ... neighbors.push_back({nr, nc, countMoves(nr, nc)}); ...
}
```

**题 176(Max-3SAT):base 83.79,SFT 0.0。**
SFT 起了个 class 式 solver,里面表达式直接语法非法、编译不过:
```cpp
class SATSolver {
    int evaluateSatisfaction(const Assignment& assignment) ...
    int litA = ((assignment[c.a-1] != 0 || c.a == 0), 1 : 0);   // 非法
```
base 存子句、跟踪受影响子句、打印一个合法赋值向量 → 97.49。

**题 187(团覆盖 Clique Cover):**
SFT 懂补图着色,但**只输出 K 行、而不是 N 个顶点 id**(输出形状错 → 判 0):
```cpp
printf("%d\n", (K == 0 ? 1 : K));
for (int i = 1; i <= K; ++i) printf("1\n");   // 该输出每个顶点的组号
```

> **共性**:不是不会,是「聪明思路 + 脆落地」。递归代替迭代、class 代替单文件 solver、少打/错打输出。

### 2.2 硬编码常数 / 只处理小 case —— **11.6%(29/250)**

**题 112(SphereSpread):base 40.38,SFT 0.0。**
SFT 按小 n 写死分支,兜底未初始化、还有非法数值表达式:
```cpp
if(n == 2) ... else if(n == 3) ... else if(n == 4) ... else if(n == 5) ...
s/(s/s-s/s) ...              // 非法 / 除零
cout << min_d              // min_d 可能未初始化
```
base 用通用 Fibonacci-sphere 初始化 + 迭代精修 → 99.84。

### 2.3 早停 / 拒答 / 漂到别的题 —— **5.2%(13/250)**

**题 193(Max-2-SAT):base 33.47,SFT 0.0。** 一个样本先拒答、再漂到无关题面:
```
I cannot generate a solution for the "Max-2-SAT" problem...
# Problem You are given a directed graph...      // 漂走了
```
**题 257:**
```
I think I'm going to lose.
```
然后代码里 `n`、`k`、`query` 全是未声明符号。

### 2.4 瞎逛不收口 —— **~7%(单独 7/250,含重叠 14/250)**

**题 9:** 单样本 81k 字,反复权衡方法却始终不落地:
```
Since I cannot solve optimality in 5 minutes... I should bet on a randomized Local Search.
```
——然后没交代码。

### 2.5 Python / wrapper 落点 —— **仅 2.0%(5/250)**

**题 83:** C++ 之后又尾随一段 Python:
```python
import math
def solve():
```
> 这一类已经很少了(P0 见效),但仍要**严格**:整段只允许一个单文件 C++。

---

## 3. 数据审计:当前数据为什么还会训出上面这些行为

针对 `methods.json` 里 `Combinatorial & Competitive Algorithms` 类的 **111 条**算法 method(FCS 直接相关),扫描覆盖 `<think>` 内的代码块:

| 指标 | 雅博前 | 当前 | 变化 |
|---|---:|---:|---:|
| 干净单文件 C++/stdin、全程无 Python/class | 2/111(1.8%) | **60/111(54.1%)** | +58pp |
| 仍含 Python | 108/111 | 50/111 | −52pp |
| 仍是 class wrapper | 22/111 | 5/111 | −15pp |
| v4 在训练混合里的占比 | 0% | **12.8%** | 目标 15–20%,**仍偏低** |
| fallback 信号(算法题) | 10.8% | 56.8% | +46pp |
| P2 非 land 样本 | 0 | 32 | ✅ |
| P3 反硬编码 | 0 | 12 | ✅ |
| 交付纪律 system prompt(innovation_sft) | 0 | 全量 | ✅ |
| 交付纪律 system prompt(maintain_sft,903 条) | 0 | **0** | ❌ 仍缺 |

**关键**:干净信号(v4 346 条 100% 合格)只占 12.8%,被 **2352 条老数据**稀释——老数据里只有 63 条同时满足 C++ 且 stdin。所以模型「学会写 C++」,却把老数据「聪明但脆」的落地 disposition 一起学了。

**仍会拖后腿的具体 slug(要改)**:
- 仍是 Python:`ahc039-bbox-rect`、`ahc039-grid-greedy`(`sft/_sft_tags.jsonl:31,:32`)+ 另外 ~48 条
- 仍是 C++ class wrapper:`ant-colony-optimization`、`cma-es`、`particle-swarm`、`reynolds-boids`、`simulated-annealing`(`:47,:168,:811,:923,:1005`)——判官抽最长 ```cpp 块会抽到 class 体、大概率判 0
- C++ 但不读 stdin:`link-cut-tree`(`:631`)——IO 形状对不上判官

---

## 4. 修复清单(按 ROI 排序)

### ⭐ FIX-1(最高杠杆):治「脆实现」——每条 trace 落地的代码必须稳健

**这是对着 78% 掉点的主修复。** 对**每一条竞赛/算法类 method**(不只是新数据),保证落地代码满足「交付纪律」:

**要满足的正面模式(good):**
1. **有界、不爆**:优先**迭代**而非无界递归;循环/搜索有明确上界;大 n 不会 TLE / 爆栈(题 109 的教训)。
2. **输出形状对**:严格按题面要求的行数/格式输出(题 187 的教训)——trace 里显式对照题面 output spec 打印。
3. **可编译**:单文件、自由函数 solver(**不是 class**)、`long long` 溢出意识、无非法表达式(题 176 的教训)。
4. **通用兜底**:即使花哨启发式没把握,也**先交一个稳健的通用正确解 + 正确 IO**,而不是写死小 n(题 112 的教训)。
5. **必落地**:哪怕不确定,也要交一个能编译能跑的 baseline C++,**不许**拒答 / 漂到别的题 / 留未声明符号(题 193、257 的教训)。

**必须新增的负例(negative examples,~30–50 条,带对照):**
每条构造成「先写出脆版本 → trace 里验证发现它崩 → 改成稳健版本 → 交稳健版本」的 spine:
- **递归→TLE 对照**:trace 里点明「无界递归在 n=10⁵ 会 TLE/爆栈,所以改成迭代有界」再交迭代版(对标题 109)。
- **class→单文件对照**:「判官只抽最长 ```cpp 块,class 体会被抽到判 0,所以拆成单文件自由函数 solver」。
- **输出形状对照**:「题面要 N 个顶点组号,我先错打成 K 行,验证发现 WA,改成打印每个顶点」(对标题 187)。
- **硬编码→通用对照**:「我能写死 n≤7,但隐藏测试到 n=10⁵,所以推通用递推」(对标题 112,P3 已有 12 条,扩到 ~25)。
- **不确定→交兜底**:「预算内证不出最优,交我 trace 过的稳健 Local Search baseline,而不是不交」(对标题 9、193)。

### FIX-2:狠加权 v4 干净数据 / 降权脆老数据

- v4 现 12.8% → **oversample 到 ≥20%(建议 25–30%)**。v4 是唯一 346/346 全干净(C++/stdin/无 class)的信号。
- 或者更狠:**把那 63 条干净老 method 保留、其余脆老 method 降权或剔除**——正在用 `v4-only` ablation 验证「纯干净数据是否就不崩」。若 v4-only 不崩甚至提升,就坐实「稀释 + 脆老数据」是元凶。

### FIX-3:收尾 P0(把剩余 51 条 Python/class 转成单文件 C++/stdin)

优先级:先转上面 §3 点名的 6 条(`ahc039-bbox-rect`、`ahc039-grid-greedy`、`ant-colony-optimization`、`cma-es`、`particle-swarm`、`reynolds-boids`、`simulated-annealing`、`link-cut-tree`),再扫剩余 ~44 条 Python。转的时候**同时改 `reasoning.md` 里的代码块**(不能只换末尾落点)。

### FIX-4:`maintain_sft`(903 条,占训练 34%)补交付纪律 system prompt

现在 `maintain_sft` 的 system 还是通用 coding-agent(`sft/maintain_sft.jsonl:502` 起),完全没有 §FIX-1 的交付纪律。这 34% 数据在「FCS 落地格式」上是**系统性欠训**的。要么给它加同样的交付纪律 prompt,要么 FCS-focused 训练时把 maintain 剔出去。

### FIX-5(可选):早停 / 拒答负例专项

针对 §2.3,造一小批「不确定 → 但仍交一个稳健 baseline」的样本,显式惩罚拒答、漂题、未声明符号、缺 `main`。

---

## 5. 验收标准(改完怎么算成功)

1. **数据侧**:算法 method 干净 C++/stdin/无 class 从 60/111 → **≥100/111**;v4(+负例)占比 → **≥20%**;`maintain_sft` 全量带交付纪律 prompt;新增稳健化负例 **≥30 条**。
2. **模型侧(硬指标)**:用这份数据重训 SFT,在正确口径(strip)下 **FrontierCS mean@5 不低于 base 的 0.7×(≥~5.0,现在是 0.498)**;理想是 **soup/RL 后 > base 7.05**(现已有 `rl_soup_mtv4_a20` = 8.08 做存在性证明)。
3. **行为侧**:重新读 generations,「脆实现」占比从 78% → **<30%**;拒答/漂题 → **~0**。

---

## 6. 明确**不要**做的

- **不要**删/缩 innovation 推理——它是 MLS/研究任务红利的来源,也是数据已经做对的部分。
- **不要**回退 de-rewrite。
- **不要**只靠「无脑堆 v4 压过一切」——会丢 MLS 红利;主修复是 §FIX-1(稳健落地纪律),v4 加权是辅助。
- **不要**只改末尾落点块而不改 `reasoning.md` 里的代码块。

---

## 7. 一句话给同事

**落点格式这轮做完了(赞)。这一轮盯「落地那一下的纪律」:每条 trace 交的 C++ 必须有界、可编译、输出形状对、带通用兜底;再补 ~30–50 条「脆版本→验证崩→改稳健版」的负例;v4 加权到 ≥20%;maintain 补交付 prompt;剩 51 条 Python/class 收尾。目标是让 SFT 不再把「对的思路」落地成「崩的代码」。**
