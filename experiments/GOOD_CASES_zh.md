# 精选创新 Case(同题对比,严格对照,含诚实排除清单)

> 入选标准(四条全查):(a) 同题双方都有成功样本;(b) 分差跨样本稳定、非一发运气;(c) 机制是**可引用的设计差异**,不是"编译更常过";(d) 非评测器伪影(每个高分都做过 hack 检查)。
> 主报告:`CASE_STUDY_clean_wd03_zh.md`(本文件是其 §6.9 的全文版)。

---

## Case 1(最强,本波新发现)— AHC025:base 只会"排序分桶",我们的 RL 模型发明了"在线学权重 + 装箱优化"

**题目**:N 件未知重量物品,只能用天平比较两个不相交子集共 Q 次,分成 D 组使组和方差最小。交互式判题,无 evaluator 花招空间。

**逐样本分数(performance;有效样本原始分越低越好)**:

| 模型 | 5 样本 perf | 最好原始分 |
|---|---|---|
| base(clean_start) | [780,780,**1085**,780,780](1/5 有效) | 6.77e9 |
| **wd03+RL(step5)** | [**1102**,954,**1112**,780,916](**4/5 有效**) | **5.47e9** |
| base+RL 对照(clean_rlstart) | [780,780,780,993,834](2/5) | 9.98e9 |
| 内部基线:纯轮转 `i%D`(不用查询) | 916 | 1.23e10 |

**base 方案**(唯一成功样本):把全部查询预算花在**序数排序**上,然后蛇形分桶:
```cpp
bool cmp_less(const int& a, const int& b) { ... cout << "1 1 " << a << " " << b << endl; ... }
mergeSort(&p[0], 0, N-1);
ans[item_idx] = i % D;
```
base+RL 对照最好的样本也是同族(固定步长取对、胜场计数排序、`i%D`);它的另一个样本甚至在注释里**说出了正确想法却没做**:
```cpp
// Note: A full optimal solution would analyze results to estimate weights
```

**我们的方案**(设计族在 3/5 样本独立复现):**放弃排序,从子集和比较里在线估计基数权重**(感知机式乘法更新),再对真实目标函数(Σload²)做 LPT 贪心 + 爬山:
```cpp
if (conflict) {
    double alpha = 0.05; // Learning rate
    if (result == '<') {
        for (int idx : L) w[idx] *= (1.0 - alpha);
        for (int idx : R) w[idx] *= (1.0 + alpha);
    } ...
// LPT 贪心 + 爬山最小化 Σload²
if (new_sq < old_sq) { cur_loads[u_bin] -= w_id; ... }
```
另一样本还加了自适应衰减学习率与非对称更新(`alpha = 0.05/(1+0.1*q_idx)`,匹配小步强化、矛盾大步纠正);第三个样本自己点破机制:
```cpp
// This acts as a gradient descent / reinforcement learning to estimate relative weights
```

**为什么是好 case**:设计族 3/5 复现(均值 972.8 vs base 841 vs base+RL 833);**内部对照单调兑现创新增量**——纯轮转 1.23e10 → base 排序法 6.77e9 → 我们学权重法 **5.47e9**;交互式判题无伪影空间。

出处:`FrontierSmith/outputs/cc_eval_rlfsx_cl_wd03_a10_step5_thinking_32k_both_vllm/shard_0/samples.jsonl`(alebench, ahc025, s0/s1/s2)vs `cc_eval_clean_start_...`(s2)、`cc_eval_clean_rlstart_...`(s3/s4)。

---

## Case 2(旗舰,对照已升级)— AHC046:把"滑行"建模为图的边跑 BFS vs 逐格曼哈顿走

**题目**:20×20 冰面按序访问 40 个目标;动作 = 走一格(M)或滑到撞墙(S),动作数越省分越高。

**分数**:旗舰 `rlafter_rl_soup_mtv4_a20` [116,116,**1119**,116,411] vs 同源 base+RL 对照 `rlafter_rl_start` [116,116,**547**,116,116];头对头(双方唯一干净成功样本)**1119 vs 547**。本波 base+RL 最好 638,也是曼哈顿贪心。

**对照方案**:只用 M 逐轴走直线,完全无视滑行原语:
```cpp
if (abs(dx) > abs(dy)) { best_dir = dx > 0 ? 'D' : 'U'; ... }
for (int k = 0; k < num_moves; k++) out.emplace_back('M', actual_dir);
```
**我们的方案**:对每个目标在 **{4 个单步 M} ∪ {4 个滑行 S}** 的复合动作图上 BFS,最短动作数路径回溯:
```cpp
// 2. Try all 4 Slides (S)
while(true) {
    int nr = dest_r + dr[k]; ...
    dest_r = nr; dest_c = nc;      // 滑到边界才停,算 1 步
}
move_type[dest_r][dest_c] = 'S';   // BFS 边带动作类型
```
可复述的算法洞察:每步滑行常抵 10+ 格,BFS 在扩展动作空间上保证动作数最优。

**Caveat(如实)**:双方各只有 1 个干净成功样本,一致性弱于 Case 1;此案旧版(主报告 §4.2)对照是裸 base,本次已核实**对 base+RL 对照依然成立**,框架升级。

出处:`FrontierSmith/outputs/cc_eval_rlafter_rl_soup_mtv4_a20_thinking_32k_both_vllm/shard_0/samples.jsonl`(ahc046 s2)vs `cc_eval_rlafter_rl_start_.../`(s2)。

---

## Case 3(前轮,公平框架,跨代复现)— MLS causal-treatment-effect:正交化 CATE 估计 vs 朴素 T-learner

**题目**:有混杂的 CATE 估计,按 PEHE/ATE 误差打分。

**分数**:前轮 `method_soup10` **0.2606** vs `q35_start` **0.0549**(+375%,三个混杂数据集全赢);r3 代独立复现 methodtraj_r3 **0.2962**/methodv4_r3 **0.2775**(ATE 误差 1.503→**0.105,14×**)。两代四个 build 复现同一方法学倾向。

**base**:自称 X-Learner,实为两回归相减的朴素 T-learner(混杂下有偏)。
**我们**:Neyman 正交伪结局的 doubly-robust DR-learner(cross-fit + 倾向裁剪),r3 版为 Robinson 残差化 R-learner,生成日志原文:
```
"""Debiased X-Learner (DR-Learner with residualization).
 ... 2. R-Learner: residualize from marginal outcome model"""
```
**Caveat(如实)**:这是 SFT/soup 阶段的赢,框架 = "vs start(均无 RL)";RL 之后 MLS 上会同质化,不能外推到 base+RL 对照。

出处:主报告 §4.1、`CASE_STUDY_r3_zh.md` B2;生成日志 `FrontierSmith/outputs/cc_eval_all_r3_methodtraj_v4_r3_a10/mls/task_logs/causal-treatment-effect.log`(引文已复核)。

---

## 诚实排除清单(查过、不够格)

- **AHC015**(本波最大均值 gap):三档分数是同一个 1-ply 贪心的**实现保真度**差异(base 有状态跟踪 bug;三模型的正确实现收敛到逐字节相同的 70787141)——correctness 不是 innovation。
- **AHC039**:我们的"随机子集包围盒+边带收缩"设计漂亮且对裸 base(0/5)成立,但 **base+RL 对照拿到 1402,死于严格对照**。
- **AHC008 旗舰 944**:真策略 2.6× 于躺平分,但 1/5 样本、其余有效样本恰等于躺平分——一发运气。
- **AHC016**(旧 §4.2 案):旗舰的 start 对照拿到 1268,严格对照下不成立。
- **Research 全 64 题重排**:排除已审计 3 题后无一达 (a) 级;top gap = PySR 幻觉参数少(可靠性)/单样本超参运气/**新发现 evaluator artifact:`qknorm` 的 100 分是逐字节交回题面 baseline**,在"vs baseline 加速比 clamp 到 100"的 metric 下 1.0009× 吃满分——**"复交参考基线"型漏洞**,与 fused_linear_jsd 的"未初始化显存复用"型并列入黑名单。

## 汇报口径建议

Case 1 四条 bar 全过,作 headline;Case 2/3 各带一条如实 caveat 联合使用。三案共同讲的故事:**创新倾向(在线学习、扩展动作空间、方法学正交化)是训练进去的、跨题材/跨代可复现的,且在严格对照下兑现为分数**;排除清单和两个 evaluator artifact 的披露保证了这套 case 经得起审。
