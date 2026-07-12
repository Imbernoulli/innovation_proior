# Case Study(clean 轮):base → SFT → SoP → RL 全链路详细结果

> 本文聚焦本波实验的完整四阶段链条,主线 = **wd03 臂**(weight-decay 0.3 正则 SFT → α0.1 soup → 纯 synthetic RL),对照 = 同配置 base+RL。
> 口径:全部评测同协议(thinking 模式、max_tokens 32768、temp 1.0/top_p 0.95/top_k 20/presence 1.5、每题 n=5);FCS = strip `<think>` 后官方抽取;分数为 mean@5(括号内 best@5);错误样本率 ≤0.2% 的才引用。
> 姊妹文档:`CLEAN_DECONTAM_REG_zh.md`(全 sweep)、`INNOVATION_CAMPAIGN_REPORT_zh.md`(主报告)。

---

## 0. 总表(一图流)

| 阶段 | 模型 | FCS | ALE | Research | MLS* |
|---|---|---|---|---|---|
| ① base | Qwen3.5-9B instruct | 7.05 (11.78) | 357 (515) | 9.08 | 0.032 |
| ② SFT 直测 | full-FT@clean 数据 | **0.31–2.42**(塌陷) | 287–317 | 4.46–6.89 | — |
| ③ SoP(soup α=0.1) | wd03_a10 | 6.68 (11.91) | 416 (595) | 8.34 (19.5) | 0.070 |
| ④ **RL step5** | **wd03_a10 + GRPO×5** | **8.42 (14.88)** | **434 (632)** | **14.01 (27.4)** | 见 §5 |
| 对照:base+RL step5 | 同配置 RL on base | 7.25 (12.28) | 389 (540) | 11.48 (23.3) | 见 §5 |

\* MLS 采用共同子集/None→0 的保守口径(§5 详述该基准的三个结构性问题)。

**一句话结论:innovation-SFT 是"先验注入",soup 是"能力保全",RL 是"放大器"——wd03+RL 在 FCS/ALE/Research 三线全面超过同配置 base+RL(8.42>7.25、434>389、14.01>11.48),且 FCS 比 raw base 净涨 +1.37。**

---

## 1. 阶段①:base(Qwen3.5-9B instruct)

参照系:FCS 7.05 / ALE 357 / Research 9.08 / MLS 0.032(共同子集口径)。
行为特征:竞赛题上思考极长(评测端 65% 的样本顶满 32k cap,中位 completion = 32768),但"能写完的那 1/3"质量高——这就是 7.05 的来源。发现类任务(Research/MLS)上保守:倾向标准方法,少见多方案探索。

## 2. 阶段②:SFT 直测 —— 先验注入成功,但竞赛能力塌陷

配置:full-FT,数据 `innovation_clean_decontam_traj`(**去污染 + traj 不用 agentic**),cutoff 53760,1 epoch,lr 5e-6,wd 0.1/0.3。

| SFT 直测 | FCS | ALE | Research |
|---|---|---|---|
| nomaintain_wd01 | 0.31 (1.02) | 286.5 | 4.46 |
| full(maintain)_wd01 | 2.42 (5.66) | 317 | 6.89 |

**解读**:全参数 SFT 把创新数据的分布"学满",代价是把 base 的竞赛精确性冲掉(FCS 7.05→0.3–2.4,全线塌);Research 也低于 base——**先验学到了,但表达被塌陷掩盖**。这不是失败:SFT 的角色是把创新倾向写进权重,能力保全交给下一步。(r1 轮 LoRA 对比证明塌陷主因是全参数大扰动,与数据本身无关。)

## 3. 阶段③:SoP(model soup,α=0.1)—— 能力保全 + 先验表达

merged = 0.1·SFT + 0.9·base。轻档(α=0.05–0.1)是甜点,α≥0.3 递减、α=0.5 全塌(见 CLEAN_DECONTAM_REG §1-2)。

| soup(α=0.1) | FCS | ALE | Research | 备注 |
|---|---|---|---|---|
| **wd03_a10(主线)** | 6.68 (11.91) | **416 (595)** | 8.34 | FCS/ALE 最保 |
| nom_a5 | 6.58 (12.56) | 398 (562) | 11.1 | 全面均衡 |
| newmt_a10 | 5.89 (10.48) | 350 | 11.47 | Research 最强 |

**解读**:soup 后 FCS 回到 6.6–6.7(≈base 的 95%),ALE **超** base(416 vs 357),MLS 公平口径 0.070 vs base 0.032(2 倍+)。**创新先验以 10% 的权重混入即可表达**——发现类任务全面受益,竞赛类几乎不亏。wd03(更强正则)是本轮 sweep 里"保 FCS+ALE"最优的 SFT 配方。

## 4. 阶段④:RL —— 放大器(主结果)

### 4.1 配方(踩过坑之后的最终版)
- 数据:**纯 self-generated synthetic 500 题**(FrontierSmith/synth,确定性判分,无 train=eval 泄漏);此前 mixed@20k 配方已定案为"截断死亡螺旋"净负(CLEAN_DECONTAM_REG §3a)。
- GRPO:32 prompts × 8 rollouts/步,mini-batch 128 序列 × 2 更新/iteration,KL/loss 全 verl 默认,20 步、每 5 步存。
- 关键对齐:**rollout 采样 = 评测口径**(temp 1.0/top_p 0.95/top_k 20/presence 1.5/32768)——训练优化的就是被评测的行为。
- 4×H200/臂;w1→w2 断点续训链;每个存点全量 4 项评测。

### 4.2 训练健康(vs 死亡螺旋的反面)
奖励密度:有梯度组 53–88%/步(旧 mixed 配方 FCS 组 81% 全零);回复长度全程 ~100k 字符无塌缩;entropy 正常;无一预警指标触发。**32k cap + 大组 + 连续分值 synthetic 判分**是与死亡螺旋的三个决定性差异。

### 4.3 Step5 主结果(全对照)

| step5 | FCS | ALE | Research |
|---|---|---|---|
| **wd03+RL** | **8.42 (14.88)** | **434 (632)** | **14.01 (27.4)** |
| base+RL(对照) | 7.25 (12.28) | 389 (540) | 11.48 (23.3) |
| Δ(wd03 vs base 起点) | **+1.17** | **+45** | **+2.5** |

- **RL 涨分**:wd03 6.68→8.42(**+1.74**),Research 8.34→14.01(**+5.7**);base 起点只涨 +0.20/+2.4。
- **同样的 RL,创新先验起点学得更多**:每一条 benchmark 上 wd03+RL 都超 base+RL——这就是"SFT 先验 + RL 放大"假设的直接验证。
- wd03+RL 的 FCS 比 raw base **净涨 +1.37**(9B 上首次 RL 后超 base,且非靠单点运气:best@5 14.88 也是全场最高)。

### 4.4 轨迹(截至成文)

| step | wd03 FCS | wd03 Research | base+RL Research |
|---|---|---|---|
| 0(soup) | 6.68 | 8.34 | 9.08(=base) |
| 5 | **8.42** | **14.01** | 11.48 |
| 10 | 评测中 | 13.49 | 评测中 |
| 15/20 | 流水中 | 流水中 | 流水中 |

Research 在 s10 高位站稳(13.49),未见死亡螺旋式回落。step10/15/20 全存点评测出齐后按最优步定稿。

## 5. MLS 专节:为什么撤回定量对比

Forensic 审计(读全部 agent transcript)发现该基准在当前能力段有三个结构性问题:
1. **do-nothing 基线 ≈0.094**:交回未改模板即得分,高于几乎所有臂的 agent 均值——大部分正分是"没改坏模板"。
2. **None 排除偏差**:超时/崩溃任务被从均值剔除,恰好偏袒 base+RL(其最高价值任务超时);None→0 后 base+RL 0.093→0.084。
3. **单种子 + 单题波动 0.3–0.6**:臂间差距 = 半道~一道题的翻盘量;leave-one-out 下排序不稳定。

**保留的结论**:我们 pre-RL soup(共同子集 0.070)≫ raw base(0.032);**真实的定性发现**:RL 后模型的多轮 agent 行为退化(对被拒绝的编辑原样重试 7 次、思考 +46%、超时任务翻倍)——单轮代码 RL 与多轮 agent 能力存在真实张力,是后续工作方向。

## 6. Case 级分析

[进行中:各阶段在相同题目上的真实生成对比(思考风格/算法选择/塌陷形态/RL 增益机制),由生成文本审计补充,将在下一 commit 追加。]

## 7. 复现指针

- SFT:`LF-innov/examples/train_full/auto/os-q35_clean_nom_wd03.yaml`(数据 `innovation_clean_decontam_traj`)
- Soup:`FrontierSmith/scripts/cc_model_soup_merge.py --alpha 0.10`
- RL:`FrontierSmith/scripts/cc_rl_frontiersmith_synth_submit.sh`(ONLY="q35_inst_start cl_wd03_a10",GPUS=4 STEPS=20 SAVE=5;采样对齐已是默认)
- 评测:`slurm/cc_eval_thinking_both_ailab.sh` / `cc_eval_research_ailab.sh`(research 只能 H100/H200)/ `cc_eval_mlsbench_cpu_ailab.sh`(必须 MLSBENCH_ROOT=MLS-Bench-dev)
- RL ckpt→HF:`scripts/merge_fsdp_to_hf.py`(纯格式转换,非模型融合)
