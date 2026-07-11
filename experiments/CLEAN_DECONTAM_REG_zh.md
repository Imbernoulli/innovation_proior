# 去污染 + 正则化轮次结果(clean round)

> 目标:在**去污染(decontamination)**的数据上重训,验证 (1) 去污染掉不掉分、(2) maintain 数据的作用、(3) 各种正则手段能否让 SFT 少塌陷/保住竞赛力(FCS)。
> 数据统一为 `innovation_clean_decontam_traj`(去污染 + **traj 不用 agentic**)+(maintain 臂另加 `innovation_wave2_clean`)。
> 口径:FCS = strip `<think>` 后官方抽取;base 参照 **FCS 7.05 / ALE 356.6 / Research 9.08 / MLS 0.038**。soup `merged = α·SFT + (1-α)·base`,LoRA merge `scale·ΔW`。

---

## 1. 去污染 A/B:maintain vs nomaintain(soup,pre-RL)

| soup | FCS | ALE | Research | MLS |
|---|---|---|---|---|
| **nomaintain a5** | 6.6 | **398** | **11.1** | **0.101** |
| nomaintain a10 | 6.4 | 368 | 10.2 | 0.061 |
| full(maintain) a5 | 6.7 | 330 | 10.0 | 0.045 |
| full(maintain) a10 | 6.1 | 366 | 10.0 | 0.061 |
| a30 / a50(两臂) | 4.5–4.8 / 1.7–2.2 | — | 掉 | 掉 |
| **base** | 7.05 | 356 | 9.08 | 0.038 |

**结论**
- **去污染不掉分**:最佳轻档 soup(a5)在**发现类全面 ≥ base**(Research 10–11.1 > 9.08、MLS 0.045–0.101 > 0.038、ALE 到 398),FCS 6.6–6.7 ≈ base(仅微降)。
- **发现类 nomaintain ≥ maintain**:Research/MLS/ALE 上 nomaintain 全面更高;maintain 在 SFT-直测 FCS 上更稳(2.4 vs 0.3),但 **soup 后被 base 本身竞赛力覆盖,冗余**,反而略稀释发现类信号。
- 甜点在**轻档 α=0.05–0.10**;α≥0.3 开始塌,α=0.5 严重塌。

---

## 2. 正则化 sweep(全在 nomaintain + 去污染 + traj 上;目标=少塌保 FCS)

轻档(soup α=0.1 / LoRA scale=0.1),格式 **mean@5 / best@5** —— 参照:nomaintain-a10 基线 FCS 6.41/11.58、ALE 368/487、Research 10.2;**base FCS 7.05/11.78、ALE 357/515、Research 9.08**:

| 变体 | FCS | ALE | Research |
|---|---|---|---|
| **weight decay 0.3** (a10) | **6.68 / 11.91** | **416 / 595** | 8.34 / 19.5 |
| weight decay 0.5 (a10) | 6.53 / 11.86 | 414 / **665** | 9.49 / 22.7 |
| 低 LR 2e-6 (a10) | 6.40 / 11.59 | 362 / 539 | **10.77 / 24.8** |
| 低 LR 1e-6 (a10) | 5.70 / 11.45 | 367 / 517 | 7.97 / 20.6 |
| 新代码 baseline (a10) | 6.16 / 10.99 | 393 / 569 | 9.87 / 20.1 |
| 新 maintain (a10, §4) | 5.89 / 10.48 | 350 / 491 | **11.47 / 24.0** |
| LoRA r32 (s0.1) | 5.98 / 11.55 | 369 / 500 | 8.60 / 19.3 |
| LoRA r64 (s0.1) | 5.76 / 12.20 | 337 / 449 | 7.64 / 19.3 |
| LoRA r8 (s0.1) | 5.36 / 9.58 | 376 / 549 | 8.85 / 18.5 |
| LoRA r16 (s0.1) | 5.26 / 9.93 | 360 / 490 | 9.03 / 20.2 |

**结论**
- **保 FCS/ALE 选 wd0.3–0.5**(FCS 6.5–6.7 ≈ base、ALE 414–416 双超基线);**保 Research 选低 LR 2e-6 或新 maintain**(10.8–11.5 > base 9.08)。没有单一变体三线全赢——wd0.3 的 Research(8.3)反而低于 base。
- **LoRA 低秩也能保住 FCS**(s0.1 档 5.3–6.0);**关键是 merge scale(0.1 vs 0.5)而非 rank**——重档(s0.5)全塌到 1.1–3.0。
- 重档(soup a50 / LoRA s0.5)一律塌陷(FCS 0.6–4.2),印证"小扰动"才保能力。NEFTune、大 wd 在重档更差。
- 备注:dropout 此前一直关(LoRA `lora_dropout=0.0`、模型自身无 dropout);为标准正则可选 0.05–0.1。
- **NEFTune a10 补测中**:此前 soup 文件损坏(disk-full 期写坏,safetensors MetadataIncompleteBuffer),从未出过分;已重 soup + 重评(FCS/ALE/Research)。

---

## 3. RL A/B(clean soup a10 上 GRPO)

### 3a. mixed 数据 @20k cap(FrontierCS172+synth10+ALE40)—— 全程塌,根因已定
按 step 的 FCS mean@5(**剔除判题 infra 错误后的修正值**;括号=修正前 headline):

| step | full_a10(maintain) | nomaintain_a10 | base+RL(start) |
|---|---|---|---|
| pre-RL | 6.15 | 6.41 | 7.05 |
| 15 | 4.05 | 4.13 | 5.28 |
| 20 | 5.16 | 4.08 | **4.92**(1.57 纯为 infra 假象,23% 错误样本) |
| 40 | **3.06**(2.36) | **3.30**(2.83) | 5.44 |

**Debug 结论(forensic agent,证据在训练 log + rollout dump + eval samples 三层)**:
1. **主因 = 20k 截断死亡螺旋**。SFT-soup 起点在 FCS 上自然思考 16–20k+ token,早期 80–94% rollout 被截断→无代码→0 分(reward 无截断豁免);组里唯一"写短了才写完"的 rollout 独占全部 advantage(28/40 步出现 adv_max=2.475,恰为 1-of-8 非零组的代数签名)→ 策略学到"只想 2–6k token"——step 11–16 相变式塌缩,之后钉死在 ~6k,对精确算法题致命。
2. **放大器 = FCS reward 太稀疏**:81% 的 FCS 组全零(无梯度),幸存信号只编码"写完了",不编码"写得好"。
3. **方向盘 = 部分分题(ALE+NP-hard)接管晚期梯度**:变短后唯一稳定得分的行为是"快速贪心启发式",ALE 得分率 0.05→0.53——解释 ALE↑(366→490)FCS↓ 的不对称。不是字面 reward hacking:s40 的输出 think 结构/代码块完好,是**思考深度**塌了,不是格式。
4. **排除**:逐题归一(单纯 /100,仿射无害)、格式崩坏、KL/clip 病理、字面刷分;base start 几乎不塌(s15 5.28→s40 5.44)——**受害者恰是思考更长的我们的模型**。
5. **赢家(synth-only @32k)对照**:32k 下模型完整思考放得下,选择压力反转——**得分 rollout 中位长度反而是零分组的 3 倍**,reward 有梯度(多赢家组、adv_max~1.5),entropy 不塌(≥0.70)。同一模型 20k=−50% FCS、32k=+19%,截断 cap 就是分水岭。

**新 4 组 RL 的预警指标**(每 patrol 检查):赢家-长度差(得分组中位长 <0.5× 零分组、连续 3 步 → 杀);长度地板(mean 掉破前 5 步均值一半而 score 未翻倍 → step15 前杀);组退化率(>60% 有奖组是 1-of-8、持续 5 步 = 信号太稀);entropy<0.55(step20 前)= 锁死前兆;eval 端 median completion <4k = 浅思考签名。

### 3b. 纯 synthetic 500 题 @32k(= 唯一赢过 base 的 RL 配方,重启中)
- 依据:rlfsx_q35_inst_start(job 10728651,4 GPU,20 步 8h41)最终 **FCS 7.61 > base 6.38**,无 train=eval 泄漏。
- 2 GPU 尝试全灭:32k 长回复下 update_actor step2 CUDA OOM(5/5,130GB 占用再要 15.6GB),且 62min/步。**32k 必须 4 GPU**。
- 已按赢家原配置(4 GPU/480G/32768 resp/45056 model_len/gpu_mem0.4/offload 关/mb1)重提 6 臂:start 对照 + cl_nom_a5/a10/a20 + cl_newmt_a10 + cl_wd03_a10;20 步、每 5 步存,每个存点全 4 项评测。
- 观察:我们的 SFT soup 在 synth 上的回复远长于 start(clip 0.875 vs 0.19–0.31)——创新倾向的直接体现,RL 是否能把它压回有效长度是本轮看点。

---

## 4. 新 maintain 数据优化(LlamaFactory commit 494ff82)

修了两个 loss-mask bug:(1) `loss:null` 之前被当成 False → **静默把 maintain 例子整条零 loss 训练**(等于没训);(2) folded 轮不再被注入空 `<think>`(避免推理时不存在的空-think 历史 conditioning)。

A/B(同配方,唯一差异=新代码):
- 轻档:newmt_a10 FCS 5.9 ≈ 原 full a10 的 6.1(FCS/ALE 上没占优)
- **重档:newmt_a50 FCS 4.2 vs 原 full a50 的 2.2** → **maintain 修复在重档(更多 SFT 权重)才显威**。

---

## 5. 扩展到 Qwen3.6-35B-A3B(MoE,进行中)

用上面的最佳配方在 35B-A3B 上做 6-way(clean nomaintain + 去污染 + traj):
- **3 LoRA**:r32 / r16 / r32+wd0.3 → merge s0.1
- **3 全参数**(ds_z3 offload+nopin,4 GPU,1200G):full+wd0.1 / full+wd0.3 / full+lr2e6 → soup a5/a10

**35B base 参照 = FCS 8.95/15.53、ALE 448.7/636**(910/910 完整,错误样本仅 0.9%)。LoRA 首批分(mean@5/best@5):

| 设置 | FCS | vs base | ALE |
|---|---|---|---|
| **LoRA r32 s0.1** | **9.83 / 16.76** | **+0.88(+10%)** | 447(持平) |
| LoRA r32+wd0.3 s0.1 | 9.72 / 16.89 | +0.77 | 413(−36) |
| LoRA r16 s0.1 | 9.13 / 15.34 | +0.18 | **495(+46)** |

**里程碑:35B 上我们的 clean-data SFT 首次做到 FCS 净涨**(9B 上所有配方最多只能"少掉");r32_s01 FCS +10% 且 ALE 不掉,r16 换一种权衡(FCS 持平、ALE +46)。全参数 wd01/wd03/lr2e6 已训完 + soup 已合,评测因"全力让路给 RL"暂缓,RL 上齐后补。[进行中]

---

## 复现

- 数据:`LF-innov/data/innovation_clean_decontam_traj.jsonl`(去污染,已剔 agentic);maintain 臂加 `innovation_wave2_clean.jsonl`。
- 训练配置:`LF-innov/examples/train_full/auto/os-q35_a100_clean_{full,nomaintain}_wd01.yaml` + reg 变体 `os-q35_clean_nom_*.yaml` / `lora_q35_clean_nom_r*.yaml`。
- 评测:FCS/ALE = `FrontierSmith/slurm/cc_eval_thinking_both_ailab.sh`;Research = `cc_eval_research_ailab.sh`(RESEARCH_DATA 用**绝对路径** `data/frontiercs/research.parquet`);MLS = `cc_eval_mlsbench_cpu_ailab.sh`。
- 分数落盘:`FrontierSmith/outputs/cc_eval_clean_clean_*` / `cc_eval_clnom_*`。

姊妹文档:`INNOVATION_CAMPAIGN_REPORT_zh.md`(主报告)、`CASE_STUDY_r3_zh.md`。
