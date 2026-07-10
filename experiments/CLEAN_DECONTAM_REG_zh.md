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

轻档(soup α=0.1 / LoRA scale=0.1)FCS/ALE —— **nomaintain-a10 基线 = FCS 6.4 / ALE 368**:

| 变体 | FCS | ALE |
|---|---|---|
| **weight decay 0.3** (a10) | **6.7** | **416** |
| weight decay 0.5 (a10) | 6.5 | 414 |
| 低 LR 2e-6 (a10) | 6.4 | 362 |
| 新代码 baseline (a10) | 6.2 | 393 |
| LoRA r32 (s0.1) | 6.0 | 369 |
| LoRA r64 / r8 / r16 (s0.1) | 5.8 / 5.4 / 5.3 | 337–376 |

**结论**
- **weight decay 0.3 在轻档最保 FCS + ALE**(6.7 / 416,双双超基线),是最有效的额外正则;NEFTune、大 wd 在**重档**反而更差。
- **LoRA 低秩也能保住 FCS**(s0.1 档 5.3–6.0);**关键是 merge scale(0.1 vs 0.5)而非 rank**——重档(s0.5)全塌到 1.1–3.0。
- 重档(soup a50 / LoRA s0.5)一律塌陷(FCS 0.6–4.2),印证"小扰动"才保能力。
- 备注:dropout 此前一直关(LoRA `lora_dropout=0.0`、模型自身无 dropout);为标准正则可选 0.05–0.1。
- **缺口**:reg sweep 的 Research/MLS 仍在补测,补齐后判断"保 FCS 的同时创新有没有迁移"。

---

## 3. RL A/B(clean soup a10 上 GRPO;对照 base+RL = clean_rlstart 5.44)

| RL 臂 | FCS | ALE |
|---|---|---|
| full_a10(maintain) step40 | 2.36 | 490 |
| nomaintain_a10 step40 | 2.83 | 462 |
| base+RL | 5.44 | — |

**结论**:RL 后 **nomaintain ≥ maintain**(2.83 vs 2.36),但**两臂 FCS 都低于 base+RL 5.44** → **RL 不提 FCS、maintain 在 RL 下也无优势**(与 r3 一致:RL 把 pre-RL 的 6+ 拉回,同质化)。ALE 则被 RL 抬高(462–490)。

---

## 4. 新 maintain 数据优化(LlamaFactory commit 494ff82)

修了两个 loss-mask bug:(1) `loss:null` 之前被当成 False → **静默把 maintain 例子整条零 loss 训练**(等于没训);(2) folded 轮不再被注入空 `<think>`(避免推理时不存在的空-think 历史 conditioning)。

A/B(同配方,唯一差异=新代码):
- 轻档:newmt_a10 FCS 5.9 ≈ 原 full a10 的 6.1(FCS/ALE 上没占优)
- **重档:newmt_a50 FCS 4.2 vs 原 full a50 的 2.2** → **maintain 修复在重档(更多 SFT 权重)才显威**。

---

## 5. 扩展到 Qwen3.6-35B-A3B(MoE,进行中)

用上面的最佳配方在 35B-A3B 上做 6-way(clean nomaintain + 去污染 + traj):
- **3 LoRA**:r32 / r16 / r32+wd0.3
- **3 全参数**(ds_z3 offload,4 GPU):full+wd0.1 / full+wd0.3 / full+lr2e6

下游:LoRA→merge s0.1、full→soup a5/a10 → 4 benchmark → 跨设置挑最优。[训练中]

---

## 复现

- 数据:`LF-innov/data/innovation_clean_decontam_traj.jsonl`(去污染,已剔 agentic);maintain 臂加 `innovation_wave2_clean.jsonl`。
- 训练配置:`LF-innov/examples/train_full/auto/os-q35_a100_clean_{full,nomaintain}_wd01.yaml` + reg 变体 `os-q35_clean_nom_*.yaml` / `lora_q35_clean_nom_r*.yaml`。
- 评测:FCS/ALE = `FrontierSmith/slurm/cc_eval_thinking_both_ailab.sh`;Research = `cc_eval_research_ailab.sh`(RESEARCH_DATA 用**绝对路径** `data/frontiercs/research.parquet`);MLS = `cc_eval_mlsbench_cpu_ailab.sh`。
- 分数落盘:`FrontierSmith/outputs/cc_eval_clean_clean_*` / `cc_eval_clnom_*`。

姊妹文档:`INNOVATION_CAMPAIGN_REPORT_zh.md`(主报告)、`CASE_STUDY_r3_zh.md`。
