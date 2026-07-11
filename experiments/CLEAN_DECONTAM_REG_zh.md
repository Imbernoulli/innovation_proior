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

### 3a. mixed 数据 @20k cap(FrontierCS172+synth10+ALE40)—— 全程塌
按 step 的 FCS mean@5(best@5):

| step | full_a10(maintain) | nomaintain_a10 | base+RL(start) |
|---|---|---|---|
| pre-RL | 6.10 (10.99) | 6.41 (11.58) | 7.05 (11.78) |
| 5 | 4.81 (8.98) | 5.78 (10.24) | — |
| 10 | 4.90 (9.67) | — | — |
| 15 | 4.00 (9.51) | 4.13 (9.67) | 5.25 (10.66)* |
| 20 | 5.15 (10.73) | 4.08 (8.59) | 1.57 (3.68)* |
| 40 | 2.36 (5.22) | 2.83 (5.60) | 5.44 (11.18) |

\* start_step15/20 的分数含判题 infra 超时计 0(step20 有 236/910=26%!),**已修 resume 逻辑重评中**,1.57 大概率被低估。
**结论**:20k cap 的 mixed RL 对所有臂都是净负(pre-RL 6.1–6.4 → step20 4–5 → step40 2.4–2.8);ALE 被 RL 抬高(step40 462–490)。回复长度塌缩(→5–7k)+ clip 掉到 0.03,复现了 RL 考古的"response collapse"。

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

LoRA 首批分(mean@5/best@5;35B base 评测中,暂无参照):

| 设置 | FCS | ALE |
|---|---|---|
| **LoRA r32 s0.1** | **9.83 / 16.76** | 447 |
| LoRA r32+wd0.3 s0.1 | 9.72 / 16.89 | 413 |
| LoRA r16 s0.1 | 9.13 / 15.34 | **495** |

9B 最好档才 6.7,35B LoRA 轻档直接 9.7–9.8 —— 净增益要等 35B base 分出来才能定,但绝对值已远超 9B 全家。全参数 wd01/wd03/lr2e6 均已训完(loss 0.98→0.93),soup a5/a10 评测中。[进行中]

---

## 复现

- 数据:`LF-innov/data/innovation_clean_decontam_traj.jsonl`(去污染,已剔 agentic);maintain 臂加 `innovation_wave2_clean.jsonl`。
- 训练配置:`LF-innov/examples/train_full/auto/os-q35_a100_clean_{full,nomaintain}_wd01.yaml` + reg 变体 `os-q35_clean_nom_*.yaml` / `lora_q35_clean_nom_r*.yaml`。
- 评测:FCS/ALE = `FrontierSmith/slurm/cc_eval_thinking_both_ailab.sh`;Research = `cc_eval_research_ailab.sh`(RESEARCH_DATA 用**绝对路径** `data/frontiercs/research.parquet`);MLS = `cc_eval_mlsbench_cpu_ailab.sh`。
- 分数落盘:`FrontierSmith/outputs/cc_eval_clean_clean_*` / `cc_eval_clnom_*`。

姊妹文档:`INNOVATION_CAMPAIGN_REPORT_zh.md`(主报告)、`CASE_STUDY_r3_zh.md`。
