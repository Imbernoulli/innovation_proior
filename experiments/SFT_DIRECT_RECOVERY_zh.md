# SFT 直测恢复实验(round2 / c2):不 soup 的纯 SFT 能不能不塌?

> 问题 (c):full-FT 直测 FCS 从 base 7.05 塌到 0.25–2.40,能否靠 SFT 配方优化让**不做 average 的 SFT 模型本体**直测可用?
> 成功标准:直测 FCS ≥ 6.0(base 85%),且发现类(Research/ALE)≥ soup a10 档。
> 口径:FCS = strip-think 官方抽取,per-题 mean@5 再宏平均(dedup last-wins);anchors **base FCS 7.05 / ALE 357 / Research 9.08**。
> 姊妹文档:`CLEAN_DECONTAM_REG_zh.md`(clean 轮注册表,soup/轻档结果)。

---

## 1. 存量直测盘点(此前只有 2 个直测数据点)

已有(来自 clean 轮):

| 模型(直测,不 soup) | FCS | ALE | 备注 |
|---|---|---|---|
| base(Qwen3.5-9B-bf16) | **7.05** | 357 | anchor |
| sft_q35_clean_full_wd01(含 maintain) | 2.40 | — | maintain=wave2_clean 1352 ex,38% 占比 |
| sft_q35_clean_nomaintain_wd01 | 0.25 | — | 塌得最狠 |

**其余全部变体的直测都缺**(clean 轮只评了 soup a10/a50、LoRA s0.1/s0.5)。本轮已补交(2026-07-14):

| 直测 eval | 模型 | job |
|---|---|---|
| clnom_wd03_sft | sft_q35_clean_nom_wd03 | 11169298 |
| clnom_wd05_sft | sft_q35_clean_nom_wd05 | 11169297 |
| clnom_lr1e6_sft | sft_q35_clean_nom_lr1e6 | 11169294 |
| clnom_lr2e6_sft | sft_q35_clean_nom_lr2e6 | 11169295 |
| clnom_neft5_sft | sft_q35_clean_nom_neft5 | 11169293 |
| clnom_neft10_sft | sft_q35_clean_nom_neft10 | 11169299 |
| clnom_newcode_sft | sft_q35_clean_nom_newcode_wd01 | 11169292 |
| clfull_newmt_sft | sft_q35_clean_full_newmt_wd01 | 11169296 |
| clnom_lora_r{8,16,32,64}_s10 | s=1.0 merge(=adapter 原样,直测口径) | merge 11169313 → eval 11169356-59 |

注:所有变体目录缺 `preprocessor_config.json`/`video_preprocessor_config.json`(LF 不保存,vLLM 必需),已统一 `cp -n` 补齐。LoRA s=1.0 merge 用新 wrapper `fs/cc_merge_lora_clean_nom_s10.sh` → `models_sft/lora_q35_clean_nom_r*_s10_merged`。

## 2. 新配方(batch 1,4 个 × 2 H200,2026-07-14 提交)

机制假设:直测塌陷的最大单因素 = **maintain(竞赛)数据占比**(full 2.40 ≫ nomaintain 0.25,8 倍);其次是"扰动大小"(LoRA 低秩 / 低 LR / 大 wd 都是同一方向)。

| run | 配方 | 假设 | 训练 job | 直测 eval |
|---|---|---|---|---|
| sft_q35_c2_maint2x_wd01 | traj2225 + wave2×2(maintain 55%),其余同 full_wd01(4 GPU,eff batch 128 与 full_wd01 一致) | maintain 剂量↑ → 直测 FCS↑ | 11169532 | fix 11169546 → 11169547 |
| sft_q35_c2_maint4x_wd01 | traj2225 + wave2×4(maintain 71%) | 剂量-响应第二点 | 11169533 | fix 11169548 → 11169549 |
| sft_q35_c2_swa_wd01 | 同 full_wd01 但 save_steps=4(4 GPU,28 步),训后**纯轨迹均匀平均**(不含 base)avgtail(后 4 ckpt)/ avgall(全部 7) | 若"训练内 average"(≈EMA/SWA)能恢复直测,则 soup 收益可内化进 SFT | 11169531 → swa-build 11169543 | 11169544(tail)/ 11169545(all) |
| sft_q35_c2_twostage_mt | 从 nomaintain_wd01 ckpt 出发,wave2_clean 单独回炉 1ep @lr2e-6 | 顺序效应:竞赛力放训练**末端**恢复 | 11169530 | fix 11169550 → 11169551 |

数据复制用 dataset_info.json 别名(innovation_wave2_clean_r2/r3/r4 → 同一文件),零磁盘开销。
调度备注:2026-07-14 06:00–18:00 della 维护窗;twostage/swa 以短 walltime 在维护前 backfill 抢跑(03:47 起跑),maint2x/4x 挂依赖(峰值 ≤8 H200)维护后自动起;全部 pli 评测和 LoRA merge(cpu)也要等 18:00 后。
训练状态:twostage **训完**(22 步,loss 0.35→0.30,04:36)、swa **训完**(28 步,loss 0.79→0.69,7 个 ckpt 4/8/…/28 全在,04:51);maint2x/4x 维护后起。
batch 2(待存量证据落地后选):wd03/wd05×maintain 组合、lr2e6×maintain、LoRA r128 或 2ep、wd1.0、maintain 换通用竞赛数据。

## 3. 结果表(直测,不 soup)

(评测进行中,待填)

## 4. 结论

(待填)

---
*更新:2026-07-14 提交全部 batch;下次更新=存量直测分数落地。*
