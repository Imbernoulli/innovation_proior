# Qwen3.6-35B-A3B (MoE) 线重启:SFT/soup/LoRA 评测补齐 + RL 可行性与 smoke

日期:2026-07-14。本文档记录 35B 线重启的三件事:(1) 评测欠账补齐;(2) 35B MoE 上跑 GRPO RL 的可行性分析;(3) RL smoke 配置与结果。

模型:`models/Qwen3.6-35B-A3B`(arch `Qwen3_5MoeForConditionalGeneration`,model_type `qwen3_5_moe`,35.95B 总参 / A3B 激活,256 experts,GDN 混合线性注意力,bf16 权重 67.0 GiB,单卡 H200 141G 可放下)。

## 1. 评测全表(FCS/ALE,thinking 32k,mean@5)

基线:**base FCS 8.947 / ALE 448.7**(182 题,910 samples,complete)。

全表已落地(2026-07-14 深夜,分片管线 + agg 全部跑通,182 题/910 samples 口径):

| 模型 | TAG | FCS mean@5 | ALE perf | 备注 |
|---|---|---|---|---|
| base Qwen3.6-35B-A3B | q36_35bA3b_base | 8.947 | 448.7 | 基线 |
| lora r16 s0.1 | q36_lora_r16_s01 | 9.134 | 494.6 | |
| **lora r32 s0.1** | q36_lora_r32_s01 | **9.826** (+9.8%) | 447.4 | **FCS 冠军**,RL 起点 |
| lora r32 wd03 s0.1 | q36_lora_r32_wd03_s01 | 9.720 | 412.9 | |
| full-FT wd01 直测 | q36full_wd01_sft | 2.496 | 302.3 | 塌陷(同 9B 故事) |
| full-FT wd03 直测 | q36full_wd03_sft | 2.194 | 321.6 | 塌陷 |
| full-FT lr2e6 直测 | q36full_lr2e6_sft | 8.067 | 405.8 | 低 LR 直测不塌,但仍低于 base |
| soup wd01 a5 | q36full_wd01_a5 | 8.185 | 468.8 | |
| soup wd01 a10 | q36full_wd01_a10 | 8.388 | 493.7 | |
| soup wd03 a5 | q36full_wd03_a5 | 9.397 | 497.4 | FCS+ALE 双升 |
| soup wd03 a10 | q36full_wd03_a10 | 8.445 | **533.8** | |
| soup lr2e6 a5 | q36full_lr2e6_a5 | 8.146 | **536.9** | ALE 冠军 |
| **soup lr2e6 a10** | q36full_lr2e6_a10 | **9.676** | 497.2 | **均衡冠军**(FCS+0.73 且 ALE+48.5) |

读法:35B 复刻 9B 规律——full-FT 直测塌 FCS(除 lr2e6 低学习率档),轻档扰动(soup a5/a10、LoRA s0.1)普遍 FCS≥base 且 ALE 上升。**FCS 单指标冠军 = r32_s01(9.83);均衡冠军 = lr2e6_a10(9.68/497)**;lr2e6_a10 是 RL 第二臂/后续 soup-RL 的天然候选。

### 1.1 "不 soup" 对照组(2026-07-14 应用户要求补做)

"不 soup" = 训练产物**不做任何平均/缩放**直接测,共两类:

1. **full-FT 直测 ×3**(q36full_{wd01,wd03,lr2e6}_sft,已在上表排队)——全参 SFT 权重原样;soup a5/a10 是它与 base 的 5%/10% 插值。
2. **LoRA s=1.0 满强度合并 ×3**(新增)——`merge_lora_scaled.py` 的 s 就是 LoRA 的"soup 旋钮":W = W_base + **s**·(α/r)·BA。已完成的 r16/r32/r32_wd03 全部是 **s=0.1**(≈轻档汤等价物);s=1.0 = 训练出的 adapter 原样合并。**s=1.0 vs s=0.1 直接回答"不 average 差多少"**。

结果(已落地):

| 模型 | TAG | FCS | ALE | Research |
|---|---|---|---|---|
| lora r16 **s=1.0** | q36_lora_r16_s10 | 5.635 | 420.6 | — |
| lora r32 **s=1.0** | q36_lora_r32_s10 | 4.189 | 333.8 | **16.44(全场最高)** |
| lora r32_wd03 **s=1.0** | q36_lora_r32_wd03_s10 | 4.337 | 354.9 | — |

**答案:"不 average"在 FCS/ALE 上差很多**(r32:9.83→4.19,-57%;ALE 447→334),full-FT 直测更惨(2.2-2.5)。但 **Research 反转:s=1.0 拿到全场最高 16.44** —— FCS-vs-Research 目标冲突,s 是 trade-off 旋钮而非单调优劣。

### 1.2 s 旋钮中段扫描(2026-07-14 深夜,应用户要求)

两端已知:s=0.1(FCS 9.83/ALE 447)vs s=1.0(FCS 4.19/ALE 334/Research 16.44)。找 FCS-vs-Research 膝点(9B 的 soup-α 甜点在 0.05–0.1;LoRA 低秩约束温和,膝点可能右移):
- 合并:**11193936**(`cc_merge_lora_q36_35b_sweep.sh`,r32 × s=0.2/0.3/0.5 → `models_sft/lora_q36_35bA3b_clean_nom_r32_s{02,03,05}_merged`,四件套+磁盘护栏)
- 评测(afterok merge,**off-ailab:pli H100 TP=2**,分片管线;提交脚本 `scripts/cc_eval_35b_ssweep_submit.sh`):

| TAG | FCS/ALE 链(s0/s1/w2/w2/agg) | Research 链 |
|---|---|---|
| q36_lora_r32_s02 | 11193943/944/945/946/947 | 11193948/949/950/951/952 |
| q36_lora_r32_s03 | 11193953/954/955/956/957 | 11193958/959/960/961/962 |
| q36_lora_r32_s05 | 11193963/964/965/966/967 | 11193968/969/970/971/972 |

风险:pli-low 队列深(80 pending,fairshare 0,tp2probe 已排 9h+)——若堵死,切回 ailab 一行命令:`EVAL_PART=ailab EVAL_GRES=gpu:1 EVAL_TP=1 MERGE_JID=11193936 bash scripts/cc_eval_35b_ssweep_submit.sh`(先 scancel pli 链)。

合并作业:11170474(`cc_merge_lora_q36_35b_s10.sh`,CPU 分区,复用 s01 同一路径 merge_lora_scaled.py + Qwen3_5MoeForConditionalGeneration 自动类选择,含四件套拷贝与 <500G 磁盘护栏);评测 afterok 挂在其后。输出 → `models_sft/lora_q36_35bA3b_clean_nom_{r16,r32,r32_wd03}_s10_merged`(每个 ~67G,共 ~200G)。

对照阅读(9B 先例):s 缩放与 soup alpha 同为"扰动强度"旋钮,预期 s=1.0 FCS 低于 s=0.1(若反之,则 35B 上 LoRA 低秩约束本身已足够温和,不需要缩)。

Research track(64 题 runnable 口径,全部从零 → 2-way 分片):base s0/s1 11171441/442(agg 11171445)、r32_s01 s0/s1 11171446/447(agg 11171450)、r32_s10 s0/s1 11171451/452(agg 11171455);best soup 待 FCS/ALE 落地后补交。

注:
- 4 个 soup 评测 07-11 曾跑到一半被外部 cancel(35B 线暂停),同 TAG 重交 = resume,历史 samples 复用。
- **2-way 分片加速(07-14 应用户要求)**:所有**从零开始**的 FCS/ALE 与 research 评测改为 NUM_SHARDS=2 成对作业(SHARD_IDX=0/1,各 1 GPU,同 TAG,按题目 idx 交错切分),wall-clock 减半、GPU 时数不变;shard 输出各写 `shard_{0,1}/samples.jsonl` + `summary_shard.json`,链尾 CPU 聚合作业(`slurm/cc_eval_agg_shards.sh`)按 (data_source, ground_truth, sample_idx) last-wins 合并、过完整性闸(910/320 样本、错误≤12/20)后用同一 driver 的 resume 模式写出**规范格式的顶层 summary.json**(空任务列表,无需 vLLM)。**有真实历史 samples 的 resume(wd01_a10 / wd03_a5 / wd03_a10)保持单 shard 不动**,避免 shard 划分变化浪费已有样本;wd01_a5 的 582 条历史"samples"全是死 judge error(无可复用),按从零处理并入分片。提交脚本:`scripts/cc_eval_35b_sharded_submit.sh`。
- **吞吐顺手提**:分片 FCS/ALE 作业 MAX_NUM_SEQS 64→128(GDN 混合架构仅 10/40 层持 KV,H200 上 ~60G KV 池远不止 64 并发)、CONCURRENCY 64→96、REQUEST_TIMEOUT 2400→3600(高并发下排队请求等更久)。research 保持 CONCURRENCY=4(评测器与 vLLM 共卡)。
- 全部 GPU 作业受 07-14 06:00–18:00 EDT 维护窗(root_277,全部 ailab/pli 节点)压着,预计 18:00 起陆续开跑。
- 读分纪律:引用前查 samples.jsonl 的 error 数(eval-resume-error-zero-bug)。
- 结果落地后更新命令:见文末"更新本表"(分片 TAG 的 summary.json 由 agg 作业写出,读法不变)。

**当前 winner(已完成口径):lora_q36_35bA3b_clean_nom_r32_s01_merged**(FCS 9.826,首个 FCS 净增益 +9.8%,ALE 持平 base)。RL smoke 以它为起点。

## 2. RL 可行性分析(GRPO on 8×H200)

复用 9B 已验证链:`slurm/cc_rl_frontiersmith_synth.sh` → `scripts/run_verl_grpo_frontiercs_qwen35_9b.sh`(verl fork + vLLM 0.23.0 + transformers 5.12.1,synth-500 数据,offline reward,无 judge server)。栈内对 qwen3_5_moe 的支持逐项核验:

### 2.1 逐项核验结论

| 项 | 结论 |
|---|---|
| vLLM 0.23 arch 支持 | `Qwen3_5MoeForConditionalGeneration` + `Qwen3_5MoeMTP` 均在 registry |
| verl monkey_patch | `verl/models/transformers/monkey_patch.py:270,495` 显式支持 `qwen3_5_moe`(remove_padding forward patch) |
| GDN prefill backend | runner 自动对 `qwen3_5_moe` 设 `gdn_prefill_backend=triton`(已有代码,无需改) |
| ckpt 格式(关键) | LF 存盘为 **UNFUSED** per-expert 键(`mlp.experts.N.gate_proj.weight`,31666 键);transformers 5.12 live module 是 **FUSED** 3D(`mlp.experts.gate_up_proj`,1026 键)。5.12 的 `conversion_mapping`(qwen3_5_moe_text ← qwen2_moe,MergeModulelist)在 from_pretrained 时自动 unfused→fused;**tiny 模型 round-trip 实测 EXACT MATCH**(2026-07-14,.venv-vllm023) |
| FSDP→vLLM 权重同步 | 代码级 trace(2026-07-14)结论 **无 blocker**:fork 为 verl 0.8.0.dev,走 checkpoint-engine/CUDA-IPC 路径(`fsdp_workers.py rollout_mode` → 8 GB IPC bucket → vLLM `load_weights`);名字翻译全部由 vLLM 端负责,vLLM 0.23 `qwen3_5.py` 的 `fused_expert_params_mapping` 原生接受 HF fused 3D 键(`experts.gate_up_proj→w13_weight`,dim -2 切 w1/w3,无需转置);最大单 tensor(gate_up_proj [256,1024,2048] fp32)= **2.0 GiB < 8 GB bucket**;verl 的 MoE weight-loader patch 对本 arch 是无害 no-op(vLLM 0.23 的 `process_weights_after_loading` 幂等) |
| vLLM sleep/wake | 默认 `free_cache_engine=True` + `enable_sleep_mode=True` → update_actor 期间 vLLM **sleep level 2(权重销毁 + KV 释放)**,每步经 IPC 重同步;GMU 只决定 awake 阶段的池子大小 |
| in-tree 先例 | `verl/examples/grpo_trainer/run_qwen3_vl_30b_vllm_fsdp_npu.sh` = Qwen3-VL-30B-**A3B**(fused-expert MoE)+ FSDP2 + vLLM colocated,同一代码路径的存在性证明 |
| MTP 头 | base 有 19 个 mtp.* 键(HF 类不含 → 忽略);全部 SFT/soup/LoRA merged ckpt 无 mtp(LF drop,同 9B 已知) |
| 四件套 | 全部 12 个 35B 构建目录已补齐 preprocessor/processor/video_preprocessor_config.json + chat_template.jinja(从 base 拷,cp -n) |
| GDN fast path | .venv-vllm023 缺 flash-linear-attention/causal-conv1d → actor 侧 GDN 走 torch fallback(慢但正确;9B 同栈全程如此) |

### 2.2 显存 budget(141 GiB/GPU,8-way FSDP;verl trainer 持 **fp32** 参数)

| 项 | /GPU | 备注 |
|---|---|---|
| fp32 params | 16.7 G | 35.95B×4B / 8(verl FSDP 默认 model_dtype fp32) |
| fp32 grads | 16.7 G | |
| Adam fp32 m+v | 33.5 G → **offload 到 CPU** | resident 会复刻 9B 的 save-OOM;CPU 侧共 268 G(节点 1.5 T,job 申 960 G) |
| activations (micro=1, ckpting, seq≤42k, hidden 2048/40 层) | ~10–25 G | expandable_segments:True |
| vLLM awake 池(rollout 阶段) | GMU 0.65 = 91.6 G | TP=1:bf16 权重 67 G + KV ~24 G;GDN 混合架构仅 10/40 层 full-attn,KV 小。awake 阶段合计 91.6+16.7(+grads)≈128 G < 141 G |
| update_actor 阶段 vLLM | **≈0**(sleep level 2,权重销毁) | 每步重同步(update_weights,9B 实测 7–8 s) |
| 权重同步 bucket | 8 G 瞬时 | update_weights_bucket_megabytes=8192;最大 tensor 2.0 GiB fp32 |

结论:**可行,无需改代码**。核心配置 = `ACTOR_PARAM_OFFLOAD=False, ACTOR_OPTIMIZER_OFFLOAD=True, REF_PARAM_OFFLOAD=True, micro_batch=1, TP=1, GMU=0.65, response 32768 / model_len 45056`。OOM 退避序列:GMU 0.65→0.55;response 32768→24576;TP=1→2(权重减半但 DP 引擎减半,GMU 0.35)。
Watch item(smoke 验证点):第 2 次权重同步正常(step2 reward 非退化、无 `weight_loader` AttributeError)——verl 的 MoE loader patch 对本 arch 不生效,依赖 vLLM 0.23 自身幂等性(代码已核,smoke 实证)。

## 3. RL smoke

- 提交脚本:`FrontierSmith/scripts/cc_rl35b_synth_submit.sh`(新;knobs 见脚本头)
- 起点模型:`models_sft/lora_q36_35bA3b_clean_nom_r32_s01_merged`(当前 winner)
- 配置:8×H200,TP=1,GMU=0.65,STEPS=3,SAVE_FREQ=2,TB=8 × RN=4(32 seqs/step),mini=8,micro=1,response 32768 / model_len 45056,eval-matched sampling(top_p .95 / top_k 20 / presence 1.5),ACTOR_PARAM_OFFLOAD=False / ACTOR_OPTIMIZER_OFFLOAD=True / REF_PARAM_OFFLOAD=True,save_contents=["model"]
- **作业:11169862 `rl35b_r32s01_smoke`**(2026-07-14 提交;维护窗 18:00 结束后开跑)。ckpt → `FrontierSmith/checkpoints/rl_frontiersmith_synth/rl35b_r32s01_smoke/`
- SUCCESS 判据:global_step_2 完整 model-only ckpt 落盘(~67 G bf16,shard 齐全)+ 真实步时报告;save-OOM/加载失败按 2.2 退避序列迭代

### 3.0 全自动流水线(2026-07-14 应用户"最高优先跑通 RL"指令)

不需要人盯,四层 slurm 依赖链自驱(状态流水账:`FrontierSmith/logs/rl35b_pipeline_status.log`):

1. **smoke → checker**(`scripts/rl35b_smoke_check_and_advance.sh`,afterany 挂在 smoke 后):按 SUCCESS 判据验证 ckpt 完整性(8 rank shards + ≥55G)+ 抽取真实步时/显存入日志。
2. **FAIL → 自动退避重提**:区分 **host-RAM OOM**(slurm oom_kill 无 CUDA OOM → 同级重提 MEM=1450G;若已 1450G 仍 host-OOM → ALARM 停)与 **GPU OOM**(退避序列 L2 GMU0.55 → L3 +resp24576 → L4 TP2+GMU0.35 → 尽头 ALARM),每级自带下一个 checker。
3. **PASS → 立即自动提交正式 20 步**:用通过级别的 knobs,`STEPS=20 TB=32 RN=8 MB=16 save_freq=5 KEEP=10 model-only WALL=23:59`,**r32s01 + base 对照臂**同时排队,各带 w2/w3 afterany 断点续训窗(launcher window-guard 保证到步即空转退出)。
4. **PASS → 每臂一个 ckpt-eval waiter**(`scripts/rl35b_ckpt_eval_waiter.sh`,CPU 48h,10min 轮询):每个完整存点(s5/10/15/20)自动链 export(CPU `merge_fsdp_to_hf`,bf16 fused 键,→ `models_rl/rl35b_<arm>_s<N>_hf`)→ 四件套补拷 → **off-ailab 分片 FCS/ALE**(pli H100,TP=2,`--account=goedelprover --qos=pli-low`,2 shards×2GPU + w2 + agg)→ s20 加 research 分片链。**每次轮询与提交前 df 检查,<500G 写 ALARM 并停**。
   - pli TP2 路径预验证探针:cc-eval-35b-tp2probe(VALIDATE=1,base 35B,TAG=q36_35b_tp2probe,不污染正式输出)。

### 3.2 smoke 迭代记录

| 尝试 | job | 配置 | 结果 |
|---|---|---|---|
| #1 | 11169862 | TP1 GMU0.65 resp32768 **mem=960G** | **HOST-RAM OOM**(29min,oom_kill in `ref_log_prob`;非 GPU:vLLM TP1@GMU0.65 init + 整轮 rollout 均成功 → GPU knobs 已验证)。根因:fp32 Adam offload 268G + ref fp32 CPU 参数 140G + Ray object store(~cgroup 30%)+ 8 worker host 开销 > 960G |
| #2 | **11176508** | 同 #1 GPU knobs,**MEM=1450G(整节点)** | 排队中;checker 11176509 挂后 |

教训入库:8 卡 35B RL 作业 host 内存必须要整节点(submit 脚本已改默认 GPUS≥8 → 1450G)。曾按错误前提(以为 GPU OOM)排的 L2 级 smoke 11176328/29 已撤。

### 3.1 smoke 结果(待填)

| step | gen (s) | update_actor (s) | step total (s) | max_mem alloc/reserved |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 (+save) | — | — | — | — |
| 3 | — | — | — | — |

参照系:9B winner(rlfsx_q35_inst_start,4×H200,32×8@32k)step≈1100–1190s,gen≈310–365s,update_actor≈660–685s,update_weights≈7–8s。

## 4. 下一步:20 步正式 RL 建议配置

smoke 通过后(以 smoke 实测步时修正):

```
STEPS=20 SMOKE=0 TB=32 RN=8 MB=16 ONLY=r32s01 bash scripts/cc_rl35b_synth_submit.sh
# 对照组(评测欠账出分后再定):ONLY=base 或 best-soup
```

- TB=32×RN=8=256 seqs/step,mini=16(128 seqs/update,2 updates/step)——与 9B synth-RL 口径一致,可比
- save_freq=5,KEEP=10,save_contents=["model"](35B ckpt ~67 G,磁盘纪律:df<500G 停手)
- 20 步预计横跨 >1 个 23:59 窗口 → 用 FRESH_START=0 续窗(launcher 自带 window-guard)
- RL 后评测:`sbatch --time=06:00:00 --job-name=cc-eval-35b-<tag> --export=ALL,MODEL_PATH=<merged_hf>,TAG=<tag> slurm/cc_eval_thinking_both_ailab.sh`

## 5. 在跑/排队作业清单(2026-07-14 提交)

2026-07-14 加速改造(用户嫌 6.5h 慢)后的现役清单;被替换的单 shard PENDING 作业(11169248/252–256/279/280、11169679–684、11170476–482)已 scancel。每条分片链 = s0+s1(GPU,6h)→ s0w2+s1w2(afterany resume,3.5/4h)→ agg(CPU,写顶层 summary.json)。

| Job | TAG | 内容 |
|---|---|---|
| 11169249/250/251 | q36full_{wd01_a10,wd03_a5,wd03_a10} | 真实 resume,单 shard 保留(6h) |
| 11171396–400 | q36full_wd01_a5 | 分片链(历史全 error → 从零) |
| 11171401–405 / 406–410 / 411–415 | q36full_{wd01,wd03,lr2e6}_sft | full-FT 直测分片链 ×3 |
| 11171416–420 / 421–425 | q36full_lr2e6_{a5,a10} | lr2e6 soup 分片链 ×2 |
| 11171426–430 / 431–435 / 436–440 | q36_lora_{r16,r32,r32_wd03}_s10 | LoRA s=1.0 分片链 ×3 |
| 11171441–445 / 446–450 / 451–455 | research {base, r32_s01, r32_s10} | research 分片链 ×3(GMU 0.62) |
| ~~11169862~~ | rl35b_r32s01_smoke #1 | HOST-RAM OOM @960G(见 §3.2) |
| **11176508** | rl35b_r32s01_smoke #2 | MEM=1450G 重提 |
| 11176509 | rl35b_check_L1 | smoke checker(afterany;PASS→自动正式 20 步双臂+waiters,FAIL→自动退避) |
| 11176510 | cc-eval-35b-tp2probe | pli H100 TP=2 评测路径探针(VALIDATE=1) |
| 11170474 | cc-merge-lora-q36-35b-s10 | LoRA s=1.0 合并 ×3 — **已完成**(7m27s,3×66G 已验证) |

注:wd01_a5 的 582 条历史 samples 全部是死 judge infra error(reward=None)——`_record_compatible` 对 error 记录返回 False → 分片重跑不浪费任何有效样本;其 shard_0 旧 error 记录在 agg 的 last-wins 合并中被新记录覆盖。

## 6. 更新本表

```bash
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs
for d in cc_eval_q36full_*_thinking_32k_both_vllm cc_eval_q36_lora_*_thinking_32k_both_vllm cc_eval_q36_35bA3b_base_thinking_32k_both_vllm; do
  python - "$d" <<'PY'
import json,sys
d=sys.argv[1]
try: j=json.load(open(f'{d}/summary.json'))
except FileNotFoundError: print(f'{d}: PENDING'); raise SystemExit
m=j['metrics']; print(d, 'FCS', round(m['frontiercs']['reward']['mean@5'],3), 'ALE', round(m['alebench']['performance']['mean@5'],1), f"({j['complete_problem_count']}题)")
PY
done
```
