# SFT 完成后的执行手册（agentic 消融 4B）

前置：两臂训完（`models_sft/agentic_ablation_4b/{withag,noag}` 根目录 = 最终权重，
save_only_model，另有 checkpoint-11/22/33/44/55/66 中间 ckpt）。

## 1. Soup（α=0.1 / 0.2，两臂 × 2 = 4 个合并体）

⚠️ Qwen3.5-4B 有 MTP head——必须用容忍 key 不对称的合并脚本：

```bash
cd /srv/home/bohanlyu/innovation_proior/training/FrontierSmith
V=.venv-gpublaze/bin/python
BASE=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*)
for ARM in withag noag; do for A in 10 20; do
  $V scripts/cc_model_soup_merge.py \
    --sft /srv/home/bohanlyu/models_sft/agentic_ablation_4b/$ARM \
    --base "$BASE" --alpha 0.$A \
    --out /srv/home/bohanlyu/models_sft/agentic_ablation_4b/soup_${ARM}_a$A
done; done
```
（脚本路径若不对，用 `experiments/scripts/train/cc_model_soup_merge.py`；参数名以
`--help` 为准，历史语义：merged = α·SFT + (1−α)·BASE，逐 SFT key 混合、SFT-only key 复制。）

## 2. 评测（与 base 同协议：eval_split_local.sh，32k thinking / presence 1.5 / n=5 / y26）

6 个模型（2 raw + 4 soup），每个一条命令，GPU 4,5（RL 若在训则排队/换 5,7 组合）：

```bash
cd /srv/home/bohanlyu/innovation_proior/training/FrontierSmith
for M in withag noag soup_withag_a10 soup_withag_a20 soup_noag_a10 soup_noag_a20; do
  VLLM_RPC_TIMEOUT=600000 EXTRA_VLLM_ARGS="--no-enable-prefix-caching" GPUS=4,5 \
    bash scripts/gpublaze/eval_split_local.sh \
    /srv/home/bohanlyu/models_sft/agentic_ablation_4b/${M#withag}... # 路径按实际
done
```

⚠️ serve 口径（base r1-r3 的血泪）：vllm 0.21 V1 默认开 prefix caching，但 Qwen3.5 的
mamba/GDN 架构下它是实验路径，长序列段会让 `sample_tokens` RPC 停滞 >120s → 引擎自杀
（`EngineDeadError`，客户端表现为成批 APIConnectionError）。**必须
`--no-enable-prefix-caching` + `VLLM_RPC_TIMEOUT=600000`**；失败可无限 RESUME=1 续跑
（默认开），已完成样本不重做。
实际执行时逐个跑（串行），TAG 命名 `q35_4b_<model>`。MLS/research 用 agent 的链
（把 8006 服务换成对应模型权重重启即可，qwen3_xml parser + RPC 120s 保留）。

## 3. 判读（README 锚点）

- 主判据 FCS：两臂差 < 噪声底 1.0 ⇒ agentic 无关紧要；noag 高 ⇒ 判负成立；withag 高 ⇒ 翻案。
- 复验机制指标：两臂生成的 think 长度分布（base 锚点对照）；ALE-40 看 abs>0 防失败地板。
- 同口径纪律：全部数字只和本机 188 题/40 题/22 任务口径互比，不与 ailab/Princeton 历史表混排。

## 4. 注意

- 评测期间 SFT 卡（0-3）空出，可给 RL 扩到 4 卡或并行第二个评测 serve。
- 磁盘：6 个模型 ×8.5G + 12 个中间 ckpt ≈ 155G，评测完删除 soup 的中间产物与多余 ckpt
  （保留每臂 final + 判读选中的 ckpt）。
