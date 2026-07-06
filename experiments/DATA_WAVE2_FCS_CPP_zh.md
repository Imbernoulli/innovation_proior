# 数据 Wave-2：FrontierCS-style C++ / 拒绝采样 + Codex 黑盒生成

> 本轮为修 FrontierCS(FCS)/ALE 掉分而造的**可验证**训练数据的一次收口。根因见
> [DATA_FIX_FCS_LANDING_zh.md](DATA_FIX_FCS_LANDING_zh.md) / [PIPELINE_FINDINGS_zh.md](PIPELINE_FINDINGS_zh.md)：
> 掉分不是推理差，是**落点错**——method 数据 98% 落点是 Python、51% 是 class，而 FCS 只给
> 「单文件 C++ 读 stdin」打分。所以本轮全部对着 **C++ 落点 + 竞赛选手收尾纪律 + 真验证**造数据。

## 一、这一波是什么（三条独立引擎，全部可验证）

| 引擎 | 模型 | 做法 | 验证 | 产物 |
|---|---|---|---|---|
| **on-policy 拒绝采样** | Qwen3.6-27B | 在 hard 可验证 RL 训练题上 rollout，4→8→16 逐步加采样，只留过验证的 | 代码 compile+run 测例；数学 answer-match；reasoning/IF 官方 reward | `_hardcp/traces/*.jsonl` |
| **tier-2 兜底解题** | DeepSeek V4 Pro | 把 27B「打满还做不出」的硬失败交给更强模型解，只留过验证的 | 同上 | `_hardcp/traces/*.deepseek.jsonl` |
| **Codex 黑盒造数据** | Codex `gpt-5.5` | 自建仿 FCS 分布的困难题 list，每题一个 `codex exec`，让它自产 context+reasoning+answer，它自己编译验证 | Codex 自验 + 落地后 g++ 编译 | `_fcs_codex/gen/<id>/` |

数学里「金标没法判」的题（多值/方程/区间/文字，占 math 约 15–29%）用 **DeepSeek V4 Flash 关思考当 LLM judge**
兜底判等价（对抗测试证明其严格，不放错解进数据）。

## 二、统计（本次收口）

**Wave-2 主集** `sft/innovation_wave2_sft.jsonl.gz`（758 条，ShareGPT 格式）：

| domain | 条数 | 说明 |
|---|---|---|
| code (C++) | 119 | 单文件 C++ 读 stdin，过测例 |
| math | 92 | 含 flash-judge 救回的不可判金标 |
| reasoning | 397 | Guru 6-domain |
| ifollow | 141 | IFEval 约束 |
| fcs_codex (C++/Py/Triton) | 9 | Codex 黑盒，algorithm 7 + research_cpu 1 + research_gpu 1 |

reasoning 长度 **中位数 33,277 字符**（直接对着「SFT 比 base 短 22×」的欠推理根因）。

**V4 / cpv4b 竞赛 C++ 集** `sft/innovation_v4_sft.jsonl.gz`（346 条）：**100% 单文件 C++ 读 stdin、100% 有 debug/自验环节**、reasoning 中位数 17,465 字符。源目录 `data_v4/*/`(context+reasoning+train_answer) 早已入库，但**从未进过 SFT 混合**（见 FCS_LANDING P0），本次把它 build 成可直接 mix 的 `.gz`。

**额外 method 数据点（回忆补齐，源目录已提交）**：本轮为解题另造的若干 method 轨迹
（`methods/{b3lyp-hybrid,caspt2,casscf,ccsdt-gold-standard,multilinear-formula-lb,pir-cell-probe-lb,variational-tst,ginzburg-landau-superconductivity,wilson-operator-product-expansion,parke-taylor-mhv-amplitude,adiprasito-huh-katz-matroid-hodge}`）。
它们有 context/reasoning/answer，但**尚缺 `train_answer.md`**（需 discovery-writeup 一步）才进训练——先入库存证。

## 三、文件清单（本 commit）

- `sft/innovation_wave2_sft.jsonl.gz` — 主集 758 条（拒绝采样 + Codex）。
- `sft/innovation_v4_sft.jsonl.gz` — V4 竞赛 C++ 346 条。
- `data_v4/_fcs_codex/` — Codex 黑盒题库 `problems.jsonl` + 产物 `gen/<id>/{context,reasoning,answer}.md`。
- `tools/hardcp_rollout.py` — 拒绝采样 driver（vLLM 27B / DeepSeek / Codex 后端，query 亲和路由，per-domain 预算）。
- `tools/fcs_codex_gen.py` — Codex 黑盒批量生成器。
- `tools/assemble_wave.py` — 把 keepers + Codex 汇成 ShareGPT jsonl。
- `tools/driver_watchdog.sh` — driver 自愈 supervisor。
- `methods/…` — 上述 method 轨迹源目录。

原始 rollout keepers 在 `data_v4/_hardcp/`（gitignore，体量大），已汇进上面的 `.gz`，故不单独入库。

## 四、如何加入训练

1. 解压：`gunzip -k sft/innovation_wave2_sft.jsonl.gz sft/innovation_v4_sft.jsonl.gz`。
2. 在 LLaMA-Factory `dataset_info.json` 注册为独立 dataset，或拼进现有 `innovation_sft.jsonl`。
3. 权重：C++ 落点是主修复，`innovation_v4_sft` 建议 oversample 到 ≥15–20%（见 FCS_LANDING P0）。
4. 复现：`python tools/assemble_wave.py`（需 `data_v4/_hardcp/traces`）、`python sft/build_v4.py`。

## 五、未尽 / 仍在跑

- 三条引擎仍在后台产出（27B 已切 **code-only**，DeepSeek tier-2、Codex 生成器）；本文件是一次**快照收口**，下一波会更多、更偏 C++。
- method 数据点待补 `train_answer.md` 再进训练。
- Codex 自造题目无独立 oracle，靠 Codex 自验；要更严可把带真实测例的 code worklist 也过一遍 Codex。
