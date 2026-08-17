# 来源价值审计（source-value audit）— 153 条抽样

**问题**：附录 claim「决定性步骤靠非 primary 来源支撑，光看论文推不出来」在全语料上成立多少？老统计（`tmp/count_sources.py`）只数“提到”，不看“依赖”，不能用。

**方法**：从 1,213 条 method 分层抽 153 条（42 条老标签为 author_self_account + 111 条按 5 大类比例），6 个 subagent 逐条读 notes + reasoning.md，必要时 grep refs/src 原文，分四类：
- **A route-forcing**：至少一个决定性步骤依赖非 primary 来源，只看论文看不出为什么这样走
- **B corroborating**：非 primary 来源用于核对公式/数字/定义，决定性步骤论文里就有
- **C decorative**：来源只是历史/腔调/词汇，删掉不影响任何推理
- **D single-source**：实质上是论文改写，没有非 primary 来源参与

判据：A vs B —— “只读 primary 能否看出这一步是被逼出来的？”能→B。B vs C —— “删掉所有非 primary 提法，是否有推理步失去依据？”否→C。

## 结果

| | A | B | C | D |
|---|--:|--:|--:|--:|
| **总体 n=153** | 32 (21%) | 93 (61%) | 4 (3%) | 24 (16%) |
| Empirical ML n=95 | 11 | 66 | 1 | 17 |
| Math & Physics n=23 | 11 | 9 | 1 | 2 |
| Theory n=14 | 5 | 6 | 1 | 2 |
| Combinatorial n=10 | 3 | 3 | 1 | 3 |
| Applied & Eng n=11 | 2 | 9 | 0 | 0 |

按 notes 存在性：`notes=none` 15 条 → 14 条 D；有 source_matrix 56 条 → A 6 / B 47 / D 3；有 synthesis 60 条 → A 19 / B 35 / D 6。

按老标签：老 `author_self_account` 42 条 → 只有 6 条 A、34 条 B（自述找了但没用到刀刃上）；老 `ancestors_only` 30 条 → 11 A（比例最高，因为 primary 常缺失，只能从 ancestor 推）。

## 解读

1. **C（轶事装饰）不是主要问题**：只有 4 条。用户担心的“帮不上 reasoning 的东西”在数据里已经很少。
2. **主要问题是 D**：16%，几乎全部（14/15）是 `notes=none` —— 目录里只有 primary tex，从没找过第二来源，主要集中在 Empirical ML 的近年论文（chain-of-thought, react, gpipe, medusa, xlnet, streaming-llm, distilbert, constitutional-ai, movement-pruning, unipc, least-to-most, focal-frequency-loss, megatron-lm, gridsynth…）。
3. **B 是大头（61%）**：来源真找了、真读了，但落笔时决定性步骤仍来自论文。其中一部分是本身就该 B（论文自己把反例/动机写全了，如 g-pbgd / topk-ef / bohb）；另一部分是**有自述材料却没用上**（老 self_account 标签里 34 条 B）——这是最有提升空间的一批。
4. **假 A**：`hastad-switching-lemma`、`gelfand-representation`、`nash-embedding` 等 sources.md 自认“没读原文靠 standard knowledge”；`ip-equals-pspace`、`nll-entropy`、`gign`、`holevo-bound` 是 primary 拿不到（paywall/死链）被迫从二手推。这些是薄 trace 不是 win。
5. **附带发现的坏文件**：`methods/myerson-auction/refs/roughgarden-myerson-lemma.pdf` 是 300 字节 HTML 406 错误页却被当来源引用；`methods/holevo-bound/refs/holevo-1973*.pdf` 是 paywall 占位。

## 老统计为什么错

`tmp/count_sources.py` 用正则数 arXiv 号 / URL / “Author, 2019” 模式，`basis` 标签靠 notes 表格里的类别字样。回答的是“notes 登记了几类材料”，不是“trace 依赖哪些材料”。`source_class_mentions.self_account=577` 就是字面 mention 数。

## 建议的清洗顺序（待确认）

1. **D 类补来源**（~16% ≈ 190 条全量估计）：对 `notes=none` 的方法跑一次 source search（自述/官方 repo/作者博客/OpenReview 回复），把非 primary 的 load-bearing 材料补进 reasoning，同时补 notes。
2. **老 self_account 标签里的 B**（42 抽样中 34 条 → 全量约 66 条）：材料已在 refs/，只需把 trace 的决定性步骤改成沿着自述走。
3. **假 A**：补 primary 或标记为薄。
4. 清理坏 refs 文件。
5. 论文里 §3 “Sources” 段和附录首段的数字改用本审计口径（A 21% / B 61% / D 16%），不再引用老的 82/716/624。

产物：`experiments/source_value_audit/merged.jsonl`（153 条，含 class/key_step/evidence/category/basis_old），`batch_*.jsonl` 原始输出，`PROMPT.md` 审计指令。
