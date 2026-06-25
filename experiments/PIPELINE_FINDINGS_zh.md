# 管线与根因：两个已坐实的深层发现（本轮调查）

> 配套：[MODEL_FAILURE_ROOTCAUSE_zh.md](MODEL_FAILURE_ROOTCAUSE_zh.md)（15 失败模式全表）、[DATA_REMEDIATION_zh.md](DATA_REMEDIATION_zh.md)（数据补救）。
> 本文只记两件**从真实模型输出 + 代码逐行核对**坐实的事，每件都给「是什么 / 证据 / 怎么修」。

---

## 发现一：rewrite-leakage 不是字符串泄露，是**结构性「反推改写」立场**泄露

**结论**：训练数据里**没有**任何「写作过程/拒答/rewrite」的字面串（穷举扫了全部 1925 个 reasoning/answer/train_answer + 680 trajectory rung，0 命中）。模型那句 *"I cannot complete this thought because the next thinking is empty / nothing to rewrite"* 不是抄来的——它是**数据构造立场**在行为上的回流。

**证据**：
- `paper-to-reasoning` skill 的组织原则是**反推**：从已知论文倒推一条「第一次发现」的轨迹，且**"never betray that a finished paper exists"**、**"End by landing on the method"**、让结论「**inevitable**」。
- 实测结构指纹：读过的轨迹**~100% 都 land 预定方法**；**零真实弃局**（没有一条以「这条路不行，我放弃/换了个不同的方法」结尾）；撞墙都是装饰性的，必然收敛到既定终点；不确定性是**表演**的。
- 行为回流（真实输出）：`q35_a100_methodtraj_sft` 6/720 样本在自己 `<think>` 跑空时吐出 *"the original thinking appears garbled... nothing to rewrite meaningfully"* / *"To properly rewrite and summarize this thinking..."* / *"the next thinking is empty"*——模型把数据的**任务**（「给你一段原始思考，改写/总结成一篇自信的发现」）学进去了；推理时没有「原始思考」可改，跑空就幻觉「原始思考是空的/坏的」然后拒答。措辞由 base 模型的拒答腔提供（(iii) 通道），**根因是 (ii) 反推改写立场**。
- 同源的第二个症状：**断言但不验证**。720 条 SFT 输出里 `"guaranteed"` 占 28.9%、`"is correct"` 占 15.4%，但 **80% 得 0**。`fcs_xor_sidon` 那条断言「This confirms the construction holds for larger inputs」——假的、从没验。因为数据从不示范「检查→发现错→放弃」。

**修复（深层）**：
1. **去掉「改写已知答案 / never betray a finished paper / 必然 land」的组织立场**；注入**真实的失败/弃局轨迹**：试 → 验证 → 发现错 → 退回 → land 到一个**不同/更差/不确定**的结果（而不是每条都成功收口）。
2. **要求「能改变终点」的真验证**：narrator 跑一个**结果不预定**的具体检查（worked example / 反例 / 正确性测试），允许它**改写后续走向**——把「confirm the construction holds」变成挣来的，不是断言的。
3. 顺带（廉价）：清掉 ~285 文件里的 in-frame 违例（`the paper`/`the authors`/arXiv），它们是同一立场的**字面残迹**（见发现二之外的清理任务，已在批量修）。

---

## 发现二：Qwen3 多轮 think 处理 → 我们的 fork 与折叠多半多余（含一处需现场确认的配置）

**背景**：observed bug = *"the next thinking is empty"*，**集中在 methodtraj**（折叠最多的模型）。

**逐行核对的事实**：
1. **官方 Qwen3/Qwen3.5 模板把历史轮 `<think>` 整块剥掉**（不是留空标签）——本地模板 + Qwen3.5-9B README "No Thinking Content in History" + Qwen3 tech report 一致。
2. **我们 fork（`feat/per-turn-loss-mask`）在 SFT 路径下并不剥**：`mask_history:false` + `enable_thinking:true` 时 `remove_thought` 两个循环都不跑（`template.py:447-454`）。所以 build_sft 折叠塞进历史的**空 `<think>\n\n</think>` 被原样训练**（built 文件里 2738 个），方向上喂出了 *"thinking is empty"*。
3. **修复已落地**：`build_sft.fold_think` 改为**整块删除** `<think>...</think>`（不再留空标签）→ 空 think 块 2738 → 0，与官方模板一致。（commit `1bbba31c`）
4. **Mode 1 vs Mode 2 / 要不要 fork**：MLS 推理是**单 user 轮 + 结果走 `role="tool"`**（`base.py:1434` / `models.py:512-517`）；而 `models.py:495-502` 对 qwen3（非 deepseek/openrouter）**把历史轮 `<think>` 在客户端就丢掉**。所以**推理时模型看不到上一步的 think**。⇒ 正确的多轮表示 = **单 user + observation 结果 + 历史 think 剥掉**；Mode 1（保留 think）与 Mode 2（把结果伪造成 `human` user 边界）**都不完全对**。**fork 的 per-turn-loss 折叠主要是为了支撑 Mode 2，可以退役**，改用上游 LLaMA-Factory + 在 build_sft 里直接产出「单 user + observation + 历史 think 已剥」的自然多轮。
5. **一处需现场确认**：若 vLLM 起服务时带了 `--reasoning-parser`，think 会被解析成独立 `reasoning_content`（于是被 (4) 的客户端丢弃）；若没带，think 内联在 content 里会被保留。这决定多轮历史到底「剥」还是「留」。**定下 build_sft 改造前先核对这一项**（看评测起服务脚本的 vLLM 参数）。

**结论**：单轮评测（FCS/ALE，主战场）无多轮问题，method 数据天然对齐；多轮（agentic/trajectory）按 (4) 改成上游 + 自然多轮（历史 think 剥），退役 fork，待 (5) 确认后定稿。

---

## 即时行动状态
- ✅ `build_sft.fold_think` 整块删除（commit `1bbba31c`）。
- 🔄 in-frame 违例清理：285 文件，已修 ~74，余 211 在批量修（workflow）。
- ⏭ 结构性深层修：注入失败/弃局 + 真验证（改 `paper-to-reasoning` 立场 + 造对应数据）。
- ⏭ 竞赛数据规模化：用 HF 现成蒸馏+筛选集（OpenCodeReasoning-2 C++ 等，见 research）。
- ⏭ build_sft 统一改造（退 fork / 单 user+observation / 剥历史 think），待 vLLM reasoning-parser 配置确认。
