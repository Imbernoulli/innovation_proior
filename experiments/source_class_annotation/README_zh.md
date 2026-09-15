# 来源类别标注（2026-09-14）

**为什么有这一轮**：2026-09-08 22:08 PDT 一次磁盘清理会话 `rm -rf` 了 `innovation_proior/methods`（22.8 GB）和 `data_v4/_hardcp`（35.7 GB）。git 跟踪的 `results/*.md`、404 个单元的 notes、170 个单元的 refs 已用 `git checkout` 恢复；`.gitignore` 排除的 `methods/**/{notes,refs,src,code}` 中约 780 个单元的材料本机已无副本。

**恢复**：`recover_from_transcripts.py` 扫全部 Claude Code agent transcript（10,646）与 Codex session（5,563），凡完整 `cat`/Read 过的文件按最新一次读取原样写回：698 个 notes、134 个 refs 文本、50 个 src、14 个 code；清单 `recovered_files_manifest.json`。

**标注**：`annotate_workflow.js`（Sonnet，每单元一个 agent；低置信度二读）。输入 = 现存 + 恢复的 bundle、8 月 svfix 三波 agent transcript 摘要、results/。每类记录（ancestor / explainer / self_account / code / other）分 `present`（bundle 里有）与 `used`（trace 真依赖，附一句证据），`basis` 记录用到了哪些材料。结果 `labels.json`：1,185 / 1,185 单元；present A 883 / E 640 / C 587 / S 386；used A 830 / E 289 / C 504 / S 209；104 个只能靠 results/ 标注。论文 Figure 2.2 的 Venn 由 `overleaf_paper/figures/venn_counts.json` 生成。

旧的 7 月启发式标注（`tmp/labels/`，脚本按 notes 里的类别字样计数）与本轮在有材料的单元上：ancestor 一致约 79%，explainer 73%，self-account 76%（脚本只数了作为"骨干"的自述，少了一半）。
