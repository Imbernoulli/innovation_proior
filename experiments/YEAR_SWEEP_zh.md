# 测试时年份扫描:模型到底有没有学到"站在某一年做研究"

> 2026-09-11 立项(用户提议);设计与口径先写,数字落地即追加,已有数字不改。

## 0. 为什么做

这条线的"innovation 先验"在数据上就是**时间条件化**:SFT 语料 2885 条,**100%** 以 `It is now year {历史年份}.` 开头(年份 1690–2026,集中在 2019–2025),教模型"站在方法诞生的那一年,提出当时是新的东西"。
RL 数据与主评测(FCS 算法轨、研究轨、ALE、MLS)统一用 `It is now year 2026.`;而所有 idea / 判断力类评测(`liveidea_gen`、`review_weakness`、openreview 打分、Tang & Yang)**没有任何 system prompt**,一直是在训练分布之外测的。

所以一个直接的问题是:**模型用了这个年份吗?** 如果学到了时间条件,把年份换成 1900 / 2100,输出应当系统性地移动;底座没见过这个前缀,应当基本不动(或只表现为分布外扰动)。这既是对机制的检验,也是"为什么判断力那层没有增益"的一个候选解释。

## 1. 设计

- **变量只有一个**:system prompt 里的年份 Y ∈ {1900, 1950, 2000, 2025, 2050, 2100};Y = 2026 就是记分卡里的现有数字,不重跑。
- **模板逐字与 2026 运行一致**:FCS / 研究轨 / ALE 用 `It is now year {Y}. You are a good researcher.`(`EVAL_SYS_PROMPT_MODE=short`,与 2026 运行日志打印的 system message 相同);MLS 用 `MLSBENCH_SYS_PREFIX="It is now year {Y}."`(与 `mls21.sh` 相同)。采样协议不变:`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768, N=5, thinking on`。
- **臂**:9B 三臂 `base9b_v2c`(底座,必需对照)、`lo32nm_a10`(SFT)、`rlv5_lo32nm_a10_s20`(RL);4B 三臂 `base4b` / `4b_lo32nm_a10` / `rlv5_4b_lo32nm_a10_s20` 随后。
- **bench**:FrontierCS 算法轨(153 题)+ ALE-40(f0/f1 分片)、研究轨(53 题,r0/r1)、MLS21(单次)。idea 类(`liveidea_gen`、Tang 288 seed set、`openreview_novel`)排在主 bench 之后,用 `gen_client.py` 新增的 `GEN_SYSTEM_PROMPT` 环境变量注入同一模板。
- **读法**:横轴年份、每臂一条曲线。先看 unparsed / 崩溃率与输出 token 长度随年份的变化,再看分数;臂内对 2026 基线的配对差用 `pairprob.py`(同题自助,★ = 95% CI 严格不含 0)。**底座那条曲线是必需对照**:1900 对底座也是分布外输入。

## 2. 预先登记的几种可能

| 结果形状 | 读法 |
|---|---|
| 三臂都平 | 模型不理年份;时间条件化没有被学到(或被 RL 洗掉)。照写 |
| 底座平、SFT/RL 随年份变 | 学到了时间条件;方向是否合理要看 1900 下是否主动放弃现代算法 |
| 三臂都变、形状一样 | 分布外扰动,不是时间条件 |
| 只有 RL 臂变 | RL 阶段的 2026 前缀让模型对该位置敏感,但未必是"年份语义" |

## 3. 复现

```bash
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
bash $D/scripts/year_sweep_submit.sh 1900 base9b_v2c lo32nm_a10 rlv5_lo32nm_a10_s20   # 每臂年份 5 组:f0 f1 r0 r1 mls21,双投 ailab/gpu
# 输出:cc_eval_<arm>_y<Y>_thinking_32k_both_vllm / cc_eval_<arm>_y<Y>_research_thinking_32k_vllm / cc_mls21_<arm>_y<Y>
cd experiments/scripts/agg
python3 allstats.py frontiercs base9b_v2c base9b_v2c_y1900 base9b_v2c_y2000 base9b_v2c_y2100 --json
python3 mls_merge.py base9b_v2c_y1900 lo32nm_a10_y1900 rlv5_lo32nm_a10_s20_y1900
```

## 4. 进度

| 时间 | 事件 |
|---|---|
| 2026-09-11 19:4x | 第一波投出:1900 / 2000 / 2100 × 9B 三臂,45 组(作业 13759476–13759568,`$D/logs/year_sweep_jobids.txt`) |
| 2026-09-11 20:09 | 首个作业 `ev-base9b_v2c_y1900-f0` 起跑,日志打印的 system message 为 `It is now year 1900. You are a good researcher.`,与设计一致 |
| 2026-09-11 21:0x | **MLS 全部 21 题在 1 s 内 `agent_failed`**(`mls21-base9b_v2c_y1900` 13759484)。根因不是年份:整个 `/scratch/gpfs/CHIJ/st3812/` 被删(磁盘空闲从 2.3T 跳到 18T),MLS-Bench 的 `vendor/{data,external_packages,images}` 在 della 上所有副本都是指向那里的软链。与模型无关,FCS/ALE/研究轨作业不受影响。已 scancel 其余 8 个未起跑的 mls21 作业 |
| 2026-09-11 21:4x | **MLS 依赖重建于 `$D/mlsvendor/`**:9 个容器镜像从 HF `Bohan22/MLS-Bench-Tasks` 下回(≈23G);8 个外部包从工作区副本按 pinned commit 重新 clone(工作区里全部"脏文件"核实为 pre_edit/mid_edit 运行时生成物);`scikit-learn` 存根重建;数据用 `data_scripts.devstray.bak` 在登录节点重新准备(sklearn OpenML/AIF360、badge OpenML 6/44/46)。我自己 `mlsroot/vendor/` 下三个失效软链改指新位置;提交脚本加 `MLSBENCH_DATA_ROOT`。失败的输出目录与锁**改名保留**(`*.failed_20260911*`),未删。烟测:`mls21-base9b_v2c_y1900` 重投 13761801/13761802 |
| 2026-09-12 00:41 | 烟测 13761801 起跑(排队 2.5h)。前 4 题:2 题 scored、**2 题在 8–17 s 内 `agent_failed`**——`causal-discovery-discrete`(conda worker `ModuleNotFoundError: causallearn`)与 `ml-anomaly-detection`(`FileNotFoundError .../bohan/MLS-Bench-dev/vendor/data/adbench/6_cardio.npz`)。根因同一处:`mlsroot/holdout` 仍是指向 bohan `MLS-Bench-dev/holdout` 的软链,holdout 里的 `dgp.py` 用 `Path(__file__).resolve().parents[2]` 求项目根,于是落回 bohan 树,再经其失效的 `vendor/*` 软链。也就是说 21 题里凡是 host 侧打分器要读 `vendor/` 的都会中招(grep 全部 21 题的 holdout:只有这两题 + `ml-ensemble-boosting` 的 sklearn 数据) |
| 2026-09-12 00:43 | **修复**:`mlsroot/holdout` 改为 bohan holdout 的真实副本(329K,原软链改名 `holdout.bohan_symlink_20260912` 保留),项目根现在解析到 `$D/mlsroot`;ADBench 四个数据集(6_cardio / 38_thyroid / 30_satellite / 32_shuttle .npz)从 GitHub `Minqi824/ADBench` 下到 `$D/mlsvendor/data/adbench/`。登录节点验证:`dgp.gen_input` 四个环境形状正常;conda 环境 `mlsbench-causal-bnlearn` 内 `import pgmpy, causallearn` 通过。提交脚本加 `MLS_ONLY=1` 开关(只投 mls21,避免把仍在排队的 ev-* 重复投递)。让烟测继续跑完以暴露其余 17 题是否还有洞,再整体重投 `mls21-base9b_v2c_y1900` 与其余 8 个 |
| 2026-09-12 01:28 | 烟测 13761801 跑到 20/21 题:修复后起跑的 16 题全部 `scored`(其中 ml-active-learning / ml-calibration / ml-ensemble-boosting / ml-subgroup-calibration-shift / optimization-multi-objective / causal-obs-linear-non-gaussian 为 0.0,2026 底座同题多数也是 0.0);修复前起跑的 2 题秒败;`optimization-nas` `agent_failed` 是**模型行为**——agent 对话超过 vLLM 40960 token 上下文被 400 拒绝、无输出可评(2026 底座该题 scored 0.0),不是依赖洞,照记不修。依赖层面已无新洞 → 立即 `MLS_ONLY=1` 重投其余 8 个 mls21(13769375–13769390 双投号);`mls21-base9b_v2c_y1900` 烟测锁改名 `.lock.smoke_20260912`,等烟测作业结束后输出目录同样改名再整体重投 |
| 2026-09-12 01:57 | **holdout 修复被绕过**:重投的 `mls21-lo32nm_a10_y1900`/`mls21-rlv5_lo32nm_a10_s20_y1900` 起跑后 causal-discovery-discrete / ml-anomaly-detection 仍在 6–15 s 内 `agent_failed`。原因:`mlsroot/tasks/<task>/` 下的 `edits/`、`parser.py`、`config.json` 本身也是指向 bohan 树的软链,`mid_edit.py` 里 `Path(__file__).resolve().parents[3]` 先落到 bohan 根,再进 bohan 的 holdout(而非我的副本)。修复:21 题的任务目录全部换成解引用的真实副本(`cp -rL`,共 2.6M;原软链目录移到 `mlsroot/tasks.bohan_links_20260912/` 保留),核实三题的 mid_edit/parser 现在都解析到 `$D/mlsroot`。scancel 13769375/13769377,其锁与输出目录改名(`.retry1_tasklinks_20260912` / `.failed_tasklinks_20260912`),重投 13769899/900、13769901/902;其余 6 个 PD 的 mls21 起跑时即用新目录,不动 |
