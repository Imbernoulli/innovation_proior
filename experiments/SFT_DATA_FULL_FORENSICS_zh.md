# SFT 数据全量法证(clean 轮):数据、训练、生成三层逐一实看

> 任务:**不预设机制**,把 clean 轮 SFT(创新数据 + maintain 数据)保存下来的一切——数据文件、训练配置、训练产物、训练日志、评测生成——真正读一遍,先报告"看到了什么",最后才给解读。
> 方法:纯读盘(不占 GPU、不动在跑作业);tokenizer 用训练同款(`FrontierSmith/models/Qwen3.5-9B-bf16`);数据精读共 **39 个完整样本**(traj 16 + wave2 15 + maintain_r3 8+3 条轨迹)、生成精读 **30+ 条**;全量扫描均为 python 流式统计。
> 姊妹文档(结论不重复,只交叉引用):`CLEAN_DECONTAM_REG_zh.md`(分数注册表)、`AVERAGE_INNOVATION_zh.md`(§3 已做生成端塌陷分桶:83.2% 真编译错、think 20× 缩短)、`SFT_DIRECT_RECOVERY_zh.md`(c2 直测恢复,进行中)、`DATA_WAVE2_FCS_CPP_zh.md`(wave2 造数流水线)。
> 撰写:2026-07-14。事实与推断分开标注;与既有文档矛盾处在 §7.3 单列。

---

## ① 资产清单(全)

### 1.1 数据(`LF-innov/data/`,dataset_info 条目 → 文件)

| dataset_info 条目 | 文件 | 行数 | mtime | 在 clean 轮的角色 |
|---|---|---|---|---|
| `innovation_clean_decontam_traj` | innovation_clean_decontam_traj.jsonl | **2225** | 07-08 13:55 | **创新主数据**(所有 clean 臂共用) |
| `innovation_wave2_clean` | innovation_wave2_clean.jsonl | **1352** | 07-08 11:28 | **maintain 臂追加数据**(full/newmt/c2) |
| `innovation_wave2_clean_r2/r3/r4` | → 同一文件(别名) | — | 07-14 加入 | c2 轮 maintain 2×/4× 过采样(零磁盘开销) |
| `innovation_clean_decontam` | innovation_clean_decontam.jsonl | 2698 | 07-08 11:28 | 未被任何 clean 配置引用(non-traj 版,1840 unique prompts) |
| `innovation_maintain_r3` | maintain_r3.jsonl | 903 | 07-06 | **不在 clean 轮**(r3 轮 maintain;本报告读作对照) |
| (r3 系列) | innovation_methodtraj_v4_r3.jsonl 等 | 2225 | 07-06 | traj_clean 的直接前身(见血缘) |

**血缘(全部 md5 实测)**:
- `traj_clean` 与 `methodtraj_v4_r3` **prompt 集完全相同**(1713 unique/1713 重合)、1547 条 2-turn 行 byte 级相同;**678 条多轮行不同**——r3 的 folded 轮(loss:false、无 think)在 clean 版里部分变为 **loss:true + 保留 think**(见 §2.3)。即"decontam"没有删掉任何题,改的是多轮折叠/训练面。
- `wave2_clean` = `wave2_r3`(758 条全保留)+ **594 条新增**(cp 175、math 171、empty 248)。新增 cp 是 Codeforces 式题面("## Problem Statement…easy/hard version");与 `innovation_v4(_r2)` 无重合。
- traj 的 346 条 "expert competitive programmer 2025" 样本 = `innovation_v4_r2.jsonl` 的 346 条(md5 全重合)= v4/cpv4b 竞赛 C++ 集。
- traj ∩ wave2 prompt 重合 = 0。

### 1.2 训练配置(`LF-innov/examples/`)

- 主配置:`train_full/auto/os-q35_a100_clean_{full,nomaintain}_wd01.yaml`(07-08)、`os-q35_clean_full_newmt_wd01.yaml`(07-09)。
- reg 变体(07-09):`os-q35_clean_nom_{lr1e6,lr2e6,wd03,wd05,neft5,neft10,newcode_wd01}.yaml`;LoRA:`train_lora/auto/lora_q35_clean_nom_r{8,16,32,64}.yaml`。
- c2 轮(07-14):`os-q35_c2_{maint2x,maint4x,swa,twostage_mt}_…yaml`。
- 35B:`os-q36_35bA3b_clean_nom_full_{wd01,wd03,lr2e6}.yaml` + `lora_q36_35bA3b_clean_nom_*`。
- 启动脚本 `LF-innov/cc-sft-innov.sh`(默认 gpu:2;07-08 两主 run 用了 gpu:4);manifest `LF-innov/logs/sft_clean_matrix_manifest.txt`(job 10830101/10830102)。

### 1.3 训练产物(`models_sft/`,均含 trainer_state.json / trainer_log.jsonl / training_loss.png / all_results.json / README.md / checkpoint-N)

14 个 q35 clean SFT 目录 + 12 个 LoRA merge(r8/16/32/64 × s01/s05/s10)+ 24 个 soup 目录 + 6 个 q36-35B 目录 + 6 个 c2 目录(maint2x/4x 07-14 尚在训/队列,swa/twostage 已完)。注意 soup 目录里的 trainer_state 是从对应 SFT 拷来的(不是独立训练)。

### 1.4 训练日志(`FrontierSmith/logs/`)

`sft_clean_full_wd01-10830101.{out,err}`、`sft_clean_nom_wd01-10830102.*`、`cc-sft-nom_*-1086*.*`(lr/wd/neft/lora 8 个)、`cc-sft-clean_full_newmt-10869719.*`、`cc-sft-clean_nom_newcode-10869737.*`、`cc-mlora-*`(merge)。日志含完整 tokenized example 预览(input+labels)、Num examples、进度条、保存记录。

### 1.5 评测生成(`FrontierSmith/outputs/`,共 103 个 clean/clnom 目录)

- 直测(本报告重点):`cc_eval_clean_nom_wd01_sft_*` 与复评 `cc_eval_clean_clean_nomaintain_wd01_sft_*`;`cc_eval_clean_full_wd01_sft_*` 与复评 `cc_eval_clean_clean_full_wd01_sft_*`;base=`cc_eval_clean_start_*`。每个 `shard_0/samples.jsonl` 910 行(FCS 860 + ALE 50)。
- soup α 网格、LoRA s01/s05、newmt/newcode/wd/lr/neft a10/a50、research/MLS 目录见 `CLEAN_DECONTAM_REG_zh.md` §复现。newmt/wd03 等**直测** eval 07-14 刚提交(job 11169292-99),截稿未落分;q36 直测 `cc_eval_q36full_wd0{1,3}_sft_*` 只有 shard,无 summary(未完成)。
- 07-14 的 LoRA s10 merge 已在盘(`lora_q35_clean_nom_r*_s10_merged`),eval 未落。

### 1.6 版本控制事实(git,`LF-innov/`)

- 分支 `feat/per-turn-loss-mask`,仅两个 commit:063cfe9(06-22 clone 基线)→ 494ff82("fix(data): treat loss:null as train + stop injecting empty think into loss-masked turns",作者日期 07-05)。
- **reflog 实证:本 checkout 直到 07-09 09:32 才 fast-forward 到 494ff82**。据 sacct 起跑时间,各 run 实际执行代码为:
  - **旧代码**(<09:32):full_wd01(07-08 15:34)、nomaintain_wd01(07-08 17:15)、nom_lr2e6(07-09 09:14)、nom_lr1e6(09:16);
  - **新代码**(>09:32):nom_lora32/8(09:45)、lora16/64(10:02/10:04)、**nom_wd03(10:28)、nom_wd05(10:35)、neft5(10:39)、neft10(10:44)**、newmt(11:27)、newcode(11:32)。
- `data/dataset_info.json` 的全部 innovation/clean 条目至今是**未提交的工作区改动**(514 行 insertions)。

---

## ② 数据画像

### 2.1 量化总表(tokenizer=训练同款,逐条实测)

| | traj(创新主数据) | wave2_clean(maintain 臂) | maintain_r3(对照,未用) |
|---|---|---|---|
| 条数 | 2225 | 1352 | 903 |
| 例子 token p10/p50/p90/max | 5.8k / 11.7k / 19.3k / 49.9k | 3.0k / 15.3k / 43.7k / 60.1k | 0.13k / 3.8k / 78.1k / 158.2k |
| **超 cutoff 53760 条数** | **0** | **41(3.0%)**(cp 层 30) | 276(30.6%) |
| 入 loss 的 gpt 轮 / token | 2737 轮 / **20.08M** | 1352 轮 / **24.94M** | (无 loss 标注)9.36M |
| loss 轮 think token p50/p90 | 3,466 / 8,664 | **14,415 / 41,655** | 0 / 46(基本无 think) |
| think 自查标记密度(次/1k tok)p50 | **0.00**(researcher 层);0.27(cp 层) | **7.51(cp)/ 2.68(math)/ 2.56(empty)** | 3.49 |
| (对照)base 在 FCS eval 的自然 think | — p50 19,543 tok,密度 **4.93** — | | |
| 最终答案语言 | **python 1769(79.5%)/ cpp 416(18.7%)**/ 其他 40 | prose(math/puzzle/IF)1023 / **cpp 301** / py 9 / 其他 19 | prose 671 / 代码块 232 |
| 缺 `</think>` 的末轮 | 0 | 0 | (think 大多缺省) |

标记词表:Wait/Hmm/verify/Actually/But wait/double-check/sanity check/Let me test/recheck 等。**核心反差(事实)**:创新数据的 think 密度 ≈ 0——是**写好的光滑叙述**;wave2-cp(拒绝采样 rollout)7.5/1k、base 自然思考 4.9/1k——**创新数据在教一种和模型原生推理完全不同的"无自查、短 5 倍"的思考风格**,而全数据里唯一"长思考+高自查+C++"的成分是 wave2 的 303 条 cp(只在 maintain 臂里)。

**loss 口径**:full 臂 45.0M loss tokens 中 wave2 占 **55%**(按条数只占 38%——`SFT_DIRECT_RECOVERY` 引 38%,按 token 应为 55%);nomaintain 臂 20.1M 全部为 traj。

### 2.2 模板化/重复(全量扫描)

- **traj think 开头三大模板家族覆盖 35.9% 的 loss:true 轮**:`**Reading the problem and pinning the contract.**`×288、`OK, let me think this through from scratch`×161、`Let me start from …` 十余变体 ×240+(distinct 开头 1339)。
- **wave2 think 开头 "Here's a thinking process"×483 轮(35.7%)**,另有 `The user wants me to solve a logic puzzle`×103 等;prompt 侧是少数几个谜题模板的大量实例化(属性网格 232、ARC 式 90、谓词链 128、猛犸象排序 41…),但 1352 条 human prompt **无一重复**。
- traj:1713 unique prompts / 2225 行;678 条多轮行是 166 个题的**嵌套前缀切片**(同一对话切成 4/6/8…轮各存一条,最多同题 12 条)。由此 **14.5% 的 loss 字符被训练两遍**(同一轮内容在两个切片里都 loss:true,max 重复=2)。
- traj ∩ wave2 = 0;wave2 内部无重复。

### 2.3 结构与 loss 落点(mask_history:false + per-turn loss,已对日志 label 预览核验)

- 2-turn(traj 1547 + wave2 1352):**prompt 全 -100,`<think>` 起全部进 loss**(日志 `label_ids` 预览实证:labels 恰从 `<think>` 开始)。think+答案都训。
- traj 多轮两种互斥格式(全量扫描):**512 条 human-反馈 ladder**——只最后一个 gpt 轮 loss:true;前面 gpt 轮全 loss:false 且 **0 条含 think**(think 被剥除)。**166 条 observation-反馈轨迹**——**所有** gpt 轮 loss:true 且全含 think(即史轮的 think 留在上下文里训练;推断:推理时 Qwen 模板会剥史轮 think,此处是训推不一致点,但属设计选择)。
- **旧代码 run 的实际渲染差异(事实,来自 commit 494ff82 diff + reflog 时间)**:07-08 的 full_wd01/nomaintain_wd01(以及 lr1e6/lr2e6)在 512 条 ladder 的 1207 个 folded 轮上被注入 `<think>\n\n</think>` 空块——模型被条件在推理时不存在的"空 think 历史"上;07-09 09:32 后的 run(wd03/wd05/neft/lora/newmt/newcode)无此问题。
- **loss:null bug(fix 1)在 clean 轮两个文件上都没有触发**(实测:traj 2737+1207、wave2 1352 个 gpt 轮全部带显式 loss 标注,absent=0)——见 §7.3 与既有文档的出入。
- maintain_r3(对照):**无任何 loss 字段**;350 条 OpenHands 轨迹**没有 gpt 轮**(助手侧全为 function_call),ET True/False 混排(653/250)。若与带标注文件混训,恰是 fix 1 描述的触发场景——但它不在 clean 轮。

### 2.4 traj 精读记录(16 条完整精读;详录见下,引文均原文)

样本选择:最长(183k 字符)/最短(4.1k)/中位(35.4k)+ 4 条 cp + 4 条多轮(含 24 轮最长)+ 随机各年份。**要点**:

1. **数据本质 = "重演式蒸馏"**:把已知论文/纪录/竞赛解重写成第一人称现场推导。反复出现"周期人格当场发明已有方法"腔:*"The method is BADGE"*、*"I propose the R-learner"*、*"The method I propose is **Weyl's equidistribution theorem** … the technique I call **Weyl differencing**"*(1916 人格用 Weyl 名字"自己提出")、*"I'll name the new one … Muon"*。
2. **think 质量两极**:好的一端有真实数值验证与回溯(R-learner 样本:*"Now wait — look again at what I derived, *before* I divided"*;BEiT 样本用 ε 序列数值核 ELBO 极限;**石子合并样本的 int 溢出叙述 `536870911 = INT_MAX/4` 经本次实际编译运行逐位吻合**)。差的一端是零验证的光滑成稿(最短样本、中文镜像对称样本——见 §6 异常 1)。
3. **cp 样本(346 条 v4)共用固定"两幕 bug 剧"**:先写带缺陷版 → 最小反例 trace → 精确归因修复 → 第二个 bug(off-by-one/溢出/播种)→ *"Edge cases, deliberately, because this is where counting code dies."* → *"I convinced myself the *idea* is right by … and the *code* is right by …"* → causal recap。S08/S09/S10/S11 逐段同构、措辞复用。
4. **答案形态**:99.7% 是"解释散文 + 代码"(没有 code-only);多数样本 `</think>` 后散文与 think 大段语义重复,cp 样本的完整代码在 think 和答案里**逐字出现两遍**。
5. **多轮 ladder**:human 反馈轮只有指标表(`Measured results — baseline:xxx` + markdown 表),无"improve"指令;**最后一轮(唯一 loss:true)的"改进"从未被回喂验证**——训练目标是"根据前史数字提出下一个方法",不是"提出被证实更好的方法"。
6. **溯源残留**:`// ale-49: Reconfiguration Routing`、`getenv("ALE_BASELINE")`(评测 harness 开关直接留在"提交解"里)、`# U-net design by @brendanh0gan`、`"By @fernbear.bsky.social"`、`modded-nanogpt PR #199`、*"found by an agentic coding-agent search (AutoEvolver, via Claude/Opus 'aspiration prompting')"*(gpt 轮内出现 Claude/Opus 模型自指)。
7. **年份人格大面积穿帮(多轮尤甚)**:system 2011 引 2020 论文(GraN-DAG)、2016 讲 Lion/Muon、2019 讲 Gemma-2;2-turn research 样本则大多对齐。
8. **编译级正确性印象**:纯算法 cp 样本大多可编译且正确;但存在:O(n²logn) 交付给 n≤2e5/TL2s 的必 TLE 解(自辩 *"fine for the intended scale"*)、BEiT 样本宣称 self-contained 却依赖方法体为 `...` 的 scaffold stub、优化器样本缺 `load_state_dict`、speedrun 样本 FP8 backward 里 `grad_w` 未定义(故意片段化)。

### 2.5 wave2_clean 精读记录(15 条)

1. **内容全部是"有唯一可验证答案的常规解题"**——名为 innovation_wave2 实为 maintain:谜题(属性网格/ARC/谓词链)、IFEval 约束、竞赛数学(\boxed)、Codeforces/AtCoder C++ 全题面、少量多语种通用 chat。与 `DATA_WAVE2_FCS_CPP_zh.md` 的造数记录(27B 拒绝采样 + DeepSeek 兜底 + Guru/ifeval)一致。
2. **think 是全语料里最接近 base 原生风格的部分**:cp 样本 "Wait" 上百次(一条 126k 字符样本 193 次),逐样例验算、反例回溯;math 样本有 *"*Self-Correction/Verification check using algebraic simplification first:*"*。
3. **蒸馏口癖**:483 轮以 *"Here's a thinking process"* 开场(traj 为 0、base 生成为 0)——单一来源改写腔的指纹。
4. **正确性**:抽验的 math 结论(59/64、5/9、f(n)=n、1/2)均正确;cp 未逐题判定但无语法级缺陷;例外见 §6(挥手式论证 1 条、错位 2 条、退化 3 条)。

### 2.6 maintain_r3 对照读(未用于 clean 轮)

500 条 2-turn(其中 250 条 <500 字符微模板题 + 250 条长知识论述)+ 350 条 OpenHands SWE 轨迹(38–384 轮,真实 exit-code observation,以 `finish` 成功收尾)+ 35 条 pi 轨迹 + 18 条 app 构建。微模板家族程序化核验 50 条全对,但**水壶题 3 条数学不可解仍给肯定答案、12 条 plan-order 题答案恒为 "Critical path length: 3 units"(全错)**;OpenHands ET=True 子集的 think 有"探索前已知修法"的先知式泄漏(推断:事后注入 think 的折叠流程残留)。

---

## ③ 训练指标(逐 run)

### 3.1 主表(trainer_state 实测;loss 为 logging_steps=5 的原始点)

| run | 代码版 | GPU×GBS | 步数 | loss 首→末(均值) | grad_norm 首→末 | Num examples(日志) | 备注 |
|---|---|---|---|---|---|---|---|
| **full_wd01** | 旧 | 4×128 | 28 | 0.794→0.693(0.723) | 2.00→0.45 | **3,577**(=2225+1352,零丢弃) | 07-08 |
| **nomaintain_wd01** | 旧 | 4×128 | 18 | 1.062→0.951(0.993) | 3.53→0.74 | **2,225**(零丢弃) | 07-08 |
| **full_newmt_wd01** | 新 | **2×64** | 56 | 0.787→0.688(0.698) | 2.41→0.75 | 3,577 | 与 full_wd01 同数据 |
| nom_lr2e6 | 旧 | 2×64 | 35 | 1.085→0.958(0.994) | 5.05→0.93 | 2,225 | |
| nom_lr1e6 | 旧 | 2×64 | 35 | 1.088→0.991(1.021) | 5.86→1.22 | 2,225 | |
| nom_wd03 | 新 | 2×64 | 35 | 1.075→0.916(0.962) | 2.71→0.77 | 2,225 | |
| nom_wd05 | 新 | 2×64 | 35 | 1.075→0.916(0.962) | 2.65→0.77 | 2,225 | 与 wd01 曲线重合 |
| nom_neft5 / neft10 | 新 | 2×64 | 35 | 1.075→0.916(0.962) | ≈同上 | 2,225 | **与无 NEFT 基线逐点重合** |
| nom_newcode_wd01 | 新 | 2×64 | 35 | 1.075→0.916(0.962) | 2.65→0.81 | 2,225 | 新代码基线 |
| nom_lora8/16/32/64 | 新 | 2×32 | 70 | 1.09-1.11→0.92-0.96 | 0.2-0.5 | 2,225 | 40 步后平台 |
| q36-35B full wd01/wd03 | 新 | — | 18 | 0.980→0.893(0.925) | 1.54 | — | wd01≡wd03 曲线 |
| q36-35B full lr2e6 | 新 | — | 18 | 0.989→0.930(0.950) | 1.57 | — | |
| c2_swa_wd01(07-14) | 新 | 4×128 | 28 | 0.794→0.693 | 1.83 | 3,577 | full_wd01 复刻+7 ckpt |
| c2_twostage_mt(07-14) | 新 | — | 22 | **0.35→0.304** | 1.48 | 1,352(wave2 only) | 从 nomaintain ckpt 续训 |

### 3.2 从曲线里读到的事实

1. **所有曲线光滑单调、无尖峰、无二段下降、无发散**;grad_norm 单调衰减。**塌陷不是优化事故**(没有 loss blow-up 可指)。
2. **零样本丢弃**:日志 `Num examples` 与文件行数逐一相符(3,577/2,225);traj 无一条超 cutoff,wave2 有 41 条被右截断(见 §6)。
3. **wave2 token 对模型异常"便宜"**:full(含 wave2)整体 loss 0.72 ≪ nomaintain(纯 traj)0.99;更硬的证据是 c2_twostage——**nomaintain ckpt 从未见过 wave2,直接在 wave2 上续训,首点 loss 只有 0.35**。traj(重写文体)loss ≈1.0,wave2(拒绝采样 rollout+高模板谜题)≈0.35-0.5。SFT 的梯度预算里,"便宜"的 wave2 占 55% loss token 但提供的更新量远小于 traj——**创新文体是 full 臂里真正大改权重的部分**(此句为推断,损失差为事实)。
4. **NEFTune 无效(事实+机理)**:neft5/neft10/无-NEFT 三条曲线在 1e-4 量级内逐点重合(step5 均 1.0754,neft5 1.0754148 vs 基线 1.0753704);training_args.bin 里 `neftune_noise_alpha=5.0` 确实生效地记录了。机理(推断,可代数验证):NEFT 噪声幅度 = α/√(L·d),L≈12k、d≈4096 时每元素 ≈0.0008——超长序列把噪声稀释到无。CLEAN_DECONTAM §2 里 NEFT 行本质上是 wd01 基线的重复测量。
5. **wd 在 35 步内对 loss 不可见**:wd05 与 wd01(newcode)曲线逐点相同,wd03 只在第 4 位小数偏离——wd 变体的分数差异只能来自权重收缩本身,不来自拟合程度(事实)。
6. LoRA 曲线 40 步后进入平台(0.92–0.96),末端无过拟合迹象;1 epoch 内任何 run 都看不到过拟合信号(没有 eval loss,只能从 train loss 形状判断,弱证据)。

---

## ④ 配置核对(全字段 diff)

以 `os-q35_a100_clean_nomaintain_wd01.yaml` 为基准,**所有 clean yaml 的差异全部列举**(输出目录除外):

| 配置 | 与基准的全部差异 |
|---|---|
| clean_full_wd01 | `dataset` + `innovation_wave2_clean` |
| clean_full_newmt_wd01 | 同 full(**yaml 无差异**;差异在代码版与 GPU 数,见 §1.6/§3.1) |
| nom_lr1e6 / lr2e6 | 仅 lr 1e-6 / 2e-6 |
| nom_wd03 / wd05 | 仅 weight_decay 0.3 / 0.5 |
| nom_neft5 / neft10 | 仅 + neftune_noise_alpha 5/10 |
| nom_newcode_wd01 | **无差异**(纯代码版 A/B) |
| lora_r{8..64} | finetuning_type lora、ds_z3→z2、GA 32→16、lr 5e-6→1e-4、warmup 0.1→0.05、wd→0、rank/alpha=r/2r、dropout 0 |
| c2_maint2x/4x | dataset 加 wave2 别名 ×1/×3 |
| c2_swa | + save_steps 4(存 7 个 ckpt) |
| c2_twostage | model=nomaintain ckpt,dataset=wave2 only,lr 2e-6 |

**共同项(全部 run 一致,逐项核过)**:template **qwen3**、mask_history **false**、cutoff_len **53760**、无 packing、无 train_on_prompt、val_size 0、cosine+warmup0.1、bf16+fa2+liger、ds_z3(LoRA 为 z2)、save_only_model、seed 42。tokenizer:eos=`<|im_end|>`、pad=`<|endoftext|>`,vocab 248044,训练与模型目录同源;visual 塔冻结(可训 8.95B/9.41B)。日志仅一条模板警告:"You are using reasoning template, please add `_nothink` suffix if the model is not a reasoning model."(qwen3 reasoning 模板,预期内)。

**"不是故意设的"差异(核对结论)**:yaml 层面干净,没有暗改;**真正的隐藏变量有三个,全部不在 yaml 里**——
(i) **代码版本**(§1.6:lr 变体=旧代码,wd/neft/lora/newmt/newcode=新代码;CLEAN_DECONTAM §2 的 reg 对比表里 lr 行和 wd 行之间因此叠加了一个未声明的代码差);
(ii) **GPU 数/有效 batch**(07-08 两主 run 4 卡 GBS128,其余全部 2 卡 GBS64→步数、warmup 步长、每步样本混合都不同;newmt vs full_wd01 的 A/B 同时带上了 GBS 128→64);
(iii) **NEFTune 实际无效**(§3.2-4)。

---

## ⑤ 生成 ↔ 数据 对应实例(逐字引文 + 双侧 grep 计数)

生成侧:`shard_0/samples.jsonl`,`text` 不含 `<think>` 开标签(模板预填),`</think>` 出现=思考完整。频数分母 860(FCS 行)。

### 5.0 直测行为总表(与 AVERAGE_INNOVATION §3 相互印证,数字独立复算)

| 模型 | FCS mean@5 | %得分>0 | completion p50 | %撞 32k | %think 闭合 | think p50 字符 |
|---|---|---|---|---|---|---|
| base | 7.05 | 15.6% | 32768 | 64.5% | 42.4% | 66,406 |
| **nomaintain 直测** | **0.25 / 0.31**(两次复评) | 1.4% | **2,650** | 3.1% | 86.3% | **4,437** |
| **full 直测** | **2.40 / 2.42** | 6.2% | 7,306 | 37.3% | 52.7% | 6,304 |
| soup nom_a10 | 6.41 | 15.2% | 32768 | 63.0% | 42.4% | 66,023 |
| LoRA r32 s0.1 | 5.98 | 13.6% | 32768 | 61.3% | 43.8% | 56,871 |
| LoRA r32 s0.5 | 2.36 | 7.8% | 4,022 | 7.7% | 91.4% | 10,714 |

base 的 %cpp fence 恰等于 %think 闭合(42.4)——**base 只要想完就必出代码,零分主要死于 64.5% 撞 cap**;SFT 的死法相反(想得短、交得整齐、代码编译不过)。soup/LoRA-s0.1 把 base 的长 think 人格几乎原样恢复(开头 `The user wants a C++` 310/290 行 vs base 314、NOM 1)。

### 5.1 实例一:wave2 蒸馏口癖 "Here's a thinking process" → full 直测生成(**最干净的单来源对应**)

- 数据侧:wave2_clean 483 个 gpt 轮以 `<think>\nHere's a thinking process` 开场(traj 0 次)。
- 生成侧:**full 直测 22/23 行**(两次复评)以 *"Here's a thinking process:\n\n1. **Understand the…"* 开场;**nomaintain(没训 wave2)只有 2 行;base 0 行;soup/LoRA-s01 0 行;LoRA-s05 13 行**。
- 判定:只在含 wave2 的臂高发、开头逐字同款——直接的数据→行为迁移(事实)。

### 5.2 实例二:wave2 行 441 的支架 token 泄漏 → full 生成泄漏 `<model_answer>`

- 数据侧(wave2_clean 行 441,math 层,答案结尾原文):`Therefore, the measure of angle $\angle ABC$ is $60^\circ$.\n\n<model_answer>\n\boxed{60^\circ}`——评测支架标签直接留在训练答案里。
- 生成侧:full 直测 p101_s3 的 think 内出现 `<|user_input>` / `<model_answer>` 支架 token(base/nom/soup 均 0)。
- 判定:1 条训练样本即可解释该行为方向(support 弱但串完全特异);另 `<|vq_43510|>` 类退化 token 无数据侧来源,属权重扰动本身(推断)。

### 5.3 实例三:v4 竞赛 cp 风格 → "bits/stdc++ 短 think 快交付"

- 数据侧:traj 的 407 个轮含 `#include <bits/stdc++.h>`(集中在 346 条 v4 cp,think 中位仅 ~4.4k token、自查密度 0.27/1k,答案"说明+单块 C++");wave2 仅 8。
- 生成侧:答案以 ```` ```cpp\n#include <bits/stdc++.h> ```` 开头:**NOM 244/860(28%)**、FULL 166、LoRA-s05 350,vs **base 23**。同时 NOM 的 completion 中位 2,650 token、think 4.4k 字符——**与 v4 cp 数据的"短想快交"的形态一致,而 base 从不这样**。
- 判定:风格(短 think + bits/stdc++ + 说明支架)可溯源;但**具体的坏代码内容不来自数据**——数据侧代码大体可编译,生成侧是幻觉拼贴(`MAX_N` 未定义、`qpow` 未声明、main 内嵌套函数)。即"学到形,丢了实"(前半事实,后半推断)。

### 5.4 实例四:数据里没有的退化行为(负对照,同样重要)

生成侧 NOM 特有的三种高频坏形态,在两个训练文件里 **0 命中**:
- 裸 `#include`(无 code fence)开头的答案:NOM 98–117 行(其中仅 4 行得分)vs 数据侧 0——**畸形 fence 是 SFT 权重扰动的涌现退化,不是模仿**;
- think 频道倾倒代码(text 以 `#include`/```` ``` ````开头):NOM 48 行 vs 数据 0;
- `fread(q + i, sizeof(int), 1, stdin)` 式二进制读 stdin:NOM 6-8 行、FULL 2-9 行 vs 数据 0(推断:来自预训练分布的回流)。
- 另:FULL 2 行把 checker 合同写进解题代码(`resultFile << "Ratio: ";`)——wave2 的 23 处 "Ratio:" 全是数学用语、traj 0 处,**该合同不在这两份训练文件里**(FrontierCS checker 契约知识,来源为题面或预训练,未定)。

### 5.5 实例五:自查行为的消失可从数据密度直接预言

- 数据侧:traj think 自查密度 0.13/1k(researcher 层中位 0);wave2-cp 7.45/1k;base 自然 4.84/1k。
- 生成侧(AVERAGE §2.2 独立测量):"Wait"/10k 字符 base 16.13 → SFT 直测 **0.71**;本次复核 NOM think 里假验证盛行(*"Wager 1: others = "000", answer = 1 → choose 1 (mistakes = 0) ✓"*——✓ 全是编的;`if (maxMistakeRate > maxMistakeRate)` 自比较)。
- 判定:SFT 后自查签名密度掉到与 traj 数据同数量级;而"假装验证"的表演形态恰似 traj 的"口述式自验证"叙事(§2.4-3),只是没有真实运行支撑(前半事实,后半推断)。

---

## ⑥ 异常清单(无论是否与塌陷相关)

**数据侧**
1. **wave2 行 1198/1276:整段退化循环样本,且 token 数恰在 57.3k > cutoff 53760**——行 1198 最终答案 371,335 字符、20,920 词却只有 **14 个唯一词**("internacionalmente contemporáneamente…" 无限循环);行 1276 同族(16 个唯一词大写循环)。二者 loss:true,训练时被右截断——**模型在 ~50k token 的退化循环上训练且无 EOS 收尾**。行 1256(45.6k tok,未截断)半退化,think 内含自白 *"This is insane. I'll stop generating random long words."*。三条源头都是"每词 ≥15 字符"的不可满足 IFEval 约束。
2. **wave2 41 条(3.0%)超 cutoff**(cp 层 30 条):右截断切掉最终答案与 EOS——3% 的 maintain 臂样本教"想到一半戛然而止"。
3. wave2 行 1351/1352:cp system("single self-contained C++17 … stdin/stdout")与任务(Python/Triton kernel)**错位**;1352 的 think 残留容器口供 *"CUDA is unavailable in this container"*。
4. wave2 行 866:think 尾部 ~30 行 "Done. / Proceeds. / Output matches request." 循环填充。
5. traj 行 706(最短样本):**实质错误**——"Picard-Fuchs 验证"比较两个代数恒等式(残差恒 0 的验证表演);断言 q<z 与其自身代码实测输出相反(q=1.0078e-5 > z=1e-5,本次实跑确认)。
6. traj 行 1306:n≤2e5、TL2s 的题交付 O(n²logn) 必 TLE 解,文本自辩 *"fine for the intended scale"*。
7. traj 行 1250:`// ale-49:…`、`getenv("ALE_BASELINE")` 评测工件留在"提交解"内;prompt Background 直接点名答案方法(考点泄漏式出题)。
8. traj 年份人格穿帮成批(2011↔2020 论文、2016↔Lion/Muon、2019↔Gemma-2);行 1864 首轮 prompt 的 record 值(1.5098)与后续轮(1.5028628969)不一致;行 1864 gpt 轮内出现 "Claude/Opus" 自指。
9. traj 行 1608:代码注释自相矛盾("highest marginal variance" 注释配 `argmin` 实现);方法与回喂标签 `baseline:notears_mlp` 名实不符(文本自认)。
10. maintain_r3(未用,备案):水壶题 3 条数学不可解却给肯定答案(行 274 think 列出循环 "6, 2, 8, 4, 0" 后断言 "The target 3 appears in this cycle.");12 条 plan-order 题答案恒 "Critical path length: 3 units"(全错);OpenHands think 先知式泄漏;行 501 轨迹含 9 次空参数工具调用失败循环。

**训练/工程侧**
11. **代码版本混杂**(§1.6):同一张 reg 对比表里 lr 变体=旧代码、wd/neft 变体=新代码;newmt vs full 的 A/B 同时叠加 GBS 128→64。
12. **NEFTune 无效**(§3.2-4):neft5/neft10 与基线曲线逐点重合——两个" NEFT run"实为基线重复。
13. dataset_info.json 全部自定义条目长期处于未提交状态(514 行工作区改动,易被 checkout/reset 抹掉)。
14. wave2 按条数 38%、按 loss token **55%**——文档口径引用的 38% 低估了 maintain 臂的实际梯度占比。

**评测侧**
15. 直测 eval 存在新旧两套目录(`cc_eval_clean_*` 与 `cc_eval_clean_clean_*`),FCS 结论两次复评一致(0.249/0.311、2.396/2.418);但旧 full 目录的 ALE `overall_absolute_score=0.0` 与逐样本均值 286.5 并存——**"ALE 全零"的真相是 50/50 行全部落在判题器每题的常数兜底分**(546,−80,…),即模型程序零贡献,不是 harness 全挂(每 run 仅 3–11 行真 infra error,全部 `status=timeout` 计 0)。
16. q36-35B 直测 eval 只有 shard 无 summary(未完成);07-14 提交的 newmt/wd03/lora-s10 直测批截稿未落分(见 `SFT_DIRECT_RECOVERY_zh.md` §1)。

---

## ⑦ 事实综合(先事实,后推断)

### 7.1 看到的核心事实链

1. **"创新数据"(traj 2225)与 FCS 的要求在三个维度上系统性反向**:落点语言(79.5% Python vs FCS 只判 C++)、思考长度(loss 轮 think p50 3.5k token vs base 在 FCS 上自然 19.5k)、思考风格(自查密度 ~0 的**写好的光滑叙述** vs base 4.9/1k 的真回溯)。它教的是"短想、不自查、说明书式交付"。
2. **maintain 臂(wave2 1352)是全配方里唯一供给"长思考+高自查+竞赛 C++"的成分**(303 条 cp,think p50 36.7k token、自查 7.5/1k),且按 loss token 占 full 臂 55%。full 直测 FCS 2.4 vs nomaintain 0.25 的 8 倍差,与这两组画像逐项吻合;full 直测里保留的 37% 撞-cap 长思考行为、22 行 "Here's a thinking process" 开场,都能在 wave2 里找到逐字来源。
3. **生成端塌陷形态 = "学到形,丢了实" + 涌现退化**:可溯源的是形(短 think、bits/stdc++、说明支架、wave2 口癖、支架 token);**不可溯源的是质**——裸 include、think 倾倒代码、未定义标识符幻觉拼贴、假验证 ✓,两份训练文件 0 命中。数据侧代码大体可编译,生成侧 83% 真编译错(AVERAGE §3)。
4. **训练动力学无事故**:曲线光滑、无丢样、无截断(traj 侧)、无过拟合尖峰。塌陷不是"训炸了",是"训成了"——它忠实地朝数据分布移动了。
5. **工程混淆项三件**(代码版、GPU/GBS、NEFT 无效)叠在 reg sweep 的解释上,但不改变主对比(full/nom 同为旧代码 4 卡;两次独立评测复现)。

### 7.2 机制解读(标注:推断)

- 塌陷的最大单因素是**思考行为的替换**:2225 条 0-自查、5 倍短的重演式 think,在 5e-6 全参 1 epoch 下足以覆写 base 在竞赛题上的"长推理+自验证"策略;而精确 C++ 的成功率恰恰依赖后者。语言落点(79.5% Python)加重了 C++ 生成的退化,但 nomaintain 直测里 67% 的答案仍带 cpp fence——**模型知道该写 C++,是写不对**,所以"风格塌"重于"语言塌"。
- maintain/wave2 的作用机制更像**行为锚**而非"能力数据":它的 token 对模型极便宜(twostage 首点 loss 0.35),提供的梯度小,但让"长想+自查+单块 C++"的模式在混合分布里仍有 55% 的 loss 权重,于是 full 保住了部分 base 行为(2.4 vs 0.25)。它未能完全兜住,可能因为其中真正对口 FCS 的只有 303 条 cp(8.5% 条数),且 30 条被截断、3 条是退化循环(反向教学)。
- 传统正则(wd/lr/NEFT)不针对以上任何一条——与 CLEAN_DECONTAM §2 里它们只在 soup 后小幅改善直测无解一致;LoRA-s0.1/soup-a0.1 有效是因为在权重空间上把"行为替换"的幅度直接砍小(AVERAGE §2 的 α 曲线)。

### 7.3 与既有文档矛盾/需修订处(如实列)

1. **CLEAN_DECONTAM §4 的 newmt 机制表述**:"`loss:null` 之前被当成 False → 静默把 maintain 例子整条零 loss 训练(等于没训)"——**在 clean 轮不成立**:两份文件所有 gpt 轮都带显式 loss 标注(实测 absent=0),fix 1 从未在 clean 轮触发。full_wd01 与 newmt 的真实差异 =(a)fix 2(folded 轮不再注入空 `<think>`,只影响 traj 的 512 条 ladder)+(b)GBS 128→64。newmt_a50 4.2 vs full_a50 2.2 的改善应重新归因到 (a)/(b),不能归因于"maintain 从没训过→训上了"。
2. **CLEAN_DECONTAM §2 reg 表内隐藏代码差**:lr1e6/lr2e6 行(旧代码)与 wd03/wd05/neft 行(新代码)不同代码版;幸有 newcode_wd01(6.16 vs 旧 nom a10 6.41)标定该差 ≈0.25 量级,未推翻排序,但表格口径应加注。
3. **"maintain 38% 占比"**(SFT_DIRECT_RECOVERY §1):按条数对,按 loss token 是 **55%**。
4. **"traj 不用 agentic"**(CLEAN_DECONTAM 開頭):traj 内实有 166 条 observation-反馈轨迹全轮参训(含史轮 think 留在上下文)——"不用 agentic"仅指剔除 OpenHands 式数据,表述宜精确化。
5. **NEFT 行**(CLEAN_DECONTAM §2):neft5/neft10 = 基线重复测量,其分数差纯属评测噪声,不宜作为"NEFTune 更差/更好"的证据。
6. c2 的 maint2x/4x 剂量实验(SFT_DIRECT_RECOVERY §2)按 loss-token 口径实为 55%→71%→83%(非 55%/71% 的条数口径);且加倍的同时会把行 1198/1276 的退化循环也训 2/4 遍——建议先剔除这 3 条(+41 条超长截断)再加倍。

### 7.4 一句话

**数据本身就是答案**:创新 traj 教了"短、不自查、Python 优先、说明书式"的推理人格,maintain(wave2)是唯一的反向锚但只有 8.5% 对口 C++ 且带着 3 条退化循环和 3% 截断;全参 SFT 无事故地学会了这一切——FCS 上塌的正是被替换掉的"长想+真自查"行为,而 soup/LoRA 之所以能救,是把这次行为替换在权重上冲淡,而不是修复了数据。

---

### 附:本报告全部证据路径

数据 `LF-innov/data/{innovation_clean_decontam_traj,innovation_wave2_clean,maintain_r3,…}.jsonl`;配置 `LF-innov/examples/train_{full,lora}/auto/`;产物 `models_sft/sft_q35_clean_*`(trainer_state/trainer_log/training_loss.png);日志 `FrontierSmith/logs/{sft_clean_*,cc-sft-*,cc-mlora-*}`、manifest `LF-innov/logs/sft_clean_matrix_manifest.txt`;生成 `FrontierSmith/outputs/cc_eval_clean*_thinking_32k_both_vllm/shard_0/samples.jsonl`;git 证据 `LF-innov` reflog + commit 494ff82;精读样本提取副本(临时)`/tmp/claude-372967/…/scratchpad/`。
