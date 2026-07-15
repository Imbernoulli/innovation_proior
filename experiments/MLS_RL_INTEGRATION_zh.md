# MLS-Bench 智能体训练 × verl GRPO 集成报告

日期:2026-07-14。目标:把 MLS-Bench(多步工具使用 + Apptainer 任务环境 + 确定性打分)接入我们的 verl GRPO 训练栈,并在 20 个 dev CPU 任务上完成端到端冒烟。

---

## 1. 架构选型:方案 A(原生 AgentLoop,token 级)✅ vs 方案 B(外挂 harness 打 rollout server)❌

**选了 A:自定义 `MLSBenchAgentLoop`(verl AgentLoop 子类)+ 每 episode 一个 MLS 环境子进程。**

关键事实(全部读代码验证过,不是猜的):

1. **我们的 verl fork 已经全员走 AgentLoop**。`ray_trainer.py` 中 `async_rollout_mode = True`、"sync mode is deprecated"——现有单轮 FrontierCS/synth 训练其实就是 `single_turn_agent`。因此加一个新 agent loop 是**纯增量**:注册走 `actor_rollout_ref.rollout.agent.agent_loop_config_path`(hydra 配置文件),按样本用 parquet 的 `agent_name` 列路由。不传这个 override、不用这种 parquet 的旧运行**一个字节都不变**。
2. **方案 B 的 token 轨迹无法保真还原**。B 要把未改动的 `mlsbench agent`(OpenAI chat.completions 客户端)打到 verl 的 rollout HTTP 端点,再事后从消息列表重新分词得到 trajectory。两处致命:
   - Qwen3.5 的 chat template 会**丢弃历史轮的 `<think>` 内容**(`loop.index0 > ns.last_query_index` 才保留),MLS 客户端也不回传 reasoning;重新分词得到的序列 ≠ 实际采样的 token 序列,多轮 credit assignment 直接失真(采样出的思考 token 全部丢失)。
   - 每次 API 调用重发全历史,序列不是"单条连续增长"的,无法拼成 verl 需要的 `prompt_ids + response_ids(+response_mask)` 连续轨迹。
3. **Qwen3.5 工具调用格式是 XML 风格**(`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>...</function>\n</tool_call>`,即 vLLM 的 `qwen3_coder` 格式),不是 hermes JSON;verl 自带的 hermes ToolParser 解析不了。且该模板对"只有 tool 消息的列表"直接 raise(`No user query found`),所以 verl 库存 `ToolAgentLoop` 的增量分词路径也会崩——必须自己按模板字面量手工渲染 delta。这两点 A 方案里都以最小代码解决(自写 XML parser + 手工 delta),B 方案则要依赖 vLLM server 端解析器 + 消息重放,更脆。
4. **掩码正确性**:A 方案 token 级生成(`AsyncLLMServerManager.generate` token-in-token-out),`response_mask` 构造与库存 `ToolAgentLoop` 同构:采样出来的 assistant token 记 1,tool 响应/nudge 的 delta token 记 0。**mask=1 的每一个 token 都是模型真实采样的**,这就是正确的多轮 credit assignment。

**环境侧完全复用 MLS-Bench-dev 原代码,但放到每 episode 一个子进程里**(详见 §3 掩码/隔离)。这不是 A/B 折中,而是 A 的实现细节:环境(workspace、edit/test/submit/undo、Apptainer 执行、leaderboard、评分)在 worker 子进程内原样跑 `InteractiveAgent` 的机器;token 生成与轨迹在训练进程内。发现的硬约束:MLS 各任务的 `mid_edit.py`/`parser.py` 会 `sys.path.insert + import dgp`(每任务一个同名模块),**同一进程跑多个任务会互相污染 `sys.modules`**(实测:causal-* 任务的 dgp 被缓存后,ml-* 任务全崩)。官方 eval harness 也正是每任务一个子进程。子进程另送两个好处:episode 超时可整组 `killpg`(连带挂死的 apptainer);worker 用**验证过的 eval conda python**(带 pgmpy 等依赖,`causal-discovery-discrete` 的 host 侧数据生成需要它,训练 venv 没有)。

## 2. 改动文件清单(全部新增,零修改现有文件)

| 路径 | 作用 |
| --- | --- |
| `FrontierSmith/verl/verl/experimental/agent_loop/mlsbench_agent_loop.py` | **新文件**(verl 树内,但不被任何现有模块 import;仅当 agent_loop_config_path 指到它才加载)。`MLSBenchAgentLoop`:token 级多轮循环、Qwen3.5 XML 工具解析(schema 感知类型转换)、手工 delta 渲染、episode 超时/fail-soft、reward 写入 `AgentLoopOutput.reward_score`(→ `rm_scores`,自动短路 reward manager,无需自定义 reward fn)、每 episode JSONL 日志。所有旋钮走 `MLS_RL_*` 环境变量。 |
| `FrontierSmith/scripts/mlsbench_rl_episode_worker.py` | **每 episode 环境子进程**(stdio JSON-RPC;协议 fd 用 `dup(1)` 保留,再 `dup2(2,1)` 把 mlsbench 的 print 全部改道 stderr,防止污染协议)。命令:`init`(建 InteractiveAgent + setup_workspace + build_initial_prompt,返回 system/initial prompt + 工具 schema)、`dispatch`(WorkspaceTools.dispatch,内部兜底 try/except)、`finalize`(未 submit 则 `submit(n=-1,_force=True)` + `record_zero_if_no_finals` + `evaluate_task` 打分 + **打分后清掉本 episode 在共享 leaderboard.csv 里的行**,防 O(n²) 膨胀;精确按唯一 model 名匹配,baseline 行不动)、`cleanup`、`quit`。文件底部带 `WorkerClient`(阻塞客户端,loop 与 parquet 脚本共用,stdlib-only)。 |
| `FrontierSmith/config/mlsbench_agent_loop.yaml` | agent loop 注册表:`name: mlsbench_agent → _target_: ...MLSBenchAgentLoop`。 |
| `FrontierSmith/scripts/prepare_mlsbench_rl_parquet.py` | 数据准备:每任务起一个隔离 worker,取真实初始对话(system+user,含 str_replace 版工具说明),写 `data/mlsbench_rl/train[_smoke].parquet`(列:prompt/data_source=mlsbench_rl/agent_name=mlsbench_agent/reward_model/extra_info.task)+ token 统计 report。 |
| `FrontierSmith/slurm/cc_rl_mlsbench.sh` | GRPO 训练 launcher(仿 cc_rl_frontiersmith_synth.sh):设 `MLS_RL_*` env,调 **同一个** `run_verl_grpo_frontiercs_qwen35_9b.sh`,只追加两个 hydra override(`agent_loop_config_path`、`agent.num_workers=2`)。冒烟默认:4 任务 × n=2、2 GPU、3 步、save_freq=2。 |
| `FrontierSmith/scripts/cc_rl_mlsbench_submit.sh` | 提交 helper(CPUS=GPU×8、MEM=GPU×120G 自动缩放)。 |
| `FrontierSmith/scripts/mlsbench_rl_episode_test.py` | 无训练 harness 测试:真 `MLSBenchAgentLoop` + FakeServerManager(vLLM `/v1/completions` 收 token-id 数组),跑完整 episode,打印每轮 token 数、mask 分段、reward,并断言 mask 不变量。 |
| `FrontierSmith/slurm/cc_mlsbench_rl_episode_test.sh` | 上者的 sbatch 包装(1 GPU:起本地 vLLM → 跑 episode 测试)。 |
| `FrontierSmith/data/mlsbench_rl/{train,full,train_smoke,full_smoke}.parquet` | 20 任务全量 + 4 任务冒烟数据。 |

**未改动**:`agent_loop.py`(含 ROLLOUT_PRESENCE_PENALTY patch)、`tool_agent_loop.py`、reward_score/*、所有现有 launcher。`bash -n` 全过;`import verl.experimental.agent_loop` 后确认 `mlsbench_agent_loop` **不在** `sys.modules`(增量性验证)。MLS-Bench-dev 本体零修改。

## 3. 数据 / 奖励 / 掩码设计

**数据**:每行一个任务;`prompt` 存真实初始对话(拿来做长度过滤与展示),rollout 时 loop 在自己的新 workspace 上重建同一 prompt(模板确定性 ⇒ 内容一致,且行号与本 episode 的 workspace 严格对齐)。20 任务 prompt token:min 5991 / max 22029(`optimization-hyperparameter-search`);工具 schema 在 rollout 时再加约 1.2k。冒烟 4 任务(clustering/anomaly/causal-discovery/dim-reduction)max 8222 → `MAX_PROMPT_LENGTH=14336` 足够。

**奖励**:episode 结束(submit 成功 / 步数用尽 / token 预算用尽 / 无工具调用)→ worker `finalize`:未 submit 但有 test 历史则强制提交最后一次(与 eval 的 auto-submit 同口径)→ `evaluate_task(task, model=vllm/rl-<uuid12>)` 得确定性 task score ∈ [0,1](与 `mlsbench score` 完全同一代码路径)。每 episode 唯一 model 名 ⇒ 并发安全(leaderboard 有 fcntl 锁)。**任何环节炸掉 → 0.0 分,episode 决不带崩训练**(worker 崩溃、超时、评分异常均兜底)。分数天然按任务归一([0,1],由任务自身 baseline anchors 校准),所以**不需要** FS_PERTASK_REWARD_NORM。奖励经 `reward_score → rm_scores` 进 batch,`extract_reward` 直取,reward manager 自动短路。

**掩码**(正确性红线):
```
response:      [assistant 采样 token][tool 响应 delta][assistant 采样][nudge delta][assistant 采样]...
response_mask: [1,1,...,1           ][0,0,...,0      ][1,...,1     ][0,...,0    ][1,...,1     ]
```
- assistant 段 = `AsyncLLMServerManager.generate` 实际返回的 token id,原样入轨迹(含 `<think>`、`<tool_call>`、结尾 `<|im_end|>`),mask=1。
- tool/nudge delta = 按模板字面量手工渲染 `"\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"` 后 `encode(add_special_tokens=False)`,mask=0。
- 终止条件:回复预算(delta 放不下也终止)、生成非干净停止(尾 token 非 im_end/eos)、submit 完成、步数预算、nudge 用尽仍无工具调用、episode 墙钟超时(默认 2700s,超时 killpg worker,reward=0)。
- 工具结果超长中段截断(默认 8000 字符,头尾各半)。
- 并发控制:每个 AgentLoopWorker 进程内 `asyncio.Semaphore` 限并发 `test()`(默认 3-4);冒烟 2 个 worker ⇒ 全节点 ≤6-8 个并发 apptainer 测试。

**与 eval 口径的已知差异**(有意为之):RL 轨迹里历史轮的 `<think>` 保留在上下文中(它们是被采样的 token,必须留在轨迹里);eval 时客户端+模板会丢历史 think。这是所有 verl 多轮训练的标准做法,属训练/推理上下文分布的既知偏差。

## 4. 怎么跑

```bash
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

# 1) 数据(已生成;换任务集时再跑)
source .venv-vllm023/bin/activate
python scripts/prepare_mlsbench_rl_parquet.py                    # 20 任务 → data/mlsbench_rl/train.parquet
python scripts/prepare_mlsbench_rl_parquet.py --suffix _smoke \
    --tasks ml-clustering-algorithm ml-anomaly-detection causal-discovery-discrete ml-dimensionality-reduction

# 2) 无训练 harness 测试(1 GPU;起 vLLM + 跑真实 episode + 校验 mask/score)
sbatch --export=ALL,TASKS="ml-clustering-algorithm ml-anomaly-detection",EPISODES=1 \
    slurm/cc_mlsbench_rl_episode_test.sh

# 3) GRPO 冒烟(2 GPU,4 任务 × n=2,3 步)
scripts/cc_rl_mlsbench_submit.sh mlsrl_smoke1 2 3
# 观察:logs/rl-mls-mlsrl_smoke1-<jid>.out(训练)、
#       outputs/mls_rl/mlsrl_smoke1/episode_logs/episodes_*.jsonl(每 episode 一行:task/reward/轮次/token 数/耗时)
#       outputs/mls_rl/mlsrl_smoke1/episode_logs/worker_*.log(每 episode 的 MLS 环境 stderr)
# ckpt: checkpoints/rl_mlsbench/mlsrl_smoke1/global_step_2/
```
常用 env 旋钮(提交时 `--export` 或改 launcher 默认):`MLS_RL_MAX_STEPS`(默认 8)、`MLS_RL_MAX_TESTS`(默认 1)、`MLS_RL_EPISODE_TIMEOUT`(默认 3000s)、`MLS_RL_MAX_CONC_TESTS`、`TRAIN_BATCH_SIZE/ROLLOUT_N/NGPU`、`MAX_PROMPT_LENGTH/MAX_RESPONSE_LENGTH`(20 任务全量需 ≥24k/配套 MAX_MODEL_LEN)。

## 5. 测试结果(真实数字)

### 5.0 静态/协议层(login 节点,已过)
- XML 工具解析单测:edit(str_replace/replace 带 int 行号强转)、test(空参)、submit(n=2)全对;`content` 参数换行精确还原。
- worker 协议端到端:init(system 3170 字符 / initial 18653 字符 / 4 工具)→ 非法 edit 返回 `ERROR: old_str not found...`(step_count 正确 +1)→ 无测试时 submit 正确拒绝 → finalize reward=0.0(`No metric values found`)→ prune 只删本 episode 行、15 条 baseline 行原封不动。
- 增量性:`import verl.experimental.agent_loop` 不加载本模块;所有 launcher `bash -n` 通过。
- 数据:20/20 任务 parquet 构建成功(隔离 worker 后;同进程构建会因 dgp 污染挂掉 11/20——这就是子进程隔离的实证)。

### 5.1 无训练 harness 测试(job 11169846,1×H200,della-i19g1)✅ 通过
真 `MLSBenchAgentLoop` 代码路径(FakeServerManager 仅把 token 数组打到本地 vLLM `/v1/completions`),两个完整 episode:

**Episode 0 — ml-clustering-algorithm:reward = 0.3797(真实非零 MLS 任务分!)**
- elapsed 90.1s(生成 76.2s + 工具 4.1s),12 轮;prompt 7328 tok,response 14336 tok(assistant mask=1:13201;tool mask=0:1135)。
- mask 分段(A=assistant/1,T=tool/0):`A3176 T161 A2571 T161 A2199 T161 A1739 T161 A478 T491 A3038` — 严格交替,首 token mask=1,断言全过。
- 行为:4 次 `str_replace` edit(全部 `OK: Replaced 1 occurrence`,行号/可编辑区跟踪正确)→ `test()`(Apptainer 真跑,3 个 sklearn 脚本并行,拿到真实指标 `ari_blobs=0.9389...`)→ 第 11 轮生成时顶到 response 预算截断(finished=response_budget)→ finalize 强制提交 test#1 → `evaluate_task` 得 0.3797。**即 episode 被截断也能正确拿分**(fail-soft 语义符合设计)。
- 结束后 leaderboard 中本 episode 行已自动 prune,15 条 baseline 行原封不动。

**Episode 1 — ml-anomaly-detection:reward = 0.0(任务侧已知基建问题,非集成 bug)**
- elapsed 37.6s,16 轮,7 步 1 test,模型走完 edit→test→undo→edit→submit 全流程(done=true,finished=submitted);prompt 6813,response 7426(assistant 5887 / tool 1539),mask 分段同样严格交替。
- test 返回 `[BUDGET CHECK FAILED] ... ModuleNotFoundError: dgp`:该任务 `budget_check.py` 在**容器内**加载 `edits/mid_edit.py`,后者 `import dgp`(host-only 的 holdout 模块,容器内路径解析到 `/holdout/...` 不存在)。**官方 eval 同样命中**(上一轮 eval 的 ml-calibration / ml-missing-data-imputation 日志里就有同样的 `[BUDGET CHECK FAILED]`)——属 MLS-Bench-dev 任务基建的既有行为,与 RL 集成无关。全 20 CPU 任务排查:仅 `ml-anomaly-detection`、`ml-missing-data-imputation` 两个任务命中"budget_check 容器内加载 import dgp 的 mid_edit"组合(确定性 0 分);已反馈路径,冒烟任务集换成了无此问题的 4 个任务。

### 5.2 GRPO 冒烟(job 11170133,2×H200,della-i19g1)✅ COMPLETED
配置:4 任务(clustering/dim-reduction/causal-discovery/subgroup-calibration)× n=2 = 8 episodes/步,3 步,`MAX_PROMPT/RESPONSE=14336/14336`,`MLS_RL_MAX_STEPS=8 MAX_TESTS=1 EPISODE_TIMEOUT=1500s`,save_freq=2。全程 45:50 墙钟(维护窗前 backfill 挤进去的)。

- **步耗时**:step1 345.6s → step2 306.5s → step3 ~29.5min(被一个 causal episode 的 1500s 超时收割拖满;见下)。训练循环 3/3 完成,**update_actor 每步正常完成**(grad 更新 + ckpt 均落盘)。
- **奖励(per-step rollout dump,`outputs/rl_mlsbench_rollout/mlsrl_smoke1bf/*.jsonl`)**:
  - step1: `[0,0,0,0,0,0,0,0.123]` mean=0.0154
  - step2: `[0,0,0,0,0,0.3875,0,0]` mean=0.0484
  - step3: 全 0
  - **非退化**:step1/2 各有一个 GRPO 组内 (0, >0) 对 → 优势非零、有真实梯度信号;但整体稀疏(28 个 episode 里 2 个非零)。稀疏来源与 eval 一致:base 模型在这些任务上本来就大多 0 分(上轮 eval 20 任务 mean 0.078,13/20 为 0),再叠加 14336 response 预算截断(9/28 episodes `response_budget` 截断)。这是**模型能力+预算问题,不是集成问题**——分数管道全通(0.123/0.3875 都是真实 MLS task score)。
- **episode 统计(28 = 24 train + 4 final-val)**:19/28 跑了真实 Apptainer test;结束原因 submitted=7 / step_budget=11 / response_budget=9 / episode_timeout=1;**0 个 episode 报错**,1 个超时(causal-discovery 的重型 test 超 1500s,被 killpg 收割,0 分,训练无感继续——fail-soft 实测生效)。episode 墙钟 min 36s / mean 125s / max 1500s;assistant token 平均 10202/episode。
- **ckpt**:`checkpoints/rl_mlsbench/mlsrl_smoke1bf/global_step_2/`(36G)+ `global_step_3/`(36G),`latest_checkpointed_iteration.txt=3`,模型-only 口径与生产一致。
- **卫生检查(事后审计)**:4 个训练任务 + 2 个 eptest 任务的 `leaderboard.csv` 中 `vllm/rl-*` 残留行 = **0**(打分后自动 prune 全部生效;并发下 WAL replay 自愈了一次外部覆盖)。verl 树自昨日起被改文件 = 仅 `mlsbench_agent_loop.py`(纯新增)。

## 6. 社区对照验证(verl 上游 + Slime,2026-07-14)

按用户要求,对我们 Qwen3.5 多轮工具处理的三个关键决策做了外部对照(verl 已迁移到 `verl-project/verl`;Slime = `THUDM/slime`)。逐项结论:

### 6.1 决策一:XML 工具解析器自写 —— 与上游已合并实现同构,且修掉了一个上游仍开着的坑
- 上游 **已合并** `@ToolParser.register("qwen3_coder") class Qwen3XMLToolParser`(`verl/experimental/agent_loop/tool_parser.py`,随 PR **#6779** Continuous Token 进入 main;改编自 Qwen3-Coder 官方 `qwen3coder_tool_parser.py`)。与我们的 `parse_qwen_xml_tool_calls` 对照:正则族相同(`<tool_call>(.*?)</tool_call>` DOTALL + `<function=` + `<parameter=`)、参数值**恰剥一个首尾 `\n`** 的语义相同、按 schema 类型转换(int/float/bool/object)相同。差异:上游额外容忍未闭合的截断调用(`<tool_call>(.*?)$` 兜底)并把无标签全文兜底成函数体;我们更严格(不完整调用不执行 → 走 nudge/终止)——RL 训练场景下严格更安全,保留。
- **真问题(已修)**:verl issue **#4757** + RFC **#6424**(均 open)指出 agent loop 缺失推理引擎在 chat-completions 里 decode→tool-parse 之间的 **reasoning parser** 步骤:`<think>` 里写的假 tool_call 会被当真执行、污染对话史、教坏模型(#6223/#6252 报了下游 benchmark 回归)。上游连 `Qwen3XMLToolParser` 也还没剥 think(修复 PR **#6434** 尚未合并)。我们的 parser 原本同样暴露 → **已修**:新增 `strip_reasoning()`(按 vLLM qwen3 reasoning parser 语义:首个 `</think>` 前全是 reasoning;未闭合=整轮 reasoning=无调用;残余完整 `<think>…</think>` 段一并剥除),解析只对剥后内容进行。验证:5/5 单测过(含"think 里假调用+正文真调用只取真调用"、"老行为会先执行假 submit"的回归断言);对 24 条真实冒烟 rollout 输出重放,新旧提取结果 24/24 一致(冒烟里该坑没触发,但已关死)。**此点我们现在比上游 main 更严格,与上游官方修复方向(#6434:generate→decode→reasoning parser→tool parser)一致。**

### 6.2 决策二:tool 反馈按模板字面渲染增量(含 `<|im_end|>` 后补 `\n`)—— 与上游已合并修复逐字节一致
- verl PR **#6921**(**已合并** 2026-07-06)修的正是这个:`ToolAgentLoop` 增量拼 token 时,生成停在 `<|im_end|>`,模板的后继分隔符 `\n`(Qwen id 198)在每个 assistant→tool 边界被静默丢一个,"rolled-out token sequence diverges from apply_chat_template of the equivalent full conversation, compounding each turn"。修复=在 tool 轮首补回分隔符、mask=0。我们 `_delta_ids` 的前缀 `"\n"` 与之等价(且我们的实现自始就带)。
- Continuous Token(RFC **#6719**,PR **#6779** 已合并)里的 `QwenContinuousTokenBuilder._merge_non_assistant_token_ids`(`verl/utils/tokenizer/continuous_token.py`)同样:"前缀末尾是 `<|im_end|>` 就插入缺失的 `\n` 再拼非 assistant token"。SGLang 侧同类 bug 见 issue **#3720**(eos 后丢换行)。
- 上游对"模板对 tool-only 列表 raise"的解法是**合成前缀 diff**(`render_delta_token_id([合成 system,user,assistant], tool_messages)` 渲染后减去前缀),我们是 Qwen3.5 模板字面量直写——输出等价,上游做法泛化到多模型家族,我们的做法对单一模板更简单直接(scale 到别的模型家族时应改用上游 continuous_token 层,我们 fork 目前没有该层)。

### 6.3 决策三:assistant=引擎原样 token id、历史 think 留在轨迹 —— 社区同构做法,且 Option B 的代价被 Slime 证实
- verl issue **#6854**(open)一字不差问了我们的问题("Qwen3.5 模板会剥掉历史轮 `<think>`,token-in-token-out 不重渲则 think 累积,重渲则前缀校验碎裂,怎么办?")——上游 legacy ToolAgentLoop 与 Continuous Token 路径都选择 **token 原样进轨迹、决不重渲已采样内容**(`merge_assistant_tokens` 直接拼 `assistant_token_ids`),think 累积被接受为 on-policy 保真的代价。与我们一致。
- **Slime**(THUDM/slime):(a) 自营多轮(`examples/retool/generate_with_retool.py`)= 纯 token 拼接:发 `input_ids` 给 SGLang `/generate`,取 `output_token_logprobs` 的 token id 原样 append(`trainable=True`),观测文本 `tokenizer(next_obs)` 后 append(`trainable=False`),逐轮 `max_new_tokens` 钳到剩余预算——与我们逐项同构(loss_mask 即我们的 response_mask,预算钳制即我们的 per-turn max_tokens)。(b) **外接 harness 路径**(`slime/agent/trajectory.py`,服务 claude_code/codex)被迫实现了一整套 token 漂移分类机(`CLEAN/REALIGN/FORK` + `_common_prefix_len`),其 docstring 明说 "a replayed turn rarely re-tokenizes byte-for-byte: TITO round-trips and chat-template re-rendering both perturb the ids";漂移段只能降级为 loss_mask=0 重放或 fork 掉。**这正是我们否掉方案 B 的理由的独立工程实证**:外挂 harness + 重分词必然漂移,要么丢训练信号(REALIGN 置 0)要么断轨(FORK)。
- 佐证配置:issue **#6223** 报告 **Qwen3.5-9B 在 vLLM token-in-token-out 多轮工具下正常**(不稳的是 35B-A3B,`enable_thinking=False` 时格式漂移,另见 #6252)——我们用的正是 9B+TITO 这一被社区确认可行的组合;将来上 35B-A3B 需先复查该 issue 进展。

### 6.4 结论
三个决策全部与社区收敛方向一致:决策二/三与 verl 已合并代码(#6921、#6779)及 Slime 自营路径同构;决策一与上游已合并 parser 同构且比 main 多修了 #4757(上游修复 #6434 还没合并)。本轮唯一改动 = 自己文件里加 `strip_reasoning`(纯增量);已排队 no-train episode 复验(job 11175235,维护窗 18:00 后跑)。若日后 rebase 到新 verl,可直接切换到上游 `qwen3_coder` parser + continuous_token 层并给它补 think-strip。

## 7. 已知限制与 scale-up 计划

1. **GPU 空转**:test() 跑 apptainer 时 rollout GPU 闲置(agentic RL 同步范式的固有成本)。缓解:加大 batch 内 episode 并发(生成与工具执行天然交错)、后续可上 verl 的 partial rollout / async 训练。
2. **每步墙钟由最慢 episode 决定**:CPU 任务单次 test 数分钟~15 分钟不等;`MLS_RL_EPISODE_TIMEOUT` 是硬上界。全量训练建议 `MAX_STEPS=8-12, MAX_TESTS=1-2, EPISODE_TIMEOUT≈2×最慢任务 test 时长`。
3. **上下文预算**:保留历史 think ⇒ 每轮 ~1-4k think token,8 步 episode 很容易顶到 response 预算(冒烟 14336)。Scale-up:response 24k-32k(参考 synth 32k 赢家配置:4 GPU / gpu_mem 0.4 / offload off),或降低 thinking 长度(presence penalty env 已有管道)。
4. **并发 apptainer 上限**:256 并发不可能;当前 = num_workers × MLS_RL_MAX_CONC_TESTS(冒烟 2×3)。全量(32×8=256 episodes/步)必须:(a) 拉高 episode 并发但限 test 信号量(test 是瓶颈,其余轮次纯 GPU);(b) 或分波次 train_batch 16×8、多节点。实测单节点 8 cores/GPU×4=32 核,scikit-learn 类任务 compute≈0.25-1 核等效,建议 conc_tests ≤ 12/节点。
5. **MLS-Bench-dev 侧写入**:episode 的 leaderboard 行已在打分后自动清除;`logs/<task>/vllm/rl-<uuid>/agent/` 下每 episode 留 messages.jsonl(排查利器,但量大;全量训练前可加 `MLS_RL_NO_MLS_LOGS` 或定期清理)。workspace 默认即删(`MLS_RL_KEEP_WORKSPACE=1` 可留)。
6. **评测与训练同任务** = 在 dev 集上过拟合是预期用法(开发期);正式实验要按任务 split(train.parquet 换成训练 split,MLS eval 留 holdout)。
7. **单种子**:episode 内 seeds=[42](与 eval CPU 口径一致);奖励噪声主要来自任务本身的随机性,GRPO 组内对比可部分抵消。
8. **奖励稀疏**(冒烟实测 2/28 非零):base 模型在这些任务上本来就大多 0 分。缓解优先级:(a) response 预算 24k-32k(9/28 episode 是被截断的);(b) `MAX_TESTS=2` 给一次迭代机会(eval 里 max_tests=3 时非零率更高);(c) 加大 `ROLLOUT_N`(组内出现非零对的概率随 n 上升);(d) 从 SFT/RL 过的更强起点模型开始(rlafter 系 MLS 0.116)。警惕 RL 大表的截断死亡螺旋教训:response 预算必须给足。
9. **episode 超时被 kill 的 worker** 可能在 leaderboard 留下 non-final 行(带唯一 model 名,无害不影响任何查询;本次冒烟恰好 0 残留)。可定期 `grep -c "vllm/rl-"` 审计。
10. **budget_check/dgp 容器内失败**(`ml-anomaly-detection`、`ml-missing-data-imputation` 两任务确定性 0 分,eval 同样命中)→ 应上报 MLS-Bench-dev 维护者修 `budget_check.py` 的 mid_edit 加载路径(容器内 `_PROJECT_ROOT` 解析到 `/`)。

（冒烟数字已落地;后续修订随实验推进。）
