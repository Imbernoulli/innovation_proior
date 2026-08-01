# 32K 长序列 GRPO RL / 全参 SFT 训练效率手册（Qwen3.5-9B & Qwen3.6-35B-A3B）

> 硬约束：生成/输出长度必须 ≥ 32768 (32K)。以下所有措施都 **不缩短** response/output，只降显存 + 提吞吐。
> 所有代码结论均标注 `file:line`，基于对本仓库实际代码的阅读与 <2min import 可行性测试（未跑重训练/长编译）。

## 0. 环境事实（已实测）

| 用途 | venv | torch / CUDA / py | flash_attn | liger | fla | 备注 |
|---|---|---|---|---|---|---|
| **RL actor + vLLM rollout** | `FrontierSmith/.venv-vllm023` | **2.11.0+cu130 / cu13 / 3.12** | ✗ 未装 | ✗ | ✗ | actor 落回 sdpa；vLLM 自带 `vllm.vllm_flash_attn`（rollout 已用 FLASH_ATTN） |
| 旧 vLLM/杂用 | `FrontierSmith/.venv` | 2.6.0+cu124 | ✗ | – | – | 非 RL 主路径 |
| **全参 SFT (LLaMA-Factory)** | `envs/sft_lf`（=`research_overlay`） | 2.10.0+cu128 / cu12.8 | **✓ 2.8.3.post1** | ✓ | ✗ | SFT 侧已装好 FA2+liger |

关键结论提前说：**SFT 全参侧（LLaMA-Factory）已经是优化态**——`envs/sft_lf` 里 flash_attn 2.8.3.post1 + liger_kernel 都在，LF 配置默认 `flash_attn: fa2` + `enable_liger_kernel: true`（如 `LF-innov/examples/train_full/cc_q3_innovonly_2ep.yaml:32-33`）。**所有效率缺口都在 RL (verl) 侧的 `.venv-vllm023`。**

模型结构（`models/Qwen3.6-35B-A3B/config.json` → `text_config`，实读）：
- 40 层 = **30 层 linear_attention (GDN) + 10 层 full_attention**（`full_attention_interval=4`，`layer_types` 统计）。
- `num_attention_heads=16, head_dim=256, hidden_size=2048`，MoE `num_experts=256 / per_tok=8`。
- **`vocab_size=248320`（约 24.8 万，极大）** → 32K 下 logits 张量是主要激活尖峰（见 B6）。
- `max_position_embeddings=262144`，32K response 完全在范围内。

---

## A. 动态 batch（use_dynamic_bsz）怎么工作 + 推荐设置

### A.1 机制（token 预算微批）
入口 `prepare_dynamic_batch` → `rearrange_micro_batches`（`verl/verl/utils/seqlen_balancing.py:348-424`）：

1. 逐条算 **有效长度** = `attention_mask.sum(dim=1)`（`seqlen_balancing.py:380`）。
2. **硬约束**：`assert max_token_len >= max_seq_len`（`:382-384`）——单个微批的 token 预算必须 ≥ 本批最长序列，否则直接抛错。
3. 微批数 `num_micro_batches = min(batch_size, ceil(total_seqlen / max_token_len))`（`:387`）；开 SP/PP 时按 dp_group 取 MAX 对齐（`:391-396`）。
4. 用 **Karmarkar-Karp 差分法** 按“工作量”均衡切分（`get_seqlen_balanced_partitions`，`:213/253`），工作量 = `24576*L + L²`（`calculate_workload`，`:46`，即近似注意力 FLOPs，L² 项对 32K 长序列权重很大）。
5. `use_dynamic_bsz_balance=True` 时把大微批放两端、小微批居中，减少 warm-up/cool-down 气泡（`:406-416`）。

### A.2 `max_token_len` 公式（实读，两处一致）
- **训练 update_policy**：`max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size`（`verl/verl/workers/actor/dp_actor.py:559`）。
- **compute_log_prob（old/ref logp）**：`max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size`（`dp_actor.py:467`）。

含义：Ulysses SP 会把每条序列沿序列维切到 `SP` 张卡上，所以 **每卡实际前/反向的 token 负载 ≈ `ppo_max_token_len_per_gpu`**，而 `max_token_len`（=×SP）是“整条序列在集合里的总预算”。因此约束 A.1-2 展开为：

> **`ppo_max_token_len_per_gpu × ulysses_sp ≥ (max_prompt + max_response)` = MAX_MODEL_LEN**

35B 现值：`24576 × 2 = 49152 ≥ 45056` ✓（正是记忆里的 “MAXTOKLEN×SP ≥ MAX_MODEL_LEN”，成立）。这正是 35B 靠 Ulysses SP=2 把每卡 token 负载从 34816 压到 24576、腾出 ~30GB 的原理。

### A.3 是不是现在的标准默认？
是。dp_actor 侧字段 `use_dynamic_bsz` / `ppo_max_token_len_per_gpu` 是一等公民（`dp_actor.py:75, 558-560`），生成态默认配置里 torchtitan 引擎甚至把 `ppo_max_token_len_per_gpu` 默认设 32768（`verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:455`）。FSDP 路径出于兼容默认 `use_dynamic_bsz=false`（`_generated_ppo_trainer` 里 `oc.select:...,false`），但对变长 RL，动态 bsz 是 verl 官方推荐路径。我们脚本已对 35B 打开（`scripts/run_verl_grpo_frontiercs_qwen35_9b.sh` 的 `USE_DYNAMIC_BSZ`、`ref.log_prob_use_dynamic_bsz`、`rollout.log_prob_use_dynamic_bsz` 全部随 `$USE_DYNAMIC_BSZ` 联动）。

### A.4 推荐值
- **保持** `use_dynamic_bsz=True`（三处：actor / ref.log_prob / rollout.log_prob，脚本已联动）。
- `ppo_max_token_len_per_gpu`：**目前 24576 是被 sdpa 激活量压出来的保守值**。装上 flash-attn（B1）后 O(n²) 激活消失，可上调到 `≈ 2×(prompt+resp)/SP` 以内的更大值（例如 SP=2 时 32768），单微批塞更多 token → 提升 MFU、减少微批数。上调前务必仍满足 A.2 约束。
- `ulysses_sp`：35B=2（每序列 2 卡分摊）是长序列显存的关键，保留。9B 若不 OOM 保持 1（SP>1 有 all-gather 通信开销）。

---

## B. 32K 训练效率“确定性收益”排行（大→小，全部保持 32K 输出）

> 现状已开且正确的（**不用动**）：`use_remove_padding=True`（序列打包/varlen，脚本已设）、`enable_gradient_checkpointing=True`（`fsdp_workers.py:495-496`，`use_reentrant=False`）、35B `optimizer_offload=True` + `ref.param_offload=True`、`expandable_segments`、`VERL_CKPT_OFFLOAD_TO_CPU=0`（存档改从 GPU 落盘的补丁）。

### B1. 【最大】actor FlashAttention-2：sdpa → flash_attention_2
- **是什么**：actor（transformers/FSDP 前反向）的 10 层 full_attention 目前走 sdpa，32K 下是 O(n²)。verl 默认 `attn_implementation="flash_attention_2"`（`fsdp_workers.py:387`），但 `.venv-vllm023` 里 `import flash_attn` 失败，脚本自动回落 sdpa（`run_verl_grpo_frontiercs_qwen35_9b.sh` 的 attn 探测块 → `ACTOR_ATTN_IMPL=sdpa`）。
- **现在开没开**：**没开（sdpa）**。这是最大的效率损失，且不止是速度问题——见“正确性警示”。
- **正确性警示（重要）**：verl 的 **Ulysses SP 与 rmpad varlen 都是围绕 flash-attention 的 `_flash_attention_forward` 实现的**（`verl/models/transformers/monkey_patch.py:87-155` 的 `_ulysses_flash_attention_forward`，`:146` 调 `_flash_attention_forward`；`:530-536` 把它 patch 进 transformers）。当 `attn_implementation=sdpa` 时，模型走的是 sdpa 集成、**不经过这个被 patch 的函数**；同时 rmpad 把整批打包成一条 `attention_mask=None` 的长序列（`dp_actor.py:164-251`），sdpa 对打包序列只能整段因果注意 → 既 O((Σnnz)²) 又有跨样本串扰风险。也就是说，**当前 35B 的 `rmpad + Ulysses SP=2 + sdpa` 组合是脱离 verl 已验证支持路径的**。装上 flash-attn 会同时修好“慢”和“路径正确性”。
- **可行性 / 精确安装法**：
  - **无现成 wheel**：pip 索引可达（列到 `flash-attn 2.8.3.post1`），但 PyPI 只有 sdist；GitHub 预编译 wheel 按 torch/cuda 命名。仓库缓存的 `flash_attn-2.8.3-cp312`（`~/.cache/pip/wheels/...`）与 `wheelhouse/*cu12torch2.8*` 均为 **cu12/torch2.8** 构建。**实测**：把它的 `flash_attn_2_cuda*.so` 在本 venv 下 `ctypes.CDLL` → `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`，即 **torch 2.11 的 c10::cuda ABI 与 torch2.8 构建不兼容，铁定加载失败**。
  - **两条落地路径**：
    - **（A）先找官方 cu13 wheel**：flash-attn 近版本已出 cu13 轮子，命名形如 `flash_attn-2.8.3-cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`。本 venv 参数：**py3.12 / torch2.11 / cu13 / cxx11abi=True**（实测 `torch._C._GLIBCXX_USE_CXX11_ABI == True`）。若在 flash-attn GitHub releases 找到完全匹配的，直接 `pip install <该whl>`（秒级，零编译，最稳）。
    - **（B）源码构建（保底一定能成）**：CUDA 13 工具链本机可用（`module load cudatoolkit/13.0`，`/usr/local/cuda-13.0` 存在；`ninja 1.13.0` 已装）。命令：
      ```bash
      source FrontierSmith/.venv-vllm023/bin/activate
      module load cudatoolkit/13.0            # 提供 nvcc（当前 PATH 只有 cuda-12.4）
      export CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=32
      pip install flash-attn==2.8.3 --no-build-isolation   # --no-build-isolation 复用 venv 内 torch2.11
      ```
      预计 1–3h 编译（H200=sm_90，`torch.cuda.get_arch_list()` 含 sm_90/sm_100，覆盖 H200/未来卡）。构建期长，**放后台或单独 srun 编译节点**，勿占训练窗口。
    - **（C）零构建实验旁路**：`vllm.vllm_flash_attn` 已随 vLLM 装好且**与本 venv torch2.11 ABI 兼容**（实测导出 `flash_attn_varlen_func`，就是 rollout 的 FLASH_ATTN 后端）。可 monkey-patch transformers 的 `_flash_attention_forward` 指向它。省编译，但它是 vLLM 的 cute/CUTLASS FA，函数签名/return 语义与 flash_attn 略有出入，属实验路径，需小样本核对数值。
- **启用开关（装好后）**：脚本已自动探测——`import flash_attn` 成功即自动 `ACTOR_ATTN_IMPL=flash_attention_2`；或显式 `ACTOR_ATTN_IMPL=flash_attention_2 bash scripts/cc_rl35b_synth_submit.sh ...`。落到 `+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2`（`fsdp_workers.py:387-389,462-467`）。
- **显存/速度预期**：10 层 full-attn 的注意力从 O(n²)→O(n)，32K 下这些层激活显存与算力都大降；解锁 A.4 上调 token 预算。恢复 Ulysses SP 的官方支持路径。
- **正确性 caveat**：装完务必冒烟对齐 reward/kl，确认切换后梯度与旧 sdpa run 走势一致（尤其 GQA num_kv_heads=2、Ulysses 的 head 复制 `monkey_patch.py:119-121`）。

### B2. 【很大，且几乎零成本】GDN 层的 flash-linear-attention (fla) + causal_conv1d
- **是什么**：40 层里 **30 层是 GDN linear_attention**，占 3/4。`.venv-vllm023` 里 `fla`/`flash_linear_attention`/`causal_conv1d` **全部没装**（实测 import 均失败），transformers 5.12 的 qwen3_5 GDN 只能走**纯 torch 参考实现的 gated-delta-rule 递推**（慢，与记忆里“GDN 走 torch fallback”一致）。
- **现在开没开**：没开。因为它命中 30/40 层，端到端 actor 提速幅度可能 **比 flash-attn(10 层) 还大**。
- **可行性（最香的一点）**：**fla 是纯 Triton 库**（Triton 3.6.0 已在本 venv），`pip index` 显示 `flash-linear-attention 0.5.1` 可装，**无 CUDA 编译、无 ABI 风险**：
  ```bash
  source FrontierSmith/.venv-vllm023/bin/activate
  pip install flash-linear-attention        # 纯 triton，秒级
  pip install causal-conv1d                 # 可选，短卷积核；这个是 CUDA 扩展，同样需 cudatoolkit/13.0，装不上可跳过（GDN 短卷积会回落 torch）
  ```
- **启用**：transformers 的 qwen3_5 GDN 在 import 到 fla 时自动改用其 chunked kernel（按可用性 dispatch），无需改 verl flag。装完用 LF-innov 里现成的 `fla_gdn_correctness_test.py` / `fla_gdn_fp32_diag.py` 复核数值。
- **预期**：30 层 GDN 递推 Triton 化，长序列前反向显著提速。
- **caveat**：务必跑上面那两个 fla 数值对齐脚本，确认 fla kernel 与 torch fallback 在 bf16/fp32 下一致（fla 版本与 transformers 5.12 的 GDN 接口需匹配；不一致就锁一个可用版本）。

### B3. 【中】提升 vLLM rollout 并发（rollout 占 step 大头）
- **是什么**：一个 step 的 256 条 rollout（TB×RN，各 ≤32K）是长尾。`MAX_NUM_SEQS` 在 `run_verl_grpo_frontiercs_qwen35_9b.sh` 默认 **8**，35B 提交脚本没覆盖它 → 实际 8（verl 默认其实是 1024，`verl/trainer/config/rollout/rollout.yaml:68`）。
- **现在**：并发只有 8，偏低。混合 GDN 架构只有 10 层有 KV → **KV cache 很小**，有空间加并发。
- **启用**：`MAX_NUM_SEQS=16`（或 24/32）作为环境变量传入提交脚本；配合 `enable_chunked_prefill=True`（默认已开，`rollout.yaml:71`）。
- **预期**：解码吞吐近线性提升（受 GMU/KV 预算约束）。**不改任何长度**。
- **caveat**：并发×KV 若逼近 GMU 上限会触发 vLLM 抢占重算，按显存回退。

### B4. 【中】CUDA graph 保持开（enforce_eager=False）
- verl rollout 默认 `enforce_eager=False`（`rollout.yaml:38`），即 **CUDA graph 默认开**，我们脚本没关它——**保持现状即可**。32K 是解码密集，CUDA graph 对 decode 吞吐帮助明显。**不要**为省唤醒期显存去设 enforce_eager（会拖慢生成）。

### B5. 【中，需测】切 FSDP2（strategy=fsdp2 + offload_policy）
- **是什么**：现在是 FSDP1（`verl/trainer/config/actor/dp_actor.yaml:26` `strategy: fsdp`，代码注释 TODO 切 fsdp2）。FSDP2（`fsdp_workers.py:608-630`）提供 **per-parameter 粒度的 `CPUOffloadPolicy(pin_memory=True)`**（`:613-614`），与计算重叠比 FSDP1 的粗粒度 `param_offload/optimizer_offload` 更好，还解锁 `reshard_after_forward` 调参、`TiledMLP`（`fsdp_workers.py:363-364` 要求 fsdp2）。
- **收益**：offload 的 H2D/D2H 更能被算力盖住 → 有机会**关掉 optimizer_offload 换更快的 optimizer step**，或降低 35B 存档时的 host-RAM 峰值。
- **启用**：`actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2`，并用 `fsdp_config.offload_policy=true` 取代粗粒度 offload。
- **caveat**：改动面大；与 `VERL_CKPT_OFFLOAD_TO_CPU` 存档补丁、35B 权重同步路径都要在 3-step 冒烟里重验。**先小样本再上正式窗口。**

### B6. 【视情况】248k 词表 logits 尖峰的 fused-CE —— 现结论：本 MoE 暂不可用
- **动机**：`vocab=248320`，rmpad 后单微批 logits ≈ `nnz × 248320 × 2B`（nnz=24576 时约 12GB），是 actor 最大激活尖峰。
- **verl `use_fused_kernels`**：**对 qwen3_5_moe 不生效**——`monkey_patch.py:255-266` 只为 VL 模型（qwen2_vl / qwen3_vl(_moe) / glm4v）接了 fused forward，`else` 分支保持默认 forward（`:253-254`），文本 MoE 拿不到融合 CE。
- **liger（`use_liger`，`fsdp_workers.py:471-474`）**：`.venv-vllm023` 未装 liger，且 liger 对 **qwen3_5_moe 这种极新架构大概率没有对应 patch**，`_apply_liger_kernel_to_instance` 可能是空操作。
- **结论**：**暂列“待查/多半不可用”**，别当既得收益。真要压这块显存，靠 A.4 的 token 预算 + B1 flash-attn 降激活更实在。

### B7. 【不建议】torch.compile 重新打开 —— 实测基本无收益，别冒存档挂起风险
- **实测**：dp_actor 只把 `torch.compile` 套在 `entropy_from_logits` 与 `calculate_sum_pi_squared` 两个 helper（`dp_actor.py:86-90, 102-107`），**不含主 forward/backward**。而我们脚本 `entropy_coeff=0`、`calculate_entropy`/`calculate_sum_pi_squared` 均未开 → 这两个被编译的函数**热路径根本不调用**。
- **结论**：为 35B 重开 torch.compile **提速≈0**，却要冒动态 shape + Inductor 在 step-1 存档 barrier 挂起 straggler 的老坑（记忆已记）。**维持 `USE_TORCH_COMPILE=False`**（这也直接回答了任务里“能否安全重开 torch.compile”：不值得）。

### B8. 【仅 OOM 兜底】activation offload
- `enable_activation_offload`（`fsdp_workers.py:634-635`，`model.enable_activation_offload`）把激活卸到 CPU，能再省显存，但加 H2D/D2H → **变慢**。仅当 B1/B5 后仍 OOM 时才开，别默认用。

---

## C. 推荐的“高效 35B 32K”配置 diff（vs 现状）

现状基线：`STEPS=20 ... SMOKE=0 bash scripts/cc_rl35b_synth_submit.sh`，即
`resp=32768 / model_len=45056 / GMU=0.65 / dynbsz=True / SP=2 / maxtok/gpu=24576 / TP=1 / opt-offload=True / ref-offload=True / torch.compile=False / MAX_NUM_SEQS=8(继承)`。

### 第 1 步（零/低风险，先做，预计最大收益）
```bash
# 1) GDN 30 层 Triton 化（纯 triton，零 ABI 风险）
source FrontierSmith/.venv-vllm023/bin/activate
pip install flash-linear-attention          # + 可选 causal-conv1d（需 module load cudatoolkit/13.0）
python LF-innov/fla_gdn_correctness_test.py  # 数值对齐后再上训练

# 2) rollout 并发 8→16（KV 小，安全）
#    在提交时加环境变量：
MAX_NUM_SEQS=16 STEPS=20 SMOKE=0 ONLY=r32s01 bash scripts/cc_rl35b_synth_submit.sh
```

### 第 2 步（装 flash-attn，最大速度+正确性收益）
```bash
# 优先找官方 cu13/torch2.11/cp312/abiTRUE 预编译 wheel，直接 pip install；
# 找不到就源码编译（后台/单独节点，勿占训练窗口）：
module load cudatoolkit/13.0
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=32 \
  pip install flash-attn==2.8.3 --no-build-isolation
```
装好后提交（脚本自动探测 flash_attn → flash_attention_2；并把每卡 token 预算上调）：
```bash
ACTOR_ATTN_IMPL=flash_attention_2 \
MAXTOKLEN=32768 \        # flash-attn 消除 O(n²) 激活后可上调；仍满足 32768×2 ≥ 45056
MAX_NUM_SEQS=16 \
STEPS=20 SMOKE=0 ONLY=r32s01 bash scripts/cc_rl35b_synth_submit.sh
```

### 配置项对照表

| 项 | 现状 | 推荐 | 依据 file:line | 影响 |
|---|---|---|---|---|
| actor attn | `sdpa`（回落） | **`flash_attention_2`** | `fsdp_workers.py:387`; `run_..._9b.sh` attn 探测块 | 10 层 O(n²)→O(n)；修 Ulysses/rmpad 正确性路径 |
| GDN kernel | torch fallback | **装 `flash-linear-attention`(Triton)** | 实测 import 失败；`text_config.layer_types`=30 linear | 30/40 层提速，零 ABI 风险 |
| `MAX_NUM_SEQS` | 8（继承） | **16–32** | `run_..._9b.sh` 默认 8；`rollout.yaml:68` | rollout 吞吐近线性↑ |
| `ppo_max_token_len_per_gpu` | 24576 | **32768**（flash 后） | `dp_actor.py:559`；A.2 约束 | 微批更满、MFU↑ |
| `use_dynamic_bsz` / SP | True / 2 | **保持** | `dp_actor.py:558-560, 467` | 长序列每卡 token=maxtok/gpu |
| CUDA graph | on（默认） | **保持 on** | `rollout.yaml:38` enforce_eager=False | decode 吞吐 |
| `use_torch_compile` | False | **保持 False** | `dp_actor.py:86-90`（仅套 entropy，热路径不用） | 重开≈0 收益，反招存档挂起 |
| strategy | fsdp(1) | fsdp2（先冒烟） | `dp_actor.yaml:26`；`fsdp_workers.py:608-630` | offload 重叠更好，可能省 optimizer step |
| grad ckpt / rmpad / opt-offload | 已开 | 保持 | `fsdp_workers.py:495`; `dp_actor.py:65` | — |

### 9B 侧
9B 单卡放得下，一般 `SP=1`（避免 all-gather 开销）、`use_dynamic_bsz` 视显存决定；**同样受益于 flash-attn(B1) 与 fla(B2)**——它们在 `.venv-vllm023` 里生效，9B/35B 共用该 venv。SFT（LLaMA-Factory，`envs/sft_lf`）已是 FA2+liger+FSDP2 padding-free 优化态，无需再动。

---

## 附：一句话可行性判定
- **flash-attn 装不装得上**：装得上，但**没有能直接用的预编译 wheel**（缓存/wheelhouse 全是 cu12/torch2.8，实测 c10::cuda ABI 符号缺失加载失败）；要么找官方 cu13/torch2.11 wheel，要么 `module load cudatoolkit/13.0` 源码编译（1–3h）。CUDA 13 工具链、ninja、cxx11abi=True 均已就绪，**可行性=高，只是要花一次编译**。
- **更省事的邻居**：`flash-linear-attention`（纯 Triton，秒装，管 30/40 层）建议**第一优先**先上，性价比最高。
