# 实验代码索引

镜像自 `/scratch/gpfs/CHIJ/bohan/fs` 的 FrontierSmith / ThetaEvolve 实验栈中、与本 campaign 直接相关的脚本。仅作存档与可读性参考（路径、集群细节按原环境硬编码，非开箱即用）。各文件顶部有详细 docstring。

## `eval/` — 评测 harness
| 文件 | 作用 |
|---|---|
| `cc_eval_thinking_both_ailab.sh` | FrontierCS + ALE-Bench 统一 thinking 评测（vLLM serve，含 `max_model_len` 计算，q3-base bug 即在此 L113-128） |
| `eval_qwen35_base_vllm_request.py` | 上面 harness 的请求/抽码/打分实现（`strip_think` + 抽最长 ```cpp 块） |
| `cc_eval_theta_openevolve_ailab.sh` | ThetaEvolve（circle-packing 等）进化搜索评测 |
| `cc_eval_ttt_discover_openevolve_ailab.sh` | 「TTT」AC3 评测——实为 delegate 到 ThetaEvolve harness（见文档 §5.4） |
| `cc_eval_mlsbench_cpu_ailab.sh` | MLS-Bench 20 CPU 任务评测（`CONCURRENCY=20` 并行一波） |
| `cc_submit_mlsbench_cpu.sh` | MLS-Bench 提交器 `<MODEL_DIR> <TAG>` |
| `mlsbench_run_cpu_tasks.py` | MLS-Bench 在 job 内的 worker pool（含 stderr/stdout JSON 修复） |
| `start_vllm_server.sh` | 通用 vLLM 起服务 |

## `agg/` — 结果聚合
| 文件 | 作用 |
|---|---|
| `track13.py` | method/methodtraj 矩阵聚合（FCS/ALE/Theta/TTT） |
| `cc_results_full.py` / `cc_results_table.py` | innovonly/innovmaint 矩阵聚合 |
| `parse_openevolve_best.py` | 从 OpenEvolve 输出抽 `best_combined_score` + seed-floor 判别 |
| `noise_theta_agg.py` | Theta/TTT 多 seed 噪声分析（Cohen's d / CI 分离） |

## `train/` — soup 合并 + RL
| 文件 | 作用 |
|---|---|
| `cc_model_soup_merge.py` | 权重平均 `α·SFT+(1-α)·START`，容忍 MTP key 不对称 |
| `frontiercs.py` | verl 的 FrontierCS 奖励（judge 超时软失败、`strip_think`） |
| `rlm_amplify_v3_ailab.sh` | RL amplify 快配置启动脚本（关 actor offload、response 20k、save_freq=5） |
| `rlm_amplify_v3_submit.sh` | RL amplify 批量提交器 |

## `orchestrate/` — 矩阵编排
| 文件 | 作用 |
|---|---|
| `cc_orchestrator.py` | 幂等 DAG 控制器，跑 families×ratios×data×soups 矩阵 + 统一评测，自愈 |
| `cc_watchdog.sh` | flock 单例看门狗，编排器死了重启、0 任务卡住时踢一脚 |
