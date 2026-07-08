# Complexity Analysis: Innovation-Trained Qwen3.5-9B vs Base

## 1. Methodology

This analysis was computed directly from raw sample JSONL files and MLS task logs under `/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs`. The script stripped any `<think>...</think>` block before fenced-code extraction.

For FCS rows, the extractor used the final fenced `cpp`/`c++` code block and split rows with `data_source=frontiercs`. For ALE and Research rows, it used the final fenced `python`/`py` code block and split rows with `data_source=alebench` or `frontiercs_research`. For MLS, each task log was one attempted solution; ANSI color codes were stripped and lines matching `^\+\s+\d+\s+\|` were collected as agent-added Python code.

`radon` was not importable, so Python cyclomatic complexity uses the requested manual proxy: `1 + if/for/while/ternary/except/comprehension decisions + and/or boolean joins`. C++ cyclomatic complexity uses `1 + if/for/while + && + ||`.
Python AST nodes are `len(ast.walk(ast.parse(code)))`; C++ AST_nodes is the requested brace-pair proxy. LOC counts non-blank, non-comment lines. Imports/includes and function calls are counted as distinct names. Advanced-technique keywords are counted as distinct listed keywords present case-insensitively in the code.

Solutions with empty extracted code or Python parse errors were skipped for complexity aggregation and counted below. Cells with fewer than 5 parsed solutions are flagged `LOW_N`. Standard deviations are sample standard deviations (0.00 for N=1).

Resolved raw sources:

| group | model | kind | resolved_source |
|---|---|---|---|
| OURS | r3_methodtraj_v4_r3_a10 | samples | `cc_eval_all_r3_methodtraj_v4_r3_a10/fcsale/shard_0/samples.jsonl` |
| OURS | r3_methodv4_r3_a10 | samples | `cc_eval_all_r3_methodv4_r3_a10/fcsale/shard_0/samples.jsonl` |
| BASE | retest_start | samples | `cc_eval_retest_start_thinking_32k_both_vllm/shard_0/samples.jsonl` |
| BASE | clean_start | samples | `cc_eval_clean_start_thinking_32k_both_vllm/shard_0/samples.jsonl` |
| OURS | r3_methodv4_r3_a20 | samples | `cc_eval_r3_methodv4_r3_a20_research_thinking_32k_vllm/shard_0/samples.jsonl` |
| BASE | retest_start | samples | `cc_eval_retest_start_research_thinking_32k_vllm/shard_0/samples.jsonl` |
| OURS | r3_methodtraj_v4_r3_a10 | mls_logs | `cc_eval_all_r3_methodtraj_v4_r3_a10/mls/task_logs` |
| OURS | r3_methodv4_r3_a10 | mls_logs | `cc_eval_all_r3_methodv4_r3_a10/mls/task_logs` |
| BASE | q35_start_devfix | mls_logs | `cc_mlsbench_cpu_q35_start_devfix/task_logs` |

No requested source resolved as missing.

Skip counts:

| benchmark | model | attempted | parsed | empty/no_code | parse_errors | json_errors |
|---|---|---:|---:|---:|---:|---:|
| FCS | BASE:clean_start | 860 | 443 | 417 | 0 | 0 |
| FCS | BASE:retest_start | 860 | 450 | 410 | 0 | 0 |
| FCS | OURS:r3_methodtraj_v4_r3_a10 | 860 | 681 | 179 | 0 | 0 |
| FCS | OURS:r3_methodv4_r3_a10 | 860 | 656 | 204 | 0 | 0 |
| ALE | BASE:clean_start | 50 | 1 | 48 | 1 | 0 |
| ALE | BASE:retest_start | 50 | 1 | 48 | 1 | 0 |
| ALE | OURS:r3_methodtraj_v4_r3_a10 | 50 | 0 | 48 | 2 | 0 |
| ALE | OURS:r3_methodv4_r3_a10 | 50 | 0 | 50 | 0 | 0 |
| Research | BASE:retest_start | 320 | 220 | 40 | 60 | 0 |
| Research | OURS:r3_methodv4_r3_a20 | 320 | 231 | 40 | 49 | 0 |
| MLS | BASE:q35_start_devfix | 20 | 3 | 0 | 17 | 0 |
| MLS | OURS:r3_methodtraj_v4_r3_a10 | 20 | 0 | 0 | 20 | 0 |
| MLS | OURS:r3_methodv4_r3_a10 | 20 | 4 | 1 | 15 | 0 |

## 2. Per-Benchmark Complexity Tables

### FCS

| model | LOC | AST_nodes | cyclomatic | imports | fn_calls | technique_kws | score | N_parsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS:r3_methodtraj_v4_r3_a10 | 78.45 +/- 56.04 | 18.14 +/- 16.47 | 26.65 +/- 25.56 | 3.50 +/- 2.28 | 14.18 +/- 10.36 | 0.06 +/- 0.25 | 4.622 | 681/860 |
| OURS:r3_methodv4_r3_a10 | 77.89 +/- 54.51 | 17.71 +/- 14.87 | 26.23 +/- 23.64 | 3.47 +/- 2.30 | 14.38 +/- 11.05 | 0.04 +/- 0.19 | 4.554 | 656/860 |
| BASE:clean_start | 78.44 +/- 61.62 | 18.34 +/- 16.66 | 30.65 +/- 29.71 | 4.07 +/- 2.49 | 16.08 +/- 12.47 | 0.07 +/- 0.25 | 7.054 | 443/860 |
| BASE:retest_start | 82.50 +/- 64.75 | 18.54 +/- 15.86 | 31.92 +/- 28.43 | 4.22 +/- 2.37 | 16.99 +/- 12.82 | 0.05 +/- 0.22 | 6.382 | 450/860 |

### ALE

| model | LOC | AST_nodes | cyclomatic | imports | fn_calls | technique_kws | score | N_parsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS:r3_methodtraj_v4_r3_a10 | n/a | n/a | n/a | n/a | n/a | n/a | 384.100 | 0/50 LOW_N |
| OURS:r3_methodv4_r3_a10 | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 | 0/50 LOW_N |
| BASE:clean_start | 4.00 +/- 0.00 | 56.00 +/- 0.00 | 2.00 +/- 0.00 | 0.00 +/- 0.00 | 4.00 +/- 0.00 | 0.00 +/- 0.00 | 356.580 | 1/50 LOW_N |
| BASE:retest_start | 3.00 +/- 0.00 | 33.00 +/- 0.00 | 2.00 +/- 0.00 | 0.00 +/- 0.00 | 2.00 +/- 0.00 | 0.00 +/- 0.00 | 417.500 | 1/50 LOW_N |

### Research

| model | LOC | AST_nodes | cyclomatic | imports | fn_calls | technique_kws | score | N_parsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS:r3_methodv4_r3_a20 | 78.42 +/- 79.18 | 516.14 +/- 799.45 | 9.02 +/- 11.34 | 3.32 +/- 1.48 | 11.46 +/- 10.02 | 0.30 +/- 0.68 | 10.322 | 231/320 |
| BASE:retest_start | 78.07 +/- 56.63 | 472.48 +/- 417.64 | 8.54 +/- 10.29 | 3.37 +/- 1.49 | 11.80 +/- 10.66 | 0.30 +/- 0.69 | 9.080 | 220/320 |

### MLS

| model | LOC | AST_nodes | cyclomatic | imports | fn_calls | technique_kws | score | N_parsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS:r3_methodtraj_v4_r3_a10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0/20 LOW_N |
| OURS:r3_methodv4_r3_a10 | 118.25 +/- 54.46 | 938.50 +/- 632.75 | 10.75 +/- 7.18 | 0.25 +/- 0.50 | 13.25 +/- 5.25 | 0.75 +/- 0.96 | n/a | 4/20 LOW_N |
| BASE:q35_start_devfix | 119.33 +/- 48.42 | 859.33 +/- 832.62 | 13.33 +/- 19.66 | 13.33 +/- 13.50 | 15.00 +/- 12.12 | 0.67 +/- 0.58 | n/a | 3/20 LOW_N |

## 3. OURS-BASE Delta Table

Deltas are pooled across parsed solutions from all requested tags in each group for that benchmark, then `OURS - BASE` is computed. Score is omitted for MLS because task logs do not carry per-task scores directly.

| benchmark | ours_tags | base_tags | LOC | AST_nodes | cyclomatic | imports | fn_calls | technique_kws | score | N_ours | N_base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCS | r3_methodtraj_v4_r3_a10, r3_methodv4_r3_a10 | retest_start, clean_start | -2.31 | -0.52 | -4.85 | -0.66 | -2.26 | -0.01 | -2.130 | 1337/1720 | 893/1720 |
| ALE | r3_methodtraj_v4_r3_a10, r3_methodv4_r3_a10 | retest_start, clean_start | n/a | n/a | n/a | n/a | n/a | n/a | -194.990 | 0/100 | 2/100 |
| Research | r3_methodv4_r3_a20 | retest_start | 0.35 | 43.67 | 0.48 | -0.04 | -0.34 | -0.01 | 1.241 | 231/320 | 220/320 |
| MLS | r3_methodtraj_v4_r3_a10, r3_methodv4_r3_a10 | q35_start_devfix | -1.08 | 79.17 | -2.58 | -13.08 | -1.75 | 0.08 | n/a | 4/40 | 3/20 |

## 4. Honest Headline

Complexity-only headline:
Delta rows pool the parsed requested OURS tags and subtract the pooled parsed requested BASE tags for each benchmark.
- FCS: LOC lower (-2.31), cyclomatic lower (-4.85), imports+technique_kws fewer (-0.67), fn_calls fewer (-2.26).
- ALE: LOC n/a, cyclomatic n/a, imports+technique_kws n/a, fn_calls n/a.
- Research: LOC higher (+0.35), cyclomatic higher (+0.48), imports+technique_kws fewer (-0.05), fn_calls fewer (-0.34).
- MLS: LOC lower (-1.08), cyclomatic lower (-2.58), imports+technique_kws fewer (-13.00), fn_calls fewer (-1.75).
Overall, the numbers lean toward more focused low-LOC moves in 2/4 benchmarks (negative LOC with non-higher cyclomatic), not a uniform recombination pattern.
These are complexity directions only; they do not imply novelty or quality.
