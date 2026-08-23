| task | tier | data MB | setup s | overlay MB | overlay inodes | repair | probe g | agent g |
|---|---|---:|---:|---:|---:|:--:|---:|---:|
| s41467-025-63412-3 | GPU24 | 22.9 | 10.4 | 161.1 | 5152 |  | -1.0000 | -1.0000 |
| s41467-025-65557-7 | GPU24 | 39.1 | 9.9 | 52.8 | 4296 |  | -1.0000 | -0.2505 |
| s41587-024-02428-4 | GPU24 | 9.0 | 12.0 | 291.9 | 4056 |  | -1.0000 | - |
| s41592-022-01709-7 | CPU | 287.3 | 8.0 | 220.7 | 2392 |  | -1.0000 | -1.0000 |
| s41592-023-02124-2 | GPU24 | 7.9 | 14.2 | 378.7 | 4270 |  | -1.0000 | - |
| s42256-022-00468-6 | GPU24 | 2.7 | 37.0 | 1148.9 | 5364 |  | -1.0000 | -0.6794 |
| s42256-023-00611-x | GPU24 | 2.8 | 36.9 | 1623.4 | 25804 |  | -1.0000 | -1.0000 |
| s42256-023-00627-3 | GPU24 | 17.2 | 32.4 | 85.6 | 2051 |  | -1.0000 | -1.0000 |
| s42256-024-00833-7 | GPU24 | 4.8 | 0.0 | 0.0 | 3 |  | -1.0000 | - |
| s43588-024-00689-2 | CPU | 85.4 | 47.3 | 458.9 | 9869 | yes | -1.0000 | -1.0000 |
| s43588-024-00716-2 | GPU24 | 5.8 | 2.8 | 1.7 | 372 |  | -1.0000 | -1.0000 |
| s43588-025-00920-8 | GPU24 | 10.9 | 0.0 | 0.0 | 3 |  | -1.0000 | -0.6441 |

**Totals for 12 tasks**: data 0.50 GB, overlays 4.42 GB / 63632 inodes (5302 per task), total login-node setup 3.5 min.
**Agent batch `nb9b_r8`**: scored 9/12; Match-SOTA (g>=0) 0.0%, Surpass-SOTA (g>0.1) 0.0%, mean g -0.8416, distinct values 4.

## Two runs of the SAME model (Qwen3.5-9B) on the SAME 12 tasks

| task | 5-round g | 8-round g | stable? |
|---|---:|---:|:--:|
| s41467-025-63412-3 | none | -1.0000 | **NO** |
| s41467-025-65557-7 | -0.7839 | -0.2505 | **NO** |
| s41587-024-02428-4 | -1.0000 | none | **NO** |
| s41592-022-01709-7 | -1.0000 | -1.0000 | yes |
| s41592-023-02124-2 | -0.7768 | none | **NO** |
| s42256-022-00468-6 | none | -0.6794 | **NO** |
| s42256-023-00611-x | -0.2320 | -1.0000 | **NO** |
| s42256-023-00627-3 | none | -1.0000 | **NO** |
| s42256-024-00833-7 | -1.0000 | none | **NO** |
| s43588-024-00689-2 | none | -1.0000 | **NO** |
| s43588-024-00716-2 | none | -1.0000 | **NO** |
| s43588-025-00920-8 | none | -0.6441 | **NO** |

**11/12 tasks changed materially between two runs of the same model.** 5-round: 6/12 scored (3 at the -1.0 floor, 3 graded), mean g -0.7988. 8-round: 9/12 scored (6 at the floor, 3 graded), mean g -0.8416.

Reference-mode (non-LLM baselines, same official scorer): `s43588-024-00689-2` **g = +0.0517**, `s41592-022-01709-7` **g = -0.4698**, empty submission **g = -1.0** on all 17 tasks — the tasks themselves grade smoothly; the ceiling here is the 9B, not the benchmark.
