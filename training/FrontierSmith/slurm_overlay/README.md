# slurm_overlay — zy7019 replacements for scripts with inherited-path defaults

Several scripts in `slurm/` and `experiments/scripts/` resolve a tool, venv, or data
root through `${VAR:-<a path under bl3615's home or /scratch/gpfs/CHIJ/bohan>}`. Run
from another account those defaults are wrong, and the failure is usually SILENT rather
than loud. This class has bitten six times:

| default | symptom |
|---|---|
| `VLLM_VENV` | `exec: vllm: not found` |
| `MLSBENCH_ROOT` | `PermissionError [Errno 13]`, surfacing as mass `agent_failed` |
| `MLSBENCH_PY` | `ModuleNotFoundError` (deap/sklearn/pydot/pgmpy/causallearn), silent |
| merge script `ROOT` + `.venv-vllm023` | no `tensordict`; `verl.model_merger` dies before doing work |
| `upload-large-folder` | writes `.cache` INTO the source dir — fails on a read-only tree |
| README `base_model` | a local absolute path; HF rejects the commit as invalid metadata |

The overlays here point at zy7019-owned copies instead. They are additions, not edits:
the originals are left alone so a run from bl3615's account still behaves as before.

## Files

- **`cc_merge_ckpt_cpu_zy7.sh`** — merge a verl FSDP checkpoint into an HF model.
  Overlay of `slurm/cc_merge_ckpt_cpu.sh`. Note it needs a **GPU** despite being pure
  state-dict shuffling: these checkpoints bake `attn_implementation=flash_attention_2`
  into `config.json` and transformers refuses to dispatch FA2 on CPU. Submit to a GPU
  partition with `ALLOW_GPU=1`.

  Also worth knowing: the merger hard-casts to bf16
  (`verl/model_merger/fsdp_model_merger.py:169,181`) while the checkpoint shards are
  fp32 (4 × 9.41 GB for a 9B actor). A bf16 round-trip changes 100% of elements in the
  trained tensors — measured relative error 2.6e-6 (`dt_bias`) to 9.9e-4 mean / 3.8e-3
  max (`conv1d.weight`, i.e. at the bf16 floor), against exactly 0.0 for the frozen
  vision tower, which is the control proving the rest is real trained precision. This is
  harmless for eval and serving (vLLM loads bf16 anyway) but not for restarting RL from
  a merged dir, or for souping an RL checkpoint at small α.

- **`hf_upload_remaining.sh`** — serialized HF uploads. Serialization is mandatory:
  creating a repo while an `upload-large-folder` is in flight trips the 1000 request /
  5 minute limit. Reads auth from `~/.cache/huggingface/token` (0600); never put a token
  on a command line on a shared login node.
