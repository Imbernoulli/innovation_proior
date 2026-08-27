# v2b — 9B SFT wave on della/ailab (2026-08-27)

The wave that re-trains on the **fixed** agentic corpus and on the **time+researcher-only**
system prompt. Five arms, one epoch each. Everything here is the real submitted config,
copied back from `/scratch/gpfs/CHIJ/ziran/innov_v2_multi/configs/`.

## Arms

| job config | finetuning | weight decay | dataset |
|---|---|---|---|
| `full_wd01_withag_v2b.yaml` | full | 0.1 | `innov_v2` + `maintain_w2w3` |
| `full_wd03_maint.yaml` | full | 0.3 | `innov_v2` + `maintain_w2w3` |
| `full_wd03_nomaint.yaml` | full | 0.3 | `innov_v2` |
| `lora_r32_maint.yaml` | LoRA r32 | 0 | `innov_v2` + `maintain_w2w3` |
| `lora_r32_nomaint.yaml` | LoRA r32 | 0 | `innov_v2` |

The two pairs are exact controls: within each pair the configs differ ONLY in the
`dataset:` line and `output_dir:` — diff them and nothing else moves.

`full_wd01_*` has no `nomaint` twin, so wd × maintenance is not a complete 2×2. That
cell was not requested; adding it is one more 4-hour job.

Datasets (registered in `$D/data/dataset_info.json`, sharegpt format):
- `innov_v2` — `sft/innovation_sft.jsonl` from `sft/build_sft.py`, 2885 rows
  (method 1184 / traj_folded 463 / traj_full 164 / agentic_folded 736 / v4 338).
- `maintain_w2w3` — 5814 rows. `maintain_hard` (4917) is a STRICT SUBSET of it:
  all 4917 appear in w2w3 byte-identically, w2w3 adds 897. The previous wave's
  `hardmaint` arm was "maintenance = hard subset", NOT "no maintenance".

## Batch size — read this before changing `--gres`

Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × NGPU`.
The proven recipe is **128** = 1 × 32 × 4. `gradient_accumulation_steps` in these YAMLs
is set for **4 GPUs**. Running on 2 GPUs without doubling it to 64 silently halves the
effective batch to 64 and the run is no longer the same recipe.

## Wall-clock (measured, not estimated)

From cancelled job 13043700, `full_wd01_withag_v2b` on 4×H200:

```
Total optimization steps = 68        # 8699 rows / effective batch 128
26/68 [1:15:04<1:57:50, 168.34s/it]  # 173.2 s/step averaged over the first 26
```

→ ~3h16m of training + ~2 min preprocessing + the 9B save ≈ **3h25m**. Submitted with
`--time=04:00:00`, i.e. ~35 min of margin. `save_strategy: epoch` with 1 epoch means the
ONLY checkpoint is written at the end, so a timeout loses the whole run, not the tail.

Do not size this off the older 30-minute figure — that was `pure_noag` (2149 rows, no
maintenance), a quarter of this data.

## Speed levers NOT taken (no memory headroom measurement yet)

- `ds_z3_config.json` has `"overlap_comm": false` — ZeRO-3 gradient reduction is not
  overlapped with backward. Turning it on usually helps but roughly doubles the reduce
  bucket. The config lives in a read-only tree; using our own copy is required.
- `gradient_checkpointing: true` is the biggest single lever (~30-40%), but at
  `cutoff_len: 53760` it is also what makes the run fit.

Both are blocked on the same thing: nobody has recorded per-GPU memory during a
successful run. Record `nvidia-smi` next wave, then decide with numbers.

## Submit

```bash
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
SB="--partition=ailab --account=chij --qos=short --gres=gpu:4 --cpus-per-task=32 --mem=480G --time=04:00:00"
sbatch $SB --job-name=sft-ft-wd01-maint \
  --export=ALL,HF_HOME=$D/.hf_ft \
  $D/configs/launch_sft.sh $D/configs/full_wd01_withag_v2b.yaml
```

`HF_HOME` must differ per concurrently-running arm. LlamaFactory keys its tokenized
dataset cache on the dataset combo, so two arms sharing a combo (the full-FT and LoRA
maint arms both use `innov_v2,maintain_w2w3`) race each other under
`overwrite_cache: true`. `launch_sft.sh` honours a pre-set `HF_HOME` for exactly this.

## Next

Each arm gets soups at **α=0.10 and α=0.20** (10 models), then FCS / ALE-40 / Research /
MLS. The base9b anchor MUST be re-run in the same wave: this wave changed the eval口径
(MLS previously had no time conditioning at all AND the wrong edit-tool contract), so
these numbers cannot be placed next to the frozen table.
