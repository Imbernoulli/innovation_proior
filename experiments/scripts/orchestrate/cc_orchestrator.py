#!/usr/bin/env python3
"""Persistent matrix orchestrator for the innovation-prior SFT campaign.

For each family (qwen3, qwen35) and each base<->instruct premerge ratio it:
  1. premerge: build the starting model = alpha*instruct + (1-alpha)*base   [CPU]
  2. eval(start)                                                            [GPU x3: FCS+ALE, Theta, TTT]
  3. for data in {innovonly, innovmaintain}:
       sft(start, data)                                                     [GPU 4x]
       eval(sft)
       for pa in postmerge_alphas:
         postmerge: soup = pa*sft + (1-pa)*start                           [CPU]
         eval(soup)

It keeps at most MAX_GPU GPU-jobs and MAX_CPU CPU-jobs of its own in flight,
backfills as they finish, is idempotent (skips any step whose output already
exists), and resumable (state in cc_orchestrator_state.json). Run it as a
background process; it submits real sbatch jobs and exits when the matrix is done.

  python cc_orchestrator.py            # run the controller loop
  python cc_orchestrator.py --plan     # print the task DAG and exit
"""
import glob, json, os, subprocess, sys, time

ROOT = "/scratch/gpfs/CHIJ/bohan/fs"
FS   = f"{ROOT}/FrontierSmith"
LF   = f"{ROOT}/LF-innov"
SFT_OUT = f"{ROOT}/models_sft"
MERGE_SCRIPT = f"{FS}/scripts/cc_model_soup_merge.py"
ENVPY = f"{ROOT}/envs/sft_lf/bin/python"
AUTO_YAML = f"{LF}/examples/train_full/auto"
STATE = f"{ROOT}/cc_orchestrator_state.json"
LOG   = f"{ROOT}/cc_orchestrator.log"

# ---- matrix ---------------------------------------------------------------
FAMILIES = {
    "q3":  {"instruct": f"{FS}/models/Qwen3-8B",      "base": f"{FS}/models/Qwen3-8B-Base"},
    "q35": {"instruct": f"{FS}/models/Qwen3.5-9B-bf16", "base": f"{FS}/models/Qwen3.5-9B-Base"},
}
PREMERGE_ALPHAS  = [0.0, 1.0]   # base + instruct ONLY (no avg, per user)
DATASETS = {"method": "innovation_method", "methodtraj": "innovation_method_traj"}   # method-only + method+traj (drop agentic)
POSTMERGE_ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7]   # sft fraction (added small fractions per user)
THETA_TASK = "circle_packing_modular"
TTT_TASK   = "third_autocorr_inequality"   # AC3: discriminates models (AC2 seed-dominated even after band fix)
ENABLE_THETA = True    # circle-packing fix VALIDATED (job 10095615: model_beat_seed, best 1.05 > seed 0.96).
ENABLE_TTT   = True    # re-enabled with AC3 (validated discriminating: model 0.317->0.506, beats seed). AC2 band fix kept for RL.
EVAL_ITERS = 20   # AC3/circle: ~20 iters reduces diff-parse variance on thinky SFT checkpoints
FCS_LIMIT, ALE_LIMIT, NSAMP = 40, 8, 5

MAX_GPU = 200   # effectively uncapped: ailab is our reserved 144-GPU partition; submit everything ready
MAX_CPU = 60    # merges are cheap+short; let many queue
POLL = 90
RETRY_CAP = 4          # retries before a task is parked as 'dead'
DEAD_COOLDOWN = 1800   # sec; a 'dead' task is resurrected after this, so the matrix never permanently stalls

def alpha_tag(a):  # 0.2 -> a20
    return "a%02d" % round(a * 100)

def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# ---- completion markers (a step is "done" when its output exists) ---------
def model_done(d):   return os.path.exists(f"{d}/config.json") and bool(glob.glob(f"{d}/*.safetensors"))
def fcsale_done(tag):return os.path.exists(f"{FS}/outputs/cc_eval_{tag}_thinking_32k_both_vllm/summary.json")
def theta_done(tag): return (not ENABLE_THETA) or bool(glob.glob(f"{ROOT}/ThetaEvolve/outputs/cc_eval_theta_{tag}_*/**/summary.json", recursive=True))
def ttt_done(tag):   return (not ENABLE_TTT) or bool(glob.glob(f"{ROOT}/ThetaEvolve/outputs/cc_eval_theta_ttt_{tag}_*/**/summary.json", recursive=True))

# ---- submit helpers (return slurm jobid string) ---------------------------
def sb(args, parsable=True):
    cmd = ["sbatch", "--parsable"] + args if parsable else ["sbatch"] + args
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        log(f"  sbatch FAILED: {out.stderr.strip()[:200]}"); return None
    return out.stdout.strip().split(";")[0]

def submit_merge(sft_dir, base_dir, alpha, out_dir, name):
    os.makedirs(f"{FS}/logs", exist_ok=True)
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:50:00
#SBATCH --output={FS}/logs/%x-%j.out
#SBATCH --error={FS}/logs/%x-%j.err
set -uo pipefail
{ENVPY} {MERGE_SCRIPT} --sft "{sft_dir}" --base "{base_dir}" --alpha {alpha} --out "{out_dir}"
"""
    p = f"{ROOT}/.orch/{name}.sh"; os.makedirs(f"{ROOT}/.orch", exist_ok=True)
    open(p, "w").write(script)
    return sb([p])

SFT_TEMPLATE = """### auto-generated SFT config
model_name_or_path: {model}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
dataset: {dataset}
template: qwen3
mask_history: false
cutoff_len: 53760
max_samples: 1000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
output_dir: {out}
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 5.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.0
bf16: true
flash_attn: fa2
enable_liger_kernel: true
gradient_checkpointing: true
use_reentrant_gc: false
ddp_timeout: 180000000
val_size: 0.0
"""

def submit_sft(model_dir, dataset, out_dir, name):
    os.makedirs(AUTO_YAML, exist_ok=True)
    y = f"{AUTO_YAML}/{name}.yaml"
    open(y, "w").write(SFT_TEMPLATE.format(model=model_dir, dataset=dataset, out=out_dir))
    return sb(["--gres=gpu:4", "--cpus-per-task=32", "--mem=300G", "--time=06:00:00",
               f"--job-name={name}", f"{LF}/cc-sft-innov.sh", f"examples/train_full/auto/{name}.yaml"])

def submit_eval_fcsale(model_dir, tag, name):
    return sb([f"--job-name={name}",
               f"--export=ALL,MODEL_PATH={model_dir},TAG={tag},FRONTIERCS_LIMIT={FCS_LIMIT},ALEBENCH_LIMIT={ALE_LIMIT},N_SAMPLES={NSAMP},EVAL_RESEARCHER_YEAR=2026",
               f"{FS}/slurm/cc_eval_thinking_both_ailab.sh"])

def submit_eval_theta(model_dir, tag, name):
    return sb([f"--job-name={name}", f"{FS}/slurm/cc_eval_theta_openevolve_ailab.sh",
               model_dir, tag, THETA_TASK, str(EVAL_ITERS)])

def submit_eval_ttt(model_dir, tag, name):
    # ttt wrapper execs sbatch itself -> run via bash, capture the jobid it prints
    out = subprocess.run(["bash", f"{FS}/slurm/cc_eval_ttt_discover_openevolve_ailab.sh",
                          model_dir, tag, TTT_TASK, str(EVAL_ITERS)], capture_output=True, text=True)
    for tok in out.stdout.split():
        if tok.isdigit(): return tok
    log(f"  ttt submit parse fail: {out.stdout.strip()[:120]} {out.stderr.strip()[:120]}"); return None

# ---- build the task DAG ---------------------------------------------------
# task = dict(id, kind in {merge,sft,eval}, gpu bool, done_fn, deps[list of task ids], submit_fn)
def build_tasks():
    T = {}
    def add(tid, **kw): T[tid] = dict(id=tid, **kw)

    for fam, mm in FAMILIES.items():
        inst, base = mm["instruct"], mm["base"]
        for a in PREMERGE_ALPHAS:
            at = alpha_tag(a)
            start_tag = f"{fam}_{at}"
            if a == 1.0:   start_dir, premerge_id = inst, None
            elif a == 0.0: start_dir, premerge_id = base, None
            else:
                start_dir = f"{SFT_OUT}/start_{start_tag}"
                premerge_id = f"merge:start_{start_tag}"
                add(premerge_id, kind="merge", gpu=False, deps=[],
                    done_fn=(lambda d=start_dir: model_done(d)),
                    submit_fn=(lambda sd=inst, bd=base, aa=a, od=start_dir, nm=f"om-start-{start_tag}":
                               submit_merge(sd, bd, aa, od, nm)))
            sdeps = [premerge_id] if premerge_id else []
            # eval the starting model (3 evals)
            for ev, dfn, sfn in (("fcsale", fcsale_done, submit_eval_fcsale),
                                 ("theta", theta_done, submit_eval_theta),
                                 ("ttt", ttt_done, submit_eval_ttt)):
                add(f"eval:{ev}:{start_tag}", kind="eval", gpu=True, deps=list(sdeps),
                    done_fn=(lambda t=start_tag, f=dfn: f(t)),
                    submit_fn=(lambda md=start_dir, t=start_tag, s=sfn, nm=f"oe-{ev}-{start_tag}": s(md, t, nm)))
            # SFT on each dataset
            for dk, ds in DATASETS.items():
                sft_tag = f"{fam}_{at}_{dk}"
                sft_dir = f"{SFT_OUT}/sft_{sft_tag}"
                sft_id = f"sft:{sft_tag}"
                add(sft_id, kind="sft", gpu=True, deps=list(sdeps),
                    done_fn=(lambda d=sft_dir: model_done(d)),
                    submit_fn=(lambda md=start_dir, dd=ds, od=sft_dir, nm=f"os-{sft_tag}": submit_sft(md, dd, od, nm)))
                for ev, dfn, sfn in (("fcsale", fcsale_done, submit_eval_fcsale),
                                     ("theta", theta_done, submit_eval_theta),
                                     ("ttt", ttt_done, submit_eval_ttt)):
                    add(f"eval:{ev}:{sft_tag}", kind="eval", gpu=True, deps=[sft_id],
                        done_fn=(lambda t=sft_tag, f=dfn: f(t)),
                        submit_fn=(lambda md=sft_dir, t=sft_tag, s=sfn, nm=f"oe-{ev}-{sft_tag}": s(md, t, nm)))
                # postmerge soups + their evals
                for pa in POSTMERGE_ALPHAS:
                    pt = alpha_tag(pa)
                    soup_tag = f"{sft_tag}_soup{pt}"
                    soup_dir = f"{SFT_OUT}/soup_{soup_tag}"
                    soup_id = f"merge:soup_{soup_tag}"
                    add(soup_id, kind="merge", gpu=False, deps=[sft_id],
                        done_fn=(lambda d=soup_dir: model_done(d)),
                        submit_fn=(lambda sd=sft_dir, bd=start_dir, aa=pa, od=soup_dir, nm=f"om-{soup_tag}":
                                   submit_merge(sd, bd, aa, od, nm)))
                    for ev, dfn, sfn in (("fcsale", fcsale_done, submit_eval_fcsale),
                                         ("theta", theta_done, submit_eval_theta),
                                         ("ttt", ttt_done, submit_eval_ttt)):
                        add(f"eval:{ev}:{soup_tag}", kind="eval", gpu=True, deps=[soup_id],
                            done_fn=(lambda t=soup_tag, f=dfn: f(t)),
                            submit_fn=(lambda md=soup_dir, t=soup_tag, s=sfn, nm=f"oe-{ev}-{soup_tag}": s(md, t, nm)))
    return T

# ---- controller -----------------------------------------------------------
def my_active_jobids():
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", "bl3615"), "-h", "-o", "%i %t"],
                         capture_output=True, text=True)
    ids = {}
    for ln in out.stdout.splitlines():
        p = ln.split()
        if len(p) >= 2: ids[p[0].split("_")[0]] = p[1]
    return ids

def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}
def save_state(s):
    json.dump(s, open(STATE, "w"), indent=1)

def main():
    T = build_tasks()
    if "--plan" in sys.argv:
        kinds = {}
        for t in T.values(): kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
        log(f"PLAN: {len(T)} tasks -> {kinds}")
        for tid in sorted(T): print("  ", tid, "deps=", T[tid]["deps"])
        return
    state = load_state()  # tid -> {jobid, status}
    log(f"orchestrator start: {len(T)} tasks; MAX_GPU={MAX_GPU} MAX_CPU={MAX_CPU}")
    while True:
        active = my_active_jobids()
        # reconcile + count
        gpu_in = cpu_in = 0
        done = 0
        for tid, t in T.items():
            st = state.get(tid, {})
            if t["done_fn"]():
                if st.get("status") != "done": st["status"] = "done"; state[tid] = st
                done += 1; continue
            jid = st.get("jobid")
            if jid and jid in active:
                (gpu_in := gpu_in + 1) if t["gpu"] else (cpu_in := cpu_in + 1)
                st["status"] = "running"; state[tid] = st
            elif jid and st.get("status") in ("submitted", "running"):
                # job left queue but output absent -> failed; allow retry (cap retries)
                st["retries"] = st.get("retries", 0) + 1
                st["status"] = "failed" if st["retries"] < RETRY_CAP else "dead"
                st["last_fail"] = time.time()
                state[tid] = st
            elif st.get("status") == "dead" and (time.time() - st.get("last_fail", 0)) > DEAD_COOLDOWN:
                # SELF-HEAL: a dead task is resurrected after a cooldown so a transient
                # failure (or a since-fixed bug) can never permanently stall the matrix.
                st.update(status="pending", retries=0, jobid=None)
                state[tid] = st
                log(f"  RESURRECT dead task {tid} (cooldown elapsed)")
        # submit ready tasks under caps
        for tid, t in T.items():
            st = state.get(tid, {})
            if st.get("status") in ("done", "running", "dead"): continue
            if st.get("jobid") and st.get("jobid") in active: continue
            if not all(T[d]["done_fn"]() for d in t["deps"]): continue
            if t["gpu"] and gpu_in >= MAX_GPU: continue
            if (not t["gpu"]) and cpu_in >= MAX_CPU: continue
            jid = t["submit_fn"]()
            if jid:
                st.update(jobid=jid, status="submitted"); state[tid] = st
                (gpu_in := gpu_in + 1) if t["gpu"] else (cpu_in := cpu_in + 1)
                log(f"  submit {tid} -> job {jid}  (gpu_in={gpu_in} cpu_in={cpu_in})")
        save_state(state)
        log(f"progress: {done}/{len(T)} done | gpu_in={gpu_in} cpu_in={cpu_in}")
        if done == len(T):
            log("ALL TASKS DONE"); return
        time.sleep(POLL)

if __name__ == "__main__":
    main()
