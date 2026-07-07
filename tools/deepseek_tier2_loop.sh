#!/bin/bash
# Continuous DeepSeek V4 Pro tier-2: keep rescuing the 27B's GENUINE hard-failures as they
# accumulate. Each cycle -> (1) rebuild failed_worklist.jsonl per domain from the current traces
# (genuine failures = passed False & not dropped_easy & no error, joined to problem content),
# (2) run the deepseek driver which RESUMES (skips ids already in *.deepseek.jsonl) so only FRESH
# failures get solved, (3) sleep, repeat. Self-healing, singleton-locked. Math ungradeable gold is
# judged by DeepSeek V4 Flash inside the verifier.
set -u
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
REPO=/srv/home/bohanlyu/innovation_proior
VENV=/srv/home/bohanlyu/sesl/.venv
LOG="$SC/deepseek_tier2_loop.log"
CYCLE_SLEEP=${CYCLE_SLEEP:-600}      # wait between refresh cycles

export TMPDIR="$SC"
export CUDA_VISIBLE_DEVICES=
mkdir -p "$SC"

LOCKDIR="$SC/deepseek_tier2_loop.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[ds-loop $(date -u)] another deepseek_tier2_loop already running; exiting" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
trap 'exit 0' INT TERM

source "$VENV/bin/activate" || exit 1
cd "$REPO" || exit 1

rebuild_worklists() {
  python3 - <<'PY'
import json, os, glob
BASE='data_v4/_hardcp/traces'
def gf(r): return (not r.get('passed')) and (not r.get('dropped_easy')) and (r.get('error') is None)
tot=0
for d in ('math','code','reasoning','ifollow'):
    failed=set()
    for f in [f'{BASE}/{d}.jsonl']+sorted(glob.glob(f'{BASE}/{d}.*.jsonl')):
        if 'deepseek' in f or 'poe' in f or not os.path.exists(f): continue
        for l in open(f):
            l=l.strip()
            if not l: continue
            try: r=json.loads(l)
            except: continue
            if gf(r): failed.add(r['id'])
    wl=f'data_v4/_hardcp/{d}/worklist.jsonl'
    by_id={json.loads(l)['id']:json.loads(l) for l in open(wl) if l.strip()} if os.path.exists(wl) else {}
    have=[by_id[i] for i in failed if i in by_id]
    with open(f'data_v4/_hardcp/{d}/failed_worklist.jsonl','w') as w:
        for p in have: w.write(json.dumps(p)+'\n')
    tot+=len(have)
print(f'  failed-worklist total: {tot}')
PY
}

echo "[ds-loop $(date -u)] START continuous tier-2 (cycle sleep ${CYCLE_SLEEP}s)" >> "$LOG"
while true; do
  echo "[ds-loop $(date -u)] cycle: rebuild failed worklists" >> "$LOG"
  rebuild_worklists >> "$LOG" 2>&1
  echo "[ds-loop $(date -u)] cycle: run deepseek driver (resume skips done)" >> "$LOG"
  python tools/hardcp_rollout.py \
    --backend deepseek --worklist failed_worklist.jsonl \
    --domains math code reasoning ifollow \
    --concurrency 16 --query-concurrency 16 \
    --max-budget 8 --max-tokens 32000 --easy-threshold 1.1 --request-timeout 1800 \
    >> "$LOG" 2>&1
  echo "[ds-loop $(date -u)] cycle done; sleep ${CYCLE_SLEEP}s" >> "$LOG"
  sleep "$CYCLE_SLEEP"
done
