#!/bin/bash
# Re-verify every problem with the correct harness (auto-detect program vs stdout mode).
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/synth
verify_one() {
  d="$1"; id=$(basename "$d")
  if [ -f "$d/evaluator.py" ]; then
    out=$(python3 harness/validate_pyproblem.py "$d" 2>/dev/null)
  else
    out=$(python3 harness/validate_problem.py "$d" --keep-testdata 2>/dev/null)
  fi
  v=$(echo "$out" | python3 -c "import sys,json;print(json.load(sys.stdin)['verdict'])" 2>/dev/null || echo "ERR")
  echo "$v $id"
}
export -f verify_one
ls -d problems/*/ | xargs -P 8 -I{} bash -c 'verify_one "$@"' _ {} | sort
