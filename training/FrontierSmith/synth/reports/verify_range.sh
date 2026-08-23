#!/bin/bash
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/synth
verify_one(){ d="problems/$1"; [ -d "$d" ] || { echo "MISSING $1"; return; }
  if [ -f "$d/evaluator.py" ]; then H=validate_pyproblem.py; else H=validate_problem.py; fi
  v=$(python3 harness/$H "$d" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['verdict'])" 2>/dev/null||echo ERR); echo "$v $1"; }
export -f verify_one
printf '%s\n' "$@" | xargs -P 8 -I{} bash -c 'verify_one "$@"' _ {} | sort
