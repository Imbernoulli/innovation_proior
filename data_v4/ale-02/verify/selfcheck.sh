#!/usr/bin/env bash
# Self-verify: for seeds 1..20, generate, run solver+baseline, score both,
# confirm feasibility (score>0) and that the solver's mean strictly beats the
# first-fit baseline mean (and that empty output scores 0).
set -e
cd "$(dirname "$0")"
D=$(mktemp -d)
sumS=0; sumB=0; n=0; feasok=1; beateach=0
printf "%-5s %8s %8s %8s %7s\n" seed sol base empty ratio
for s in $(seq 1 20); do
  python3 gen.py "$s" > "$D/in_$s.txt"
  ./sol      < "$D/in_$s.txt" > "$D/out_$s.txt" 2>/dev/null
  ./baseline < "$D/in_$s.txt" > "$D/base_$s.txt" 2>/dev/null
  : > "$D/empty_$s.txt"
  S=$(python3 score.py "$D/in_$s.txt" "$D/out_$s.txt")
  B=$(python3 score.py "$D/in_$s.txt" "$D/base_$s.txt")
  E=$(python3 score.py "$D/in_$s.txt" "$D/empty_$s.txt")
  R=$(python3 -c "print(f'{$S/$B:.3f}')" 2>/dev/null || echo NA)
  printf "%-5s %8s %8s %8s %7s\n" "$s" "$S" "$B" "$E" "$R"
  if [ "$S" -le 0 ]; then feasok=0; fi
  if [ "$S" -gt "$B" ]; then beateach=$((beateach+1)); fi
  sumS=$((sumS+S)); sumB=$((sumB+B)); n=$((n+1))
done
echo "----"
python3 -c "print(f'mean sol={$sumS/$n:.2f}  mean base={$sumB/$n:.2f}  ratio={$sumS/$sumB:.4f}  beat_each={$beateach}/$n  feasible_all={$feasok}')"
rm -rf "$D"
