import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/cpv4-prefix-sum-count_sol"
GEN = os.path.join(HERE, "gen.py")
BRUTE = os.path.join(HERE, "brute.py")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
mism = 0
total = 0
examples = []
for seed in range(1, N + 1):
    inp = subprocess.run([sys.executable, GEN, str(seed)], capture_output=True, text=True).stdout
    sol = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
    brute = subprocess.run([sys.executable, BRUTE], input=inp, capture_output=True, text=True).stdout.strip()
    total += 1
    if sol != brute:
        mism += 1
        if len(examples) < 8:
            examples.append((seed, sol, brute, inp))

print(f"TOTAL={total} MISMATCHES={mism}")
for seed, sol, brute, inp in examples:
    print(f"--- seed={seed} sol={sol!r} brute={brute!r}")
    print(inp.rstrip())
