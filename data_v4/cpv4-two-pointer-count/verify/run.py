import subprocess, sys, importlib.util, io

# Load gen and brute as modules by exec, but simpler: call them as subprocesses.
SOL = "/tmp/cpv4-two-pointer-count_sol"
GEN = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-two-pointer-count/verify/gen.py"
BRUTE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-two-pointer-count/verify/brute.py"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
fails = 0
shown = 0
for s in range(1, N + 1):
    inp = subprocess.run([sys.executable, GEN, str(s)], capture_output=True, text=True).stdout
    sol = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
    bru = subprocess.run([sys.executable, BRUTE], input=inp, capture_output=True, text=True).stdout.strip()
    if sol != bru:
        fails += 1
        if shown < 8:
            shown += 1
            print(f"MISMATCH seed={s} sol=[{sol}] bru=[{bru}]")
            print("input repr:", repr(inp))
print(f"TOTAL FAILS: {fails} / {N}")
