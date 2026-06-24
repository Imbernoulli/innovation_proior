import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/cpv4-dijkstra-overflow_sol"
GEN = os.path.join(HERE, "gen.py")
BRUTE = os.path.join(HERE, "brute.py")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
fails = 0
for s in range(1, N + 1):
    inp = subprocess.run([sys.executable, GEN, str(s)], capture_output=True, text=True).stdout
    a = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
    b = subprocess.run([sys.executable, BRUTE], input=inp, capture_output=True, text=True).stdout.strip()
    if a != b:
        fails += 1
        if fails <= 5:
            print(f"MISMATCH seed={s} sol=[{a}] brute=[{b}]")
            print("--- input ---")
            print(inp)
            print("-------------")
print(f"TOTAL={N} FAILS={fails}")
sys.exit(1 if fails else 0)
