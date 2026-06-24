import subprocess, sys

SOL = "/tmp/cpv4-dp-knapsack-greedytrap_sol"

def run(cmd, inp=None):
    return subprocess.run(cmd, input=inp, capture_output=True, text=True).stdout.strip()

total = 0
mismatch = 0
first_bad = None
for gen in ["gen.py", "gen_big.py"]:
    for s in range(1, 501):
        inp = run(["python3", gen, str(s)])
        a = run([SOL], inp=inp)
        b = run(["python3", "brute.py"], inp=inp)
        total += 1
        if a != b:
            mismatch += 1
            if first_bad is None:
                first_bad = (gen, s, a, b, inp)

print(f"TOTAL={total} MISMATCHES={mismatch}")
if first_bad:
    g, s, a, b, inp = first_bad
    print("FIRST BAD:", g, s, "sol=", a, "brute=", b)
    print("INPUT:\n" + inp)
