import subprocess, sys

SOL = "/tmp/cpv4b-dp-1d-negzero_sol"
GEN = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-1d-negzero/verify/gen.py"
BRUTE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-1d-negzero/verify/brute.py"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
mism = 0
shown = 0
for s in range(1, N + 1):
    inp = subprocess.run(["python3", GEN, str(s)], capture_output=True, text=True).stdout
    o1 = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
    o2 = subprocess.run(["python3", BRUTE], input=inp, capture_output=True, text=True).stdout.strip()
    if o1 != o2:
        mism += 1
        if shown < 8:
            shown += 1
            print(f"MISMATCH seed={s} sol={o1!r} brute={o2!r}")
            print("INPUT:", inp.strip())
print(f"TOTAL={N} MISMATCHES={mism}")
