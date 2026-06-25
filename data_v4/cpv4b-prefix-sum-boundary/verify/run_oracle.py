import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/cpv4b-prefix-sum-boundary_sol"

def main():
    ncases = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    mism = 0
    for seed in range(1, ncases + 1):
        inp = subprocess.run([sys.executable, os.path.join(HERE, "gen.py"), str(seed)],
                             capture_output=True, text=True).stdout
        out_sol = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
        out_brute = subprocess.run([sys.executable, os.path.join(HERE, "brute.py")],
                                   input=inp, capture_output=True, text=True).stdout.strip()
        if out_sol != out_brute:
            mism += 1
            print(f"MISMATCH seed={seed} sol={out_sol!r} brute={out_brute!r}")
            print("INPUT:\n" + inp)
            if mism >= 10:
                break
    print(f"CASES={ncases} MISMATCHES={mism}")

main()
