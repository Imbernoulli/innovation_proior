import subprocess, sys, importlib.util, io, random

SOL = "/tmp/cpv4-prefix-sum-boundary_sol"
HERE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-prefix-sum-boundary/verify"

# import gen.main-like logic by exec'ing gen2.py's generation, but simplest: call gen.py as subprocess.
def gen(seed):
    return subprocess.run(["python3", f"{HERE}/gen2.py", str(seed)],
                          capture_output=True, text=True).stdout

def brute(inp):
    return subprocess.run(["python3", f"{HERE}/brute.py"],
                          input=inp, capture_output=True, text=True).stdout.strip()

def sol(inp):
    return subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mism = 0
    cases = 0
    for s in range(1, N + 1):
        inp = gen(s)
        if not inp.strip():
            continue
        cases += 1
        o1 = sol(inp)
        o2 = brute(inp)
        if o1 != o2:
            mism += 1
            print(f"MISMATCH seed={s} sol=[{o1}] brute=[{o2}]")
            print("input:\n" + inp)
            if mism >= 6:
                break
    print(f"CASES={cases} MISMATCHES={mism}")

if __name__ == "__main__":
    main()
