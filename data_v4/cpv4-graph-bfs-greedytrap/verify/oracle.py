import subprocess, sys, importlib.util, io, random, os

VDIR = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/cpv4-graph-bfs-greedytrap_sol"

# import brute.main as a function operating on stdin/stdout strings
def run_brute(inp):
    # exec brute in a subprocess of THIS python (no shell re-init), feeding inp
    p = subprocess.run([sys.executable, os.path.join(VDIR, "brute.py")],
                       input=inp, capture_output=True, text=True)
    return p.stdout.strip(), p.stderr

def run_sol(inp):
    p = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    return p.stdout.strip(), p.stderr

def gen_input(genfile, seed):
    p = subprocess.run([sys.executable, os.path.join(VDIR, genfile), str(seed)],
                       capture_output=True, text=True)
    return p.stdout

def main():
    gens = ["gen.py", "gen2.py", "gen3.py"]
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    total = 0
    mism = 0
    for genfile in gens:
        for seed in range(1, per + 1):
            inp = gen_input(genfile, seed)
            # sanity: input must be all-numeric tokens
            toks = inp.split()
            if not all(t.lstrip("-").isdigit() for t in toks):
                print("BAD GEN OUTPUT", genfile, seed, repr(inp[:80]))
                continue
            a, ae = run_sol(inp)
            b, be = run_brute(inp)
            total += 1
            if be.strip():
                print("BRUTE ERR", genfile, seed, be.strip()[:200])
                mism += 1
            elif a != b:
                mism += 1
                print("MISMATCH", genfile, seed, "sol=", a, "brute=", b)
                print("INPUT:\n" + inp)
                if mism >= 6:
                    print("TOTAL", total, "MISM", mism); return
    print("TOTAL", total, "MISMATCHES", mism)

main()
