import subprocess, sys, os, random
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/fcs-nt-01_x"
BRUTE = os.path.join(HERE, "brute.py")
GEN = os.path.join(HERE, "gen.py")

def run_sol(inp):
    return subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()

def run_brute(inp):
    return subprocess.run([sys.executable, BRUTE], input=inp, capture_output=True, text=True).stdout.strip()

def gen(seed):
    return subprocess.run([sys.executable, GEN, str(seed)], capture_output=True, text=True).stdout

def check(inp, label=""):
    s = run_sol(inp)
    b = run_brute(inp)
    if s != b:
        print("MISMATCH", label)
        print("INPUT:\n" + inp)
        print("SOL:", s, " BRUTE:", b)
        return False
    return True

ncases = int(sys.argv[1]) if len(sys.argv) > 1 else 500
fails = 0

# Explicit edge cases.
edges = [
    "1\n0 1\n",                 # x ≡ 0 (mod 1): every int works, smallest 0
    "1\n5 7\n",                 # single congruence
    "1\n-3 7\n",                # negative remainder -> normalize to 4
    "2\n2 4\n0 6\n",            # x≡2(4),x≡0(6): 2 mod gcd2 =>2-0=2 not div by 2? gcd=2,diff=2 ok
    "2\n2 4\n1 6\n",            # contradiction: diff=1 not div by gcd 2
    "2\n0 6\n0 8\n",            # both 0 -> x=0
    "3\n1 2\n2 3\n3 5\n",       # coprime CRT -> 23
    "3\n2 6\n8 12\n2 4\n",      # non-coprime consistent
    "2\n1000000000 1\n0 1\n",   # large remainder mod 1 -> 0
    "1\n999999999 1000000000\n",# single big
    "2\n0 2\n1 2\n",            # same modulus, conflicting -> -1
    "2\n1 2\n1 2\n",            # same modulus, same rem -> 1
    "4\n0 2\n0 3\n0 4\n0 5\n",  # all 0 -> 0
    "5\n1 2\n1 3\n1 4\n1 5\n1 6\n", # all ≡1 -> 1 (lcm 60)
]
for i, e in enumerate(edges):
    if not check(e, f"edge#{i}"):
        fails += 1

# Random fuzz.
for seed in range(ncases):
    inp = gen(seed)
    if not check(inp, f"seed={seed}"):
        fails += 1
        if fails > 5:
            break

print(f"DONE: {ncases} random + {len(edges)} edges, fails={fails}")
sys.exit(1 if fails else 0)
