import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/cpv4-strings-kmp-negzero_sol"

def canon(s):
    # Normalize: split into tokens line by line, drop trailing empty lines.
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    # First line = count; remaining tokens (order matters) = positions.
    if not lines:
        return ("", [])
    count = lines[0].strip()
    rest = " ".join(lines[1:]).split()
    return (count, rest)

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fails = 0
    for s in range(start, start + N):
        inp = subprocess.run([sys.executable, os.path.join(HERE, "gen.py"), str(s)],
                             capture_output=True, text=True).stdout
        o1 = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout
        o2 = subprocess.run([sys.executable, os.path.join(HERE, "brute.py")],
                            input=inp, capture_output=True, text=True).stdout
        if canon(o1) != canon(o2):
            fails += 1
            if fails <= 6:
                print("MISMATCH seed=", s)
                print("--in--\n" + inp)
                print("--sol--\n" + o1)
                print("--brute--\n" + o2)
                print("canon sol:", canon(o1))
                print("canon brute:", canon(o2))
    print("TOTAL=%d FAILS=%d" % (N, fails))

if __name__ == "__main__":
    main()
