import subprocess
SOL = "/tmp/cpv4-greedy-exchange-negzero_sol"

def run(inp):
    return subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()

cases = [
    # (input, expected, label)
    ("5\n2 1 2 1 3\n20 10 40 30 50\n", "120", "documented sample"),
    ("0\n\n\n", "0", "empty n=0"),
    ("1\n3\n-7\n", "0", "single negative gig"),
    ("1\n3\n0\n", "0", "single zero gig"),
    ("1\n0\n100\n", "0", "positive but deadline 0 (no valid day)"),
    ("1\n1\n100\n", "100", "single positive deadline 1"),
    ("3\n1 1 1\n5 6 7\n", "7", "all deadline 1, keep best"),
    ("4\n-1 -2 -3 -4\n5 5 5 5\n".replace("-1 -2 -3 -4","0 0 0 0"), "0", "all deadline 0"),
    ("3\n2 2 2\n10 20 -5\n", "30", "negative ignored, two fit"),
    ("4\n1 2 2 1\n10 5 5 10\n", "15", "ties on deadline-1 day"),
]
ok = True
for inp, exp, label in cases:
    got = run(inp)
    status = "OK" if got == exp else "FAIL"
    if got != exp: ok = False
    print(f"[{status}] {label}: got={got} exp={exp}")
print("ALL PASS" if ok else "SOME FAILED")
