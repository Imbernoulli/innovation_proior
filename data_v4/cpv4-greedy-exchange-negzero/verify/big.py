import subprocess, time, random
SOL = "/tmp/cpv4-greedy-exchange-negzero_sol"
# Max stress: n = 2*10^5, deadlines large, payouts near 10^9 -> sum ~ 2*10^14
n = 200000
rng = random.Random(99)
d = [rng.randint(1, n) for _ in range(n)]
v = [rng.randint(1, 10**9) for _ in range(n)]
inp = f"{n}\n" + " ".join(map(str, d)) + "\n" + " ".join(map(str, v)) + "\n"
t = time.time()
out = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
print("time(s)=", round(time.time()-t, 3), "answer=", out)
print("answer > 2^31?", int(out) > 2**31)
# all deadlines = 1 -> only one gig can ever be scheduled (worst-case DSU path)
d2 = [1]*n
inp2 = f"{n}\n" + " ".join(map(str, d2)) + "\n" + " ".join(map(str, v)) + "\n"
t = time.time()
out2 = subprocess.run([SOL], input=inp2, capture_output=True, text=True).stdout.strip()
print("all-deadline-1 time(s)=", round(time.time()-t, 3), "answer=", out2, "max v=", max(v))
