import subprocess, os, sys
D = os.path.dirname(os.path.abspath(__file__))
def run(mod, stdin):
    return subprocess.run([sys.executable, os.path.join(D,mod)], input=stdin,
                          capture_output=True, text=True).stdout
tiers = ["trivial","greedy","strong","invalid"]
means = {t:0.0 for t in tiers}
rows = {t:[] for t in tiers}
for tid in range(1,11):
    ins = subprocess.run([sys.executable, os.path.join(D,"gen.py"), str(tid)],
                         capture_output=True, text=True).stdout
    fin = f"/tmp/in_{tid}.txt"; open(fin,"w").write(ins)
    for t in tiers:
        out = run(os.path.join("solutions",t+".py"), ins)
        fout=f"/tmp/out_{tid}_{t}.txt"; open(fout,"w").write(out)
        r = subprocess.run([sys.executable, os.path.join(D,"verify.py"), fin, fout, fin],
                           capture_output=True, text=True).stdout
        val = 0.0
        for line in r.strip().splitlines():
            if "Ratio:" in line:
                val = float(line.split("Ratio:")[1].split()[0])
        rows[t].append(round(val,3)); means[t]+=val/10
print("tid:      ", list(range(1,11)))
for t in tiers:
    print(f"{t:8s}", rows[t], "mean=%.4f"%means[t])
print("strong-greedy = %.4f (need >=0.06)"%(means['strong']-means['greedy']))
print("strong        = %.4f (need <=0.92)"%means['strong'])
print("greedy-trivial= %.4f (need >=0.03)"%(means['greedy']-means['trivial']))
