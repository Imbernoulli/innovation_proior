import sys, random

# 10 instances: pairwise-coprime prime-power triples (2^a, 3^b, 5^c or 7^c),
# ordered small->large product; density falls (harder / sparser) as tid grows.
TRIPLES = [
    [512, 729, 343],    # 128,024,064
    [512, 2187, 125],   # 139,968,000
    [512, 729, 625],    # 233,280,000
    [1024, 729, 343],   # 256,048,128
    [1024, 2187, 125],  # 279,936,000
    [2048, 243, 625],   # 311,040,000
    [512, 2187, 343],   # 384,072,192
    [1024, 729, 625],   # 466,560,000
    [2048, 729, 343],   # 512,096,256
    [1024, 2187, 343],  # 768,144,384
]
DENS = [0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.43, 0.42, 0.42]
K = 60

def main():
    tid = int(sys.argv[1])
    idx = (tid - 1) % 10
    rng = random.Random(500000 + tid)
    ms = TRIPLES[idx]
    dens = DENS[idx]
    out = []
    out.append("3")
    out.append(" ".join(map(str, ms)))
    out.append(str(K))
    for m in ms:
        s = max(2, int(dens * m))
        vals = set(rng.sample(range(m), s))
        vals.add(0)                      # 0 in every A_i -> 0 in T (self-beat counts once)
        vals = sorted(vals)
        out.append(str(len(vals)))
        out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
