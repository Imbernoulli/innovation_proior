# TIER: invalid
# Emits a garbage artifact: a table entry of 0, which is never a unit mod p (gcd(0,p)=p),
# so it fails feasibility immediately regardless of the rest of the output.
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0
    p = int(data[pos]); pos += 1
    g = int(data[pos]); pos += 1
    LAMBDA = int(data[pos]); pos += 1
    T = int(data[pos]); pos += 1
    targets = [int(x) for x in data[pos:pos + T]]

    out = []
    out.append("1")
    out.append("0")
    for _ in range(T):
        out.append("1 1 1")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
