import sys
from sensorfield import build_instance


def main():
    test_id = int(sys.argv[1])
    inst = build_instance(test_id)
    n, T, F_max = inst["n"], inst["T"], inst["F_max"]
    edges = inst["edges"]
    R = inst["R"]

    out = []
    out.append(str(test_id))
    out.append(f"{n} {T} {F_max}")
    out.append(str(len(edges)))
    for (u, v) in edges:
        out.append(f"{u} {v}")
    for i in range(n):
        out.append(" ".join(f"{R[i][t]:.5f}" for t in range(T)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
