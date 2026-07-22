# TIER: trivial
# Do-nothing baseline: assume the leak is LINEAR in level (alpha forced to 1)
# and fit the single proportionality constant k from the RAW, uncorrected
# daily deltas -- exactly the checker's own baseline construction. Ignores
# both the exponent identification and the conservation correction.
import sys

T_DAY = 1.0


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.001*L")
        return
    D = int(data[0])
    L0 = float(data[2])
    rows = []
    idx = 3
    for _ in range(D):
        Q = float(data[idx]); E = float(data[idx + 1]); idx += 2
        rows.append((Q, E))

    E_prev = L0
    ks = []
    for Q, E in rows:
        raw = (E_prev - E) / T_DAY
        Lref = (E_prev + E) / 2.0
        if raw > 1e-6 and Lref > 1e-6:
            ks.append(raw / Lref)
        E_prev = E
    k = sum(ks) / len(ks) if ks else 1e-4
    if k < 1e-4:
        k = 1e-4
    print("%.8f * L" % k)


if __name__ == "__main__":
    main()
