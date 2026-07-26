# TIER: strong
"""Composed resolvent (rational) sieve: sign(A - theta I) is approximated
as B * sum_i w_i (B^2 + c_i I)^-1 with B = A - theta I, i.e. a SUM of a few
cheap shifted resolvents (each a single CSOLVE, exploiting that A is
banded so a shifted solve costs about the same order as a matvec) rather
than one flat high-degree polynomial. The number of terms needed to hit a
target epsilon grows only ~log(1/gap) (best rational approximation of
x^-1/2 on [gap^2, 1]) instead of the ~1/gap degree a flat polynomial needs
-- exploiting the revealed spectral gap through composition of a matvec
with a small resolvent sum, several-fold cheaper than the flat filter on
tight-gap instances.
"""
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def read_input():
    data = sys.stdin.read().split("\n")
    data = [t for t in data if t.strip() != ""]
    idx = 0
    n = int(data[idx]); idx += 1
    theta, eps, b_bw = data[idx].split(); idx += 1
    theta = float(theta); eps = float(eps)
    gap = float(data[idx]); idx += 1
    nnz = int(data[idx]); idx += 1
    A = np.zeros((n, n))
    for _ in range(nnz):
        i, j, v = data[idx].split(); idx += 1
        i = int(i); j = int(j); v = float(v)
        A[i, j] = v; A[j, i] = v
    return n, theta, eps, gap, A


def find_shifts(theta, gap, eps, Lam):
    B_eigs = Lam - theta
    a2 = gap * gap
    xs = np.geomspace(a2, 1.0, 4000)
    target = xs ** -0.5
    target_sign = np.sign(B_eigs)
    for m in range(1, 41):
        c = np.geomspace(a2 * 0.5, 2.0, m)
        Amat = 1.0 / (xs[:, None] + c[None, :])
        w, *_ = np.linalg.lstsq(Amat, target, rcond=None)
        sign_approx = B_eigs * np.sum(
            w[None, :] / (B_eigs[:, None] ** 2 + c[None, :]), axis=1)
        err = np.max(np.abs(sign_approx - target_sign))
        if err <= eps * 0.5:
            return c, w
    return c, w  # best effort


def emit_program(theta, c_list, w_list):
    # regs: 0=v(input) 1=tmp(csolve out) 2=total 3=final
    lines = ["NVEC 4"]
    lines.append("SCALE 2 0.0 0")  # total = 0
    for ci, wi in zip(c_list, w_list):
        s = float(np.sqrt(ci))
        lines.append("CSOLVE 1 %.17g %.17g 0" % (theta, s))
        lines.append("AXPBY 2 1.0 2 %.17g 1" % wi)
    lines.append("AXPBY 3 0.5 0 0.5 2")
    lines.append("OUTPUT 3")
    sys.stdout.write("\n".join(lines) + "\n")


def main():
    n, theta, eps, gap, A = read_input()
    Lam = np.linalg.eigvalsh(A)
    c, w = find_shifts(theta, gap, eps, Lam)
    emit_program(theta, c, w)


if __name__ == "__main__":
    main()
