"""
isorun.py -- OS-sandboxed candidate runner for Format-B ("evolve a heuristic") problems.

THREAT MODEL: the candidate is untrusted, possibly-adversarial model output. A plain in-process
call lets it walk the Python stack (`sys._getframe().f_back`) to steal the evaluator's closures.
A plain SUBPROCESS is still not enough: on a shared host the child can read `/proc/<judge-pid>/mem`,
`/proc/<judge-pid>/cmdline`, and the co-located judge SOURCE (gen.py / labels / laws) off the
filesystem, then regenerate the hidden answer. (All demonstrated by adversarial review.)

FIX: run the candidate under **bubblewrap (bwrap)** in fresh user/pid/net/ipc/uts/mount namespaces:
  - `--tmpfs <SYNTH_ROOT>`  hides the entire problem tree (gen.py, checker, labels, seeds, laws) ->
    the candidate cannot read any ground-truth source.
  - `--unshare-pid --proc /proc`  gives the sandbox its OWN /proc -> the judge process is invisible,
    so `/proc/<ppid>/mem` and cmdline leaks die.
  - `--unshare-net`  no network; `--clearenv`  scrubbed env; `python3 -I -- /tmp/cand.py`  no flag
    injection, no env/site influence.
The candidate communicates ONLY via a strict text protocol: ONE JSON "public instance" on stdin ->
ONE JSON answer on stdout. Hidden data / answer / oracle state live only in the parent (the judge),
which the sandbox cannot reach.

FALLBACK BACKEND (della 2026-07: user.max_user_namespaces=0 permanently -> bwrap is dead):
**apptainer** (setuid starter, no userns needed) reproduces the same guarantees:
  - `--containall` -> the container sees ONLY the SIF filesystem + our explicit binds; the system
    bind-paths (/scratch, /projects, ...) are NOT mounted, so the synth tree is invisible. We bind
    the host /usr,/lib64,/bin (+ld.so.cache) OVER the SIF so the *host* interpreter runs unchanged,
    plus ONLY the interpreter's env prefix (ro) -- never an ancestor of SYNTH_ROOT -- and, as
    defense-in-depth, an empty dir is bind-masked over SYNTH_ROOT itself.
  - `--pid` own /proc; `--net --network none` no network (verified allowed unprivileged in setuid
    mode on della); `--cleanenv` + APPTAINERENV_* scrubbed env; same `python3 -I` protocol.
Backend selection: env ISORUN_BACKEND = auto (default: bwrap if a 1-shot probe passes, else
apptainer, else none) | bwrap | apptainer | none. Resolved once per process and cached.

If no backend works, run_candidate falls back to a plain subprocess and reports sandbox="none";
the harness's G5c gate then FAILS the problem (it detects the /proc + source-read leak), so an
unsandboxed environment cannot silently pass.

Contract for a candidate program `cand.py`:
    import sys, json
    inst = json.load(sys.stdin)          # the public instance
    ...compute...
    print(json.dumps(answer))            # the ONLY thing the evaluator reads
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_BWRAP = shutil.which("bwrap")
_APPTAINER = shutil.which("apptainer")
# hide the whole synth tree (all problem/judge source) from the sandbox
SYNTH_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)

# Any SIF works as the base image: we bind the host /usr,/lib64,/bin over it, so the
# container runs host binaries (glibc/loader coherence verified empirically on della).
_SIF_CANDIDATES = [
    os.environ.get("ISORUN_SIF") or "",
    "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.cache/apptainer/frontiercs-judge.sif",
    "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.cache/apptainer/alebench/ale-bench_cpp17-202301.sif",
    "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.cache/apptainer/alebench/rust_1.79.0-buster.sif",
]
# host pieces the *host* interpreter/toolchain output needs inside --containall
_APPTAINER_SYS_BINDS = ["/usr", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/alternatives"]

_BACKEND = None  # resolved once per process ("bwrap" | "apptainer" | "none")


def _find_sif():
    for s in _SIF_CANDIDATES:
        if s and os.path.exists(s):
            return s
    return None


def _py_prefix_binds():
    """Env prefixes the running interpreter needs, bound ro. CAUTION honored here: bind ONLY
    the env prefix (venv/conda dir), never an ancestor -- if the env lives beside SYNTH_ROOT
    (e.g. .../fs/envs/sft_lf vs .../fs/innovation_prior), an ancestor bind would expose the
    synth tree. A mask bind over SYNTH_ROOT (added last in _apptainer_cmd) backstops this."""
    exe = os.path.realpath(sys.executable or "/usr/bin/python3")
    cands = [sys.prefix, sys.base_prefix, os.path.dirname(os.path.dirname(exe))]
    out = []
    for p in cands:
        rp = os.path.realpath(p) if p else ""
        if rp and rp != "/" and not rp.startswith("/usr") and rp not in out:
            out.append(rp)
    return out


def _apptainer_cmd(inner, binds=(), mask_dir=None, workdir="/tmp", net_ns=True):
    """Build an `apptainer exec` argv running `inner` with the harness's isolation contract:
    own /proc (--pid), no host filesystem beyond explicit ro binds (--containall suppresses
    the cluster's /scratch bind-paths), scrubbed env (--cleanenv + APPTAINERENV_*), and --
    when permitted -- an empty network namespace."""
    cmd = [_APPTAINER, "exec", "--containall", "--cleanenv", "--no-home", "--pid",
           "--pwd", workdir]
    if net_ns:
        cmd += ["--net", "--network", "none"]
    for b in _APPTAINER_SYS_BINDS:
        if os.path.exists(b):
            cmd += ["--bind", f"{b}:{b}:ro"]
    for p in _py_prefix_binds():
        cmd += ["--bind", f"{p}:{p}:ro"]
    for src, dst in binds:
        cmd += ["--bind", f"{src}:{dst}:ro"]
    if mask_dir:  # LAST bind wins: empty dir over SYNTH_ROOT even if a prefix bind exposed it
        cmd += ["--bind", f"{mask_dir}:{SYNTH_ROOT}:ro"]
    return cmd + [_find_sif()] + list(inner)


def _apptainer_env():
    """Host-side env for the apptainer CLI itself; the container only receives the
    APPTAINERENV_*-prefixed (scrubbed) variables because of --cleanenv."""
    e = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"),
         "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
    for k, v in SANDBOX_ENV.items():
        e["APPTAINERENV_" + k] = v
    return e


def _probe_bwrap():
    if not _BWRAP:
        return False
    try:  # 1-shot: the userns-disabled failure mode breaks ANY bwrap invocation
        r = subprocess.run([_BWRAP, "--dev-bind", "/", "/", "--unshare-pid",
                            "--proc", "/proc", "true"], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _probe_apptainer():
    if not (_APPTAINER and _find_sif()):
        return False
    try:  # real 1-shot exec with the same bind set we will use for candidates
        r = subprocess.run(_apptainer_cmd(["/bin/true"]), capture_output=True,
                           timeout=90, env=_apptainer_env())
        return r.returncode == 0
    except Exception:
        return False


def resolve_backend():
    """Resolve + cache the sandbox backend. ISORUN_BACKEND=auto|bwrap|apptainer|none.
    Forced values are still probed so sandbox_available() never lies to the G5c gate."""
    global _BACKEND
    if _BACKEND is None:
        want = os.environ.get("ISORUN_BACKEND", "auto").strip().lower()
        if want == "none":
            _BACKEND = "none"
        elif want == "bwrap":
            _BACKEND = "bwrap" if _probe_bwrap() else "none"
        elif want == "apptainer":
            _BACKEND = "apptainer" if _probe_apptainer() else "none"
        else:  # auto: bwrap preferred (cheaper, stricter netns), apptainer fallback
            _BACKEND = ("bwrap" if _probe_bwrap()
                        else "apptainer" if _probe_apptainer() else "none")
    return _BACKEND


def sandbox_available():
    return resolve_backend() != "none"


# minimal, scrubbed environment for the sandboxed candidate (this bwrap build lacks
# --clearenv/--setenv, so we scrub via subprocess env= instead).
SANDBOX_ENV = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
_BWRAP_NETNS_ERRORS = (
    "NETLINK_ROUTE",
    "CLONE_NEWNET",
    "network namespace",
    "Operation not permitted",
)


def _bwrap_netns_failed(stderr):
    return any(s in (stderr or "") for s in _BWRAP_NETNS_ERRORS)


def _bwrap_cmd(cand_in_tmp, py, unshare_net=True):
    cmd = [_BWRAP, "--dev-bind", "/", "/",
           "--tmpfs", SYNTH_ROOT,          # hide ALL judge/problem source (gen/checker/labels/laws)
           "--tmpfs", "/tmp",
           "--ro-bind", cand_in_tmp, "/tmp/cand.py",
           "--unshare-pid"]
    if unshare_net:
        cmd.append("--unshare-net")
    return cmd + ["--unshare-ipc", "--unshare-uts", "--proc", "/proc",
                  "--die-with-parent", "--new-session", "--chdir", "/tmp",
                  py, "-I", "--", "/tmp/cand.py"]


def run_candidate(candidate_path, public_instance, timeout=20):
    """Run candidate OS-sandboxed. Return (answer_obj|None, status)."""
    try:
        payload = json.dumps(public_instance)
    except (TypeError, ValueError) as e:
        return None, f"BAD_INSTANCE({e})"

    py = sys.executable or "python3"
    backend = resolve_backend()
    tmpdir = tempfile.mkdtemp(prefix="isorun_")      # OUTSIDE synth, so the sandbox can bind it in
    try:
        cand_tmp = os.path.join(tmpdir, "cand_src.py")
        shutil.copyfile(candidate_path, cand_tmp)
        if backend == "bwrap":
            cmd = _bwrap_cmd(cand_tmp, py)
            env = dict(SANDBOX_ENV)                   # scrubbed env (bwrap here lacks --clearenv)
        elif backend == "apptainer":
            mask = os.path.join(tmpdir, "synthmask")  # empty dir bind-masked over SYNTH_ROOT
            os.mkdir(mask)
            cmd = _apptainer_cmd([py, "-I", "--", "/tmp/cand.py"],
                                 binds=[(cand_tmp, "/tmp/cand.py")], mask_dir=mask)
            env = _apptainer_env()
        else:                                        # fallback: unsandboxed (harness G5c will fail it)
            cmd = [py, "-I", "--", cand_tmp]
            env = dict(os.environ)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                env.setdefault(k, "1")
        try:
            p = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                               timeout=timeout, env=env)
            # Some managed HPC/container nodes allow bwrap mount/pid namespaces but reject
            # creating a network namespace. Retry without netns for that platform error while
            # preserving source-tree and /proc isolation. (Same safety net for apptainer's
            # --net on clusters that disallow it; della allows `--network none` unprivileged.)
            if backend == "bwrap" and p.returncode != 0 and _bwrap_netns_failed(p.stderr):
                p = subprocess.run(_bwrap_cmd(cand_tmp, py, unshare_net=False),
                                   input=payload, capture_output=True, text=True,
                                   timeout=timeout, env=env)
            elif backend == "apptainer" and p.returncode != 0 and _bwrap_netns_failed(p.stderr):
                p = subprocess.run(_apptainer_cmd([py, "-I", "--", "/tmp/cand.py"],
                                                  binds=[(cand_tmp, "/tmp/cand.py")],
                                                  mask_dir=mask, net_ns=False),
                                   input=payload, capture_output=True, text=True,
                                   timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return None, "TLE"
        except Exception as e:
            return None, f"SPAWN_ERR({e})"
        if p.returncode != 0:
            return None, f"RE({p.returncode})"
        out = (p.stdout or "").strip()
        if not out:
            return None, "EMPTY"
        cands = [out]
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines:
            cands.append(lines[-1])
        for c in cands:
            try:
                return json.loads(c), "OK"
            except Exception:
                continue
        return None, "BADJSON"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def self_check():
    import tempfile as _t
    src = "import sys,json\ninst=json.load(sys.stdin)\nprint(json.dumps({'n':inst['n']*2}))\n"
    with _t.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    ans, st = run_candidate(path, {"n": 21})
    os.unlink(path)
    assert st == "OK" and ans == {"n": 42}, (st, ans)
    print(f"isorun self_check OK (sandbox={resolve_backend()})")


if __name__ == "__main__":
    self_check()
