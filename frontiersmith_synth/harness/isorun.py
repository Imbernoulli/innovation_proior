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

If bwrap is unavailable, run_candidate falls back to a plain subprocess and reports sandbox="none";
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
# hide the whole synth tree (all problem/judge source) from the sandbox
SYNTH_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


def sandbox_available():
    return _BWRAP is not None


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
    tmpdir = tempfile.mkdtemp(prefix="isorun_")      # OUTSIDE synth, so bwrap can bind it in
    try:
        cand_tmp = os.path.join(tmpdir, "cand_src.py")
        shutil.copyfile(candidate_path, cand_tmp)
        if _BWRAP:
            cmd = _bwrap_cmd(cand_tmp, py)
            env = dict(SANDBOX_ENV)                   # scrubbed env (bwrap here lacks --clearenv)
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
            # preserving source-tree and /proc isolation.
            if _BWRAP and p.returncode != 0 and _bwrap_netns_failed(p.stderr):
                p = subprocess.run(_bwrap_cmd(cand_tmp, py, unshare_net=False),
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
    print(f"isorun self_check OK (sandbox={'bwrap' if _BWRAP else 'NONE'})")


if __name__ == "__main__":
    self_check()
