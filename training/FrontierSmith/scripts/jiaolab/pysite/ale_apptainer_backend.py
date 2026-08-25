"""jiaolab wrapper-layer: ALE-Bench APPTAINER container backend (eager, pinned).

Activated when ``ALE_BENCH_CONTAINER_BACKEND=apptainer`` (see
pysite/sitecustomize.py, which auto-imports this module and calls ``install()``).
Replaces the ``ale_bench.utils.docker_client`` context manager with an
``apptainer exec`` sandbox, so the vendored ALE-Bench harness
(``run_compile_container`` / ``run_batch_run_container`` /
``run_batch_judge_container`` / ``run_reactive_judge_container`` /
``run_gen_container`` / ``build_rust_tools``) runs UNCHANGED.

Why this exists at all, given ALE-Bench already ships an apptainer backend
(``ale_bench.utils._ApptainerContainer``, the Princeton cluster patch):

  1. FAIL-LOUD.  The shipped backend is LAZY -- it only calls
     ``subprocess.run(["apptainer", ...])`` inside ``wait()``.  Every exception
     raised from ``wait()`` is caught by the harness and converted into a
     ``COMPILATION_ERROR`` CaseResult (see run_compile_container's
     ``except Exception:`` branch), i.e. a SILENT ZERO SCORE.  A missing
     ``apptainer`` binary, an unreadable SIF, a full /tmp or an OOM-killed
     starter would therefore be scored as "the model wrote code that does not
     compile".  This backend starts the sandbox EAGERLY inside
     ``containers.run()`` (exactly like the docker daemon, and like the gpublaze
     bwrap backend), so infra failures propagate out of ``run_cases`` ->
     ``session.private_eval`` -> ``compute_score`` and surface as
     ``AleInfraError``.
  2. 1-CPU PINNING.  The harness asks docker for ``cpu_quota=100000`` (1 CPU).
     The shipped apptainer backend drops that silently.  ALE-Bench scores are
     WALL-CLOCK sensitive (TLE + the rust tester's time budget), and jiaolab is a
     128-core box shared with another user, so an unpinned sandbox is both
     unfaithful and irreproducible.  Each sandbox leases one distinct core from a
     flock pool (same mechanism as scripts/gpublaze/pysite/ale_host_backend.py,
     so the two machines' judging differ only in the sandbox implementation).

Container-equivalence mapping (per docker flag used by the harness):
  image=ale-bench:<tag>      -> $ALE_BENCH_APPTAINER_DIR/ale-bench_<tag>.sif
                                (or yimjk_ale-bench_<tag>.sif), i.e. the SAME
                                image the docker backend would pull: g++ 12.2.0,
                                boost, ac-library.
  image=rust:<tag>           -> rust_<tag>.sif  (runs the prebuilt gen/tester/vis
                                binaries from .cache/ale-bench/rust-tool-builds).
  command                    -> exec'd verbatim (list form as-is, str form via
                                /bin/sh -c, exactly like the docker daemon's
                                shell-form Cmd and the shipped apptainer shim).
  volumes={h: {bind, mode}}  -> --bind h:bind:ro|rw
  working_dir                -> --pwd, plus the shipped shim's
                                _infer_workdir_source() rule (reproduced
                                verbatim) that binds a host dir over the workdir
                                when the harness only bound children of it.
  environment                -> --env K=V (plus the shim's CARGO_HOME/RUSTUP_HOME
                                defaults for rust:* images).
  cpu_quota=100000 (1 CPU)   -> taskset -c <core> around `apptainer exec`; the
                                affinity is inherited by everything in the
                                sandbox.  Core leased from a flock pool
                                (ALE_BENCH_HOST_CPUS, default: all cores).
  mem_limit=2GiB             -> OPT-IN only (ALE_APPTAINER_MEM_LIMIT=1) via
                                prlimit --as INSIDE the container.  Off by
                                default: apptainer's Go starter reserves a huge
                                virtual address space, so an outer RLIMIT_AS
                                kills the starter itself, and an inner one needs
                                util-linux in every SIF.  Scoring-relevant memory
                                enforcement is the harness's own max-RSS check
                                (parse_profiles), which is unaffected.
  network_disabled=True      -> NOT enforced: unprivileged `apptainer --net`
                                needs a setuid install, which jiaolab does not
                                have.  Same delta as the shipped Princeton
                                apptainer shim.  Documented in
                                docs/EVAL_ON_JIAOLAB_zh.md.
  user/group_add             -> n/a: apptainer already runs as the invoking user.
  detach/wait/logs/attrs     -> eager subprocess.Popen at containers.run() time;
                                wait()/logs()/attrs["State"]["ExitCode"]/remove()
                                mirror the docker SDK surface the harness uses.
  wait(timeout)              -> on expiry the sandbox process GROUP is SIGKILLed
                                and requests.exceptions.Timeout is raised, so the
                                harness's own `except (Timeout, ConnectionError)`
                                path classifies a hung compile exactly like
                                docker did.

Failure semantics (mirrors the docker backend's fail-loud contract):
  * apptainer missing, SIF missing/unreadable, spawn failure -> raised EAGERLY
    from containers.run(), i.e. BEFORE wait(); it propagates out of
    run_compile_container/run_cases -> session.private_eval -> compute_score,
    which wraps it as AleInfraError.  It is NEVER swallowed into a fake
    COMPILATION_ERROR (that branch only catches exceptions from wait()).
  * Model-side failures (CE/TLE/RE/WA/MLE) come back as ordinary CaseResults
    exactly like the docker path -- score 0, not an exception.

The one-shot compile selftest (verl/utils/reward_score/ale_selftest.py) goes
through run_compile_container, so with this backend installed it IS the
apptainer backend selftest (real g++ 12.2.0 compiling a trivial program in the
SIF).
"""

from __future__ import annotations

import fcntl
import importlib.abc
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _requests_timeout():
    """requests.exceptions.Timeout, imported lazily (this module is imported by
    sitecustomize in MANY interpreters; requests is only guaranteed where
    ale_bench is). The harness catches exactly this type on compile timeout."""
    try:
        from requests.exceptions import Timeout
    except ImportError:  # pragma: no cover - requests is an ale_bench dep
        class Timeout(OSError):  # noqa: N801 - mimic the requests type name
            pass
    return Timeout


_IMAGE_ALE_PREFIXES = ("ale-bench:", "yimjk/ale-bench:")


# --------------------------------------------------------------------------
# SIF / cpu-pool plumbing
# --------------------------------------------------------------------------

def _sanitize_image_name(image: str) -> str:
    """Same rule as ale_bench.utils._sanitize_image_name (SIF naming contract)."""
    return image.replace("/", "_").replace(":", "_")


def _sif_dir() -> Path:
    default = Path(os.environ.get("ALE_BENCH_CACHE", Path.home() / ".cache" / "ale-bench")) / "apptainer-images"
    return Path(os.environ.get("ALE_BENCH_APPTAINER_DIR", default)).expanduser()


def _sif_for_image(image: str) -> Path:
    """Locate the SIF for a docker image name; raise (eagerly) if absent."""
    image_dir = _sif_dir()
    candidates = [image_dir / f"{_sanitize_image_name(image)}.sif"]
    if image.startswith("ale-bench:"):
        tag = image.split(":", 1)[1]
        candidates.append(image_dir / f"yimjk_ale-bench_{tag}.sif")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    candidate_list = ", ".join(str(path) for path in candidates)
    msg = (
        f"ALE-Bench apptainer backend: SIF for image {image!r} not found. Expected one of: "
        f"{candidate_list}. Pull it once with:\n"
        f"  apptainer pull {candidates[0]} docker://{'yimjk/' + image if image.startswith('ale-bench:') else image}\n"
        "Refusing to score with a missing toolchain."
    )
    raise FileNotFoundError(msg)


def _parse_cpu_spec(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _pool_cpus() -> list[int]:
    spec = os.environ.get("ALE_BENCH_HOST_CPUS", "").strip()
    if spec:
        return _parse_cpu_spec(spec)
    n = os.cpu_count() or 1
    return list(range(n))


def _pool_dir() -> Path:
    return Path(os.environ.get("ALE_BENCH_HOST_CPU_POOL_DIR", "/dev/shm/ale-bench-apptainer-cpu-pool"))


def _acquire_cpu() -> tuple[int, int] | None:
    """Lease one distinct core from the flock pool; None => run unpinned.

    Fixed-size pool of 0-byte lockfiles under /dev/shm (reused across runs, so
    no per-run temp-file growth). The lease fd is released by closing it.
    """
    cpus = _pool_cpus()
    if not cpus:
        return None
    random.shuffle(cpus)
    pool = _pool_dir()
    try:
        pool.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    for cpu in cpus:
        try:
            fd = os.open(pool / f"cpu-{cpu}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        return cpu, fd
    # Pool exhausted (more concurrent sandboxes than cores): fall back to an
    # unpinned run rather than blocking the judge.
    return None


# --------------------------------------------------------------------------
# docker-SDK-compatible apptainer container
# --------------------------------------------------------------------------

class _ApptainerContainer:
    """Implements the subset of docker.models.containers.Container the
    ALE-Bench harness uses: wait(), logs(), attrs, remove(). The process is
    started EAGERLY (at containers.run()), matching docker's detach semantics;
    wait() merely collects it."""

    def __init__(
        self,
        image: str,
        command,
        volumes: dict | None = None,
        working_dir: str | None = None,
        environment: dict | None = None,
        mem_limit: int | None = None,
        network_disabled: bool = True,
        **_: object,
    ) -> None:
        self.image = image
        self.command = command
        self.volumes = volumes or {}
        self.working_dir = working_dir
        self.environment = environment or {}
        self.mem_limit = mem_limit
        self.attrs = {"State": {"ExitCode": None}}
        self._proc: subprocess.Popen | None = None
        self._stdout = b""
        self._stderr = b""
        self._collected = False
        self._cpu_lease: tuple[int, int] | None = None
        self._workdir_tmp: tempfile.TemporaryDirectory | None = None

        # Lease the core BEFORE building argv (the taskset prefix needs it).
        self._cpu_lease = _acquire_cpu()
        try:
            argv = self._build_argv()
            # Eager start: any infra problem (apptainer missing, SIF missing,
            # spawn failure) raises HERE, out of containers.run(), so it
            # surfaces as an infra error instead of a fake compile error.
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # own process group: wait-timeout can killpg
            )
        except Exception:
            self._release_cpu()
            self._cleanup_workdir_tmp()
            raise

    # -- workdir inference (verbatim from ale_bench.utils._ApptainerContainer) --

    def _infer_workdir_source(self) -> Path | None:
        if self.working_dir is None:
            return None

        workdir = self.working_dir.rstrip("/")
        sources: list[Path] = []
        for host_path, spec in self.volumes.items():
            bind_path = spec.get("bind", "")
            if bind_path.rstrip("/") == workdir:
                return None
            if not bind_path.startswith(f"{workdir}/"):
                continue
            rel = Path(bind_path[len(workdir) + 1 :])
            host = Path(host_path).resolve()
            if not rel.parts:
                continue
            if host.is_dir():
                sources.append(host)
            else:
                base = host
                for _ in rel.parts:
                    base = base.parent
                sources.append(base)

        if sources:
            common = Path(os.path.commonpath([str(source) for source in sources]))
            if common.exists():
                return common

        self._workdir_tmp = tempfile.TemporaryDirectory(prefix="ale-bench-workdir-")
        return Path(self._workdir_tmp.name)

    def _binds(self) -> list[str]:
        binds: list[str] = []
        workdir_source = self._infer_workdir_source()
        if workdir_source is not None and self.working_dir is not None:
            binds.append(f"{workdir_source}:{self.working_dir}:rw")
        for host_path, spec in self.volumes.items():
            bind_path = spec["bind"]
            mode = spec.get("mode", "rw")
            binds.append(f"{Path(host_path).resolve()}:{bind_path}:{mode}")
        return binds

    # -- sandbox argv ------------------------------------------------------

    def _build_argv(self) -> list[str]:
        apptainer = shutil.which("apptainer") or shutil.which("singularity")
        if apptainer is None:
            raise FileNotFoundError(
                "ALE-Bench apptainer backend: `apptainer` not found on PATH. jiaolab has no "
                "usable docker (sudo-only) and no bwrap, so apptainer is the only sandbox. "
                "Refusing to score."
            )
        sif = _sif_for_image(self.image)

        cmd: list[str] = [apptainer, "exec", "--cleanenv", "--no-home"]

        env = dict(self.environment)
        # Same defaults the shipped Princeton shim applies for the rust tool image.
        if self.image.startswith("rust:") and "CARGO_HOME" not in env:
            env["CARGO_HOME"] = f"{self.working_dir or '/tmp'}/.cargo"
        if self.image.startswith("rust:") and "RUSTUP_HOME" not in env:
            env["RUSTUP_HOME"] = f"{self.working_dir or '/tmp'}/.rustup"
        for key, value in env.items():
            cmd += ["--env", f"{key}={value}"]

        if self.working_dir:
            cmd += ["--pwd", self.working_dir]
        for bind in self._binds():
            cmd += ["--bind", bind]
        cmd.append(str(sif))

        # Inner command: docker shell-form string -> /bin/sh -c (daemon behavior,
        # same as the shipped apptainer shim); list form verbatim.
        if isinstance(self.command, str):
            inner = ["/bin/sh", "-c", self.command]
        else:
            inner = [str(a) for a in self.command]

        # Opt-in address-space cap inside the container (see module docstring).
        if self.mem_limit and os.environ.get("ALE_APPTAINER_MEM_LIMIT", "0") == "1":
            inner = ["/usr/bin/prlimit", f"--as={int(self.mem_limit)}", "--"] + inner

        argv = cmd + inner

        # 1-CPU pinning replacing cpu_quota. taskset wraps the OUTER apptainer
        # process; the affinity mask is inherited by every process in the
        # sandbox (apptainer exec does not create a new cpuset).
        if self._cpu_lease is not None:
            argv = ["/usr/bin/taskset", "-c", str(self._cpu_lease[0])] + argv
        return argv

    # -- docker container API subset ----------------------------------------

    def _release_cpu(self) -> None:
        if self._cpu_lease is not None:
            _cpu, fd = self._cpu_lease
            self._cpu_lease = None
            try:
                os.close(fd)
            except OSError:
                pass

    def _cleanup_workdir_tmp(self) -> None:
        if self._workdir_tmp is not None:
            try:
                self._workdir_tmp.cleanup()
            except Exception:
                pass
            self._workdir_tmp = None

    def _kill_group(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        import signal

        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def wait(self, timeout: float | None = None) -> dict:
        assert self._proc is not None
        try:
            self._stdout, self._stderr = self._proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Mirror the docker SDK wait-timeout: kill the container, raise the
            # requests Timeout the harness catches for compile timeouts.
            self._kill_group()
            self._stdout, self._stderr = self._proc.communicate()
            rc = self._proc.returncode
            self.attrs["State"]["ExitCode"] = 128 - rc if rc is not None and rc < 0 else rc
            self._collected = True
            self._release_cpu()
            raise _requests_timeout()(f"apptainer sandbox wait timed out after {timeout}s (killed)")
        rc = self._proc.returncode
        self.attrs["State"]["ExitCode"] = 128 - rc if rc is not None and rc < 0 else rc
        self._collected = True
        self._release_cpu()
        return {"StatusCode": self.attrs["State"]["ExitCode"]}

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        parts = []
        if stdout:
            parts.append(self._stdout)
        if stderr:
            parts.append(self._stderr)
        return b"".join(parts)

    def remove(self, force: bool = False) -> None:
        # Harness always calls remove(force=True) in a finally block. If wait()
        # never ran (harness exception mid-flow), make sure nothing is left.
        if self._proc is not None and not self._collected:
            self._kill_group()
            try:
                self._stdout, self._stderr = self._proc.communicate(timeout=5)
            except Exception:
                pass
            self._collected = True
        self._release_cpu()
        self._cleanup_workdir_tmp()

    def __del__(self) -> None:  # last-resort reaper; remove() is the norm
        try:
            self.remove(force=True)
        except Exception:
            pass


class _ApptainerContainerCollection:
    def run(self, **kwargs: object) -> _ApptainerContainer:
        return _ApptainerContainer(**kwargs)


class _ApptainerClient:
    """Drop-in for the harness's `with docker_client() as client` blocks."""

    def __init__(self) -> None:
        self.containers = _ApptainerContainerCollection()

    def close(self) -> None:
        return None


@contextmanager
def apptainer_docker_client():
    """Replacement for ale_bench.utils.docker_client when backend=apptainer."""
    client = _ApptainerClient()
    try:
        yield client
    finally:
        client.close()


# --------------------------------------------------------------------------
# install: patch ale_bench.utils.docker_client (lazily, at its first import)
# --------------------------------------------------------------------------

_INSTALLED = False


def _patch_consumers() -> None:
    """Rebind `docker_client` in already-imported consumer modules (they did
    `from ale_bench.utils import docker_client`, so they hold their own ref)."""
    for name in (
        "ale_bench.tool_wrappers.case_runner",
        "ale_bench.tool_wrappers.input_generation",
        "ale_bench.tool_wrappers.code_runner",
        "ale_bench.data",
        "ale_bench.session",
    ):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "docker_client"):
            mod.docker_client = apptainer_docker_client


def _patch_utils_module(module) -> None:
    module.docker_client = apptainer_docker_client
    _patch_consumers()


class _AleUtilsImportHook(importlib.abc.MetaPathFinder):
    """Post-import hook: patch ale_bench.utils the moment anything imports it,
    so every later `from ale_bench.utils import docker_client` (case_runner,
    input_generation, data) binds THIS backend instead of docker."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "ale_bench.utils":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec_module = spec.loader.exec_module

        def _exec_then_patch(module):
            orig_exec_module(module)
            _patch_utils_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def install() -> bool:
    """Install the apptainer backend. Idempotent; returns True when active.

    Lazy: only registers a one-name meta-path hook, so processes that never
    import ale_bench pay nothing (vLLM workers etc.).
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    existing = sys.modules.get("ale_bench.utils")
    if existing is not None:
        _patch_utils_module(existing)
    if not any(isinstance(f, _AleUtilsImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _AleUtilsImportHook())
    _INSTALLED = True
    return True


def installed() -> bool:
    return _INSTALLED
