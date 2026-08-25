"""gpublaze wrapper-layer: ALE-Bench NO-DOCKER (host) container backend.

Activated when ``ALE_BENCH_CONTAINER_BACKEND=host`` (see pysite/sitecustomize.py,
which auto-imports this module and calls ``install()``). Replaces the
``ale_bench.utils.docker_client`` context manager with a bwrap-based local
sandbox, so the vendored ALE-Bench harness (``run_compile_container`` /
``run_batch_run_container`` / ``run_batch_judge_container`` /
``run_reactive_judge_container`` / ``run_gen_container`` /
``build_rust_tools``) runs UNCHANGED, but no container ever touches dockerd.

Why: the rootless dockerd on gpublaze died twice under 6-client concurrent
container bursts (fd exhaustion / wedge) on 2026-08-23, turning every ALE score
into AleInfraError. The user ruled the judging chain must not depend on docker.

Container-equivalence mapping (per docker flag used by the harness):
  image=ale-bench:<tag>      -> bwrap with the EXPORTED IMAGE ROOTFS
                                ($ALE_BENCH_CACHE/host-rootfs/<tag>, built once by
                                scripts/gpublaze/prepare_ale_host_rootfs.sh) bound
                                ro at /usr + /opt + /etc: the SAME g++ 12.2.0,
                                boost, ac-library, coreutils as the container.
  image=rust:* / httpd:*     -> host tools root (ro /usr + /etc): only the
                                prebuilt gen/tester/vis rust binaries run here
                                (glibc 2.28-built, host glibc 2.35 is forward
                                compatible; verified via ldd).
  command                    -> exec'd verbatim (list form as-is, str form via
                                /bin/sh -c, exactly like the docker daemon's
                                shell-form Cmd and the apptainer shim).
  volumes={h: {bind, mode}}  -> --ro-bind / --bind (file and dir binds).
  working_dir                -> --chdir.
  network_disabled=True      -> --unshare-all (no network namespace at all).
  user/group_add             -> n/a: bwrap already runs as the invoking user.
  cpu_quota=100000 (1 CPU)   -> taskset -c <one core>, core leased from a
                                flock-protected pool (ALE_BENCH_HOST_CPUS,
                                default: all cores) so concurrent sandboxes get
                                DISTINCT cores.
  mem_limit=2GiB             -> prlimit --as=<bytes> (RLIMIT_AS). Semantic delta
                                vs cgroup RSS limit documented in the delivery
                                report; scoring-relevant memory enforcement is
                                the harness's own max-RSS check (parse_profiles).
  detach/wait/logs/attrs     -> eager subprocess.Popen at containers.run() time
                                (docker starts eagerly too); wait()/logs()/
                                attrs["State"]["ExitCode"]/remove() mirror the
                                docker SDK surface the harness uses.
  wait(timeout)              -> on expiry the sandbox process GROUP is SIGKILLed
                                and requests.exceptions.Timeout is raised, so the
                                harness's own `except (Timeout, ConnectionError)`
                                path classifies a hung compile exactly like
                                docker did.

Failure semantics (mirrors the docker backend's fail-loud contract):
  * bwrap missing, rootfs missing/corrupt, spawn failure -> raised EAGERLY from
    containers.run(), i.e. BEFORE wait(); it propagates out of
    run_compile_container/run_cases -> session.private_eval -> compute_score,
    which wraps it as AleInfraError. It is NEVER swallowed into a fake
    COMPILATION_ERROR (that branch only catches exceptions from wait()).
  * Model-side failures (CE/TLE/RE/WA/MLE) come back as ordinary CaseResults
    exactly like the docker path -- score 0, not an exception.

The one-shot compile selftest (verl/utils/reward_score/ale_selftest.py) goes
through run_compile_container, so with this backend installed it IS the host
backend selftest (real g++ 12.2.0 compiling a trivial program inside bwrap).
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
_IMAGE_TOOLS_PREFIXES = ("rust:", "httpd:")

# Fixed sandbox env (matches the image defaults; the container had no locale).
_SANDBOX_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
}


# --------------------------------------------------------------------------
# rootfs / pool plumbing
# --------------------------------------------------------------------------

def _cache_root() -> Path:
    return Path(os.environ.get("ALE_BENCH_CACHE", Path.home() / ".cache" / "ale-bench"))


def _rootfs_for_tag(tag: str) -> Path:
    """Locate the exported image rootfs for an ale-bench:<tag> image."""
    base = Path(os.environ.get("ALE_BENCH_HOST_ROOTFS", _cache_root() / "host-rootfs"))
    rootfs = base / tag
    if not (rootfs / "usr" / "bin" / "bash").is_file():
        msg = (
            f"ALE-Bench host backend: exported image rootfs for tag {tag!r} not found at {rootfs}. "
            "Build it once with: bash scripts/gpublaze/prepare_ale_host_rootfs.sh "
            "(or set ALE_BENCH_HOST_ROOTFS). Refusing to score with a missing toolchain."
        )
        raise FileNotFoundError(msg)
    return rootfs


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
    return Path(os.environ.get("ALE_BENCH_HOST_CPU_POOL_DIR", "/dev/shm/ale-bench-host-cpu-pool"))


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
    # Pool exhausted (more concurrent sandboxes than cores): fall back to a
    # random core without a lease rather than blocking the judge.
    return None


# --------------------------------------------------------------------------
# docker-SDK-compatible host container
# --------------------------------------------------------------------------

class _HostContainer:
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

        # Lease the core BEFORE building argv (the taskset prefix needs it).
        self._cpu_lease = _acquire_cpu()
        try:
            argv = self._build_argv()
            # Eager start: any infra problem (bwrap missing, rootfs missing,
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
            raise

    # -- sandbox argv ------------------------------------------------------

    def _image_tag(self) -> str:
        return self.image.rsplit(":", 1)[1] if ":" in self.image else "latest"

    def _build_argv(self) -> list[str]:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise FileNotFoundError(
                "ALE-Bench host backend: `bwrap` not found on PATH; the no-docker sandbox "
                "requires bubblewrap. Install it or use ALE_BENCH_CONTAINER_BACKEND=docker."
            )

        args: list[str] = [bwrap, "--unshare-all", "--die-with-parent"]

        if self.image.startswith(_IMAGE_ALE_PREFIXES) and not self.image.startswith(_IMAGE_TOOLS_PREFIXES):
            # Model-code image: bind the exported image rootfs (same toolchain
            # as the container: g++ 12.2.0, boost, ac-library, GNU time...).
            rootfs = _rootfs_for_tag(self._image_tag())
            args += ["--ro-bind", str(rootfs / "usr"), "/usr"]
            if (rootfs / "opt").is_dir():
                args += ["--ro-bind", str(rootfs / "opt"), "/opt"]
            args += ["--ro-bind", str(rootfs / "etc"), "/etc"]
        elif self.image.startswith(_IMAGE_TOOLS_PREFIXES):
            # Official rust-tool image: only prebuilt gen/tester/vis binaries
            # run here (bound via volumes); host /usr + /etc suffice (buster
            # binaries, host glibc 2.35 is forward compatible).
            args += ["--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
        else:
            msg = f"ALE-Bench host backend: no rootfs mapping for image {self.image!r}."
            raise ValueError(msg)

        # usrmerge layout (both bookworm rootfs and Ubuntu host): /bin,/lib,
        # /lib64,/sbin are symlinks into /usr.
        args += [
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/sbin", "/sbin",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
        ]

        # Volume binds AFTER the tmpfs so single-file binds overlay it.
        for host_path, spec in self.volumes.items():
            bind = spec["bind"]
            mode = spec.get("mode", "rw")
            args += ["--ro-bind" if mode == "ro" else "--bind", str(Path(host_path).resolve()), bind]

        if self.working_dir:
            # The image guarantees /workdir exists; on the host-tools root
            # (rust/httpd images) nothing else creates it, and bwrap --chdir
            # into a missing dir is fatal.
            args += ["--dir", self.working_dir, "--chdir", self.working_dir]

        args.append("--clearenv")
        for key, value in {**_SANDBOX_ENV, **self.environment}.items():
            args += ["--setenv", key, str(value)]

        # Inner command: docker shell-form string -> /bin/sh -c (daemon behavior,
        # same as the apptainer shim); list form verbatim.
        if isinstance(self.command, str):
            inner = ["/bin/sh", "-c", self.command]
        else:
            inner = [str(a) for a in self.command]

        # Equivalence wrappers: 1-CPU pinning (taskset) + address-space cap
        # (prlimit --as) replacing cpu_quota / mem_limit.
        wrapped = inner
        if self._cpu_lease is not None:
            wrapped = ["/usr/bin/taskset", "-c", str(self._cpu_lease[0])] + wrapped
        if self.mem_limit:
            wrapped = ["/usr/bin/prlimit", f"--as={int(self.mem_limit)}", "--"] + wrapped

        return args + wrapped

    # -- docker container API subset ----------------------------------------

    def _release_cpu(self) -> None:
        if self._cpu_lease is not None:
            _cpu, fd = self._cpu_lease
            self._cpu_lease = None
            try:
                os.close(fd)
            except OSError:
                pass

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
            raise _requests_timeout()(f"host sandbox wait timed out after {timeout}s (killed)")
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

    def __del__(self) -> None:  # last-resort reaper; remove() is the norm
        try:
            self.remove(force=True)
        except Exception:
            pass


class _HostContainerCollection:
    def run(self, **kwargs: object) -> _HostContainer:
        return _HostContainer(**kwargs)


class _HostClient:
    """Drop-in for the harness's `with docker_client() as client` blocks."""

    def __init__(self) -> None:
        self.containers = _HostContainerCollection()

    def close(self) -> None:
        return None


@contextmanager
def host_docker_client():
    """Replacement for ale_bench.utils.docker_client when backend=host."""
    client = _HostClient()
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
            mod.docker_client = host_docker_client


def _patch_utils_module(module) -> None:
    module.docker_client = host_docker_client
    _patch_consumers()


class _AleUtilsImportHook(importlib.abc.MetaPathFinder):
    """Post-import hook: patch ale_bench.utils the moment anything imports it,
    so every later `from ale_bench.utils import docker_client` (case_runner,
    input_generation, data) binds the HOST backend instead of docker."""

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
    """Install the host backend. Idempotent; returns True when active.

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
