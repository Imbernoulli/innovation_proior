"""gpublaze wrapper-layer patches for ALE-Bench.

Loaded automatically (python imports `sitecustomize` from sys.path) when the
gpublaze wrappers prepend scripts/gpublaze/pysite to PYTHONPATH. No historical
file is edited.

Two independent patches:

1. ALE_BENCH_CONTAINER_BACKEND=host -> install the NO-DOCKER bwrap backend
   (ale_host_backend.py): gen/tester/vis + compile + case-run "containers" run
   as local bwrap sandboxes (exported image rootfs, 1-core pinning, RLIMIT_AS,
   no network), so judging never touches dockerd.

2. ALE_BENCH_DOCKER_ROOT_USER=1 -> ROOTLESS-docker uid remap (docker backend
   only; dormant under the host backend).

Why (2): the vendored ALE-Bench harness starts every judge container with
`user=os.getuid()`. That is correct on a rootful daemon, but under ROOTLESS
docker the host user is mapped to container uid 0 and `os.getuid()` (2065) is
mapped to an unrelated subordinate uid, so the container cannot write /workdir
(chowned to 0 in the image, built with --build-arg UID=0) nor the rw
bind-mounted host files. Symptom: `ld: cannot open output file a.out:
Permission denied` -> the compile selftest correctly refuses to score.

Fix: when ALE_BENCH_DOCKER_ROOT_USER=1 (set by the gpublaze wrappers), rewrite
`user` to 0 and drop `group_add` for the ALE-Bench-related images only.
Container root == the invoking host user under rootless docker, so bind-mount
writes come back owned by us -- the exact semantics the harness assumes.
"""
from __future__ import annotations

import os

# ---- no-docker (host) ALE-Bench backend ---------------------------------------
# When ALE_BENCH_CONTAINER_BACKEND=host, install the bwrap-based backend from
# ale_host_backend.py (same directory, on PYTHONPATH with this file). It
# replaces ale_bench.utils.docker_client lazily via an import hook, so the
# harness runs unchanged but no container touches dockerd. Fail loud if the
# backend cannot be installed: a silent fall-through would run the docker
# backend instead, masking the outage the user asked to eliminate.
if os.environ.get("ALE_BENCH_CONTAINER_BACKEND", "").lower() == "host":
    try:
        import ale_host_backend

        ale_host_backend.install()
    except Exception as exc:  # sitecustomize exceptions only print; make it loud
        import sys
        import traceback

        traceback.print_exc()
        print(
            f"sitecustomize: FATAL: ALE_BENCH_CONTAINER_BACKEND=host but the host backend "
            f"failed to install ({exc!r}). ALE scoring on this interpreter is NOT safe.",
            file=sys.stderr,
        )

if os.environ.get("ALE_BENCH_DOCKER_ROOT_USER") == "1":
    try:
        from docker.models.containers import ContainerCollection

        _IMAGES = ("ale-bench", "yimjk/ale-bench", "rust:", "httpd:")
        _orig_run = ContainerCollection.run

        def _patched_run(self, image=None, command=None, stdout=True, stderr=False, remove=False, **kwargs):
            name = image if isinstance(image, str) else getattr(image, "tags", [""])[0] if image is not None else ""
            if any(str(name).startswith(p) for p in _IMAGES):
                if "user" in kwargs:
                    kwargs["user"] = 0
                kwargs.pop("group_add", None)
            return _orig_run(self, image, command, stdout, stderr, remove, **kwargs)

        ContainerCollection.run = _patched_run
    except Exception:  # docker sdk absent in this interpreter -> nothing to patch
        pass
