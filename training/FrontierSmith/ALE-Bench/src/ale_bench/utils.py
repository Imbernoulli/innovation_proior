"""General utility helpers for filesystem, docker, and parsing."""

import base64
import io
import logging
import os
import random
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, overload

import cairosvg
import docker
from PIL import Image
from ahocorapy.keywordtree import KeywordTree

from ale_bench.constants import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


# Docker
def _sanitize_image_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "_")


def _remote_image_uri(image: str) -> str:
    if image.startswith("ale-bench:"):
        return f"docker://yimjk/ale-bench:{image.split(':', 1)[1]}"
    if image.startswith("docker://"):
        return image
    return f"docker://{image}"


def _apptainer_image_path(image: str) -> Path:
    image_dir = Path(os.environ.get("ALE_BENCH_APPTAINER_DIR", get_cache_dir() / "apptainer-images")).expanduser()
    candidates = [image_dir / f"{_sanitize_image_name(image)}.sif"]
    if image.startswith("ale-bench:"):
        tag = image.split(":", 1)[1]
        candidates.append(image_dir / f"yimjk_ale-bench_{tag}.sif")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if os.environ.get("ALE_BENCH_APPTAINER_ALLOW_PULL", "").lower() in {"1", "true", "yes"}:
        image_dir.mkdir(parents=True, exist_ok=True)
        target = candidates[0]
        subprocess.run(
            ["apptainer", "pull", "--force", str(target), _remote_image_uri(image)],
            check=True,
        )
        return target.resolve()

    candidate_list = ", ".join(str(path) for path in candidates)
    msg = (
        f"Apptainer image for {image!r} was not found. Expected one of: {candidate_list}. "
        "Run scripts/prepare_alebench_apptainer_images.sh on a networked login node first, "
        "or set ALE_BENCH_APPTAINER_ALLOW_PULL=1."
    )
    raise FileNotFoundError(msg)


class _ApptainerContainer:
    def __init__(
        self,
        image: str,
        command: str | list[str],
        volumes: dict[str, dict[str, str]] | None = None,
        working_dir: str | None = None,
        environment: dict[str, str] | None = None,
        **_: object,
    ) -> None:
        self.image = image
        self.image_path = _apptainer_image_path(image)
        self.command = command
        self.volumes = volumes or {}
        self.working_dir = working_dir
        self.environment = environment or {}
        self.attrs = {"State": {"ExitCode": None}}
        self._stdout = b""
        self._stderr = b""
        self._ran = False
        self._workdir_tmp: tempfile.TemporaryDirectory[str] | None = None

    def _command_args(self) -> list[str]:
        if isinstance(self.command, str):
            return ["/bin/sh", "-c", self.command]
        return [str(arg) for arg in self.command]

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

    def _run_once(self, timeout: float | None = None) -> None:
        if self._ran:
            return
        self._ran = True

        cmd = ["apptainer", "exec", "--cleanenv", "--no-home"]
        env = dict(self.environment)
        if self.image.startswith("rust:") and "CARGO_HOME" not in env:
            env["CARGO_HOME"] = f"{self.working_dir or '/tmp'}/.cargo"
        if self.image.startswith("rust:") and "RUSTUP_HOME" not in env:
            env["RUSTUP_HOME"] = f"{self.working_dir or '/tmp'}/.rustup"
        for key, value in env.items():
            cmd.extend(["--env", f"{key}={value}"])
        if self.working_dir:
            cmd.extend(["--pwd", self.working_dir])
        for bind in self._binds():
            cmd.extend(["--bind", bind])
        cmd.append(str(self.image_path))
        cmd.extend(self._command_args())

        try:
            completed = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._stdout = exc.stdout or b""
            self._stderr = exc.stderr or b""
            self.attrs["State"]["ExitCode"] = 124
            raise

        self._stdout = completed.stdout
        self._stderr = completed.stderr
        self.attrs["State"]["ExitCode"] = completed.returncode

    def wait(self, timeout: float | None = None) -> dict[str, int | None]:
        self._run_once(timeout=timeout)
        return {"StatusCode": self.attrs["State"]["ExitCode"]}

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        parts = []
        if stdout:
            parts.append(self._stdout)
        if stderr:
            parts.append(self._stderr)
        return b"".join(parts)

    def remove(self, force: bool = False) -> None:
        if self._workdir_tmp is not None:
            self._workdir_tmp.cleanup()
            self._workdir_tmp = None


class _ApptainerContainerCollection:
    def run(self, **kwargs: object) -> _ApptainerContainer:
        return _ApptainerContainer(**kwargs)


class _ApptainerClient:
    def __init__(self) -> None:
        self.containers = _ApptainerContainerCollection()

    def close(self) -> None:
        return None


@contextmanager
def docker_client() -> Generator[docker.DockerClient, None, None]:
    """Context manager for Docker client.

    Yields:
        docker.DockerClient: The Docker client.

    """
    if os.environ.get("ALE_BENCH_CONTAINER_BACKEND", "docker").lower() == "apptainer":
        client = _ApptainerClient()
        try:
            yield client  # type: ignore[misc]
        finally:
            client.close()
        return

    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


# Cache
def get_cache_dir() -> Path:
    """Get the cache directory for ALE-Bench.

    Returns:
        Path: The cache directory.

    """
    cache_dir_str = os.environ.get("ALE_BENCH_CACHE", None)
    if cache_dir_str is None:
        return DEFAULT_CACHE_DIR
    return Path(cache_dir_str).expanduser().resolve()


def clear_cache() -> None:
    """Clear the cache directory for ALE-Bench."""
    cache_dir = get_cache_dir()
    if cache_dir.is_dir():
        logger.info("Clearing cache directory: %s", cache_dir)
        shutil.rmtree(cache_dir)


# Data
def get_local_data_dir() -> Path | None:
    """Get the local data directory for ALE-Bench.

    Returns:
        Path | None: The local data directory. Returns None if not set.

    """
    data_dir_str = os.environ.get("ALE_BENCH_DATA", None)
    if data_dir_str is None:
        return None
    data_dir = Path(data_dir_str).expanduser().resolve()
    if not data_dir.is_dir():
        logger.warning("Data directory does not exist: %s", data_dir)
        return None
    return data_dir


def dir_tree(
    dir_path: Path,
    prefix: str = "",
) -> Generator[str, None, None]:
    """Generate a tree structure of the directory.

    Args:
        dir_path (Path): The path to the directory.
        prefix (str, optional): The prefix for the tree structure. Defaults to "".

    Yields:
        str: The tree structure of the directory.

    """
    if not dir_path.is_dir():
        msg = f"{dir_path} is not a directory."
        raise ValueError(msg)
    tee = "├── "
    last = "└── "
    branch = "│   "
    space = "    "
    contents = list(dir_path.iterdir())
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents, strict=True):
        yield prefix + pointer + path.name
        if path.is_dir():
            extension = branch if pointer == tee else space
            yield from dir_tree(path, prefix + extension)


def print_dir_tree(dir_path: Path) -> None:
    """Print the tree structure of the directory.

    Args:
        dir_path (Path): The path to the directory.

    """
    for line in dir_tree(dir_path):
        print(line)  # noqa: T201


# Problem
def text_image_contents_to_openai(contents: list[str | Image.Image]) -> list[dict[str, str | dict[str, str]]]:
    """Convert the contents to OpenAI format.

    Args:
        contents (list[str | Image.Image]): The contents to convert.

    Returns:
        list[dict[str, str]]: The converted contents.

    """
    openai_contents: list[dict[str, str | dict[str, str]]] = []
    for content in contents:
        if isinstance(content, str):
            openai_contents.append({"type": "text", "text": content})
        elif isinstance(content, Image.Image):
            openai_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64jpeg(content)}"},
                }
            )
        else:
            msg = "The content is not a str or a PIL.Image.Image."
            raise TypeError(msg)
    return openai_contents


@overload
def parse_statement(
    statement: str,
    images: dict[str, Image.Image | list[Image.Image]],
    ignore_video: bool = False,
    extract_video_frame: Literal["first", "last", "all"] = "all",
    return_openai: Literal[False] = False,
) -> list[str | Image.Image]: ...


@overload
def parse_statement(
    statement: str,
    images: dict[str, Image.Image | list[Image.Image]],
    ignore_video: bool = False,
    extract_video_frame: Literal["first", "last", "all"] = "all",
    return_openai: Literal[True] = True,
) -> list[dict[str, str | dict[str, str]]]: ...


@overload
def parse_statement(
    statement: str,
    images: dict[str, Image.Image | list[Image.Image]],
    ignore_video: bool = False,
    extract_video_frame: Literal["first", "last", "all"] = "all",
    return_openai: bool = False,
) -> list[str | Image.Image] | list[dict[str, str | dict[str, str]]]: ...


def parse_statement(
    statement: str,
    images: dict[str, Image.Image | list[Image.Image]],
    ignore_video: bool = False,
    extract_video_frame: Literal["first", "last", "all"] = "all",
    return_openai: bool = False,
) -> list[str | Image.Image] | list[dict[str, str | dict[str, str]]]:
    """Parse the problem statement and images and return a list of contents.

    Images are interleaved with the text in the statement.

    Args:
        statement (str): The problem statement.
        images (dict[str, Image.Image | list[Image.Image]]): The images with their names.
            The keys are the image names int the statement and the values are the images or a list of images.
        ignore_video (bool, optional): If True, ignore video frames. Defaults to False.
        extract_video_frame (Literal["first", "last", "all"], optional): The video frame to extract.
            Defaults to "all". If ignore_video is True, this argument is ignored.
            If "first", extract the first frame. If "last", extract the last frame. If "all", extract all frames.
        return_openai (bool, optional): If True, convert the contents to OpenAI format. Defaults to False.

    Returns:
        list[str | Image.Image] | list[dict[str, str | dict[str, str]]]:
            A list of contents, where each content is either a text or an image.

    """
    # Search for image names in the statement by using Aho-Corasick algorithm
    kwtree = KeywordTree(case_insensitive=False)
    for image_name, image_value in images.items():
        if isinstance(image_value, list) and ignore_video:
            continue  # Ignore video
        kwtree.add(image_name)
    kwtree.finalize()
    matches = kwtree.search_all(statement)

    # If no image names are found, return the statement as is
    contents: list[str | Image.Image] = []
    if matches is None:  # No image names found in the statement
        contents.append(statement)
        if return_openai:
            return text_image_contents_to_openai(contents)
        return contents

    # Interleave the images with the text in the statement
    matches = sorted(matches, key=lambda x: x[1])  # Sort by the start index
    current_idx = 0
    for matched_image, idx in matches:
        contents.append(statement[current_idx:idx])
        image = images[matched_image]
        if isinstance(image, list):
            video_frames: list[Image.Image] = [frame for frame in image if isinstance(frame, Image.Image)]
            if extract_video_frame == "first":
                contents.append(video_frames[0])
            elif extract_video_frame == "last":
                contents.append(video_frames[-1])
            elif extract_video_frame == "all":
                contents.extend(video_frames)
            else:
                msg = f"`extract_video_frame` must be 'first', 'last', or 'all'. Got: {extract_video_frame}"
                raise ValueError(msg)
        else:
            contents.append(image)
        current_idx = idx + len(matched_image)
    contents.append(statement[current_idx:])

    # Convert the contents to OpenAI format if requested
    if return_openai:
        return text_image_contents_to_openai(contents)
    return contents


# Session
def find_free_port(min_port: int = 9000, max_port: int = 65535) -> int:
    """Find a free port in the specified range.

    Args:
        min_port (int, optional): Minimum port number. Defaults to 9000.
        max_port (int, optional): Maximum port number. Defaults to 65535.

    Returns:
        int: A free port number.

    Raises:
        RuntimeError: If no free port is found in the specified range.

    """
    ports = list(range(min_port, max_port + 1))
    random.shuffle(ports)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError:
                continue
            else:
                return port
    msg = f"No free ports found in range {min_port}-{max_port}."
    raise RuntimeError(msg)


# Image
def base64_to_pil(base64_str: str) -> Image.Image:
    """Convert a base64 string to a PIL image.

    Args:
        base64_str (str): The base64 string of the image.

    Returns:
        Image.Image: The PIL image.

    """
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image.convert(image.mode)  # NOTE: to create a new Image instance (not subclasses like PngImageFile)


def pil_to_base64(image: Image.Image, image_format: Literal["JPEG", "PNG"] = "PNG") -> str:
    """Convert a PIL image to a base64 string.

    Args:
        image (Image.Image): The PIL image.
        image_format (Literal["JPEG", "PNG"]): The format to save the image in. Defaults to "PNG".

    Returns:
        str: The base64 string of the image.

    """
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pil_to_base64jpeg(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string of a JPEG image.

    Args:
        image (Image.Image): The PIL image.

    Returns:
        str: The base64 string of the JPEG image.

    """
    return pil_to_base64(image.convert("RGB"), image_format="JPEG")


def read_svg(svg_text: str, size: int | tuple[int, int] = 1000) -> Image.Image:
    """Read an SVG text and return a PIL image.

    Args:
        svg_text (str): The SVG text.
        size (int | tuple[int, int], optional): The size of the output image. Defaults to 1000.
            If it is an integer, the output image will be a square. If it is a tuple, (width, height) will be used.

    Returns:
        Image.Image: The PIL image of the SVG.

    Raises:
        ValueError: If the SVG text is empty.

    """
    if len(svg_text) == 0:
        msg = "SVG text is empty."
        raise ValueError(msg)
    if isinstance(size, int):
        size = (size, size)
    width, height = size
    buffer = io.BytesIO()
    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=width,
        output_height=height,
        background_color="white",
        write_to=buffer,
    )
    return Image.open(buffer).convert("RGB")
