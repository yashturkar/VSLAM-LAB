"""Install DROID-SLAM CUDA components for Blackwell (sm_120) GPUs."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
DROID_SOURCE_URL = "https://github.com/princeton-vl/DROID-SLAM.git"
DROID_SOURCE_REVISION = "2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99"


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def install_torch() -> None:
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch==2.7.1+cu128",
        "torchvision==0.22.1+cu128",
        "--index-url",
        TORCH_INDEX,
    )


def cuda_home() -> Path:
    candidates = [
        Path(value)
        for value in (os.environ.get("CUDA_HOME"), "/usr/local/cuda-12.8", "/usr/local/cuda")
        if value
    ]
    for candidate in candidates:
        nvcc = candidate / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        version = subprocess.check_output([nvcc, "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", version)
        if match and tuple(map(int, match.groups())) >= (12, 8):
            return candidate
    raise RuntimeError("CUDA toolkit 12.8 or newer is required to compile sm_120 kernels")


def extension_path(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"The droidslam environment does not provide {module}")
    return Path(spec.origin)


def has_sm120(path: Path, toolkit: Path) -> bool:
    cuobjdump = toolkit / "bin" / "cuobjdump"
    result = subprocess.run(
        [cuobjdump, "--list-elf", path], text=True, capture_output=True, check=True
    )
    return "sm_120" in result.stdout


def prepare_source(source: Path) -> None:
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", DROID_SOURCE_URL, str(source))
        run("git", "checkout", "--detach", DROID_SOURCE_REVISION, cwd=source)
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if current != DROID_SOURCE_REVISION:
        raise RuntimeError(f"Unexpected DROID-SLAM source revision in {source}: {current}")
    run("git", "submodule", "update", "--init", "--recursive", cwd=source)


def patch_lietorch_architecture(source: Path) -> None:
    setup = source / "thirdparty" / "lietorch" / "setup.py"
    text = setup.read_text(encoding="utf-8")
    text = re.sub(
        r"^\s*'-gencode=arch=compute_(?:60|61|70|75),"
        r"code=(?:sm|compute)_(?:60|61|70|75)',\s*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    sm120 = "                    '-gencode=arch=compute_120,code=sm_120',\n"
    marker = "                'nvcc': ['-O2',\n"
    if sm120 not in text:
        text = text.replace(marker, marker + sm120)
    setup.write_text(text, encoding="utf-8")


def build_environment(toolkit: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": str(toolkit),
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "FORCE_CUDA": "1",
        }
    )
    return env


def build_extension(source: Path, toolkit: Path) -> None:
    run(
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
        cwd=source,
        env=build_environment(toolkit),
    )


def install_extension(source: Path, pattern: str, destination: Path, toolkit: Path) -> None:
    builds = list(source.glob(pattern))
    if len(builds) != 1 or not has_sm120(builds[0], toolkit):
        raise RuntimeError(f"Rebuilt extension does not contain sm_120 kernels: {pattern}")
    backup = destination.with_name(f"{destination.stem}.pre-blackwell{destination.suffix}")
    if not backup.exists():
        shutil.copy2(destination, backup)
    shutil.copy2(builds[0], destination)


def install_torch_scatter(source: Path, toolkit: Path) -> None:
    scatter_source = source / "thirdparty" / "pytorch_scatter"
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-build-isolation",
        "--no-deps",
        ".",
        cwd=scatter_source,
        env=build_environment(toolkit),
    )


def main() -> None:
    install_torch()
    import torch

    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"Installed PyTorch does not include sm_120: {torch.cuda.get_arch_list()}")

    toolkit = cuda_home()
    destinations = {
        "droid_backends": extension_path("droid_backends"),
        "lietorch_backends": extension_path("lietorch_backends"),
        "lietorch_extras": extension_path("lietorch_extras"),
    }
    scatter_destination = extension_path("torch_scatter._scatter_cuda")
    if all(has_sm120(path, toolkit) for path in (*destinations.values(), scatter_destination)):
        print("Blackwell DROID-SLAM extensions are already installed")
        return

    shared_root = Path(os.environ.get("VSLAM_SHARED_ROOT", "/mnt/share/local/eph/VSLAM")).expanduser()
    source = shared_root / "baselines" / "DROID-SLAM-source"
    prepare_source(source)

    if not has_sm120(destinations["droid_backends"], toolkit):
        build_extension(source, toolkit)
        install_extension(source, "droid_backends*.so", destinations["droid_backends"], toolkit)

    lietorch_names = ("lietorch_backends", "lietorch_extras")
    if not all(has_sm120(destinations[name], toolkit) for name in lietorch_names):
        patch_lietorch_architecture(source)
        lietorch_source = source / "thirdparty" / "lietorch"
        build_extension(lietorch_source, toolkit)
        for name in lietorch_names:
            install_extension(lietorch_source, f"{name}*.so", destinations[name], toolkit)

    if not has_sm120(scatter_destination, toolkit):
        install_torch_scatter(source, toolkit)
        rebuilt_scatter = extension_path("torch_scatter._scatter_cuda")
        if not has_sm120(rebuilt_scatter, toolkit):
            raise RuntimeError("Rebuilt torch_scatter does not contain sm_120 kernels")

    print("Installed Blackwell DROID-SLAM, lietorch, and torch_scatter extensions")


if __name__ == "__main__":
    main()
