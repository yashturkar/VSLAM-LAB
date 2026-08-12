"""Install MASt3R-SLAM CUDA components for Blackwell (sm_120) GPUs."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
MAST3R_SOURCE_URL = "https://github.com/rmurai0610/MASt3R-SLAM.git"
LIETORCH_SOURCE_URL = "https://github.com/princeton-vl/lietorch.git"
MAST3R_SOURCE_REVISION = "e6f4e3d474fad0e11f561482012be864ba8c3f17"
LIETORCH_SOURCE_REVISION = "e7df86554156b36846008d8ddbcc4d8521a16554"


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
        Path(value) for value in (os.environ.get("CUDA_HOME"), "/usr/local/cuda-12.8", "/usr/local/cuda") if value
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
        raise RuntimeError(f"The mast3rslam environment does not provide {module}")
    return Path(spec.origin)


def has_sm120(path: Path, nvcc_root: Path) -> bool:
    cuobjdump = nvcc_root / "bin" / "cuobjdump"
    result = subprocess.run([cuobjdump, "--list-elf", path], text=True, capture_output=True, check=True)
    return "sm_120" in result.stdout


def patch_source(source: Path) -> None:
    setup = source / "setup.py"
    text = setup.read_text(encoding="utf-8")
    for architecture in ("60", "61", "70", "75", "80", "86"):
        text = text.replace(f'        "-gencode=arch=compute_{architecture},code=sm_{architecture}",\n', "")
    marker = '        "-O3",\n'
    sm120 = '        "-gencode=arch=compute_120,code=sm_120",\n'
    if sm120 not in text:
        text = text.replace(marker, marker + sm120, 1)
    setup.write_text(text, encoding="utf-8")

    kernels = source / "mast3r_slam" / "backend" / "src" / "gn_kernels.cu"
    text = kernels.read_text(encoding="utf-8").replace("torch::linalg::linalg_norm", "at::linalg_norm")
    kernels.write_text(text, encoding="utf-8")

    matching = source / "mast3r_slam" / "backend" / "src" / "matching_kernels.cu"
    text = matching.read_text(encoding="utf-8").replace("D11.type()", "D11.scalar_type()")
    matching.write_text(text, encoding="utf-8")


def clone(source: Path, url: str, revision: str, submodule: str) -> None:
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "init", str(source))
        run("git", "remote", "add", "origin", url, cwd=source)
        run("git", "fetch", "--depth", "1", "origin", revision, cwd=source)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if current != revision:
        raise RuntimeError(f"Unexpected source revision in {source}: {current}")
    run("git", "submodule", "update", "--init", "--depth", "1", submodule, cwd=source)


def build(source: Path, toolkit: Path) -> None:
    build_env = os.environ.copy()
    build_env.update({"CUDA_HOME": str(toolkit), "TORCH_CUDA_ARCH_LIST": "12.0"})
    run(sys.executable, "setup.py", "build_ext", "--inplace", cwd=source, env=build_env)


def install_extension(source: Path, pattern: str, destination: Path, toolkit: Path) -> None:
    builds = list(source.glob(pattern))
    if len(builds) != 1 or not has_sm120(builds[0], toolkit):
        raise RuntimeError(f"Rebuilt extension does not contain sm_120 kernels: {pattern}")
    backup = source / f"{destination.stem}.pre-blackwell{destination.suffix}"
    if not backup.exists():
        shutil.copy2(destination, backup)
    shutil.copy2(builds[0], destination)


def main() -> None:
    install_torch()
    import torch

    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"Installed PyTorch does not include sm_120: {torch.cuda.get_arch_list()}")

    toolkit = cuda_home()
    destinations = {
        "mast3r_slam_backends": extension_path("mast3r_slam_backends"),
        "lietorch_backends": extension_path("lietorch_backends"),
        "lietorch_extras": extension_path("lietorch_extras"),
    }
    if all(has_sm120(path, toolkit) for path in destinations.values()):
        print("Blackwell MASt3R-SLAM and lietorch backends are already installed")
        return

    shared_root = Path(os.environ.get("VSLAM_SHARED_ROOT", "/mnt/share/local/eph/VSLAM")).expanduser()
    mast3r_source = shared_root / "baselines" / "MASt3R-SLAM-source"
    if not has_sm120(destinations["mast3r_slam_backends"], toolkit):
        clone(mast3r_source, MAST3R_SOURCE_URL, MAST3R_SOURCE_REVISION, "thirdparty/eigen")
        patch_source(mast3r_source)
        build(mast3r_source, toolkit)
        install_extension(
            mast3r_source, "mast3r_slam_backends*.so", destinations["mast3r_slam_backends"], toolkit
        )

    if not all(has_sm120(destinations[name], toolkit) for name in ("lietorch_backends", "lietorch_extras")):
        lietorch_source = shared_root / "baselines" / "lietorch-source"
        clone(lietorch_source, LIETORCH_SOURCE_URL, LIETORCH_SOURCE_REVISION, "eigen")
        build(lietorch_source, toolkit)
        for name in ("lietorch_backends", "lietorch_extras"):
            install_extension(lietorch_source, f"{name}*.so", destinations[name], toolkit)
    print("Installed Blackwell MASt3R-SLAM and lietorch backends")


if __name__ == "__main__":
    main()
