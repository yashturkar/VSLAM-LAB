"""Configure project runtime storage outside the source checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared_storage import STORAGE_CONFIG, relocate_baseline  # noqa: E402

DEFAULT_ROOT = Path("/mnt/share/local/eph/VSLAM")


def _replace_with_link(local: Path, target: Path) -> None:
    if local.is_symlink():
        if local.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected link: {local} -> {local.resolve()}")
        return
    if local.exists():
        incoming = target.with_name(f".{target.name}.incoming-{os.getpid()}")
        if target.exists() or incoming.exists():
            raise FileExistsError(f"Cannot safely migrate {local}; target is occupied: {target}")
        shutil.copytree(local, incoming, symlinks=True)
        incoming.replace(target)
        backup = local.with_name(f"{local.name}.local-backup-{datetime.now():%Y%m%d-%H%M%S}")
        local.replace(backup)
    target.mkdir(parents=True, exist_ok=True)
    local.symlink_to(target, target_is_directory=True)


def configure(root: Path) -> None:
    root = root.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("Refusing to use a filesystem root as shared storage")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise PermissionError(f"Shared storage is not writable: {root}")
    for directory in (
        root / "baselines", root / "pixi-cache", root / "cache" / "huggingface",
        root / "cache" / "torch", root / "cache" / "triton",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    STORAGE_CONFIG.write_text(str(root) + "\n", encoding="utf-8")
    _replace_with_link(PROJECT_ROOT / ".pixi", root / "pixi-project")
    pixi_config = root / "pixi-project" / "config.toml"
    pixi_config.write_text(f'[cache]\nroot = "{root / "pixi-cache"}"\n', encoding="utf-8")

    for child in (PROJECT_ROOT / "Baselines").iterdir():
        if child.is_dir() and not child.is_symlink() and (child / ".git").is_dir():
            relocate_baseline(child, keep_backup=True)

    print(f"Shared VSLAM storage configured at {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    configure(args.root)


if __name__ == "__main__":
    main()
