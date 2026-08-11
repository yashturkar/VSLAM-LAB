"""Project-local shared storage helpers for large ignored baseline checkouts."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from path_constants import VSLAM_LAB_DIR

STORAGE_CONFIG = VSLAM_LAB_DIR / ".vslamlab-storage"


def shared_root() -> Path | None:
    if not STORAGE_CONFIG.is_file():
        return None
    value = STORAGE_CONFIG.read_text(encoding="utf-8").strip()
    if not value:
        return None
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError(f"Shared storage root must be absolute: {root}")
    return root


def shared_baseline_target(baseline_path: Path) -> Path | None:
    root = shared_root()
    return root / "baselines" / baseline_path.name if root else None


def link_existing_shared_baseline(baseline_path: Path) -> None:
    """Link an already-populated shared checkout when the workspace path is absent."""
    target = shared_baseline_target(baseline_path)
    if target is None or baseline_path.exists() or baseline_path.is_symlink():
        return
    if (target / ".git").is_dir():
        baseline_path.symlink_to(target, target_is_directory=True)


def relocate_baseline(baseline_path: Path, keep_backup: bool = False) -> Path:
    """Copy-verify-move a cloned baseline to shared storage and link it back."""
    target = shared_baseline_target(baseline_path)
    if target is None:
        return baseline_path
    if baseline_path.is_symlink():
        if baseline_path.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected baseline link: {baseline_path} -> {baseline_path.resolve()}")
        return baseline_path
    if not (baseline_path / ".git").is_dir():
        raise RuntimeError(f"Refusing to relocate a non-git baseline directory: {baseline_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (target / ".git").is_dir():
            raise FileExistsError(f"Both local and shared baseline checkouts exist: {baseline_path}, {target}")
        raise FileExistsError(f"Shared baseline target is occupied: {target}")

    incoming = target.with_name(f".{target.name}.incoming-{os.getpid()}")
    if incoming.exists():
        raise FileExistsError(f"Stale incoming baseline directory: {incoming}")
    shutil.copytree(baseline_path, incoming, symlinks=True)
    if not (incoming / ".git").is_dir():
        shutil.rmtree(incoming, ignore_errors=True)
        raise RuntimeError(f"Shared baseline verification failed: {incoming}")
    incoming.replace(target)

    if keep_backup:
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = baseline_path.with_name(f"{baseline_path.name}.local-backup-{suffix}")
        baseline_path.replace(backup)
    else:
        shutil.rmtree(baseline_path)
    baseline_path.symlink_to(target, target_is_directory=True)
    return baseline_path
