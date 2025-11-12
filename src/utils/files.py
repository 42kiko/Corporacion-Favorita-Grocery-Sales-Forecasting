"""
Utility helpers for file and directory management across the project.
Ensures consistent handling for outputs like reports, images, and temporary files.
"""

from pathlib import Path
import os
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists. Create it if missing.

    Args:
        path (Union[str, Path]): Directory path to create if not existing.

    Returns:
        Path: The resolved Path object of the created or existing directory.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(path: Path) -> Path:
    """
    Ensure that a directory exists. Creates it if missing.
    Returns the path for convenience.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Union[str, Path], pattern: str = "*") -> list[Path]:
    """
    List all files in a given directory that match the pattern.

    Args:
        directory (Union[str, Path]): Directory to search.
        pattern (str, optional): Glob pattern. Defaults to "*".

    Returns:
        list[Path]: Matching files.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    return list(directory.glob(pattern))


def clean_dir(directory: Union[str, Path], extensions: tuple[str, ...] = ()) -> int:
    """
    Delete all files in a directory (optionally filtered by extensions).

    Args:
        directory (Union[str, Path]): Directory to clean.
        extensions (tuple[str, ...], optional): File extensions to remove (e.g., ('.png', '.csv')).

    Returns:
        int: Number of deleted files.
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    deleted = 0
    for f in directory.iterdir():
        if f.is_file() and (not extensions or f.suffix in extensions):
            f.unlink()
            deleted += 1
    return deleted


def get_project_root() -> Path:
    """
    Get the absolute project root directory.
    Assumes this file is located at `src/utils/files.py`.
    """
    return Path(__file__).resolve().parents[2]

def set_project_root(marker: str = "configs") -> Path:
    """
    Ensure the current working directory is the project root.
    This is safe to run multiple times, even inside Jupyter notebooks.

    Args:
        marker (str): Folder name that must exist at the project root (default: 'configs').

    Returns:
        Path: The absolute project root path.
    """
    import os

    current = Path.cwd()
    # already at project root
    if (current / marker).exists():
        print(f"📂 Working directory already set: {current}")
        return current

    # if notebook is inside 'notebooks/', move one level up
    if current.name == "notebooks" and (current.parent / marker).exists():
        os.chdir(current.parent)
        print(f"📂 Working directory set to project root: {current.parent}")
        return current.parent

    # fallback: climb up until marker found
    for parent in current.parents:
        if (parent / marker).exists():
            os.chdir(parent)
            print(f"📂 Working directory set to project root: {parent}")
            return parent

    raise FileNotFoundError(
        f"Could not locate project root — no '{marker}' directory found above {current}"
    )