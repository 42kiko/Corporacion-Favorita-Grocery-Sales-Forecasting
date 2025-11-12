"""
Unit tests for src/utils/files.py
---------------------------------
These tests verify that the filesystem utilities behave correctly and safely.
They use pytest's tmp_path fixture for isolation.
"""

from pathlib import Path
import os
import pytest

from src.utils.files import ensure_dirs, set_project_root


def test_ensure_dirs_creates_directory(tmp_path: Path) -> None:
    """Ensure that ensure_dirs() creates a directory if it doesn't exist."""
    target_dir = tmp_path / "nested" / "folder"
    assert not target_dir.exists()

    created_path = ensure_dirs(target_dir)

    # ✅ Check directory was created
    assert created_path.exists()
    assert created_path.is_dir()


def test_set_project_root_already_at_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the current working directory already contains the marker folder, it should remain unchanged."""
    marker_dir = tmp_path / "configs"
    marker_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)
    result = set_project_root(marker="configs")

    # ✅ Verify current dir remains
    assert result == tmp_path
    assert Path.cwd() == tmp_path


def test_set_project_root_climbs_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If executed from a subfolder, it should climb up until the project root is found."""
    project_root = tmp_path / "my_project"
    (project_root / "configs").mkdir(parents=True, exist_ok=True)

    # create nested subdir (like notebooks/)
    nested_dir = project_root / "notebooks"
    nested_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(nested_dir)
    result = set_project_root(marker="configs")

    # ✅ Should detect project root
    assert result == project_root
    assert Path.cwd() == project_root


def test_set_project_root_raises_if_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If no marker directory exists, it should raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        set_project_root(marker="nonexistent_marker")