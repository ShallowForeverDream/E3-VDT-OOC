from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "submission"

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "outputs",
    "runs",
    "data",
    "datasets",
    "artifacts",
    "checkpoints",
    "models",
    "weights",
    "wandb",
    "tensorboard",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".parquet",
    ".sqlite",
    ".db",
    ".tmp",
}

INCLUDE_ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
}

INCLUDE_DIRS = {
    ".github",
    "configs",
    "demo",
    "docs",
    "examples",
    "scripts",
    "src",
    "tests",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return False
    if path.name.startswith("~$"):
        return False
    if path.is_dir():
        return rel.parts[0] in INCLUDE_DIRS
    if rel.parts[0] in INCLUDE_DIRS:
        return path.suffix.lower() not in EXCLUDE_SUFFIXES
    return str(rel) in INCLUDE_ROOT_FILES


def iter_files() -> list[Path]:
    files: list[Path] = []
    for item in ROOT.iterdir():
        if item.is_file() and should_include(item):
            files.append(item)
        elif item.is_dir() and item.name in INCLUDE_DIRS and should_include(item):
            for child in item.rglob("*"):
                if child.is_file() and should_include(child):
                    files.append(child)
    return sorted(files, key=lambda p: str(p.relative_to(ROOT)).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean course-project submission package without data/model artifacts.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for the zip package.")
    parser.add_argument("--name", default=None, help="Optional zip file name. Defaults to E3-VDT-OOC-submission-<commit>.zip")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    zip_name = args.name or f"E3-VDT-OOC-submission-{commit}.zip"
    out_path = out_dir / zip_name

    files = iter_files()
    manifest = {
        "project": "E3-VDT-OOC",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "file_count": len(files) + 1,
        "notes": [
            "This package excludes datasets, model weights, checkpoints, caches, and local paths.",
            "Large reproduction artifacts are documented in docs/REPRODUCTION_STATUS.md but not included.",
        ],
    }

    with ZipFile(out_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("SUBMISSION_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(out_path)
    print(f"[OK] packaged {len(files)} files + manifest, size={size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
