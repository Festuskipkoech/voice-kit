#!/usr/bin/env python3
"""
Voicekit model downloader.

Reads the model registry from /etc/voicekit/models.toml (baked into the image).
Reads operator config.toml only to confirm it is present before starting.

Usage:
    python download_models.py --config /data/config.toml --dest /models

Exit codes:
    0 - all models present and verified
    1 - one or more downloads failed or checksum mismatched
"""

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: 'huggingface-hub' not installed.")
    sys.exit(1)

REGISTRY_PATH = Path("/etc/voicekit/models.toml")


def load_registry(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)["models"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def model_dir(dest: Path, entry: dict) -> Path:
    """Return the directory that holds this model's files.

    For single-file models (e.g. a .gguf or .onnx) the registry path
    includes the filename, so the directory is its parent.
    For multi-file models the registry path is already the directory.
    """
    p = dest / entry["path"]
    files = entry.get("files", [])
    # single-file: path ends with the one file listed
    if len(files) == 1 and p.suffix:
        return p.parent
    return p


def check_model(dest: Path, entry: dict) -> tuple[bool, list[str]]:
    mdir = model_dir(dest, entry)
    files = entry.get("files", [])

    if not files:
        return mdir.exists() and any(mdir.iterdir()), []

    missing = [f for f in files if not (mdir / f).exists()]
    return len(missing) == 0, missing


def verify_checksums(dest: Path, entry: dict) -> bool:
    sha256_map = entry.get("sha256", {})
    if not sha256_map:
        return True

    mdir = model_dir(dest, entry)
    all_ok = True

    for filename, expected in sha256_map.items():
        if not expected:
            continue
        path = mdir / filename
        if not path.exists():
            print(f"  MISSING for checksum: {path}")
            all_ok = False
            continue
        actual = sha256_file(path)
        if actual != expected:
            print(f"  CHECKSUM MISMATCH: {filename}")
            print(f"    expected: {expected}")
            print(f"    got:      {actual}")
            path.unlink()
            all_ok = False
        else:
            print(f"  checksum ok: {filename}")

    return all_ok


def download_model(dest: Path, name: str, entry: dict) -> bool:
    files = entry.get("files", [])
    mdir = model_dir(dest, entry)
    mdir.mkdir(parents=True, exist_ok=True)

    repo = entry["repo"]
    try:
        if files:
            for filename in files:
                target = mdir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"  downloading {repo}/{filename} ...")
                hf_hub_download(
                    repo_id=repo,
                    filename=f"{entry['path']}/{filename}",
                    local_dir=str(dest),
                    local_dir_use_symlinks=False,
                )
        else:
            print(f"  downloading full repo {repo} ...")
            snapshot_download(repo_id=repo, local_dir=str(mdir))
        return True
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Voicekit model downloader")
    parser.add_argument("--config", required=True, help="path to operator config.toml")
    parser.add_argument(
        "--dest", required=True, help="destination directory for models"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    dest = Path(args.dest)

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    if not REGISTRY_PATH.exists():
        print(f"ERROR: model registry not found: {REGISTRY_PATH}")
        sys.exit(1)

    registry = load_registry(REGISTRY_PATH)
    dest.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    dest.chmod(0o777)
    
    failed = []

    for name, entry in registry.items():
        print(f"\n[model-init] {name}")

        present, missing = check_model(dest, entry)

        if present:
            print("  present - verifying checksums")
            if not verify_checksums(dest, entry):
                print("  checksum failed - re-downloading")
                if not download_model(dest, name, entry):
                    failed.append(name)
                elif not verify_checksums(dest, entry):
                    print("  checksum still failed after re-download")
                    failed.append(name)
        else:
            print(f"  missing: {missing}")
            if not download_model(dest, name, entry):
                failed.append(name)
            elif not verify_checksums(dest, entry):
                print("  checksum failed after download")
                failed.append(name)
            else:
                print("  downloaded and verified")

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("[model-init] all models ready")


if __name__ == "__main__":
    main()
