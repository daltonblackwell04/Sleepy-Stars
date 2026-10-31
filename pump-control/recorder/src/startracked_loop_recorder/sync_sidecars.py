from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .storage import file_sha256


def sidecars(root: Path):
    for camera in "ABCD":
        for path in (root / camera).glob("recording_start-*/*/*"):
            if path.suffix in {".json", ".jsonl"}:
                yield path


def sync(primary: Path, backup: Path, dry_run: bool = False) -> tuple[int, int]:
    copied = skipped = 0
    for source in sidecars(primary):
        destination = backup / source.relative_to(primary)
        if destination.exists() and file_sha256(source) == file_sha256(destination):
            skipped += 1
            continue
        print(f"{source} -> {destination}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
            shutil.copy2(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        copied += 1
    return copied, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Atomically synchronize StarSeg sidecars")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.primary.is_absolute() or not args.backup.is_absolute():
        raise SystemExit("paths must be absolute")
    copied, skipped = sync(args.primary, args.backup, args.dry_run)
    print(f"copied={copied} skipped={skipped}")
    return 0
