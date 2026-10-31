#!/usr/bin/env python3
"""Mirror back-cast StarSeg sidecars from the primary recordings to the backup.

This is the no-GPU half of the LaCie rollover. For every video that exists on
BOTH the primary record root and the backup, and whose PRIMARY sidecars have
already been re-run to the target model, it copies the three rewritten StarSeg
files to the backup (the mp4 is unchanged and never touched):

  <stem>_starseg_frames.jsonl
  <stem>_starseg_summary.json
  <stem>.json                 (meta sidecar; its starseg_summary block changed)

Backup sidecars are byte-copies of the primary ones (they already embed primary
absolute paths), so this preserves the existing backup convention exactly.

It is idempotent (skips backups already on the target model), atomic (temp file +
os.replace, original mode preserved), and gentle (nice + optional throttle). The
~846 videos that live ONLY on the backup are not handled here; run
`backcast_starseg.py --root <backup>` afterward to cover those (the overlapping
videos synced here will then be skipped as already-done).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIMARY = Path(os.environ.get("BASLER_RECORD_ROOT", "~/Videos/Basler Loop Recordings")).expanduser()
DEFAULT_BACKUP = Path(os.environ.get("BASLER_BACKUP_ROOT", f"/media/{os.environ.get('USER', 'user')}/LaCie/Basler Loop Recordings")).expanduser()
DEFAULT_MODEL = Path(os.environ.get("BASLER_STARSEG_MODEL", str(PROJECT_DIR / "models" / "best.pt"))).expanduser()
SKIP_DIRS = {"_active_segments", "preview"}


def iso(ts):
    return time.strftime("%Y%m%dT%H%M%S", time.localtime(ts)) + f"{int((ts % 1) * 1000):03d}"


def log(msg):
    print(f"[{iso(time.time())}] {msg}", flush=True)


def find_videos(root):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".mp4"):
                continue
            mp4 = Path(dirpath) / name
            stem = name[:-4]
            if mp4.parent.name != stem:
                continue
            yield mp4, stem


def summary_model(path):
    try:
        return json.loads(Path(path).read_text()).get("model")
    except Exception:
        return None


def copy_atomic(src, dst):
    src, dst = Path(src), Path(dst)
    data = src.read_bytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    mode = None
    if dst.exists():
        try:
            mode = dst.stat().st_mode & 0o777
        except OSError:
            mode = None
    if mode is None:
        mode = src.stat().st_mode & 0o777
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix="." + dst.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser(description="Sync back-cast StarSeg sidecars from primary to backup.")
    ap.add_argument("--primary", default=str(DEFAULT_PRIMARY), help="primary recording root")
    ap.add_argument("--backup", default=str(DEFAULT_BACKUP), help="backup root (e.g. LaCie)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="target model the primary must already be on")
    ap.add_argument("--nice", type=int, default=19, help="process niceness")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between videos")
    ap.add_argument("--dry-run", action="store_true", help="report what would be copied and exit")
    args = ap.parse_args()

    try:
        os.nice(args.nice)
    except OSError:
        pass

    primary = Path(args.primary)
    backup = Path(args.backup)
    model_str = str(Path(args.model).expanduser())
    if not backup.exists():
        log(f"ERROR: backup root not mounted: {backup}")
        return 2

    log(f"primary={primary}")
    log(f"backup={backup}")
    log(f"target model={model_str}")

    counts = {"copied": 0, "skip_primary_not_done": 0, "skip_backup_done": 0, "skip_no_backup": 0, "errors": 0}
    for mp4, stem in sorted(find_videos(primary), key=lambda t: str(t[0])):
        rel = mp4.parent.relative_to(primary)
        bdir = backup / rel
        psum = mp4.parent / f"{stem}_starseg_summary.json"
        if summary_model(psum) != model_str:
            counts["skip_primary_not_done"] += 1
            continue
        if not bdir.is_dir() or not (bdir / f"{stem}.mp4").exists():
            counts["skip_no_backup"] += 1
            continue
        bsum = bdir / f"{stem}_starseg_summary.json"
        if summary_model(bsum) == model_str:
            counts["skip_backup_done"] += 1
            continue
        if args.dry_run:
            counts["copied"] += 1
            continue
        try:
            for suffix in (f"{stem}_starseg_frames.jsonl", f"{stem}_starseg_summary.json", f"{stem}.json"):
                src = mp4.parent / suffix
                if src.exists():
                    copy_atomic(src, bdir / suffix)
            counts["copied"] += 1
            if counts["copied"] % 50 == 0:
                log(f"progress: {counts}")
            if args.delay > 0:
                time.sleep(args.delay)
        except Exception as e:
            counts["errors"] += 1
            log(f"ERROR {rel}/{stem}: {e!r}")

    verb = "would copy" if args.dry_run else "copied"
    log(f"done: {verb}={counts['copied']} backup_already_done={counts['skip_backup_done']} primary_not_done={counts['skip_primary_not_done']} no_backup={counts['skip_no_backup']} errors={counts['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
