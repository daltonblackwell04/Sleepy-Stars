#!/usr/bin/env python3
"""Extract diverse food-present frames from finalized Basler recordings.

Reads annotation sidecars and finalized segment metadata only. It never reads
_active_segments, never controls cameras, and never modifies recorder files.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, math, os, subprocess, sys
from collections import defaultdict, deque
from pathlib import Path

RECORD_ROOT = Path(os.environ.get('BASLER_RECORD_ROOT', '~/Videos/Basler Loop Recordings')).expanduser()
ANNOTATION_ROOT = Path(os.environ.get('BASLER_ANNOTATION_CACHE', '~/.cache/basler-annotation-interface')).expanduser() / 'annotations'
DEFAULT_OUTPUT = Path(__file__).resolve().with_name('starseg_food_training_frames')
CAMERA_QUOTA = {'C': 34, 'A': 33, 'B': 33}


def play_to_wall(t: float, pieces: list[dict], clip_start: float) -> float:
    for p in pieces:
        p0 = float(p.get('play_start') or 0)
        dur = float(p.get('play_dur') or 0)
        if dur <= 0:
            continue
        if t <= p0 + dur + 1e-9:
            frac = min(1.0, max(0.0, (t - p0) / dur))
            ws, we = p.get('wall_start_ts'), p.get('wall_end_ts')
            if ws is not None and we is not None:
                return float(ws) + frac * (float(we) - float(ws))
            break
    return float(clip_start) + float(t) * 3600.0


def load_eating_intervals() -> list[dict]:
    out = []
    for path in sorted(ANNOTATION_ROOT.glob('*.annotations.json')):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        events = sorted(data.get('feeding_events') or [], key=lambda x: float(x.get('time') or 0))
        if not events:
            continue
        meta = data.get('meta') or {}
        camera = str(meta.get('camera') or '').upper()
        trial, date = meta.get('trial'), meta.get('date')
        clip_start = meta.get('clip_start_ts')
        pieces = meta.get('pieces') or []
        duration = float(data.get('duration_seconds') or meta.get('target_duration_seconds') or 0)
        if camera not in CAMERA_QUOTA or not trial or not date or clip_start is None or duration <= 0:
            continue
        for i, ev in enumerate(events):
            if ev.get('state') != 'eating':
                continue
            a = max(0.0, float(ev.get('time') or 0))
            b = min(duration, float(events[i + 1].get('time')) if i + 1 < len(events) else duration)
            if b <= a:
                continue
            wa, wb = play_to_wall(a, pieces, float(clip_start)), play_to_wall(b, pieces, float(clip_start))
            if wb > wa:
                out.append({'trial': trial, 'date': date, 'camera': camera,
                            'play_start': a, 'play_end': b, 'wall_start_ts': wa,
                            'wall_end_ts': wb, 'annotation_path': str(path)})
    return out


def load_finalized_segments(trials: set[str]) -> dict[str, list[dict]]:
    by_cam = defaultdict(list)
    for camera in CAMERA_QUOTA:
        for meta_path in (RECORD_ROOT / camera).glob('recording_start-*/*/*.json'):
            if meta_path.name.endswith('_starseg_summary.json'):
                continue
            try:
                data = json.loads(meta_path.read_text())
                trial = str(data.get('recording_session') or meta_path.parts[-3])
                if trial not in trials:
                    continue
                source = Path(data.get('file') or meta_path.with_suffix('.mp4'))
                if not source.is_file() or '_active_segments' in source.parts or 'preview' in source.parts:
                    continue
                start = float(data.get('start_ts'))
                end = float(data.get('end_ts') or (start + float(data.get('duration_seconds') or 0)))
                if end <= start:
                    continue
                by_cam[camera].append({'camera': camera, 'trial': trial, 'start_ts': start,
                    'end_ts': end, 'frames': int(data.get('frames') or 0),
                    'fps': float(data.get('fps') or 1.0), 'source_mp4': str(source),
                    'source_metadata': str(meta_path)})
            except Exception:
                continue
        by_cam[camera].sort(key=lambda s: s['start_ts'])
    return by_cam


def segment_at(segments: list[dict], ts: float) -> dict | None:
    # Lists are modest for the one annotated trial; linear lookup is deterministic.
    for s in segments:
        if s['start_ts'] <= ts < s['end_ts']:
            return s
    return None


def candidate_times(interval: dict) -> list[float]:
    a, b = interval['wall_start_ts'], interval['wall_end_ts']
    dur = b - a
    vals = []
    # Favor feeding-state transitions, then cover long eating intervals every 5 min.
    for off in (15, 45, 90, 180):
        if a + off < b - 2:
            vals.append(a + off)
    t = a + 150
    while t < b - 30:
        vals.append(t)
        t += 300
    if dur > 60:
        vals.append(a + dur * 0.5)
        vals.append(b - 30)
    return sorted(set(round(x, 3) for x in vals if a + 2 <= x <= b - 2))


def evenly_take(rows: list[dict], n: int) -> list[dict]:
    if n >= len(rows):
        return rows[:]
    # Even temporal coverage without adjacent near-duplicates.
    idxs = []
    for i in range(n):
        idx = round(i * (len(rows) - 1) / max(1, n - 1))
        if idx not in idxs:
            idxs.append(idx)
    for idx in range(len(rows)):
        if len(idxs) >= n:
            break
        if idx not in idxs:
            idxs.append(idx)
    return [rows[i] for i in sorted(idxs[:n])]


def build_plan(intervals: list[dict], segments: dict[str, list[dict]]) -> list[dict]:
    pools = defaultdict(list)
    seen = set()
    for iv in intervals:
        for ts in candidate_times(iv):
            seg = segment_at(segments[iv['camera']], ts)
            if not seg or seg['trial'] != iv['trial']:
                continue
            offset = max(0.0, ts - seg['start_ts'])
            frame_index = int(round(offset * max(seg['fps'], 1e-6)))
            key = (seg['source_mp4'], frame_index)
            if key in seen:
                continue
            seen.add(key)
            row = {**iv, **seg, 'wall_ts': ts, 'source_offset_seconds': offset,
                   'source_frame_index': frame_index}
            pools[(iv['camera'], iv['date'])].append(row)
    selected = []
    for cam, quota in CAMERA_QUOTA.items():
        groups = {date: sorted(rows, key=lambda r: r['wall_ts'])
                  for (c, date), rows in pools.items() if c == cam and rows}
        if not groups:
            raise RuntimeError(f'No food-present finalized candidates for Camera {cam}')
        per_date = {d: evenly_take(rows, min(len(rows), math.ceil(quota / len(groups)) + 2))
                    for d, rows in groups.items()}
        queues = {d: deque(rows) for d, rows in sorted(per_date.items())}
        cam_rows = []
        while len(cam_rows) < quota and any(queues.values()):
            for d in sorted(queues):
                if queues[d] and len(cam_rows) < quota:
                    cam_rows.append(queues[d].popleft())
        if len(cam_rows) < quota:
            used = {(r['source_mp4'], r['source_frame_index']) for r in cam_rows}
            remainder = sorted((r for (c, _), rows in pools.items() if c == cam for r in rows
                                if (r['source_mp4'], r['source_frame_index']) not in used),
                               key=lambda r: r['wall_ts'])
            cam_rows.extend(evenly_take(remainder, quota - len(cam_rows)))
        if len(cam_rows) != quota:
            raise RuntimeError(f'Camera {cam}: needed {quota}, found {len(cam_rows)}')
        selected.extend(cam_rows)
    selected.sort(key=lambda r: ('CAB'.index(r['camera']), r['wall_ts']))
    for i, row in enumerate(selected, 1):
        stamp = dt.datetime.fromtimestamp(row['wall_ts']).strftime('%Y%m%dT%H%M%S')
        row['id'] = f'food_{i:03d}'
        row['output_file'] = f"images/food_{i:03d}_camera-{row['camera']}_{stamp}.jpg"
        row['selection_basis'] = 'feeding annotation state=eating; transition-biased and temporally stratified'
    return selected


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract(plan: list[dict], output: Path) -> None:
    images = output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(plan, 1):
        dst = output / row['output_file']
        if dst.exists() and dst.stat().st_size > 0:
            continue
        cmd = ['ionice', '-c3', 'nice', '-n', '19', 'ffmpeg', '-hide_banner', '-loglevel', 'error',
               '-ss', f"{row['source_offset_seconds']:.3f}", '-i', row['source_mp4'],
               '-frames:v', '1', '-q:v', '2', '-y', str(dst)]
        subprocess.run(cmd, check=True)
        if not dst.exists() or dst.stat().st_size <= 0:
            raise RuntimeError(f'Extraction failed: {dst}')
        print(f'[{i:03d}/100] {dst.name}', flush=True)


def write_manifests(plan: list[dict], output: Path) -> None:
    for row in plan:
        p = output / row['output_file']
        row['output_path'] = str(p)
        if p.exists():
            row['output_bytes'] = p.stat().st_size
            row['sha256'] = sha256(p)
    (output / 'manifest.json').write_text(json.dumps(plan, indent=2))
    fields = ['id','output_file','camera','trial','date','wall_ts','source_mp4','source_metadata',
              'source_offset_seconds','source_frame_index','annotation_path','play_start','play_end',
              'wall_start_ts','wall_end_ts','selection_basis','output_bytes','sha256']
    with (output / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(plan)
    readme = f"""StarSeg food adaptation frame set\n\nImages: {len(plan)}\nSource: finalized Basler MP4 archive only\nSelection: frames inside human feeding-state intervals labeled eating\nCamera allocation: C=34, A=33, B=33\nExtraction: full source resolution, JPEG quality 2, low CPU/I/O priority\nLabels: not yet created; these images require manual food-object annotation before training.\n\nProvenance is in manifest.json and manifest.csv. Recorder files were read only.\n"""
    (output / 'README.txt').write_text(readme)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    intervals = load_eating_intervals()
    segments = load_finalized_segments({x['trial'] for x in intervals})
    plan = build_plan(intervals, segments)
    if len(plan) != 100:
        raise RuntimeError(f'Expected 100 planned frames, got {len(plan)}')
    (args.output / 'extraction_plan.json').write_text(json.dumps(plan, indent=2))
    if not args.plan_only:
        extract(plan, args.output)
        write_manifests(plan, args.output)
    print(json.dumps({'output': str(args.output), 'planned': len(plan),
                      'cameras': {c: sum(r['camera'] == c for r in plan) for c in CAMERA_QUOTA},
                      'dates': sorted({r['date'] for r in plan})}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
