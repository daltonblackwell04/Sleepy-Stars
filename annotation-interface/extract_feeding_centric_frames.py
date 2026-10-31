#!/usr/bin/env python3
"""Extract 100 sea-star-centric crops from human feeding ethograms.

Reads finalized recorder MP4s, StarSeg sidecars, and annotation-cache ethograms
only. Never touches _active_segments, never controls cameras, and never modifies
recorder files.

Scope: Experiment 2 (trial_21+), the cohort where feeding began to be tracked.
Selection: frames whose human ethogram state is eating and that have a
Pycnopodia detection. The 100-frame set is stratified across cameras and then
chosen for positional and temporal spread.
Crop: square whose side is 2x the detection box width, centered on the box
and clamped to the source frame.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

RECORD_ROOT = Path(os.environ.get('BASLER_RECORD_ROOT', '~/Videos/Basler Loop Recordings')).expanduser()
ANNOTATION_ROOT = Path(os.environ.get(
    'BASLER_ANNOTATION_CACHE', '~/.cache/basler-annotation-interface'
)).expanduser() / 'annotations'
DEFAULT_OUTPUT = Path(__file__).resolve().with_name('feeding_centric_frames')
STAR_CLASS = 'Pycnopodia_helianthoides'
CAMERA_ORDER = ['C', 'A', 'B']
CAMERA_QUOTA = {'C': 34, 'A': 33, 'B': 33}
MIN_CONF = 0.25
MIN_BOX_WIDTH = 40.0
CANDIDATE_STRIDE_S = 60.0
MIN_SAME_CAMERA_GAP_S = 15 * 60.0
MIN_FILE_AGE_S = 120.0


def trial_number(heading: str) -> int | None:
    match = re.fullmatch(r'trial_(\d+)_', str(heading or '').strip())
    return int(match.group(1)) if match else None


def is_feeding_experiment(heading: str) -> bool:
    number = trial_number(heading)
    return number is not None and number >= 21


def play_to_wall(t: float, pieces: list[dict], clip_start: float) -> float:
    for piece in pieces:
        start = float(piece.get('play_start') or 0)
        dur = float(piece.get('play_dur') or 0)
        if dur <= 0:
            continue
        if t <= start + dur + 1e-9:
            frac = min(1.0, max(0.0, (t - start) / dur))
            wall_start, wall_end = piece.get('wall_start_ts'), piece.get('wall_end_ts')
            if wall_start is not None and wall_end is not None:
                return float(wall_start) + frac * (float(wall_end) - float(wall_start))
            break
    return float(clip_start) + float(t) * 3600.0


def merge_intervals(rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not rows:
        return []
    rows = sorted(rows)
    out = [list(rows[0])]
    for start, end in rows[1:]:
        if start <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


class IntervalIndex:
    def __init__(self, intervals: list[tuple[float, float]]):
        self.intervals = merge_intervals(intervals)
        self.starts = [a for a, _ in self.intervals]

    def contains(self, ts: float) -> bool:
        i = bisect_right(self.starts, ts) - 1
        return i >= 0 and self.intervals[i][0] <= ts < self.intervals[i][1]

    def overlaps(self, start: float, end: float) -> bool:
        if end <= start or not self.intervals:
            return False
        i = bisect_right(self.starts, end - 1e-9) - 1
        j = bisect_right(self.starts, start)
        lo = max(0, i)
        hi = min(len(self.intervals), j + 1)
        for a, b in self.intervals[max(0, lo - 1):hi]:
            if a < end and b > start:
                return True
        return False


def load_eating_intervals() -> tuple[list[dict], dict[str, IntervalIndex]]:
    out = []
    by_camera = defaultdict(list)
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
        trial = meta.get('trial')
        date = meta.get('date')
        clip_start = meta.get('clip_start_ts')
        pieces = meta.get('pieces') or []
        duration = float(data.get('duration_seconds') or meta.get('target_duration_seconds') or 0)
        if camera not in CAMERA_QUOTA or not trial or not date or clip_start is None or duration <= 0:
            continue
        for i, ev in enumerate(events):
            if ev.get('state') != 'eating':
                continue
            start = max(0.0, float(ev.get('time') or 0))
            end = min(duration, float(events[i + 1].get('time')) if i + 1 < len(events) else duration)
            if end <= start:
                continue
            wall_start = play_to_wall(start, pieces, float(clip_start))
            wall_end = play_to_wall(end, pieces, float(clip_start))
            if wall_end <= wall_start:
                continue
            row = {
                'trial': trial,
                'date': date,
                'camera': camera,
                'play_start': start,
                'play_end': end,
                'wall_start_ts': wall_start,
                'wall_end_ts': wall_end,
                'annotation_path': str(path),
            }
            out.append(row)
            by_camera[camera].append((wall_start, wall_end))
    indexes = {cam: IntervalIndex(spans) for cam, spans in by_camera.items()}
    return out, indexes


def load_overlapping_segments(indexes: dict[str, IntervalIndex]) -> list[dict]:
    now = dt.datetime.now().timestamp()
    sessions = set()
    for path in ANNOTATION_ROOT.glob('*.annotations.json'):
        name = path.name
        if '__' in name:
            sessions.add(name.split('__', 1)[0])
    rows = []
    for camera in CAMERA_ORDER:
        index = indexes.get(camera)
        if index is None:
            continue
        cam_root = RECORD_ROOT / camera
        if not cam_root.exists():
            continue
        for meta_path in cam_root.glob('recording_start-*/*/*.json'):
            if meta_path.name.endswith('_starseg_summary.json'):
                continue
            if '_active_segments' in meta_path.parts or 'preview' in meta_path.parts:
                continue
            session = meta_path.parts[-3]
            if sessions and session not in sessions:
                continue
            try:
                if now - meta_path.stat().st_mtime < MIN_FILE_AGE_S:
                    continue
                data = json.loads(meta_path.read_text())
            except Exception:
                continue
            heading = str(data.get('heading') or '')
            if not is_feeding_experiment(heading):
                continue
            source = Path(data.get('file') or meta_path.with_suffix('.mp4'))
            if not source.is_file() or '_active_segments' in source.parts:
                continue
            start = float(data.get('start_ts'))
            end = float(data.get('end_ts') or (start + float(data.get('duration_seconds') or 0)))
            if end <= start or not index.overlaps(start, end):
                continue
            jsonl = data.get('starseg_frames_jsonl')
            jsonl_path = Path(jsonl) if jsonl else source.with_name(source.stem + '_starseg_frames.jsonl')
            if not jsonl_path.is_file():
                continue
            rows.append({
                'camera': camera,
                'trial': str(data.get('recording_session') or session),
                'heading': heading,
                'start_ts': start,
                'end_ts': end,
                'frames': int(data.get('frames') or 0),
                'fps': float(data.get('fps') or 1.0),
                'width': int(data.get('width') or 0),
                'height': int(data.get('height') or 0),
                'source_mp4': str(source),
                'source_metadata': str(meta_path),
                'starseg_jsonl': str(jsonl_path),
            })
    rows.sort(key=lambda r: (r['camera'], r['start_ts']))
    return rows


def best_star(dets: list[dict]) -> dict | None:
    best = None
    best_score = -1.0
    for det in dets or []:
        if str(det.get('class_name') or '') != STAR_CLASS:
            continue
        box = det.get('box_xyxy') or []
        if len(box) != 4:
            continue
        width = float(box[2]) - float(box[0])
        if width < MIN_BOX_WIDTH:
            continue
        conf = float(det.get('confidence') or 0)
        if conf < MIN_CONF:
            continue
        if conf > best_score:
            best = det
            best_score = conf
    return best


def interval_lookup(intervals: list[dict]) -> dict[str, list[dict]]:
    by_cam = defaultdict(list)
    for row in intervals:
        by_cam[row['camera']].append(row)
    for cam in by_cam:
        by_cam[cam].sort(key=lambda r: r['wall_start_ts'])
    return by_cam


def matching_interval(by_cam: dict[str, list[dict]], camera: str, ts: float) -> dict | None:
    rows = by_cam.get(camera) or []
    starts = [r['wall_start_ts'] for r in rows]
    i = bisect_right(starts, ts) - 1
    if i >= 0 and rows[i]['wall_start_ts'] <= ts < rows[i]['wall_end_ts']:
        return rows[i]
    return None


def collect_candidates(segments: list[dict], indexes: dict[str, IntervalIndex], intervals: list[dict]) -> list[dict]:
    kept = {}
    scanned = 0
    used = 0
    by_cam = interval_lookup(intervals)
    for si, seg in enumerate(segments, 1):
        index = indexes[seg['camera']]
        try:
            with open(seg['starseg_jsonl'], encoding='utf-8') as fh:
                for line in fh:
                    scanned += 1
                    if not line or line[0] != '{':
                        continue
                    rec = json.loads(line)
                    ts = rec.get('frame_ts')
                    if ts is None or not index.contains(float(ts)):
                        continue
                    det = best_star(rec.get('detections') or [])
                    if det is None:
                        continue
                    box = [float(v) for v in det['box_xyxy']]
                    width = max(1.0, box[2] - box[0])
                    height = max(1.0, box[3] - box[1])
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    img_w = float(rec.get('width') or seg['width'] or 1)
                    img_h = float(rec.get('height') or seg['height'] or 1)
                    interval = matching_interval(by_cam, seg['camera'], float(ts))
                    bin_key = (
                        seg['camera'],
                        int(float(ts) // CANDIDATE_STRIDE_S),
                    )
                    row = {
                        'camera': seg['camera'],
                        'trial': seg['trial'],
                        'heading': seg['heading'],
                        'date': dt.datetime.fromtimestamp(float(ts)).date().isoformat(),
                        'wall_ts': float(ts),
                        'source_mp4': seg['source_mp4'],
                        'source_metadata': seg['source_metadata'],
                        'starseg_jsonl': seg['starseg_jsonl'],
                        'source_offset_seconds': max(0.0, float(ts) - seg['start_ts']),
                        'source_frame_index': int(rec.get('frame_index') or 0),
                        'image_width': int(img_w),
                        'image_height': int(img_h),
                        'box_xyxy': [round(v, 2) for v in box],
                        'box_width': round(width, 2),
                        'box_height': round(height, 2),
                        'center_x': round(cx, 2),
                        'center_y': round(cy, 2),
                        'nx': cx / img_w,
                        'ny': cy / img_h,
                        'confidence': float(det.get('confidence') or 0),
                        'model': rec.get('model') or '',
                        'annotation_path': (interval or {}).get('annotation_path', ''),
                        'play_start': (interval or {}).get('play_start'),
                        'play_end': (interval or {}).get('play_end'),
                        'wall_start_ts': (interval or {}).get('wall_start_ts'),
                        'wall_end_ts': (interval or {}).get('wall_end_ts'),
                    }
                    prev = kept.get(bin_key)
                    if prev is None or row['confidence'] > prev['confidence']:
                        kept[bin_key] = row
                        used += 1
        except Exception:
            continue
        if si == 1 or si % 50 == 0 or si == len(segments):
            print(f'scanned_segments={si}/{len(segments)} candidates={len(kept)}', flush=True)
    print(f'jsonl_lines_seen={scanned} bins_kept={len(kept)} detections_accepted={used}', flush=True)
    return list(kept.values())


def annotate_time_norm(rows: list[dict]) -> None:
    if not rows:
        return
    t0 = min(r['wall_ts'] for r in rows)
    t1 = max(r['wall_ts'] for r in rows)
    span = max(1.0, t1 - t0)
    for row in rows:
        row['nt'] = (row['wall_ts'] - t0) / span


def feature_dist(a: dict, b: dict) -> float:
    dx = a['nx'] - b['nx']
    dy = a['ny'] - b['ny']
    dt = a['nt'] - b['nt']
    dcam = 0.0 if a['camera'] == b['camera'] else 0.40
    dtrial = 0.0 if a['heading'] == b['heading'] else 0.18
    return math.sqrt(1.4 * (dx * dx + dy * dy) + 1.0 * dt * dt) + dcam + dtrial


def min_dist(row: dict, chosen: list[dict]) -> float:
    if not chosen:
        return 1e9
    return min(feature_dist(row, other) for other in chosen)


def too_close_in_time(row: dict, chosen: list[dict]) -> bool:
    for other in chosen:
        if other['camera'] == row['camera'] and abs(other['wall_ts'] - row['wall_ts']) < MIN_SAME_CAMERA_GAP_S:
            return True
    return False


def pick_group_seeds(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['camera'], row['heading'])].append(row)
    seeds = []
    used = set()
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: (r['wall_ts'], r['nx'], r['ny']))
        # Mid-interval representative, then spatially farthest if needed later.
        seed = group[len(group) // 2]
        ident = id(seed)
        if ident not in used:
            seeds.append(seed)
            used.add(ident)
    return seeds


def select_diverse(rows: list[dict], n: int = 100) -> list[dict]:
    annotate_time_norm(rows)
    by_cam = defaultdict(list)
    for row in rows:
        by_cam[row['camera']].append(row)
    for cam in by_cam:
        by_cam[cam].sort(key=lambda r: (r['wall_ts'], r['nx'], r['ny']))

    quotas = dict(CAMERA_QUOTA)
    available = {cam: len(by_cam[cam]) for cam in CAMERA_ORDER}
    leftover = 0
    for cam in CAMERA_ORDER:
        take = min(quotas[cam], available.get(cam, 0))
        leftover += quotas[cam] - take
        quotas[cam] = take
    for cam in CAMERA_ORDER:
        extra = min(leftover, max(0, available.get(cam, 0) - quotas[cam]))
        quotas[cam] += extra
        leftover -= extra
    if sum(quotas.values()) < n:
        n = sum(quotas.values())

    selected = []
    selected_ids = set()
    counts = {cam: 0 for cam in CAMERA_ORDER}

    def add(row: dict) -> bool:
        ident = id(row)
        if ident in selected_ids:
            return False
        if counts[row['camera']] >= quotas[row['camera']]:
            return False
        selected.append(row)
        selected_ids.add(ident)
        counts[row['camera']] += 1
        return True

    for seed in pick_group_seeds(rows):
        if len(selected) >= n:
            break
        add(seed)

    remaining = [row for row in rows if id(row) not in selected_ids]
    while len(selected) < n and remaining:
        best = None
        best_score = -1.0
        relaxed = []
        for row in remaining:
            if counts[row['camera']] >= quotas[row['camera']]:
                continue
            close = too_close_in_time(row, selected)
            score = min_dist(row, selected)
            if close:
                relaxed.append((score, row))
                continue
            if score > best_score:
                best = row
                best_score = score
        if best is None and relaxed:
            relaxed.sort(key=lambda item: item[0], reverse=True)
            best = relaxed[0][1]
        if best is None:
            break
        add(best)
        remaining = [row for row in remaining if id(row) not in selected_ids]

    selected.sort(key=lambda r: (CAMERA_ORDER.index(r['camera']), r['wall_ts'], r['nx'], r['ny']))
    if len(selected) > n:
        selected = selected[:n]
    return selected


def crop_window(box: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = min(max(2, int(round(2.0 * width))), int(img_w))
    crop_h = min(max(2, int(round(2.0 * width))), int(img_h))
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x0 = max(0, min(x0, int(img_w) - crop_w))
    y0 = max(0, min(y0, int(img_h) - crop_h))
    return crop_w, crop_h, x0, y0


def finalize_plan(rows: list[dict]) -> list[dict]:
    plan = []
    for i, row in enumerate(rows, 1):
        crop_w, crop_h, x0, y0 = crop_window(row['box_xyxy'], row['image_width'], row['image_height'])
        stamp = dt.datetime.fromtimestamp(row['wall_ts']).strftime('%Y%m%dT%H%M%S')
        heading = (row['heading'] or 'trial').strip('_')
        item = dict(row)
        item.update({
            'id': f'feed_{i:03d}',
            'crop_width': crop_w,
            'crop_height': crop_h,
            'crop_x': x0,
            'crop_y': y0,
            'output_file': (
                f"images/feed_{i:03d}_{heading}_camera-{row['camera']}_{stamp}"
                f"_x{int(round(row['center_x']))}_y{int(round(row['center_y']))}.jpg"
            ),
            'selection_basis': (
                'human ethogram state=eating; Experiment 2 feeding-tracking trials; '
                'StarSeg Pycnopodia detection; camera-stratified farthest-point over '
                'normalized position and time'
            ),
        })
        plan.append(item)
    return plan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(plan: list[dict], output: Path) -> None:
    images = output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(plan, 1):
        dst = output / row['output_file']
        if dst.exists() and dst.stat().st_size > 0:
            print(f'[{i:03d}/{len(plan)}] exists {dst.name}', flush=True)
            continue
        vf = f"crop={row['crop_width']}:{row['crop_height']}:{row['crop_x']}:{row['crop_y']}"
        cmd = [
            'ionice', '-c3', 'nice', '-n', '19',
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-ss', f"{row['source_offset_seconds']:.3f}",
            '-i', row['source_mp4'],
            '-frames:v', '1',
            '-vf', vf,
            '-q:v', '2',
            '-y', str(dst),
        ]
        subprocess.run(cmd, check=True)
        if not dst.exists() or dst.stat().st_size <= 0:
            raise RuntimeError(f'Extraction failed: {dst}')
        print(f'[{i:03d}/{len(plan)}] {dst.name} {row["crop_width"]}x{row["crop_height"]}', flush=True)


def write_manifests(plan: list[dict], output: Path, intervals: list[dict], candidates: int) -> None:
    for row in plan:
        path = output / row['output_file']
        row['output_path'] = str(path)
        if path.exists():
            row['output_bytes'] = path.stat().st_size
            row['sha256'] = sha256(path)
    (output / 'manifest.json').write_text(json.dumps(plan, indent=2))
    fields = [
        'id', 'output_file', 'camera', 'heading', 'trial', 'date', 'wall_ts',
        'center_x', 'center_y', 'nx', 'ny', 'box_width', 'box_height',
        'crop_width', 'crop_height', 'crop_x', 'crop_y', 'confidence',
        'source_mp4', 'source_metadata', 'starseg_jsonl', 'source_offset_seconds',
        'source_frame_index', 'annotation_path', 'selection_basis', 'output_bytes', 'sha256',
    ]
    with (output / 'manifest.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(plan)
    cameras = {cam: sum(r['camera'] == cam for r in plan) for cam in CAMERA_ORDER}
    headings = sorted({r['heading'] for r in plan})
    dates = sorted({r['date'] for r in plan})
    readme = f"""Sea-star-centric feeding frame set

Images: {len(plan)}
Source: finalized Basler MP4 archive + StarSeg sidecars + human feeding ethograms
Cohort: Experiment 2 (trial_21+), where feeding began to be tracked
Eating intervals available: {len(intervals)}
Detection candidates after 60s thinning: {candidates}
Camera allocation: C={cameras['C']}, A={cameras['A']}, B={cameras['B']}
Trials: {', '.join(headings)}
Dates: {dates[0] if dates else '?'} to {dates[-1] if dates else '?'}
Crop: square side = 2 × StarSeg box width, centered on the detection, clamped to frame
Extraction: JPEG quality 2, low CPU/I/O priority, recorder files read only

Provenance is in manifest.json and manifest.csv.
"""
    (output / 'README.txt').write_text(readme)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    intervals, indexes = load_eating_intervals()
    if not intervals:
        raise RuntimeError('No eating intervals found in annotation cache')
    segments = load_overlapping_segments(indexes)
    if not segments:
        raise RuntimeError('No finalized StarSeg segments overlap eating intervals')
    print(json.dumps({
        'eating_intervals': len(intervals),
        'eating_hours': round(sum(i['wall_end_ts'] - i['wall_start_ts'] for i in intervals) / 3600.0, 3),
        'overlapping_segments': len(segments),
        'headings': sorted({s['heading'] for s in segments}),
        'cameras': {cam: sum(s['camera'] == cam for s in segments) for cam in CAMERA_ORDER},
    }, indent=2), flush=True)

    candidates = collect_candidates(segments, indexes, intervals)
    if len(candidates) < args.count:
        raise RuntimeError(f'Only {len(candidates)} feeding detections available, need {args.count}')
    selected = select_diverse(candidates, args.count)
    plan = finalize_plan(selected)
    (args.output / 'extraction_plan.json').write_text(json.dumps(plan, indent=2))
    (args.output / 'candidate_summary.json').write_text(json.dumps({
        'candidates': len(candidates),
        'selected': len(plan),
        'by_camera': {cam: sum(r['camera'] == cam for r in candidates) for cam in CAMERA_ORDER},
        'selected_by_camera': {cam: sum(r['camera'] == cam for r in plan) for cam in CAMERA_ORDER},
        'selected_by_heading': {h: sum(r['heading'] == h for r in plan) for h in sorted({r['heading'] for r in plan})},
        'selected_dates': sorted({r['date'] for r in plan}),
    }, indent=2))
    if not args.plan_only:
        extract(plan, args.output)
        write_manifests(plan, args.output, intervals, len(candidates))
    print(json.dumps({
        'output': str(args.output),
        'planned': len(plan),
        'candidates': len(candidates),
        'cameras': {cam: sum(r['camera'] == cam for r in plan) for cam in CAMERA_ORDER},
        'headings': sorted({r['heading'] for r in plan}),
        'dates': sorted({r['date'] for r in plan}),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
