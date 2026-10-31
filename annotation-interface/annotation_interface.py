#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt, hashlib, json, mimetypes, os, re, shutil, subprocess, tempfile, threading, time, urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor

PORT = int(os.environ.get('BASLER_ANNOTATION_PORT', '8010'))
RECORD_ROOT = Path(os.environ.get('BASLER_RECORD_ROOT', '~/Videos/Basler Loop Recordings')).expanduser()
CACHE_ROOT = Path(os.environ.get('BASLER_ANNOTATION_CACHE', '~/.cache/basler-annotation-interface')).expanduser()
STATE_DIR = Path(os.environ.get('BASLER_ANNOTATION_STATE', '/tmp/basler_annotation_interface_state'))
TRIAL_QUALITY_PATH = CACHE_ROOT / 'trial_quality_flags.json'
CAMERA_ORDER = ['C', 'A', 'B']
CAMERA_GROUPS = {'C': 'nonstress', 'A': 'control', 'B': 'stress'}
BEHAVIORS = [
    {'key':'q','name':'moving','label':'moving','color':'#e53935','y':0},
    {'key':'w','name':'SA','label':'still alert (SA)','color':'#1e88e5','y':1},
    {'key':'e','name':'quiescent','label':'quiescent','color':'#fdd835','y':2},
    {'key':'r','name':'poking','label':'poking','color':'#43a047','y':3},
]
BEHAVIOR_BY_NAME = {b['name']: b for b in BEHAVIORS}
FEEDING_STATES = [
    {'key':'z','name':'eating','label':'eating','color':'#f97316','y':0},
    {'key':'x','name':'not_eating','label':'not eating','color':'#06b6d4','y':1},
]
FEEDING_BY_NAME = {s['name']: s for s in FEEDING_STATES}
TIMELAPSE_VERSION = 'smooth-v5-gaps'
# Smooth review target: at 1 real hour / playback second, 60 fps samples one
# source frame per real minute. 960 px wide is much easier for the browser to
# decode/scrub than the native 1920/2600 px frames while retaining enough detail
# for behavioral review.  All cameras are padded into the same 960x540 canvas so
# browser decode/MIME support is identical for A/B/C and the future ethogram
# slider has one stable video coordinate system.
TARGET_FPS = int(os.environ.get('BASLER_ANNOTATION_TARGET_FPS', '60'))
SAMPLE_INTERVAL_SECONDS = 3600.0 / TARGET_FPS
MAX_WIDTH = int(os.environ.get('BASLER_ANNOTATION_MAX_WIDTH', '960'))
MAX_HEIGHT = int(os.environ.get('BASLER_ANNOTATION_MAX_HEIGHT', '540'))
# Recorder start/stop leaves real-time gaps between finalized segments. We keep
# the timeline equal to true wall time by inserting "no video" placeholder time
# at the same 1 real hour / playback second scale.  A small visibility floor keeps
# tiny gaps perceptible; the residual clock error introduced by the floor is at
# most (GAP_MIN_PLAYBACK_SECONDS*3600 - real_gap) seconds for an under-floor gap.
GAP_MIN_PLAYBACK_SECONDS = float(os.environ.get('BASLER_ANNOTATION_GAP_MIN_PLAYBACK', '0.08'))
GAP_MAX_PLAYBACK_SECONDS = float(os.environ.get('BASLER_ANNOTATION_GAP_MAX_PLAYBACK', '3.0'))
# Gaps at/under this many real seconds are treated as contiguous (no padding) so
# normal back-to-back segment boundaries are not flagged as downtime.
GAP_IGNORE_SECONDS = float(os.environ.get('BASLER_ANNOTATION_GAP_IGNORE_SECONDS', '2.0'))
GAP_FILL_COLOR = os.environ.get('BASLER_ANNOTATION_GAP_COLOR', '0x101318')
ENCODER_MODE = os.environ.get('BASLER_ANNOTATION_ENCODER', 'auto').lower()
PREWARM_WORKERS = int(os.environ.get('BASLER_ANNOTATION_PREWARM_WORKERS', '1'))
AUTO_PREWARM = os.environ.get('BASLER_ANNOTATION_AUTO_PREWARM', '0').strip().lower() in ('1','true','yes','on')
CURRENT_DAY_PREWARM = os.environ.get('BASLER_ANNOTATION_CURRENT_DAY_PREWARM', '0').strip().lower() in ('1','true','yes','on')
PREWARM_MIN_INTERVAL = float(os.environ.get('BASLER_ANNOTATION_PREWARM_MIN_INTERVAL', '300'))
CACHE_MAX_LAG_HOURS = float(os.environ.get('BASLER_ANNOTATION_CACHE_MAX_LAG_HOURS', '0.15'))
# Automatic refresh cadence for newest/current dates in selected cohorts.
# 900 s = 15 minutes; generation remains serialized and low-priority.
CACHE_MAINTENANCE_INTERVAL = float(os.environ.get('BASLER_ANNOTATION_CACHE_MAINTENANCE_INTERVAL', '900'))
CACHE_MAINTENANCE_ENABLED = os.environ.get('BASLER_ANNOTATION_CACHE_MAINTENANCE', '0').strip().lower() in ('1','true','yes','on')
CACHE_MAINTENANCE_COHORTS = {
    value.strip()
    for value in os.environ.get('BASLER_ANNOTATION_CACHE_MAINTENANCE_COHORTS', 'experiment_2').split(',')
    if value.strip()
}
LEGACY_BACKFILL_DATES = {d.strip() for d in os.environ.get('BASLER_ANNOTATION_LEGACY_BACKFILL_DATES', '2026-06-21').split(',') if d.strip()}
SCAN_TTL = float(os.environ.get('BASLER_ANNOTATION_SCAN_TTL', '10'))
COMPARISON_CACHE_TTL = float(os.environ.get('BASLER_ANNOTATION_COMPARISON_CACHE_TTL', '30'))
ENCODER_THREADS = max(1, int(os.environ.get('BASLER_ANNOTATION_ENCODER_THREADS', '2')))
ANNOTATION_SCHEMA_VERSION = 2
ANNOTATION_POST_MAX_BYTES = int(os.environ.get('BASLER_ANNOTATION_POST_MAX_BYTES', str(2 * 1024 * 1024)))
LACIE_ANNOTATION_ROOT = Path(os.environ.get('BASLER_ANNOTATION_BACKUP', '~/Basler Annotation Backup/annotations')).expanduser()
# One-off user-approved cleanup: 2026-06-22 C/A/B should stop at noon.
# This keeps the annotator/timelapse scanner from re-including future finalized
# post-noon files for this date while the recorder continues running.
SPECIAL_DAY_CUTOFFS = {('2026-06-22', cam): dt.datetime.fromisoformat('2026-06-22T12:00:00').timestamp() for cam in CAMERA_ORDER}
# Hide malformed logical trials from the annotation catalog without deleting or
# modifying any recorder output. The source footage remains untouched.
HIDDEN_TRIAL_HEADINGS = {'_'}
# Recorder-safe virtual catalog splits. Source files/headings are untouched; once
# Trial 18 exceeds 72 hours, finalized segments beginning after the boundary are
# exposed as a separate annotation selection with its own sidecars/cache keys.
POST_OBSERVATION_SPLITS = {
    'trial_18_': {
        'after_hours': 72.0,
        'base_label': 'trial_18_ observation (first 72 h)',
        'post_label': 'trial_18_ post observation (>72 h)',
    },
}
AGGREGATE_DATE_IDS = {'all_first_72h', 'all_post_observation'}
scan_lock = threading.Lock()
scan_cache = {'ts': 0.0, 'items': [], 'options': {}}
selection_session_roots = {}
comparison_cache_lock = threading.Lock()
comparison_cache = {}
comparison_cache_generation = 0
generation_locks = {}
generation_semaphore = threading.Semaphore(1)
prewarm_pool = ThreadPoolExecutor(max_workers=max(1, PREWARM_WORKERS))
prewarm_lock = threading.Lock()
prewarm_jobs = {}
cache_maintenance_state = {'ts': 0, 'submitted': [], 'checked': 0, 'reason': 'not run yet'}
annotation_locks_guard = threading.Lock()
annotation_locks = {}
replica_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ann-replica')

def load_trial_quality_flags():
    try:
        data = json.loads(TRIAL_QUALITY_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def trial_quality_flag(trial, heading, flags=None):
    flags = flags if isinstance(flags, dict) else load_trial_quality_flags()
    for key in (str(heading or '').strip(), str(trial or '').strip()):
        flag = flags.get(key)
        if isinstance(flag, dict) and flag.get('quality'):
            return {'key': key, **flag}
    return None

def trial_number_from_heading(heading):
    match = re.fullmatch(r'trial_(\d+)_', str(heading or '').strip())
    return int(match.group(1)) if match else None

def trial_cohort(heading):
    number = trial_number_from_heading(heading)
    if number is not None and 1 <= number <= 18:
        return 'experiment_1'
    if number is not None and number >= 21:
        return 'experiment_2'
    return 'other'

LIGHT_REGIMES = {
    'experiment_1': {
        'light_start_hour': 4.75,
        'light_end_hour': 22.0,
        'light_label': 'Light 04:45-22:00',
        'dark_label': 'Dark 22:00-04:45',
    },
    'experiment_2': {
        'light_start_hour': 6.0,
        'light_end_hour': 18.0,
        'light_label': 'Light 06:00-18:00',
        'dark_label': 'Dark 18:00-06:00',
    },
}

def light_regime_for_cohort(cohort):
    return dict(LIGHT_REGIMES.get(cohort) or LIGHT_REGIMES['experiment_1'])


def post_observation_group(heading, canonical_trial, first_ts, segment_start_ts):
    config = POST_OBSERVATION_SPLITS.get(str(heading or '').strip())
    if not config or first_ts is None:
        return canonical_trial, None, None
    cutoff_ts = float(first_ts) + float(config['after_hours']) * 3600.0
    if float(segment_start_ts) < cutoff_ts:
        return canonical_trial, config['base_label'], cutoff_ts
    return f'{canonical_trial}__post_observation_after_72h', config['post_label'], cutoff_ts


def ffmpeg_has_encoder(name):
    try:
        encoders = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], text=True, stderr=subprocess.STDOUT, timeout=10)
        return name in encoders
    except Exception:
        return False

def ffmpeg_has_filter(name):
    try:
        filters = subprocess.check_output(['ffmpeg', '-hide_banner', '-filters'], text=True, stderr=subprocess.STDOUT, timeout=10)
        return f' {name} ' in filters
    except Exception:
        return False

def ffmpeg_encode_args():
    """Return browser-compatible H.264 args, preferring GPU encode for speed."""
    if ENCODER_MODE in ('auto', 'nvenc') and ffmpeg_has_encoder('h264_nvenc'):
        return ['-c:v','h264_nvenc','-preset','p4','-profile:v','baseline','-level','3.2','-pix_fmt','yuv420p','-g',str(max(1, TARGET_FPS // 4)),'-bf','0','-movflags','+faststart'], 'h264_nvenc'
    return ['-c:v','libx264','-profile:v','baseline','-level','3.2','-preset','ultrafast','-crf','30','-pix_fmt','yuv420p','-tune','fastdecode','-threads',str(ENCODER_THREADS),'-g',str(max(1, TARGET_FPS // 4)),'-keyint_min',str(max(1, TARGET_FPS // 4)),'-sc_threshold','0','-bf','0','-movflags','+faststart'], 'libx264-ultrafast'

def low_priority_cmd(cmd):
    """Run derived-media work at low CPU/IO priority so recording stays primary."""
    wrapped = list(cmd)
    ionice = shutil.which('ionice')
    nice = shutil.which('nice')
    if ionice:
        wrapped = [ionice, '-c3'] + wrapped
    if nice:
        wrapped = [nice, '-n', '19'] + wrapped
    return wrapped

def parse_isoish(s):
    if not s: return None
    for fmt in ('%Y%m%dT%H%M%S%f', '%Y%m%dT%H%M%S'):
        try: return dt.datetime.strptime(str(s), fmt)
        except ValueError: pass
    try: return dt.datetime.fromisoformat(str(s))
    except Exception: return None

def ts_to_local(ts):
    try: return dt.datetime.fromtimestamp(float(ts))
    except Exception: return None

def safe_rel(path):
    try: return str(Path(path).resolve().relative_to(RECORD_ROOT.resolve()))
    except Exception: return str(path)

def load_segment(meta_path):
    if meta_path.name.endswith('_starseg_summary.json'):
        return None
    try: data = json.loads(meta_path.read_text())
    except Exception: return None
    mp4 = Path(data.get('file') or meta_path.with_suffix('.mp4'))
    if not mp4.exists() or not mp4.is_file(): return None
    cam = str(data.get('camera') or '').upper()
    if cam not in CAMERA_ORDER:
        try:
            parts = meta_path.resolve().relative_to(RECORD_ROOT.resolve()).parts
            if parts and parts[0] in CAMERA_ORDER: cam = parts[0]
        except Exception: pass
    if cam not in CAMERA_ORDER: return None
    trial = str(data.get('recording_session') or '')
    if not trial:
        try: trial = meta_path.resolve().relative_to(RECORD_ROOT.resolve()).parts[1]
        except Exception: trial = 'unknown_trial'
    start = ts_to_local(data.get('start_ts')) or parse_isoish(str(data.get('started_iso') or ''))
    if start is None: return None
    dur = float(data.get('duration_seconds') or 0.0)
    end = ts_to_local(data.get('end_ts')) if data.get('end_ts') is not None else start + dt.timedelta(seconds=dur)
    if end is None or end <= start: end = start + dt.timedelta(seconds=max(dur, 0.0))
    heading = str(data.get('heading') or '')
    if heading.strip() in HIDDEN_TRIAL_HEADINGS:
        return None
    cutoff = SPECIAL_DAY_CUTOFFS.get((start.date().isoformat(), cam))
    # One-off cleanup applies only to the original mistaken trial_01_ post-noon footage.
    # Later valid trials on 2026-06-22 (trial_02_, trial_03_, ...) must still appear.
    if cutoff is not None and heading.startswith('trial_01') and end.timestamp() > cutoff:
        return None
    st = mp4.stat(); mt = meta_path.stat()
    return {
        'camera': cam, 'trial': trial, 'heading': str(data.get('heading') or ''), 'date': start.date().isoformat(),
        'start': start.isoformat(timespec='milliseconds'), 'end': end.isoformat(timespec='milliseconds'),
        'start_ts': start.timestamp(), 'end_ts': end.timestamp(), 'duration': max(0.0, end.timestamp()-start.timestamp()),
        'frames': int(data.get('frames') or 0), 'fps': float(data.get('fps') or 1.0),
        'width': data.get('width'), 'height': data.get('height'), 'mp4': str(mp4), 'meta': str(meta_path),
        'rel': safe_rel(mp4), 'mtime': max(st.st_mtime, mt.st_mtime)}

def scan_recordings(force=False):
    global selection_session_roots
    now = time.time()
    with scan_lock:
        if not force and now - scan_cache['ts'] < SCAN_TTL:
            return scan_cache['items'], scan_cache['options']
        items = []
        if RECORD_ROOT.exists():
            for cam in CAMERA_ORDER:
                cam_root = RECORD_ROOT / cam
                if cam_root.exists():
                    for meta in cam_root.glob('recording_start-*/*/*.json'):
                        seg = load_segment(meta)
                        if seg: items.append(seg)
        items.sort(key=lambda x: (x['heading'] or x['trial'], x['trial'], x['camera'], x['start_ts']))
        # Recorder restarts can split a single logical trial (same heading, e.g.
        # trial_03_) across multiple recording_start-* session directories. Keep
        # the dropdown at the logical-trial level by assigning all same-heading
        # segments to the earliest session's stable trial id. Older trials that
        # only have one session keep their current trial id, so saved annotation
        # sidecars remain compatible.
        heading_canonical = {}
        for seg in items:
            heading = (seg.get('heading') or '').strip()
            if not heading:
                continue
            cur = heading_canonical.get(heading)
            cand = (seg['start_ts'], seg['trial'])
            if cur is None or cand < cur:
                heading_canonical[heading] = cand
        for seg in items:
            heading = (seg.get('heading') or '').strip()
            canonical_start, canonical_trial = heading_canonical.get(heading, (None, seg['trial'])) if heading else (None, seg['trial'])
            group, split_label, split_cutoff_ts = post_observation_group(heading, canonical_trial, canonical_start, seg['start_ts'])
            seg['trial_group'] = group
            seg['catalog_label'] = split_label
            seg['split_cutoff_ts'] = split_cutoff_ts
        trials = {}
        quality_flags = load_trial_quality_flags()
        for seg in items:
            group = seg.get('trial_group') or seg['trial']
            source_heading = seg.get('heading') or ''
            catalog_label = seg.get('catalog_label')
            default_label = (source_heading + ' - ' if source_heading else '') + group
            tr = trials.setdefault(group, {'trial': group, 'heading': source_heading, 'source_heading': source_heading, 'label': catalog_label or default_label, 'post_observation': group.endswith('__post_observation_after_72h'), 'split_cutoff_ts': seg.get('split_cutoff_ts'), 'dates': {}, 'cameras': set(), 'segments': 0})
            tr['segments'] += 1; tr['cameras'].add(seg['camera'])
            d = tr['dates'].setdefault(seg['date'], {'date': seg['date'], 'cameras': set(), 'segments': 0, 'first': seg['start'], 'last': seg['end']})
            d['segments'] += 1; d['cameras'].add(seg['camera']); d['first'] = min(d['first'], seg['start']); d['last'] = max(d['last'], seg['end'])
        trial_list = []
        for tr in trials.values():
            dates = []
            for d0 in tr['dates'].values():
                d = dict(d0); d['cameras'] = [c for c in CAMERA_ORDER if c in d['cameras']]; dates.append(d)
            dates.sort(key=lambda x:x['date'])
            if tr.get('split_cutoff_ts') and dates:
                is_post = tr.get('post_observation', False)
                dates.insert(0, {
                    'date': 'all_post_observation' if is_post else 'all_first_72h',
                    'label': 'All post-observation footage' if is_post else 'All available first-72-hour footage',
                    'aggregate': True,
                    'cameras': [c for c in CAMERA_ORDER if c in tr['cameras']],
                    'segments': tr['segments'],
                    'first': min(d['first'] for d in dates),
                    'last': max(d['last'] for d in dates),
                })
            quality = trial_quality_flag(tr['trial'], tr.get('heading'), quality_flags)
            cohort = trial_cohort(tr.get('heading'))
            cohort_prefix = {'experiment_1': '[EXP 1] ', 'experiment_2': '[EXP 2] '}.get(cohort, '')
            label = cohort_prefix + ('⚠ POOR QUALITY — ' if quality and quality.get('quality') == 'poor' else '') + tr['label']
            trial_list.append({'trial': tr['trial'], 'heading': tr.get('heading') or '', 'label': label, 'cohort': cohort, 'light_regime': light_regime_for_cohort(cohort), 'quality': quality, 'post_observation': tr.get('post_observation', False), 'split_cutoff_ts': tr.get('split_cutoff_ts'), 'dates': dates, 'cameras': [c for c in CAMERA_ORDER if c in tr['cameras']], 'segments': tr['segments']})
        trial_list.sort(key=lambda x:x['trial'])
        options = {'record_root': str(RECORD_ROOT), 'trials': trial_list, 'camera_order': CAMERA_ORDER, 'scan_ts': now, 'auto_prewarm': AUTO_PREWARM, 'current_day_prewarm': CURRENT_DAY_PREWARM}
        # Publish immutable snapshots only after the scan is complete. Annotation
        # reads/saves use this root index without taking scan_lock, so they cannot
        # queue behind a full recorder-tree scan.
        selection_session_roots = build_selection_session_root_index(items)
        scan_cache.update(ts=now, items=items, options=options)
        with comparison_cache_lock:
            comparison_cache.clear()
        return items, options

def select_segments_from_items(items, trial, date, camera):
    if date in AGGREGATE_DATE_IDS:
        selected = [s for s in items if (s['trial'] == trial or s.get('trial_group') == trial) and s['camera'] == camera]
        selected.sort(key=lambda x:x['start_ts'])
        if not selected:
            return [], 0.0, 0.0
        return selected, selected[0]['start_ts'], max(s['end_ts'] for s in selected)
    try:
        day_start = dt.datetime.fromisoformat(date).timestamp()
    except Exception:
        raise RuntimeError(f'Invalid calendar day: {date!r}')
    day_end = day_start + 86400.0
    selected = [s for s in items if (s['trial'] == trial or s.get('trial_group') == trial) and s['camera'] == camera and s['end_ts'] > day_start and s['start_ts'] < day_end]
    selected.sort(key=lambda x:x['start_ts'])
    return selected, day_start, day_end

def select_segments(trial, date, camera):
    items, _ = scan_recordings(False)
    return select_segments_from_items(items, trial, date, camera)

def gap_play_seconds(real_seconds):
    """Map a real-time gap to its playback duration on the 1h/1s scale, clamped."""
    play = max(0.0, float(real_seconds)) / 3600.0
    play = max(GAP_MIN_PLAYBACK_SECONDS, play)
    if GAP_MAX_PLAYBACK_SECONDS > 0:
        play = min(GAP_MAX_PLAYBACK_SECONDS, play)
    return play

def compose_timeline(segs):
    """Build the ordered playback timeline from finalized segments.

    Consecutive segments separated by <= GAP_IGNORE_SECONDS are merged into a
    single contiguous "run" (so each run is still one fast ffmpeg pass).  Real
    downtime between runs becomes an explicit "gap" piece whose playback length
    keeps the timeline equal to true wall-clock time (1 real hour = 1 s), with a
    small visibility floor.  Every piece carries cumulative playback offsets so
    the same model drives video generation, wall-clock mapping, the ethogram,
    CSV export, and analytics.
    """
    segs = sorted(segs or [], key=lambda x: x['start_ts'])
    pieces = []
    play_cursor = 0.0
    coverage_seconds = 0.0
    gaps = []
    prev_end_ts = None
    run = None
    for s in segs:
        gap_real = (s['start_ts'] - prev_end_ts) if prev_end_ts is not None else 0.0
        if prev_end_ts is not None and gap_real > GAP_IGNORE_SECONDS:
            play_dur = gap_play_seconds(gap_real)
            gap_piece = {'kind': 'gap', 'play_start': play_cursor, 'play_dur': play_dur,
                         'real_dur': gap_real, 'wall_start_ts': prev_end_ts, 'wall_end_ts': s['start_ts']}
            pieces.append(gap_piece)
            gaps.append(gap_piece)
            play_cursor += play_dur
            run = None
        seg_real = max(0.0, float(s.get('duration') or (s['end_ts'] - s['start_ts'])))
        seg_play = seg_real / 3600.0
        coverage_seconds += seg_real
        if run is None:
            run = {'kind': 'run', 'play_start': play_cursor, 'play_dur': 0.0,
                   'wall_start_ts': s['start_ts'], 'wall_end_ts': s['end_ts'], 'mp4s': []}
            pieces.append(run)
        run['mp4s'].append(s['mp4'])
        run['play_dur'] += seg_play
        run['wall_end_ts'] = s['end_ts']
        play_cursor += seg_play
        prev_end_ts = s['end_ts']
    return {'pieces': pieces, 'play_total': play_cursor, 'coverage_seconds': coverage_seconds, 'gaps': gaps,
            'clip_start_ts': (segs[0]['start_ts'] if segs else None)}

def timeline_pieces_public(timeline):
    """Compact, JSON-friendly pieces for the cache sidecar and frontend."""
    out = []
    for p in timeline['pieces']:
        out.append({'kind': p['kind'], 'play_start': round(p['play_start'], 6), 'play_dur': round(p['play_dur'], 6),
                    'wall_start_ts': p.get('wall_start_ts'), 'wall_end_ts': p.get('wall_end_ts'),
                    'real_dur': round(float(p.get('real_dur') or (p.get('wall_end_ts', 0) - p.get('wall_start_ts', 0))), 3)})
    return out

def play_to_wall_ts(t, pieces, clip_start_ts):
    """Map a playback-second offset to a real wall-clock unix timestamp."""
    if clip_start_ts is None:
        return None
    if not pieces:
        return clip_start_ts + t * 3600.0
    t = max(0.0, float(t))
    for p in pieces:
        p0 = p['play_start']; p1 = p0 + p['play_dur']
        if p['play_dur'] <= 0:
            continue
        if t <= p1 + 1e-9:
            frac = (t - p0) / p['play_dur']
            frac = min(1.0, max(0.0, frac))
            ws = p.get('wall_start_ts'); we = p.get('wall_end_ts')
            if ws is None or we is None:
                return clip_start_ts + t * 3600.0
            return ws + frac * (we - ws)
    last = pieces[-1]
    return last.get('wall_end_ts') if last.get('wall_end_ts') is not None else clip_start_ts + t * 3600.0

def wall_ts_to_play(ts, pieces):
    """Map a real wall-clock unix timestamp to a playback-second offset."""
    if ts is None or not pieces:
        return None
    for p in pieces:
        ws = p.get('wall_start_ts'); we = p.get('wall_end_ts')
        if ws is None or we is None or we <= ws:
            continue
        if ts < ws:
            return p['play_start']
        if ts <= we:
            frac = (ts - ws) / (we - ws)
            return p['play_start'] + frac * p['play_dur']
    return None

def gap_play_intervals(timeline):
    """Gap intervals in playback seconds.

    Because 1 playback second equals 1 real hour, annotation event times (playback
    seconds) are numerically the same as elapsed hours used by the analytics, so
    these intervals are directly comparable to event times and behavior spans.
    """
    out = []
    for g in timeline.get('gaps', []):
        a = g['play_start']; b = g['play_start'] + g['play_dur']
        if b > a:
            out.append((a, b))
    return out

def pieces_from_meta(meta):
    """Reconstruct piecewise mapping inputs from a status/sidecar meta dict."""
    return meta.get('pieces') or [], meta.get('clip_start_ts')

def gap_intervals_from_pieces(pieces):
    out = []
    for p in pieces or []:
        if p.get('kind') == 'gap':
            a = p['play_start']; b = p['play_start'] + p['play_dur']
            if b > a:
                out.append((a, b))
    return out

def length_excluding_gaps(a, b, gaps):
    if b <= a:
        return 0.0
    total = b - a
    for (g0, g1) in gaps or []:
        lo = max(a, g0); hi = min(b, g1)
        if hi > lo:
            total -= (hi - lo)
    return max(0.0, total)

def nongap_subintervals(a, b, gaps):
    """Split [a,b] into the subintervals that do not overlap any gap."""
    if b <= a:
        return []
    segments = [(a, b)]
    for (g0, g1) in gaps or []:
        nxt = []
        for (s, e) in segments:
            lo = max(s, g0); hi = min(e, g1)
            if hi <= lo:
                nxt.append((s, e))
            else:
                if s < lo:
                    nxt.append((s, lo))
                if hi < e:
                    nxt.append((hi, e))
        segments = nxt
    return segments

def in_gap_at(t, gaps):
    for (g0, g1) in gaps or []:
        if g0 - 1e-9 <= t <= g1 + 1e-9:
            return True
    return False

def cache_key(trial, date, camera, segments, timeline=None):
    enc_args, enc_name = ffmpeg_encode_args()
    h = hashlib.sha256(); h.update(json.dumps({'version':TIMELAPSE_VERSION,'encoder':enc_name,'trial':trial,'date':date,'camera':camera,'sample':SAMPLE_INTERVAL_SECONDS,'width':MAX_WIDTH,'height':MAX_HEIGHT,'fps':TARGET_FPS,'gap_min':GAP_MIN_PLAYBACK_SECONDS,'gap_max':GAP_MAX_PLAYBACK_SECONDS,'gap_ignore':GAP_IGNORE_SECONDS}, sort_keys=True).encode())
    for s in segments: h.update(f"{s['mp4']}|{s['mtime']}|{s['start_ts']}|{s['end_ts']}|".encode())
    if timeline is not None:
        for p in timeline['pieces']:
            h.update(f"{p['kind']}|{p['play_start']:.4f}|{p['play_dur']:.4f}|{p.get('real_dur') or 0:.2f}|".encode())
    return h.hexdigest()[:20]

def timelapse_status(trial, date, camera):
    segs, day_start, day_end = select_segments(trial, date, camera)
    total = sum(max(0, min(s['end_ts'], day_end)-max(s['start_ts'], day_start)) for s in segs)
    timeline = compose_timeline(segs)
    play_total = timeline['play_total']
    key = cache_key(trial, date, camera, segs, timeline)
    name = f"{date}_{trial}_{camera}_{key}.mp4".replace('/', '_')
    out = CACHE_ROOT / 'timelapses' / name
    _, enc_name = ffmpeg_encode_args()
    gaps_public = [{'play_start':round(g['play_start'],6),'play_dur':round(g['play_dur'],6),'real_seconds':round(g['real_dur'],3),'real_minutes':round(g['real_dur']/60.0,2),'wall_start_ts':g['wall_start_ts'],'wall_end_ts':g['wall_end_ts']} for g in timeline['gaps']]
    gap_real_total = sum(g['real_dur'] for g in timeline['gaps'])
    return {'trial':trial,'date':date,'camera':camera,'segments':len(segs),'coverage_seconds':total,'coverage_hours':total/3600.0,'target_duration_seconds':play_total,'play_duration_seconds':play_total,'full_day_target_seconds':24.0,'cache_key':key,'timelapse_version':TIMELAPSE_VERSION,'encoder':enc_name,'target_fps':TARGET_FPS,'sample_interval_seconds':SAMPLE_INTERVAL_SECONDS,'max_width':MAX_WIDTH,'max_height':MAX_HEIGHT,'cached':out.exists() and out.stat().st_size>0,'video_url':'/video/'+urllib.parse.quote(name),'segments_preview':[{k:s[k] for k in ('start','end','duration','frames','rel')} for s in segs[:20]],'clip_start_ts':timeline['clip_start_ts'],'pieces':timeline_pieces_public(timeline),'gaps':gaps_public,'gap_count':len(gaps_public),'gap_real_seconds':round(gap_real_total,3),'gap_real_minutes':round(gap_real_total/60.0,2),'note':'Finalized MP4 segments only; active recorder staging is ignored. Recording gaps are padded with no-video time so the timeline equals real wall-clock time.'}

def is_current_or_future_day(date):
    try:
        return dt.date.fromisoformat(date) >= dt.datetime.now().date()
    except Exception:
        return False

def latest_cached_timelapse(trial, date, camera):
    """Return newest usable cache for this selection, even if today's exact key changed."""
    outdir = CACHE_ROOT / 'timelapses'
    if not outdir.exists():
        return None
    best = None
    for meta_path in outdir.glob('*.json'):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get('trial') != trial or meta.get('date') != date or meta.get('camera') != camera:
            continue
        video_path = Path(meta.get('output') or meta_path.with_suffix('.mp4'))
        if not video_path.exists() or video_path.stat().st_size <= 0:
            continue
        score = float(meta.get('coverage_seconds') or 0), float(meta.get('generated_at') or video_path.stat().st_mtime)
        if best is None or score > best[0]:
            name = video_path.name
            stale_meta = dict(meta)
            stale_meta.update({'cached': True, 'stale_cache': True, 'exact_cache': False, 'path': str(video_path), 'video_url': '/video/' + urllib.parse.quote(name)})
            best = (score, stale_meta)
    return best[1] if best else None

def queue_refresh_if_needed(trial, date, camera, exact_meta=None):
    if exact_meta is None:
        exact_meta = timelapse_status(trial, date, camera)
    if exact_meta.get('cached'):
        return False
    key = f'{trial}|{date}|{camera}'
    exact_key = exact_meta.get('cache_key')
    now = time.time()
    with prewarm_lock:
        job = prewarm_jobs.get(key, {})
        job_key = job.get('cache_key')
        if job.get('state') == 'running':
            return True
        # Do not let the generic refresh throttle hide newly-finalized recorder
        # segments.  If the exact cache key changed, queue the new timelapse
        # immediately; the global generation semaphore still serializes ffmpeg.
        last = float(job.get('finished') or job.get('submitted') or 0)
        same_key_recent = job_key == exact_key and now - last < PREWARM_MIN_INTERVAL
        if same_key_recent:
            return False
        prewarm_jobs[key] = {'state':'queued','trial':trial,'date':date,'camera':camera,'submitted':now,'cache_key':exact_key}
    prewarm_pool.submit(prewarm_one, trial, date, camera)
    return True
def queue_refresh_lazy(trial, date, camera):
    """Queue cache generation without synchronously scanning/building the exact key."""
    if camera not in CAMERA_ORDER:
        return False
    key = f'{trial}|{date}|{camera}'
    now = time.time()
    with prewarm_lock:
        job = prewarm_jobs.get(key, {})
        if job.get('state') in ('queued', 'running'):
            return True
        last = float(job.get('finished') or job.get('submitted') or 0)
        if now - last < PREWARM_MIN_INTERVAL:
            return False
        prewarm_jobs[key] = {'state':'queued','trial':trial,'date':date,'camera':camera,'submitted':now,'cache_key':'lazy'}
    prewarm_pool.submit(prewarm_one, trial, date, camera)
    return True

def start_prewarm_lazy(trial, date, cameras):
    submitted = []
    for camera in cameras:
        if queue_refresh_lazy(trial, date, camera):
            submitted.append(camera)
    with prewarm_lock:
        jobs = dict(prewarm_jobs)
    return {'submitted': submitted, 'workers': PREWARM_WORKERS, 'jobs': jobs, 'lazy': True}
def selected_first_cameras(selected):
    selected = (selected or '').upper()
    cams = list(CAMERA_ORDER)
    if selected in cams:
        cams = [selected] + [c for c in cams if c != selected]
    return cams

def _gap_label_text(piece):
    real = float(piece.get('real_dur') or 0.0)
    if real >= 5400:
        amount = f"{real/3600.0:.1f} h"
    elif real >= 90:
        amount = f"{real/60.0:.0f} min"
    else:
        amount = f"{real:.0f} s"
    ws = ts_to_local(piece.get('wall_start_ts')); we = ts_to_local(piece.get('wall_end_ts'))
    rng = ''
    if ws and we:
        rng = f"   {ws.strftime('%H.%M')} to {we.strftime('%H.%M')}"
    return f"NO VIDEO - recording gap {amount}{rng}"

def _render_run_clip(ffmpeg, td, idx, mp4s, enc_args):
    concat = td / f'run_{idx}.txt'
    with concat.open('w') as f:
        for mp4 in mp4s:
            f.write("file '" + mp4.replace("'", "'\\''") + "'\n")
    out_clip = td / f'piece_{idx:04d}.mp4'
    # Important order: speed the 1-fps source timestamps up first, then resample to a
    # constant browser-friendly fps.  Frequent keyframes and no B-frames keep timeline
    # scrubs and the synchronized ethogram bar responsive.
    vf = f"setpts=PTS/3600,fps={TARGET_FPS},scale={MAX_WIDTH}:{MAX_HEIGHT}:force_original_aspect_ratio=decrease:flags=bicubic,pad={MAX_WIDTH}:{MAX_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    cmd = [ffmpeg,'-hide_banner','-loglevel','error','-y','-safe','0','-f','concat','-i',str(concat),'-filter_threads','1','-filter_complex_threads','1','-vf',vf,'-an'] + enc_args + [str(out_clip)]
    subprocess.run(low_priority_cmd(cmd), check=True, capture_output=True, text=True)
    return out_clip

def _render_gap_clip(ffmpeg, td, idx, piece, enc_args, allow_text=True):
    out_clip = td / f'piece_{idx:04d}.mp4'
    dur = max(1.0 / TARGET_FPS, float(piece['play_dur']))
    base_vf = f"format=yuv420p,setsar=1"
    label_vf = base_vf
    if allow_text:
        text = _gap_label_text(piece).replace('\\', '').replace(':', '\\:').replace("'", '')
        label_vf = (base_vf + f",drawtext=text='{text}':fontcolor=0x9aa7b7:fontsize=20:"
                    "x=(w-text_w)/2:y=(h-text_h)/2")
    src = f"color=c={GAP_FILL_COLOR}:s={MAX_WIDTH}x{MAX_HEIGHT}:r={TARGET_FPS}:d={dur:.4f}"
    cmd = [ffmpeg,'-hide_banner','-loglevel','error','-y','-f','lavfi','-i',src,'-vf',label_vf,'-an'] + enc_args + [str(out_clip)]
    subprocess.run(low_priority_cmd(cmd), check=True, capture_output=True, text=True)
    return out_clip

def generate_timelapse(trial, date, camera):
    segs, _, _ = select_segments(trial, date, camera)
    if not segs: raise RuntimeError('No finalized segments found for that trial/date/camera')
    meta = timelapse_status(trial, date, camera); name = urllib.parse.unquote(meta['video_url'].split('/')[-1])
    timeline = compose_timeline(segs)
    outdir = CACHE_ROOT / 'timelapses'; outdir.mkdir(parents=True, exist_ok=True); out = outdir / name
    lock = generation_locks.setdefault(str(out), threading.Lock())
    with lock:
        if out.exists() and out.stat().st_size > 0: return out
        with generation_semaphore:
            if out.exists() and out.stat().st_size > 0: return out
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg: raise RuntimeError('ffmpeg not found')
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            enc_args, enc_name = ffmpeg_encode_args()
            can_text = ffmpeg_has_filter('drawtext')
            with tempfile.TemporaryDirectory(prefix='basler_annotation_', dir=str(CACHE_ROOT)) as td0:
                td = Path(td0)
                piece_files = []
                try:
                    for idx, piece in enumerate(timeline['pieces']):
                        if piece['kind'] == 'run':
                            piece_files.append(_render_run_clip(ffmpeg, td, idx, piece['mp4s'], enc_args))
                        else:
                            try:
                                piece_files.append(_render_gap_clip(ffmpeg, td, idx, piece, enc_args, allow_text=can_text))
                            except subprocess.CalledProcessError:
                                # Some ffmpeg builds lack a usable font for drawtext; fall back to plain fill.
                                piece_files.append(_render_gap_clip(ffmpeg, td, idx, piece, enc_args, allow_text=False))
                except subprocess.CalledProcessError as e:
                    detail = (e.stderr or e.stdout or str(e)).strip()
                    raise RuntimeError('ffmpeg timelapse generation failed: ' + detail[-2000:])
                tmp = td / 'out.mp4'
                final_list = td / 'pieces.txt'
                with final_list.open('w') as f:
                    for pf in piece_files:
                        f.write("file '" + str(pf).replace("'", "'\\''") + "'\n")
                copy_cmd = [ffmpeg,'-hide_banner','-loglevel','error','-y','-safe','0','-f','concat','-i',str(final_list),'-c','copy','-movflags','+faststart',str(tmp)]
                try:
                    subprocess.run(low_priority_cmd(copy_cmd), check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError:
                    # Stream-copy concat can reject mismatched timebases; re-encode as a fallback.
                    if tmp.exists(): tmp.unlink()
                    reenc_cmd = [ffmpeg,'-hide_banner','-loglevel','error','-y','-safe','0','-f','concat','-i',str(final_list),'-fps_mode','cfr','-r',str(TARGET_FPS),'-an'] + enc_args + [str(tmp)]
                    try:
                        subprocess.run(low_priority_cmd(reenc_cmd), check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        detail = (e.stderr or e.stdout or str(e)).strip()
                        raise RuntimeError('ffmpeg timelapse concat failed: ' + detail[-2000:])
                os.replace(tmp, out)
                m = timelapse_status(trial, date, camera); m.update({'generated_at':time.time(),'output':str(out),'source_count':len(segs),'piece_count':len(piece_files)})
                out.with_suffix('.json').write_text(json.dumps(m, indent=2))
        return out

def prewarm_one(trial, date, camera):
    job_key = f'{trial}|{date}|{camera}'
    exact_key = None
    try:
        exact_key = timelapse_status(trial, date, camera).get('cache_key')
    except Exception:
        pass
    with prewarm_lock:
        prewarm_jobs[job_key] = {'state':'running','trial':trial,'date':date,'camera':camera,'started':time.time(),'cache_key':exact_key}
    try:
        out = generate_timelapse(trial, date, camera)
        done_key = timelapse_status(trial, date, camera).get('cache_key')
        with prewarm_lock:
            prewarm_jobs[job_key].update({'state':'done','finished':time.time(),'path':str(out),'cache_key':done_key})
    except Exception as e:
        with prewarm_lock:
            prewarm_jobs[job_key].update({'state':'error','finished':time.time(),'error':str(e),'cache_key':exact_key})

def start_prewarm(trial, date, cameras):
    submitted = []
    for camera in cameras:
        if camera not in CAMERA_ORDER:
            continue
        meta = timelapse_status(trial, date, camera)
        key = f'{trial}|{date}|{camera}'
        if meta.get('cached'):
            with prewarm_lock:
                prewarm_jobs[key] = {'state':'cached','trial':trial,'date':date,'camera':camera,'finished':time.time(),'cache_key':meta.get('cache_key')}
            continue
        if queue_refresh_if_needed(trial, date, camera, meta):
            submitted.append(camera)
    with prewarm_lock:
        jobs = dict(prewarm_jobs)
    return {'submitted':submitted,'workers':PREWARM_WORKERS,'jobs':jobs}


def cache_gap_hours(exact_meta, latest_meta):
    current = float(exact_meta.get('coverage_hours') or 0.0)
    if exact_meta.get('cached'):
        cached = current
    elif latest_meta:
        cached = float(latest_meta.get('coverage_hours') or 0.0)
    else:
        cached = 0.0
    return max(0.0, current - cached)

def queue_cache_maintenance(options, reason='maintenance'):
    """Keep derived review caches close to finalized data without scanning active segments.

    Policy: maintain the newest/current date for configured cohorts within CACHE_MAX_LAG_HOURS,
    and backfill explicitly listed legacy dates (default: 2026-06-21, where A/B had
    partial caches from when this app was first created). Generation remains serialized
    by generation_semaphore and low-priority ffmpeg.
    """
    submitted = []
    checked = 0
    threshold = max(0.0, CACHE_MAX_LAG_HOURS)
    real_dates_for_trial = lambda tr: [d for d in (tr.get('dates') or []) if d.get('date') and d.get('date') not in AGGREGATE_DATE_IDS]
    # Current footage wins the one-worker queue. Virtual all-footage ranges are
    # loaded explicitly and must never block today's per-camera maintenance.
    trials = [
        tr for tr in options.get('trials', [])
        if tr.get('cohort') in CACHE_MAINTENANCE_COHORTS
    ]
    trials.sort(key=lambda tr: max((d.get('date') for d in real_dates_for_trial(tr)), default=''), reverse=True)
    for tr in trials:
        dates = real_dates_for_trial(tr)
        if not dates:
            continue
        latest_date = max((d.get('date') for d in dates), default='')
        for d in dates:
            date = d.get('date')
            if not date:
                continue
            maintain_date = date == latest_date or is_current_or_future_day(date) or date in LEGACY_BACKFILL_DATES
            if not maintain_date:
                continue
            for camera in (d.get('cameras') or tr.get('cameras') or CAMERA_ORDER):
                if camera not in CAMERA_ORDER:
                    continue
                checked += 1
                exact = timelapse_status(tr['trial'], date, camera)
                if exact.get('cached') or exact.get('segments', 0) <= 0:
                    continue
                latest = latest_cached_timelapse(tr['trial'], date, camera)
                gap = cache_gap_hours(exact, latest)
                # Legacy dates are allowed to refresh any non-exact stale cache; newest/current
                # dates refresh only once they are at least threshold hours behind finalized data.
                should_refresh = (date in LEGACY_BACKFILL_DATES and gap > 0.01) or gap >= threshold or latest is None
                if should_refresh and queue_refresh_if_needed(tr['trial'], date, camera, exact):
                    submitted.append({'trial': tr['trial'], 'date': date, 'camera': camera, 'gap_hours': round(gap, 3), 'reason': 'legacy-backfill' if date in LEGACY_BACKFILL_DATES else 'within-hour-maintenance'})
    global cache_maintenance_state
    cache_maintenance_state = {'ts': time.time(), 'submitted': submitted, 'checked': checked, 'reason': reason, 'max_lag_hours': threshold, 'cohorts': sorted(CACHE_MAINTENANCE_COHORTS), 'legacy_backfill_dates': sorted(LEGACY_BACKFILL_DATES)}
    return dict(cache_maintenance_state)

def cache_maintenance_loop():
    while True:
        try:
            _, options = scan_recordings(force=True)
            queue_cache_maintenance(options, reason='background')
        except Exception as e:
            global cache_maintenance_state
            cache_maintenance_state = {'ts': time.time(), 'submitted': [], 'checked': 0, 'reason': 'background-error', 'error': str(e)}
        time.sleep(max(60.0, CACHE_MAINTENANCE_INTERVAL))

def prewarm_latest_current_day(options):
    if not CURRENT_DAY_PREWARM:
        return {'submitted': []}
    today = dt.datetime.now().date().isoformat()
    candidates = []
    for tr in options.get('trials', []):
        for d in tr.get('dates', []):
            if d.get('date') == today:
                candidates.append((tr, d))
    if not candidates:
        return {'submitted': []}
    tr, d = candidates[-1]
    return start_prewarm(tr['trial'], d['date'], d.get('cameras') or tr.get('cameras') or CAMERA_ORDER)

def safe_annotation_key(trial, date, camera):
    raw = f"{trial}__{date}__camera-{camera}"
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', raw).strip('_')

def annotation_paths_for_selection(trial, date, camera):
    if not trial or not date or not camera:
        raise RuntimeError('trial, date, and camera are required for stable annotations')
    ann_dir = CACHE_ROOT / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    key = safe_annotation_key(trial, date, camera)
    return key, ann_dir / f'{key}.annotations.json', ann_dir / f'{key}.ethogram.csv'

def annotation_paths_for_video(video_name):
    name = Path(urllib.parse.unquote(str(video_name))).name
    if not name.endswith('.mp4'):
        raise RuntimeError('annotation video must be a cached .mp4 name')
    video_path = CACHE_ROOT / 'timelapses' / name
    if not video_path.exists() or not video_path.is_file():
        raise RuntimeError('cached timelapse video not found for annotation')
    base = video_path.with_suffix('')
    return video_path, base.with_suffix('.annotations.json'), base.with_suffix('.ethogram.csv')

def validated_recording_session_root(path, camera):
    """Return a recorder session root only when it has the expected safe shape."""
    try:
        root = Path(path).resolve()
        rel = root.relative_to(RECORD_ROOT.resolve())
    except Exception:
        return None
    if len(rel.parts) != 2 or rel.parts[0] != camera or not rel.parts[1].startswith('recording_start-'):
        return None
    return root

def segment_session_root(seg):
    camera = str(seg.get('camera') or '').upper()
    if camera not in CAMERA_ORDER:
        return None
    try:
        return validated_recording_session_root(Path(seg['mp4']).resolve().parent.parent, camera)
    except Exception:
        return None

def segment_calendar_dates(seg):
    """Yield every local calendar day touched by a finalized segment."""
    try:
        start = dt.datetime.fromtimestamp(float(seg['start_ts'])).date()
        end = dt.datetime.fromtimestamp(max(float(seg['start_ts']), float(seg['end_ts']) - 1e-6)).date()
    except Exception:
        value = str(seg.get('date') or '')
        return [value] if value else []
    dates = []
    while start <= end:
        dates.append(start.isoformat())
        start += dt.timedelta(days=1)
    return dates

def build_selection_session_root_index(items):
    """Build a validated selection -> earliest recorder-session root snapshot."""
    index = {}
    for seg in sorted(items or [], key=lambda item: float(item.get('start_ts') or 0)):
        camera = str(seg.get('camera') or '').upper()
        root = segment_session_root(seg)
        if not root:
            continue
        aliases = {str(seg.get('trial') or ''), str(seg.get('trial_group') or '')} - {''}
        for trial in aliases:
            for date in segment_calendar_dates(seg):
                index.setdefault((trial, date, camera), root)
        group = str(seg.get('trial_group') or '')
        if group and seg.get('split_cutoff_ts') is not None:
            aggregate = 'all_post_observation' if group.endswith('__post_observation_after_72h') else 'all_first_72h'
            index.setdefault((group, aggregate, camera), root)
    return index

def recording_session_root_from_cached_video(video_name, trial, date, camera):
    """Resolve a session root from server-owned timelapse metadata after restart."""
    if not video_name:
        return None
    try:
        video_path, _, _ = annotation_paths_for_video(video_name)
        metadata = json.loads(video_path.with_suffix('.json').read_text(encoding='utf-8'))
    except Exception:
        return None
    if (metadata.get('trial'), metadata.get('date'), str(metadata.get('camera') or '').upper()) != (trial, date, camera):
        return None
    for preview in metadata.get('segments_preview') or []:
        rel = preview.get('rel')
        if not rel:
            continue
        try:
            source = (RECORD_ROOT / rel).resolve()
            source.relative_to(RECORD_ROOT.resolve())
        except Exception:
            continue
        if not source.is_file():
            continue
        root = validated_recording_session_root(source.parent.parent, camera)
        if root:
            return root
    return None

def recording_session_root_for_selection(trial, date, camera, video_name=None):
    """Return the camera/session directory that contains the finalized source videos.

    An annotation is for one trial/date/camera timelapse, which may be built from many
    finalized segment MP4s.  The paired recorder copy lives one level above those
    segment directories, e.g. RECORD_ROOT/A/recording_start-.../.
    """
    camera = str(camera or '').upper()
    if not trial or not date or camera not in CAMERA_ORDER:
        return None
    root = selection_session_roots.get((str(trial), str(date), camera))
    if root:
        return validated_recording_session_root(root, camera)
    return recording_session_root_from_cached_video(video_name, str(trial), str(date), camera)

def paired_annotation_paths_for_selection(trial, date, camera, video_name=None):
    key = safe_annotation_key(trial, date, camera)
    root = recording_session_root_for_selection(trial, date, camera, video_name=video_name)
    if not root:
        return key, None, None
    return key, root / f'{key}.annotations.json', root / f'{key}.ethogram.csv'

def annotation_storage_paths(video_name=None, meta=None, trial=None, date=None, camera=None):
    meta = meta or {}
    trial = trial or meta.get('trial')
    date = date or meta.get('date')
    camera = camera or meta.get('camera')
    fallback = None
    paired = None
    video_path = None
    if video_name:
        try:
            video_path, old_json, old_csv = annotation_paths_for_video(video_name)
            fallback = (old_json, old_csv)
        except Exception:
            video_path = None
            fallback = None
    if trial and date and camera:
        key, json_path, csv_path = annotation_paths_for_selection(trial, date, camera)
        _, paired_json, paired_csv = paired_annotation_paths_for_selection(trial, date, camera, video_name=video_name)
        if paired_json and paired_csv:
            paired = (paired_json, paired_csv)
        return video_path, key, json_path, csv_path, fallback, paired
    if fallback:
        return video_path, Path(str(video_name)).stem, fallback[0], fallback[1], None, None
    raise RuntimeError('annotation storage requires trial/date/camera or a cached video name')

def sanitize_annotation_events(events, duration=None):
    clean = []
    for ev in events or []:
        try:
            t = float(ev.get('time'))
        except Exception:
            continue
        if t < 0:
            continue
        if duration is not None:
            t = min(t, max(0.0, float(duration)))
        behavior = str(ev.get('behavior') or '')
        if behavior not in BEHAVIOR_BY_NAME:
            continue
        clean.append({'time': round(t, 4), 'behavior': behavior})
    clean.sort(key=lambda x: x['time'])
    dedup = []
    for ev in clean:
        if dedup and abs(dedup[-1]['time'] - ev['time']) < 1e-4:
            dedup[-1] = ev
        else:
            dedup.append(ev)
    return dedup


def sanitize_feeding_events(events, duration=None):
    clean = []
    for ev in events or []:
        try:
            t = float(ev.get('time'))
        except Exception:
            continue
        if t < 0:
            continue
        if duration is not None:
            t = min(t, max(0.0, float(duration)))
        state = str(ev.get('state') or ev.get('feeding') or '')
        if state not in FEEDING_BY_NAME:
            continue
        clean.append({'time': round(t, 4), 'state': state})
    clean.sort(key=lambda x: x['time'])
    dedup = []
    for ev in clean:
        if dedup and abs(dedup[-1]['time'] - ev['time']) < 1e-4:
            dedup[-1] = ev
        else:
            dedup.append(ev)
    return dedup


def build_smooth_ethogram_csv(events, feeding_events, duration, meta):
    duration = max(0.0, float(duration or meta.get('target_duration_seconds') or meta.get('coverage_hours', 0) * 3600.0 or 0.0))
    step = 1.0 / max(1, TARGET_FPS)
    events = sanitize_annotation_events(events, duration)
    feeding_events = sanitize_feeding_events(feeding_events, duration)
    pieces, clip_start_ts = pieces_from_meta(meta)
    gaps = gap_intervals_from_pieces(pieces)
    def wall_clock(t):
        ts = play_to_wall_ts(t, pieces, clip_start_ts) if pieces else (clip_start_ts + t * 3600.0 if clip_start_ts is not None else None)
        loc = ts_to_local(ts) if ts is not None else None
        return loc.strftime('%Y-%m-%d %H:%M:%S') if loc else ''
    rows = ['video_time_s,source_elapsed_s,wall_clock,in_gap,behavior,behavior_code,behavior_label,color,feeding,feeding_code,feeding_label,feeding_color\n']
    idx = 0; current = None; feeding_idx = 0; current_feeding = None; n = int(duration / step) + 1
    for i in range(n + 1):
        t = min(duration, i * step)
        while idx < len(events) and events[idx]['time'] <= t + 1e-9:
            current = events[idx]['behavior']; idx += 1
        while feeding_idx < len(feeding_events) and feeding_events[feeding_idx]['time'] <= t + 1e-9:
            current_feeding = feeding_events[feeding_idx]['state']; feeding_idx += 1
        gap_flag = 1 if in_gap_at(t, gaps) else 0
        wc = wall_clock(t)
        behavior_cols = ['', '', '', '']
        feeding_cols = ['', '', '', '']
        if current:
            b = BEHAVIOR_BY_NAME[current]
            behavior_cols = [b['name'], str(b['y']), b['label'], b['color']]
        if current_feeding:
            f = FEEDING_BY_NAME[current_feeding]
            feeding_cols = [f['name'], str(f['y']), f['label'], f['color']]
        rows.append(','.join([f'{t:.4f}', f'{t*3600.0:.3f}', wc, str(gap_flag)] + behavior_cols + feeding_cols) + '\n')
        if t >= duration:
            break
    return ''.join(rows)

class AnnotationConflict(Exception):
    def __init__(self, current):
        super().__init__('stale annotation revision; reload and retry')
        self.current = current

class AnnotationReadOnly(Exception):
    def __init__(self, payload):
        super().__init__('annotation is in recovery mode and cannot be overwritten')
        self.payload = payload


def annotation_lock_for(key):
    with annotation_locks_guard:
        lock = annotation_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            annotation_locks[key] = lock
        return lock

def invalidate_comparison_cache():
    global comparison_cache_generation
    with comparison_cache_lock:
        comparison_cache_generation += 1
        comparison_cache.clear()

def annotation_log(msg):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with (STATE_DIR / 'annotations.log').open('a') as log:
            log.write('%s %s\n' % (dt.datetime.now().isoformat(timespec='seconds'), msg))
    except Exception:
        pass

def atomic_write_text(path, text):
    """Write text by replacing a same-directory temp file after fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise

def read_annotation_json(path):
    """Return (ok, data, error). Parse failure is never treated as an empty set."""
    path = Path(path)
    if not path.exists():
        return False, None, 'missing'
    try:
        raw = path.read_text(encoding='utf-8')
    except Exception as e:
        return False, None, 'unreadable: %s' % e
    if not str(raw).strip():
        return False, None, 'empty file'
    try:
        data = json.loads(raw)
    except Exception as e:
        return False, None, 'malformed JSON: %s' % e
    if not isinstance(data, dict):
        return False, None, 'JSON root is not an object'
    return True, data, None

def annotation_content_hash(data):
    payload = {
        'events': data.get('events') or [],
        'feeding_events': data.get('feeding_events') or [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]

def infer_annotation_state(data):
    explicit = data.get('state') if isinstance(data, dict) else None
    if explicit in ('untouched', 'in_progress', 'complete', 'reopened'):
        return explicit
    events = (data or {}).get('events') or []
    feeding = (data or {}).get('feeding_events') or []
    return 'in_progress' if events or feeding else 'untouched'

def next_annotation_state(previous, events, feeding_events, requested=None):
    if requested in ('untouched', 'in_progress', 'complete', 'reopened'):
        return requested
    prev = (previous or {}).get('state')
    has = bool(events or feeding_events)
    if prev == 'complete':
        return 'reopened' if has else 'complete'
    if has:
        return 'in_progress'
    return prev if prev in ('untouched', 'in_progress', 'complete', 'reopened') else 'untouched'

def revision_history_dir(json_path):
    json_path = Path(json_path)
    stem = json_path.name.replace('.annotations.json', '')
    return json_path.parent / 'revisions' / stem

def latest_archived_revision(json_path):
    hist = revision_history_dir(json_path)
    best = 0
    if hist.exists():
        for path in hist.glob('*.annotations.json'):
            match = re.match(r'(\d+)\.annotations\.json$', path.name)
            if match:
                best = max(best, int(match.group(1)))
    return best

def archive_current_revision(json_path, csv_path, revision):
    hist = revision_history_dir(json_path)
    hist.mkdir(parents=True, exist_ok=True)
    dest_json = hist / ('%06d.annotations.json' % int(revision))
    dest_csv = hist / ('%06d.ethogram.csv' % int(revision))
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    if json_path.exists() and not dest_json.exists():
        shutil.copy2(json_path, dest_json)
    if csv_path.exists() and not dest_csv.exists():
        shutil.copy2(csv_path, dest_csv)

def lacie_paths_for(json_path, csv_path, lacie_dir=None):
    if not lacie_dir:
        return None
    root = Path(lacie_dir)
    if not root.exists():
        return None
    return root / Path(json_path).name, root / Path(csv_path).name

def sibling_csv_for(json_candidate):
    path = Path(json_candidate)
    csv_path = path.with_name(path.name.replace('.annotations.json', '.ethogram.csv'))
    return csv_path if csv_path.exists() else None

def candidate_record(source, path, revision=None):
    path = Path(path)
    ok, data, err = read_annotation_json(path)
    if not ok:
        return None
    rev = int(data.get('revision') or 0) if revision is None else int(revision)
    return {
        'source': source,
        'path': str(path),
        'revision': rev,
        'saved_at': data.get('saved_at'),
        'updated_at': data.get('updated_at') or data.get('saved_at'),
        'content_hash': annotation_content_hash(data),
        'event_count': len(data.get('events') or []),
        'feeding_count': len(data.get('feeding_events') or []),
        'state': infer_annotation_state(data),
    }

def collect_recovery_candidates(json_path, paired=None, fallback=None, lacie_dir=None):
    candidates = []
    hist = revision_history_dir(json_path)
    if hist.exists():
        for path in sorted(hist.glob('*.annotations.json'), reverse=True):
            match = re.match(r'(\d+)\.annotations\.json$', path.name)
            rec = candidate_record('revision', path, int(match.group(1)) if match else None)
            if rec:
                candidates.append(rec)
    if paired and paired[0]:
        rec = candidate_record('paired', paired[0])
        if rec:
            candidates.append(rec)
    lacie = lacie_paths_for(json_path, Path(json_path).with_name(Path(json_path).name.replace('.annotations.json', '.ethogram.csv')), lacie_dir=lacie_dir)
    if lacie:
        rec = candidate_record('lacie', lacie[0])
        if rec:
            candidates.append(rec)
    if fallback and fallback[0]:
        rec = candidate_record('legacy', fallback[0])
        if rec:
            candidates.append(rec)
    return candidates

def public_annotation_view(data, key, json_path, csv_path, video_path=None, video_name=None, paired=None, recovery=None):
    view = dict(data or {})
    view.setdefault('video', video_path.name if video_path else Path(str(video_name or '')).name)
    view.setdefault('annotation_key', key)
    view.setdefault('events', [])
    view.setdefault('behaviors', BEHAVIORS)
    view.setdefault('feeding_events', [])
    view.setdefault('feeding_states', FEEDING_STATES)
    view['revision'] = int(view.get('revision') or 0)
    view['schema_version'] = int(view.get('schema_version') or ANNOTATION_SCHEMA_VERSION)
    view['state'] = infer_annotation_state(view)
    view['content_hash'] = view.get('content_hash') or annotation_content_hash(view)
    view['json_path'] = str(json_path)
    view['csv_path'] = str(csv_path)
    view['csv_exists'] = Path(csv_path).exists()
    view['read_only'] = bool(recovery)
    view['recovery'] = recovery
    view['stable_annotation'] = True
    if paired and paired[0] and paired[1]:
        view['recording_json_path'] = str(paired[0])
        view['recording_csv_path'] = str(paired[1])
        view['recording_csv_exists'] = Path(paired[1]).exists()
    return view

def replicate_committed(key, json_text, csv_text, json_path, csv_path, paired=None, lacie_dir=None):
    try:
        if paired and paired[0] and paired[1]:
            atomic_write_text(paired[0], json_text)
            atomic_write_text(paired[1], csv_text)
    except Exception as e:
        annotation_log('paired replica failed for %s: %s' % (key, e))
    try:
        lacie = lacie_paths_for(json_path, csv_path, lacie_dir=lacie_dir)
        if lacie:
            atomic_write_text(lacie[0], json_text)
            atomic_write_text(lacie[1], csv_text)
    except Exception as e:
        annotation_log('lacie replica failed for %s: %s' % (key, e))

def commit_annotation_revision(key, json_path, csv_path, data, csv_text, base_revision, paired=None, lacie_dir=None, replace_unreadable=False):
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    previous = None
    current_rev = 0
    if json_path.exists() and not replace_unreadable:
        ok, previous, err = read_annotation_json(json_path)
        if not ok:
            raise AnnotationReadOnly({'json_path': str(json_path), 'recovery': {'corrupt_canonical': True, 'error': err}})
        current_rev = int(previous.get('revision') or 0)
    if not replace_unreadable and int(base_revision) != current_rev:
        raise AnnotationConflict(public_annotation_view(previous if previous is not None else {'revision': current_rev}, key, json_path, csv_path, paired=paired))
    if json_path.exists() and not replace_unreadable:
        archive_current_revision(json_path, csv_path, current_rev)
    now = time.time()
    committed = dict(data)
    committed['revision'] = max(current_rev, latest_archived_revision(json_path), int((data or {}).get('revision') or 0)) + 1
    committed['schema_version'] = ANNOTATION_SCHEMA_VERSION
    committed['saved_at'] = now
    committed['updated_at'] = now
    committed['created_at'] = (previous or {}).get('created_at') or now
    committed['state'] = next_annotation_state(previous, committed.get('events') or [], committed.get('feeding_events') or [], committed.get('state'))
    if committed['state'] == 'complete':
        committed['completed_at'] = (previous or {}).get('completed_at') or now
    committed['content_hash'] = annotation_content_hash(committed)
    json_text = json.dumps(committed, indent=2)
    atomic_write_text(json_path, json_text)
    atomic_write_text(csv_path, csv_text)
    replica_pool.submit(replicate_committed, key, json_text, csv_text, json_path, csv_path, paired, lacie_dir)
    return committed

def migrate_missing_canonical(json_path, csv_path, source_json, source_csv, migrated_from_key):
    ok, data, err = read_annotation_json(source_json)
    if not ok:
        return None, err
    data = dict(data)
    data[migrated_from_key] = str(source_json)
    atomic_write_text(json_path, json.dumps(data, indent=2))
    csv_src = Path(source_csv) if source_csv else sibling_csv_for(source_json)
    if csv_src and csv_src.exists() and not Path(csv_path).exists():
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_src, csv_path)
    return data, None

def load_annotations_from_paths(key, json_path, csv_path, video_path=None, video_name=None, fallback=None, paired=None, lacie_dir=None):
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    paired_json = paired[0] if paired else None
    paired_csv = paired[1] if paired else None
    if json_path.exists():
        ok, data, err = read_annotation_json(json_path)
        if ok:
            return public_annotation_view(data, key, json_path, csv_path, video_path=video_path, video_name=video_name, paired=paired)
        recovery = {
            'corrupt_canonical': True,
            'error': err,
            'candidates': collect_recovery_candidates(json_path, paired=paired, fallback=fallback, lacie_dir=lacie_dir),
        }
        return public_annotation_view({'events': [], 'feeding_events': [], 'revision': 0}, key, json_path, csv_path, video_path=video_path, video_name=video_name, paired=paired, recovery=recovery)
    if paired_json and Path(paired_json).exists():
        data, err = migrate_missing_canonical(json_path, csv_path, paired_json, paired_csv, 'migrated_from_recording_json_path')
        if data is not None:
            return public_annotation_view(data, key, json_path, csv_path, video_path=video_path, video_name=video_name, paired=paired)
    if fallback and fallback[0] and Path(fallback[0]).exists():
        data, err = migrate_missing_canonical(json_path, csv_path, fallback[0], fallback[1] if len(fallback) > 1 else None, 'migrated_from_json_path')
        if data is not None:
            return public_annotation_view(data, key, json_path, csv_path, video_path=video_path, video_name=video_name, paired=paired)
    return public_annotation_view({'revision': 0, 'events': [], 'feeding_events': []}, key, json_path, csv_path, video_path=video_path, video_name=video_name, paired=paired)

def load_annotations(video_name=None, trial=None, date=None, camera=None):
    video_path, key, json_path, csv_path, fallback, paired = annotation_storage_paths(video_name=video_name, trial=trial, date=date, camera=camera)
    with annotation_lock_for(key):
        return load_annotations_from_paths(key, json_path, csv_path, video_path=video_path, video_name=video_name, fallback=fallback, paired=paired, lacie_dir=LACIE_ANNOTATION_ROOT if LACIE_ANNOTATION_ROOT.exists() else None)

def save_annotations(payload):
    meta = payload.get('meta') or {}
    video_path, key, json_path, csv_path, fallback, paired = annotation_storage_paths(video_name=payload.get('video'), meta=meta)
    try:
        duration = float(payload.get('duration') or meta.get('target_duration_seconds') or 0.0)
    except Exception:
        duration = 0.0
    events = sanitize_annotation_events(payload.get('events') or [], duration if duration > 0 else None)
    feeding_events = sanitize_feeding_events(payload.get('feeding_events') or [], duration if duration > 0 else None)
    with annotation_lock_for(key):
        if json_path.exists():
            ok, _, err = read_annotation_json(json_path)
            if not ok:
                loaded = load_annotations_from_paths(key, json_path, csv_path, video_path=video_path, video_name=payload.get('video'), fallback=fallback, paired=paired, lacie_dir=LACIE_ANNOTATION_ROOT if LACIE_ANNOTATION_ROOT.exists() else None)
                raise AnnotationReadOnly(loaded)
        data = {
            'video': video_path.name if video_path else Path(str(payload.get('video') or '')).name,
            'annotation_key': key,
            'behaviors': BEHAVIORS,
            'feeding_states': FEEDING_STATES,
            'duration_seconds': duration,
            'events': events,
            'feeding_events': feeding_events,
            'meta': meta,
            'source_fingerprint': (meta or {}).get('cache_key'),
            'state': payload.get('state'),
            'note': 'Annotation sidecar saved by the separate Basler annotation interface; recorder video files are not modified. A paired copy is also saved in the matching recording-session directory when available.'
        }
        csv_text = build_smooth_ethogram_csv(events, feeding_events, duration, meta)
        committed = commit_annotation_revision(
            key, json_path, csv_path, data, csv_text,
            payload.get('base_revision') if payload.get('base_revision') is not None else 0,
            paired=paired,
            lacie_dir=LACIE_ANNOTATION_ROOT if LACIE_ANNOTATION_ROOT.exists() else None,
        )
        view = public_annotation_view(committed, key, json_path, csv_path, video_path=video_path, video_name=payload.get('video'), paired=paired)
        view['csv_rows'] = max(0, len(csv_text.splitlines()) - 1)
        invalidate_comparison_cache()
        return view

def recover_annotations(payload):
    meta = payload.get('meta') or {}
    video_path, key, json_path, csv_path, fallback, paired = annotation_storage_paths(
        video_name=payload.get('video'), meta=meta,
        trial=payload.get('trial') or meta.get('trial'),
        date=payload.get('date') or meta.get('date'),
        camera=payload.get('camera') or meta.get('camera'),
    )
    with annotation_lock_for(key):
        loaded = load_annotations_from_paths(key, json_path, csv_path, video_path=video_path, video_name=payload.get('video'), fallback=fallback, paired=paired, lacie_dir=LACIE_ANNOTATION_ROOT if LACIE_ANNOTATION_ROOT.exists() else None)
        if not loaded.get('recovery'):
            raise RuntimeError('recovery is only allowed when the canonical annotation JSON is unreadable')
        want_path = payload.get('path')
        want_source = payload.get('source')
        want_rev = payload.get('revision')
        chosen = None
        for cand in loaded['recovery'].get('candidates') or []:
            if want_path and cand.get('path') != want_path:
                continue
            if want_source and cand.get('source') != want_source:
                continue
            if want_source == 'revision' and want_rev is not None and int(cand.get('revision') or 0) != int(want_rev):
                continue
            chosen = cand
            break
        if not chosen:
            raise RuntimeError('recovery source was not found among verified candidates')
        ok, data, err = read_annotation_json(chosen['path'])
        if not ok:
            raise RuntimeError('recovery source unreadable: %s' % err)
        if json_path.exists():
            corrupt_dest = json_path.with_name(json_path.name + '.corrupt.%d' % int(time.time()))
            try:
                shutil.copy2(json_path, corrupt_dest)
                annotation_log('preserved corrupt canonical as %s' % corrupt_dest)
            except Exception as e:
                annotation_log('failed to preserve corrupt canonical %s: %s' % (json_path, e))
        csv_src = sibling_csv_for(chosen['path'])
        duration = float(data.get('duration_seconds') or (data.get('meta') or {}).get('target_duration_seconds') or 0.0)
        events = sanitize_annotation_events(data.get('events') or [], duration if duration > 0 else None)
        feeding_events = sanitize_feeding_events(data.get('feeding_events') or [], duration if duration > 0 else None)
        csv_text = csv_src.read_text(encoding='utf-8') if csv_src else build_smooth_ethogram_csv(events, feeding_events, duration, data.get('meta') or {})
        recovered = dict(data)
        recovered['events'] = events
        recovered['feeding_events'] = feeding_events
        recovered['recovered_from'] = {'source': chosen['source'], 'path': chosen['path'], 'revision': chosen.get('revision')}
        committed = commit_annotation_revision(
            key, json_path, csv_path, recovered, csv_text,
            0,
            paired=paired,
            lacie_dir=LACIE_ANNOTATION_ROOT if LACIE_ANNOTATION_ROOT.exists() else None,
            replace_unreadable=True,
        )
        invalidate_comparison_cache()
        return public_annotation_view(committed, key, json_path, csv_path, video_path=video_path, video_name=payload.get('video'), paired=paired)



def state_duration_hours(events, duration_hours, gaps=None):
    duration_hours = max(0.0, float(duration_hours or 0.0))
    gaps = gaps or []
    events = sanitize_annotation_events(events, duration_hours if duration_hours > 0 else None)
    totals = {b['name']: 0.0 for b in BEHAVIORS}
    unannotated = 0.0
    cursor = 0.0
    current = None
    for ev in events:
        t = max(0.0, min(duration_hours, float(ev['time'])))
        if t > cursor:
            seglen = length_excluding_gaps(cursor, t, gaps)
            if current in totals:
                totals[current] += seglen
            else:
                unannotated += seglen
        current = ev['behavior']
        cursor = t
    if duration_hours > cursor:
        seglen = length_excluding_gaps(cursor, duration_hours, gaps)
        if current in totals:
            totals[current] += seglen
        else:
            unannotated += seglen
    return totals, unannotated


def wall_interval_light_dark_hours(start_ts, end_ts, light_start=4.75, light_end=22.0):
    """Split a wall-clock interval into local light/dark hours."""
    start_ts = float(start_ts)
    end_ts = float(end_ts)
    out = {'light': 0.0, 'dark': 0.0}
    if end_ts <= start_ts:
        return out
    cursor = start_ts
    while cursor < end_ts - 1e-9:
        local = dt.datetime.fromtimestamp(cursor)
        day = local.date()
        midnight = dt.datetime.combine(day, dt.time()).timestamp()
        next_midnight = dt.datetime.combine(day + dt.timedelta(days=1), dt.time()).timestamp()
        chunk_end = min(end_ts, next_midnight)
        light_a = midnight + float(light_start) * 3600.0
        light_b = midnight + float(light_end) * 3600.0
        light_seconds = max(0.0, min(chunk_end, light_b) - max(cursor, light_a))
        chunk_seconds = max(0.0, chunk_end - cursor)
        out['light'] += light_seconds / 3600.0
        out['dark'] += max(0.0, chunk_seconds - light_seconds) / 3600.0
        cursor = chunk_end
    return out


def state_window_duration_hours(events, duration_hours, window_start_ts, window_end_ts,
                                gaps=None, pieces=None, clip_start_ts=None,
                                light_start=4.75, light_end=22.0,
                                coverage_start_ts=None, coverage_end_ts=None):
    """Behavior percentages in an absolute wall-clock window.

    Playback gaps are excluded. Coverage metadata uses finalized recorded duration,
    while percentages are normalized only over moving, SA, and quiescent hours.
    Unannotated and poking footage are therefore excluded from the percentage
    denominator.
    """
    analytic_names = ['moving', 'SA', 'quiescent']
    duration_hours = max(0.0, float(duration_hours or 0.0))
    gaps = gaps or []
    pieces = pieces or []
    events = sanitize_annotation_events(events, duration_hours if duration_hours > 0 else None)
    out = {
        'window_start_ts': float(window_start_ts),
        'window_end_ts': float(window_end_ts),
        'recorded_hours': 0.0,
        'classified_hours': 0.0,
        'available': False,
        'states': {name: {'hours': 0.0, 'pct': 0.0} for name in analytic_names},
        'periods': {
            period: {
                'recorded_hours': 0.0,
                'classified_hours': 0.0,
                'available': False,
                'states': {name: {'hours': 0.0, 'pct': 0.0} for name in analytic_names},
            }
            for period in ('light', 'dark')
        },
    }
    if out['window_end_ts'] <= out['window_start_ts']:
        return out

    cursor = 0.0
    current = None
    intervals = []
    for ev in events:
        t = max(0.0, min(duration_hours, float(ev['time'])))
        if t > cursor:
            intervals.append((cursor, t, current))
        current = ev['behavior']
        cursor = t
    if duration_hours > cursor:
        intervals.append((cursor, duration_hours, current))

    for t0, t1, behavior in intervals:
        for a, b in nongap_subintervals(t0, t1, gaps):
            wall_a = play_to_wall_ts(a, pieces, clip_start_ts)
            wall_b = play_to_wall_ts(b, pieces, clip_start_ts)
            if wall_a is None or wall_b is None or wall_b <= wall_a:
                continue
            lo = max(wall_a, out['window_start_ts'], float(coverage_start_ts) if coverage_start_ts is not None else wall_a)
            hi = min(wall_b, out['window_end_ts'], float(coverage_end_ts) if coverage_end_ts is not None else wall_b)
            if hi <= lo:
                continue
            hours = (hi - lo) / 3600.0
            out['recorded_hours'] += hours
            if behavior in out['states']:
                out['states'][behavior]['hours'] += hours
            split = wall_interval_light_dark_hours(lo, hi, light_start=light_start, light_end=light_end)
            for period, period_hours in split.items():
                out['periods'][period]['recorded_hours'] += period_hours
                if behavior in out['periods'][period]['states']:
                    out['periods'][period]['states'][behavior]['hours'] += period_hours

    out['classified_hours'] = sum(value['hours'] for value in out['states'].values())
    denom = out['classified_hours']
    out['available'] = denom > 0
    for value in out['states'].values():
        value['pct'] = (value['hours'] / denom) * 100.0 if denom > 0 else 0.0
    for period in ('light', 'dark'):
        period_rec = out['periods'][period]
        period_rec['classified_hours'] = sum(value['hours'] for value in period_rec['states'].values())
        period_denom = period_rec['classified_hours']
        period_rec['available'] = period_denom > 0
        for value in period_rec['states'].values():
            value['pct'] = (value['hours'] / period_denom) * 100.0 if period_denom > 0 else 0.0
    return out


def state_disturbance_phase_duration_hours(events, duration_hours, gaps=None, pieces=None, clip_start_ts=None, first_day_start_ts=None):
    """Aggregate behavior durations into pre/during/post disturbance windows.

    Windows are absolute trial hours relative to midnight of the first recorded day:
    pre 12:00 day 1 -> 12:00 day 2, during 12:00 -> 18:00 day 2,
    post 18:00 day 2 -> 12:00 day 3.  Gaps/no-video spans are removed from
    denominators and behavior totals.
    """
    duration_hours = max(0.0, float(duration_hours or 0.0))
    gaps = gaps or []
    events = sanitize_annotation_events(events, duration_hours if duration_hours > 0 else None)
    phases = [
        {'key': 'pre', 'label': 'pre disturbance', 'start_hour': 12.0, 'end_hour': 36.0},
        {'key': 'during', 'label': 'during disturbance', 'start_hour': 36.0, 'end_hour': 42.0},
        {'key': 'post', 'label': 'post disturbance', 'start_hour': 42.0, 'end_hour': 60.0},
    ]
    out = {ph['key']: {'phase': ph['key'], 'label': ph['label'], 'start_hour': ph['start_hour'], 'end_hour': ph['end_hour'], 'recorded_hours': 0.0, 'states': {name: {'hours': 0.0, 'pct': 0.0} for name in ['moving', 'SA', 'quiescent']}} for ph in phases}
    def abs_hour(t):
        if pieces and clip_start_ts is not None and first_day_start_ts is not None:
            ts = play_to_wall_ts(t, pieces, clip_start_ts)
            if ts is not None:
                return (ts - first_day_start_ts) / 3600.0
        if clip_start_ts is not None and first_day_start_ts is not None:
            return ((clip_start_ts - first_day_start_ts) / 3600.0) + t
        return t
    cursor = 0.0
    current = None
    intervals = []
    for ev in events:
        t = max(0.0, min(duration_hours, float(ev['time'])))
        if t > cursor:
            intervals.append((cursor, t, current))
        current = ev['behavior']
        cursor = t
    if duration_hours > cursor:
        intervals.append((cursor, duration_hours, current))
    for t0, t1, behavior in intervals:
        for a, b in nongap_subintervals(t0, t1, gaps):
            ah0, ah1 = abs_hour(a), abs_hour(b)
            if ah1 <= ah0:
                continue
            for ph in phases:
                lo = max(ah0, ph['start_hour']); hi = min(ah1, ph['end_hour'])
                if hi <= lo:
                    continue
                hours = hi - lo
                rec = out[ph['key']]
                rec['recorded_hours'] += hours
                if behavior in rec['states']:
                    rec['states'][behavior]['hours'] += hours
    for rec in out.values():
        denom = rec['recorded_hours'] if rec['recorded_hours'] > 0 else 1.0
        for val in rec['states'].values():
            val['pct'] = (val['hours'] / denom) * 100.0
    return out
def light_dark_overlap_hours(wall_start_hour, wall_end_hour, light_start=4.75, light_end=22.0):
    wall_start_hour = max(0.0, min(24.0, float(wall_start_hour or 0.0)))
    wall_end_hour = max(0.0, min(24.0, float(wall_end_hour or 0.0)))
    light_start = max(0.0, min(24.0, float(light_start)))
    light_end = max(0.0, min(24.0, float(light_end)))
    if wall_end_hour <= wall_start_hour:
        return {'light': 0.0, 'dark': 0.0}
    light = max(0.0, min(wall_end_hour, light_end) - max(wall_start_hour, light_start))
    dark = (wall_end_hour - wall_start_hour) - light
    return {'light': max(0.0, light), 'dark': max(0.0, dark)}

def state_light_dark_duration_hours(events, duration_hours, clip_start_hour, gaps=None, pieces=None, clip_start_ts=None, day_start=None, light_start=4.75, light_end=22.0):
    duration_hours = max(0.0, float(duration_hours or 0.0))
    clip_start_hour = float(clip_start_hour or 0.0)
    gaps = gaps or []
    events = sanitize_annotation_events(events, duration_hours if duration_hours > 0 else None)
    totals = {period: {b['name']: 0.0 for b in BEHAVIORS} for period in ('light', 'dark')}
    period_totals = {'light': 0.0, 'dark': 0.0}
    # Anchor recorded footage to real wall time via the piece timeline so a floored
    # gap (padded slightly longer than real for visibility) does not shift the wall
    # clock of footage that comes after it.  Falls back to the linear approximation
    # when piece info is unavailable.
    def wall_hour(t):
        if pieces and clip_start_ts is not None and day_start is not None:
            ts = play_to_wall_ts(t, pieces, clip_start_ts)
            if ts is not None:
                return (ts - day_start) / 3600.0
        return clip_start_hour + t
    cursor = 0.0
    current = None
    intervals = []
    for ev in events:
        t = max(0.0, min(duration_hours, float(ev['time'])))
        if t > cursor:
            intervals.append((cursor, t, current))
        current = ev['behavior']
        cursor = t
    if duration_hours > cursor:
        intervals.append((cursor, duration_hours, current))
    for t0, t1, behavior in intervals:
        for (a, b) in nongap_subintervals(t0, t1, gaps):
            overlaps = light_dark_overlap_hours(wall_hour(a), wall_hour(b), light_start=light_start, light_end=light_end)
            for period, val in overlaps.items():
                period_totals[period] += val
                if behavior in totals[period]:
                    totals[period][behavior] += val
    return totals, period_totals
def comparison_snapshot():
    """Return one stable recorder catalog snapshot for comparison work.

    The options endpoint refreshes this snapshot explicitly. Comparisons must not
    let the short scan TTL trigger repeated full recorder-tree scans while a
    single long analytics request is still running or wait behind background
    maintenance. scan_recordings publishes these objects only after a complete
    scan, so readers can safely retain the previous immutable snapshot lock-free.
    """
    items, options, scan_version = scan_cache['items'], scan_cache['options'], scan_cache['ts']
    if items and options:
        return items, options, scan_version
    items, options = scan_recordings(False)
    return items, options, scan_cache['ts']

def compare_ethograms(trial, selected_date=None, _snapshot=None):
    snapshot = _snapshot or comparison_snapshot()
    all_items, options, scan_version = snapshot
    with comparison_cache_lock:
        generation = comparison_cache_generation
        key = (trial, selected_date or '', scan_version, generation)
        cached = comparison_cache.get(key)
        if cached and time.monotonic() - cached['ts'] < COMPARISON_CACHE_TTL:
            return cached['value']
    result = _compute_compare_ethograms(trial, selected_date, all_items, options)
    with comparison_cache_lock:
        if generation == comparison_cache_generation:
            comparison_cache[key] = {'ts': time.monotonic(), 'value': result}
    return result

def _compute_compare_ethograms(trial, selected_date, all_items, options):
    if not trial:
        raise RuntimeError('trial is required')
    tr = next((t for t in options.get('trials', []) if t.get('trial') == trial), None)
    if not tr:
        raise RuntimeError(f'trial not found: {trial}')
    cohort = tr.get('cohort') or 'other'
    light_regime = light_regime_for_cohort(cohort)
    trial_segments = [s for s in all_items if (s.get('trial') == trial or s.get('trial_group') == trial)]
    trial_start_ts = min((float(s['start_ts']) for s in trial_segments), default=0.0)
    half_windows = []
    if cohort == 'experiment_2' and trial_start_ts > 0:
        half_windows = [
            {'key': 'first_72h', 'label': 'First 72 hr', 'start_ts': trial_start_ts, 'end_ts': trial_start_ts + 72.0 * 3600.0, 'start_hour': 0.0, 'end_hour': 72.0},
            {'key': 'last_72h', 'label': 'Last 72 hr', 'start_ts': trial_start_ts + 72.0 * 3600.0, 'end_ts': trial_start_ts + 144.0 * 3600.0, 'start_hour': 72.0, 'end_hour': 144.0},
        ]
    aggregate_date = selected_date if selected_date in AGGREGATE_DATE_IDS else None
    if aggregate_date:
        dates = [aggregate_date]
        aggregate_segs = []
        for cam in CAMERA_ORDER:
            segs, _, _ = select_segments_from_items(all_items, trial, aggregate_date, cam)
            aggregate_segs.extend(segs)
        if not aggregate_segs:
            return {'trial': trial, 'selection_date': aggregate_date, 'dates': [], 'cameras': CAMERA_ORDER, 'feeding_states': FEEDING_STATES, 'axis_hours': 0.0, 'rows': []}
        first_day = dt.datetime.fromtimestamp(min(s['start_ts'] for s in aggregate_segs)).date()
        last_day = dt.datetime.fromtimestamp(max(s['end_ts'] for s in aggregate_segs)).date()
    else:
        # Aggregate range options are virtual selectors, not ISO calendar days. The
        # default whole-trial comparison continues to use stable daily sidecars.
        dates = sorted({d.get('date') for d in tr.get('dates', []) if d.get('date') and d.get('date') not in AGGREGATE_DATE_IDS})
        if not dates:
            return {'trial': trial, 'dates': [], 'cameras': CAMERA_ORDER, 'axis_hours': 0.0, 'rows': []}
        first_day = dt.date.fromisoformat(dates[0])
        last_day = dt.date.fromisoformat(dates[-1])
    axis_days = (last_day - first_day).days + 1
    display_dates = [(first_day + dt.timedelta(days=i)).isoformat() for i in range(axis_days)]
    rows = []
    analytics = []
    analytic_names = ['moving', 'SA', 'quiescent']
    first_day_start_ts = dt.datetime.combine(first_day, dt.time()).timestamp()
    for cam in CAMERA_ORDER:
        cam_items = []
        cam_totals = {name: 0.0 for name in [b['name'] for b in BEHAVIORS]}
        cam_period_totals = {period: {name: 0.0 for name in [b['name'] for b in BEHAVIORS]} for period in ('light', 'dark')}
        cam_period_recorded = {'light': 0.0, 'dark': 0.0}
        cam_date_analytics = []
        cam_phase_totals = {k: {'recorded_hours': 0.0, 'states': {name: {'hours': 0.0, 'pct': 0.0} for name in analytic_names}} for k in ('pre', 'during', 'post')}
        cam_half_totals = {
            window['key']: {
                'recorded_hours': 0.0,
                'classified_hours': 0.0,
                'states': {name: {'hours': 0.0} for name in analytic_names},
                'periods': {period: {'recorded_hours': 0.0, 'classified_hours': 0.0, 'states': {name: {'hours': 0.0} for name in analytic_names}} for period in ('light', 'dark')},
            }
            for window in half_windows
        }
        cam_unannotated = 0.0
        cam_recorded = 0.0
        for date in dates:
            try:
                _, json_path, _ = annotation_paths_for_selection(trial, date, cam)
                segs, day_start, day_end = select_segments_from_items(all_items, trial, date, cam)
                timeline = compose_timeline(segs)
                # The ethogram axis spans the padded playback timeline (recorded time +
                # no-video gap time); recorded coverage is used for denominators.
                axis = timeline['play_total']
                recorded = timeline['coverage_seconds'] / 3600.0
                gaps = gap_play_intervals(timeline)
                gaps_public = [{'play_start': round(g['play_start'], 6), 'play_dur': round(g['play_dur'], 6), 'real_seconds': round(g['real_dur'], 3)} for g in timeline['gaps']]
                if aggregate_date:
                    day_offset = 0.0
                    analysis_day_start = first_day_start_ts
                    clip_start_hour = max(0.0, (timeline['clip_start_ts'] - first_day_start_ts) / 3600.0) if segs else 0.0
                    clip_end_hour = clip_start_hour + axis
                else:
                    day_offset = (dt.date.fromisoformat(date) - first_day).days * 24.0
                    analysis_day_start = day_start
                    if segs:
                        clip_start_hour = max(0.0, (timeline['clip_start_ts'] - day_start) / 3600.0)
                        clip_end_hour = min(24.0, clip_start_hour + axis)
                    else:
                        clip_start_hour = 0.0; clip_end_hour = 0.0
                events = []
                feeding_events = []
                if json_path.exists():
                    ok, data, err = read_annotation_json(json_path)
                    if not ok:
                        cam_items.append({'date': date, 'offset_hours': day_offset, 'clip_start_hour': clip_start_hour, 'clip_end_hour': clip_end_hour, 'duration_hours': axis, 'recorded_hours': recorded, 'gaps': gaps_public, 'events': [], 'feeding_events': [], 'json_path': str(json_path), 'exists': True, 'corrupt': True, 'error': 'corrupt annotation JSON: %s' % err})
                        continue
                    events = sanitize_annotation_events(data.get('events') or [], axis if axis > 0 else None)
                    feeding_events = sanitize_feeding_events(data.get('feeding_events') or [], axis if axis > 0 else None)
                totals, unannotated = state_duration_hours(events, axis, gaps)
                period_totals, period_recorded = state_light_dark_duration_hours(events, axis, clip_start_hour, gaps, pieces=timeline['pieces'], clip_start_ts=timeline['clip_start_ts'], day_start=analysis_day_start, light_start=light_regime['light_start_hour'], light_end=light_regime['light_end_hour'])
                phase_totals = state_disturbance_phase_duration_hours(events, axis, gaps, pieces=timeline['pieces'], clip_start_ts=timeline['clip_start_ts'], first_day_start_ts=first_day_start_ts)
                half_totals = {
                    window['key']: state_window_duration_hours(
                        events, axis, window['start_ts'], window['end_ts'], gaps=gaps,
                        pieces=timeline['pieces'], clip_start_ts=timeline['clip_start_ts'],
                        light_start=light_regime['light_start_hour'], light_end=light_regime['light_end_hour'],
                        coverage_start_ts=None if aggregate_date else day_start,
                        coverage_end_ts=None if aggregate_date else day_end,
                    )
                    for window in half_windows
                }
                for name, val in totals.items(): cam_totals[name] += val
                for period in ('light', 'dark'):
                    cam_period_recorded[period] += period_recorded.get(period, 0.0)
                    for name, val in period_totals.get(period, {}).items():
                        cam_period_totals[period][name] += val
                for ph_key, ph in phase_totals.items():
                    cam_phase_totals[ph_key]['recorded_hours'] += ph.get('recorded_hours', 0.0)
                    for name in analytic_names:
                        cam_phase_totals[ph_key]['states'][name]['hours'] += ph.get('states', {}).get(name, {}).get('hours', 0.0)
                for window in half_windows:
                    key = window['key']; half = half_totals[key]; dest = cam_half_totals[key]
                    dest['recorded_hours'] += half.get('recorded_hours', 0.0)
                    dest['classified_hours'] += half.get('classified_hours', 0.0)
                    for name in analytic_names:
                        dest['states'][name]['hours'] += half.get('states', {}).get(name, {}).get('hours', 0.0)
                    for period in ('light', 'dark'):
                        dest['periods'][period]['recorded_hours'] += half.get('periods', {}).get(period, {}).get('recorded_hours', 0.0)
                        dest['periods'][period]['classified_hours'] += half.get('periods', {}).get(period, {}).get('classified_hours', 0.0)
                        for name in analytic_names:
                            dest['periods'][period]['states'][name]['hours'] += half.get('periods', {}).get(period, {}).get('states', {}).get(name, {}).get('hours', 0.0)
                cam_unannotated += unannotated
                cam_recorded += recorded
                cam_items.append({'date': date, 'offset_hours': day_offset, 'clip_start_hour': clip_start_hour, 'clip_end_hour': clip_end_hour, 'duration_hours': axis, 'recorded_hours': recorded, 'gaps': gaps_public, 'events': events, 'feeding_events': feeding_events, 'json_path': str(json_path), 'exists': json_path.exists()})
                date_denom = recorded if recorded > 0 else 1.0
                date_analytic = {'date': date, 'recorded_hours': recorded, 'unannotated_hours': unannotated, 'unannotated_pct': (unannotated / date_denom) * 100.0, 'states': {}, 'periods': {}}
                for name in analytic_names:
                    hours = totals.get(name, 0.0)
                    date_analytic['states'][name] = {'hours': hours, 'pct': (hours / date_denom) * 100.0}
                for period in ('light', 'dark'):
                    p_rec = period_recorded.get(period, 0.0)
                    p_denom = p_rec if p_rec > 0 else 1.0
                    date_analytic['periods'][period] = {'recorded_hours': p_rec, 'states': {}}
                    for name in analytic_names:
                        hours = period_totals.get(period, {}).get(name, 0.0)
                        date_analytic['periods'][period]['states'][name] = {'hours': hours, 'pct': (hours / p_denom) * 100.0}
                cam_date_analytics.append(date_analytic)
            except Exception as e:
                cam_items.append({'date': date, 'offset_hours': 0.0, 'duration_hours': 0.0, 'events': [], 'feeding_events': [], 'error': str(e), 'exists': False})
        rows.append({'camera': cam, 'dates': cam_items})
        denom = cam_recorded if cam_recorded > 0 else 1.0
        analytic = {'camera': cam, 'group': CAMERA_GROUPS.get(cam, ''), 'recorded_hours': cam_recorded, 'unannotated_hours': cam_unannotated, 'unannotated_pct': (cam_unannotated / denom) * 100.0, 'states': {}}
        for name in analytic_names:
            hours = cam_totals.get(name, 0.0)
            analytic['states'][name] = {'hours': hours, 'pct': (hours / denom) * 100.0}
        analytic['periods'] = {}
        for period in ('light', 'dark'):
            pdenom = cam_period_recorded.get(period, 0.0) if cam_period_recorded.get(period, 0.0) > 0 else 1.0
            analytic['periods'][period] = {'recorded_hours': cam_period_recorded.get(period, 0.0), 'states': {}}
            for name in analytic_names:
                hours = cam_period_totals.get(period, {}).get(name, 0.0)
                analytic['periods'][period]['states'][name] = {'hours': hours, 'pct': (hours / pdenom) * 100.0}
        analytic['dates'] = cam_date_analytics
        phase_defs = {'pre': ('pre disturbance', 12.0, 36.0), 'during': ('during disturbance', 36.0, 42.0), 'post': ('post disturbance', 42.0, 60.0)}
        analytic['disturbance_phases'] = []
        for ph_key in ('pre', 'during', 'post'):
            label, start_h, end_h = phase_defs[ph_key]
            rec = cam_phase_totals[ph_key]
            denom = rec['recorded_hours'] if rec['recorded_hours'] > 0 else 1.0
            ph_out = {'phase': ph_key, 'label': label, 'start_hour': start_h, 'end_hour': end_h, 'recorded_hours': rec['recorded_hours'], 'states': {}}
            for name in analytic_names:
                hours = rec['states'][name]['hours']
                ph_out['states'][name] = {'hours': hours, 'pct': (hours / denom) * 100.0}
            analytic['disturbance_phases'].append(ph_out)
        analytic['experiment_2_72h_windows'] = []
        for window in half_windows:
            rec = cam_half_totals[window['key']]
            window_out = {
                'key': window['key'], 'label': window['label'],
                'start_hour': window['start_hour'], 'end_hour': window['end_hour'],
                'recorded_hours': rec['recorded_hours'],
                'classified_hours': rec['classified_hours'],
                'available': rec['classified_hours'] > 0,
                'states': {}, 'periods': {},
            }
            denom = rec['classified_hours']
            for name in analytic_names:
                hours = rec['states'][name]['hours']
                window_out['states'][name] = {'hours': hours, 'pct': (hours / denom) * 100.0 if denom > 0 else 0.0}
            for period in ('light', 'dark'):
                p_rec = rec['periods'][period]['recorded_hours']
                p_classified = rec['periods'][period]['classified_hours']
                window_out['periods'][period] = {
                    'recorded_hours': p_rec,
                    'classified_hours': p_classified,
                    'available': p_classified > 0,
                    'states': {},
                }
                for name in analytic_names:
                    hours = rec['periods'][period]['states'][name]['hours']
                    window_out['periods'][period]['states'][name] = {'hours': hours, 'pct': (hours / p_classified) * 100.0 if p_classified > 0 else 0.0}
            analytic['experiment_2_72h_windows'].append(window_out)
        analytics.append(analytic)
    return {'trial': trial, 'cohort': cohort, 'selection_date': aggregate_date, 'dates': display_dates, 'first_date': display_dates[0], 'last_date': display_dates[-1], 'axis_hours': axis_days * 24.0, 'behaviors': BEHAVIORS, 'feeding_states': FEEDING_STATES, 'cameras': CAMERA_ORDER, 'camera_groups': CAMERA_GROUPS, 'light_dark': light_regime, 'experiment_2_trial_start_ts': trial_start_ts if cohort == 'experiment_2' else None, 'experiment_2_window_definition': 'fixed elapsed trial hours 0-72 and 72-144 from earliest finalized recording across cameras' if cohort == 'experiment_2' else None, 'rows': rows, 'analytics': analytics}

def pool_experiment_2_72h_analytics(analytics):
    """Pool Experiment 2 half-window hours across trials, by camera.

    Percentages are calculated after summing underlying hours, never by averaging
    trial percentages. This correctly weights partial recordings and missing footage.
    """
    analytic_names = ['moving', 'SA', 'quiescent']
    window_defs = [
        {'key': 'first_72h', 'label': 'First 72 hr', 'start_hour': 0.0, 'end_hour': 72.0},
        {'key': 'last_72h', 'label': 'Last 72 hr', 'start_hour': 72.0, 'end_hour': 144.0},
    ]
    cameras = []
    exp2 = [a for a in analytics or [] if a.get('cohort') == 'experiment_2']
    for cam in CAMERA_ORDER:
        cam_rows = [a for a in exp2 if a.get('camera') == cam]
        windows = []
        for definition in window_defs:
            pooled = {
                **definition,
                'recorded_hours': 0.0,
                'classified_hours': 0.0,
                'n_trials_with_recording': 0,
                'n_trials_with_classified': 0,
                'states': {name: {'hours': 0.0, 'pct': 0.0} for name in analytic_names},
                'periods': {
                    period: {'recorded_hours': 0.0, 'classified_hours': 0.0, 'states': {name: {'hours': 0.0, 'pct': 0.0} for name in analytic_names}}
                    for period in ('light', 'dark')
                },
            }
            for row in cam_rows:
                window = next((w for w in row.get('experiment_2_72h_windows', []) if w.get('key') == definition['key']), None)
                if not window:
                    continue
                recorded = float(window.get('recorded_hours') or 0.0)
                pooled['recorded_hours'] += recorded
                if recorded > 0:
                    pooled['n_trials_with_recording'] += 1
                classified = sum(float(window.get('states', {}).get(name, {}).get('hours') or 0.0) for name in analytic_names)
                pooled['classified_hours'] += classified
                if classified > 0:
                    pooled['n_trials_with_classified'] += 1
                for name in analytic_names:
                    pooled['states'][name]['hours'] += float(window.get('states', {}).get(name, {}).get('hours') or 0.0)
                for period in ('light', 'dark'):
                    p = window.get('periods', {}).get(period, {})
                    pooled['periods'][period]['recorded_hours'] += float(p.get('recorded_hours') or 0.0)
                    pooled['periods'][period]['classified_hours'] += sum(float(p.get('states', {}).get(name, {}).get('hours') or 0.0) for name in analytic_names)
                    for name in analytic_names:
                        pooled['periods'][period]['states'][name]['hours'] += float(p.get('states', {}).get(name, {}).get('hours') or 0.0)
            denom = pooled['classified_hours']
            pooled['available'] = denom > 0
            for value in pooled['states'].values():
                value['pct'] = (value['hours'] / denom) * 100.0 if denom > 0 else 0.0
            for period in ('light', 'dark'):
                p_denom = pooled['periods'][period]['classified_hours']
                pooled['periods'][period]['available'] = p_denom > 0
                for value in pooled['periods'][period]['states'].values():
                    value['pct'] = (value['hours'] / p_denom) * 100.0 if p_denom > 0 else 0.0
            windows.append(pooled)
        cameras.append({'camera': cam, 'group': CAMERA_GROUPS.get(cam, ''), 'n_trials': len(cam_rows), 'windows': windows})
    return {
        'cohort': 'experiment_2',
        'trial_count': len({a.get('trial') for a in exp2 if a.get('trial')}),
        'window_definition': 'fixed elapsed trial hours 0-72 and 72-144 from each trial earliest finalized recording across cameras',
        'denominator': 'summed moving, SA, and quiescent hours in each window and illumination stratum; no-video gaps, unannotated footage, and poking excluded',
        'light_regime': light_regime_for_cohort('experiment_2'),
        'cameras': cameras,
    }


def compare_intertrial_ethograms(_snapshot=None):
    """Compare same-camera ethograms across all discovered trials.

    The x-axis is elapsed trial time (hours from each trial's first recorded day),
    so C-C, A-A, and B-B rows align trial_01_, trial_02_, and future trials by
    time into trial rather than by calendar date.
    """
    snapshot = _snapshot or comparison_snapshot()
    _, options, _ = snapshot
    trials = [t for t in options.get('trials', []) if t.get('trial') and t.get('cohort') in ('experiment_1', 'experiment_2')]
    trial_results = []
    for t in trials:
        try:
            comp = compare_ethograms(t.get('trial'), _snapshot=snapshot)
            trial_results.append({
                'trial': comp.get('trial'),
                'label': t.get('label') or comp.get('trial'),
                'cohort': t.get('cohort'),
                'light_regime': comp.get('light_dark') or light_regime_for_cohort(t.get('cohort')),
                'quality': t.get('quality'),
                'dates': comp.get('dates') or [],
                'axis_hours': comp.get('axis_hours') or 0.0,
                'rows': comp.get('rows') or [],
                'analytics': comp.get('analytics') or [],
            })
        except Exception as e:
            trial_results.append({'trial': t.get('trial'), 'label': t.get('label') or t.get('trial'), 'cohort': t.get('cohort'), 'light_regime': light_regime_for_cohort(t.get('cohort')), 'quality': t.get('quality'), 'error': str(e), 'axis_hours': 0.0, 'rows': [], 'analytics': []})
    axis_hours = max([float(t.get('axis_hours') or 0.0) for t in trial_results] + [0.0])
    rows = []
    analytics = []
    for cam in CAMERA_ORDER:
        cam_trials = []
        for tr in trial_results:
            row = next((r for r in tr.get('rows', []) if r.get('camera') == cam), None)
            ana = next((a for a in tr.get('analytics', []) if a.get('camera') == cam), None)
            cam_trials.append({'trial': tr.get('trial'), 'label': tr.get('label'), 'cohort': tr.get('cohort'), 'light_regime': tr.get('light_regime'), 'quality': tr.get('quality'), 'dates': (row or {}).get('dates') or [], 'axis_hours': tr.get('axis_hours') or 0.0, 'error': tr.get('error')})
            if ana:
                flat = dict(ana)
                flat['trial'] = tr.get('trial')
                flat['trial_label'] = tr.get('label')
                flat['cohort'] = tr.get('cohort')
                flat['light_regime'] = tr.get('light_regime')
                flat['quality'] = tr.get('quality')
                analytics.append(flat)
        rows.append({'camera': cam, 'group': CAMERA_GROUPS.get(cam, ''), 'trials': cam_trials})
    return {'trials': trial_results, 'axis_hours': axis_hours, 'behaviors': BEHAVIORS, 'cameras': CAMERA_ORDER, 'camera_groups': CAMERA_GROUPS, 'light_dark_by_cohort': LIGHT_REGIMES, 'rows': rows, 'analytics': analytics, 'experiment_2_72h_summary': pool_experiment_2_72h_analytics(analytics)}

def send_file(handler, path, ctype=None):
    path = Path(path)
    if not path.exists() or not path.is_file(): handler.send_error(404); return
    size = path.stat().st_size; rng = handler.headers.get('Range'); start=0; end=max(0,size-1); status=200
    if rng and rng.startswith('bytes='):
        spec = rng.split('=',1)[1].split(',',1)[0]
        if '-' in spec:
            a,b = spec.split('-',1)
            if a: start=int(a)
            if b: end=int(b)
            status=206
    start=max(0,min(start,end)); end=max(start,min(end,size-1)); length=end-start+1 if size else 0
    handler.send_response(status); handler.send_header('Content-Type', ctype or mimetypes.guess_type(str(path))[0] or 'application/octet-stream'); handler.send_header('Accept-Ranges','bytes'); handler.send_header('Content-Length',str(length))
    if status==206: handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
    handler.end_headers()
    with path.open('rb') as f:
        f.seek(start); remaining=length
        while remaining>0:
            chunk=f.read(min(1024*1024, remaining))
            if not chunk: break
            handler.wfile.write(chunk); remaining -= len(chunk)

INDEX = """<!doctype html><html><head><meta charset='utf-8'><title>Basler Annotation Interface</title><style>
:root{color-scheme:dark;--bg:#101318;--panel:#181e27;--line:#2d3748;--text:#e5edf6;--muted:#9aa7b7;--ok:#86efac;--warn:#facc15;--err:#fca5a5}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,#172033,#0b0e13 45%);color:var(--text);font-family:Inter,system-ui,sans-serif}header{position:sticky;top:0;z-index:2;background:rgba(16,19,24,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:14px 18px}.title{display:flex;justify-content:space-between;gap:12px;align-items:baseline}h1{font-size:20px;margin:0}.safe{color:var(--ok);font-size:13px}.bar{display:grid;grid-template-columns:1.3fr 190px 140px auto auto auto;gap:10px;margin-top:12px;align-items:end}label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}select,button,input{width:100%;background:#0d1117;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px}button{cursor:pointer;background:#1d4ed8;border-color:#2563eb;font-weight:650}button.secondary{background:#263241;border-color:#3a4758}.wrap{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px}body.status-hidden .wrap{grid-template-columns:minmax(0,1fr)}body.status-hidden .side{display:none}.viewer{background:#05070a;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.35)}video{display:block;width:100%;height:var(--video-height,auto);max-height:none;background:#000;object-fit:contain}.placeholder{min-height:52vh;display:grid;place-items:center;color:var(--muted);text-align:center;padding:40px}.side{background:rgba(24,30,39,.92);border:1px solid var(--line);border-radius:14px;padding:14px;min-height:300px}pre{white-space:pre-wrap;overflow:auto;max-height:40vh;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px;color:#d5e2f0}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 8px;color:var(--muted);font-size:12px;margin:3px 4px 3px 0}.ok{color:var(--ok)}.warn{color:var(--warn)}.err{color:var(--err)}.ethowrap{border-top:1px solid var(--line);background:#090c11;padding:10px}.ethotop{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:6px;color:var(--muted);font-size:13px}.keys kbd{background:#111827;border:1px solid #374151;border-bottom-width:2px;border-radius:5px;padding:2px 6px;color:#fff;margin:0 2px}.etho{width:100%;height:220px;display:block;background:#05070a;border:1px solid var(--line);border-radius:8px;cursor:crosshair;touch-action:none}.etho.feeding-etho{height:130px;margin-bottom:10px}.compare{border-top:1px solid var(--line);margin-top:10px;padding-top:10px}.compare h3{font-size:14px;margin:0 0 6px}.compare-note{color:var(--muted);font-size:12px;margin-bottom:6px}.compare canvas{width:100%;height:210px;display:block;background:#05070a;border:1px solid var(--line);border-radius:8px}.analytics{margin-top:8px;overflow:auto}.analytics table{width:100%;border-collapse:collapse;font-size:12px}.analytics th,.analytics td{border:1px solid var(--line);padding:6px 8px;text-align:right}.analytics th:first-child,.analytics td:first-child{text-align:left}.analytics th{color:var(--muted);font-weight:650;background:#0d1117}.analytics .warn{color:var(--warn)}.annbar{display:flex;gap:8px;align-items:center;margin-top:8px}.annbar button{width:auto;padding:7px 10px}.behbtn{border-color:var(--beh-color,#3a4758)!important}.active-ind{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:5px 10px}.swatch{width:16px;height:16px;border-radius:50%;background:#64748b;border:1px solid #fff8}.annstatus{color:var(--muted);font-size:12px}.recovery{display:none;background:#3f1d1d;border:1px solid #f87171;color:#fecaca;padding:12px 16px;margin:12px 16px 0;border-radius:12px}.recovery strong{color:#fff}.recovery button{width:auto;margin:8px 8px 0 0}@media(max-width:950px){.bar{grid-template-columns:1fr}.wrap{grid-template-columns:1fr}}
.compare-camera-card{border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:10px;background:rgba(255,255,255,.025)}.compare-camera-card h4{margin:0 0 8px;color:#e5edf6}.compare-camera-card canvas{display:block;width:100%;background:#05070a;border:1px solid var(--line);border-radius:8px}</style></head><body class='status-hidden'><header><div class='title'><h1>Basler annotation interface</h1><div class='safe'>read-only app - recorder-safe - port 8010</div></div><div class='bar'><div><label>Trial</label><select id='trial'></select></div><div><label>Calendar day</label><select id='date'></select></div><div><label>Camera</label><select id='camera'></select></div><button onclick='loadTimelapse()'>Load / retry selected video</button><button class='secondary' onclick='refreshOptions()'>Refresh list</button><button id='statusToggle' class='secondary' onclick='toggleStatusPanel()'>Show status</button></div></header><div id='recoveryBanner' class='recovery'></div><div class='wrap'><main class='viewer'><div id='placeholder' class='placeholder'>Select a trial, calendar day, and camera. Selected footage is cached automatically; playable ranges appear above.<br><br>Scale: 1 real hour = 1 playback second.<br>Smooth cache: 60 fps, one source frame per real minute, GPU-accelerated fast cache when available, identical 960x540 browser-compatible MP4s for all cameras, seek-optimized keyframes.</div><video id='video' preload='auto' playsinline style='display:none'></video><div id='videoControls' class='annbar' style='display:none;padding:0 10px 10px;flex-wrap:wrap'><button class='secondary' onclick='toggleVideoPlayback()'>Play / pause</button><label style='min-width:92px'>Video size</label><input id='videoSize' type='range' min='280' max='900' step='20' value='520' style='flex:1;min-width:220px'><button class='secondary' onclick='stepVideo(-0.01)'>← -0.01 s</button><button class='secondary' onclick='stepVideo(0.01)'>+0.01 s →</button></div><div id='ethowrap' class='ethowrap' style='display:none'><div class='ethotop'><div>Feeding ethogram: eating / not eating</div><div class='active-ind'><span id='feedingActiveSwatch' class='swatch'></span><span id='feedingActiveState'>current: unannotated</span></div></div><canvas id='feedingEthogram' class='etho feeding-etho' width='1200' height='130'></canvas><div class='ethotop'><div>Behavioral state ethogram: moving / SA / quiescent / poking</div><div class='active-ind'><span id='activeSwatch' class='swatch'></span><span id='activeBehavior'>current: unannotated</span></div></div><canvas id='ethogram' class='etho' width='1200' height='220'></canvas><div class='annbar'><label style='min-width:120px'>Horizontal scrubber</label><input id='scrubber' type='range' min='0' max='1' step='0.001' value='0' style='flex:1'><span id='timecode' class='annstatus'>0.00 s</span></div><div class='annbar'><label style='min-width:120px'>Clock seek</label><input id='clockSeek' placeholder='HHMM or HH:MM[:SS]' style='max-width:190px'><button class='secondary' onclick='seekClockTime()'>Go to time</button><span id='clockStatus' class='annstatus'>Enter military time for selected calendar day.</span></div><div class='annbar keys'><span class='annstatus' style='min-width:170px'>Feeding hotkeys</span><button class='secondary behbtn' style='--beh-color:#f97316' onclick="addFeedingAnnotation('eating')"><kbd>z</kbd> eating</button><button class='secondary behbtn' style='--beh-color:#06b6d4' onclick="addFeedingAnnotation('not_eating')"><kbd>x</kbd> not eating</button></div><div class='annbar keys'><span class='annstatus' style='min-width:170px'>Behavior hotkeys · space play/pause</span><button class='secondary behbtn' style='--beh-color:#e53935' onclick="addAnnotation('moving')"><kbd>q</kbd> moving</button><button class='secondary behbtn' style='--beh-color:#1e88e5' onclick="addAnnotation('SA')"><kbd>w</kbd> SA</button><button class='secondary behbtn' style='--beh-color:#fdd835' onclick="addAnnotation('quiescent')"><kbd>e</kbd> quiescent</button><button class='secondary behbtn' style='--beh-color:#43a047' onclick="addAnnotation('poking')"><kbd>r</kbd> poking</button></div><div class='annbar'><button class='secondary' onclick='undoAnnotation()'>Undo Ctrl+Z</button><button class='secondary' onclick='redoAnnotation()'>Redo Ctrl+Y</button><button class='secondary' onclick='saveAnnotations()'>Save annotations now</button><button class='secondary' onclick='clearAnnotations()'>Clear all annotations</button><span id='annstatus' class='annstatus'>Load a timelapse to annotate.</span></div><div id='feedingComparePanel' class='compare'><h3>C/A/B feeding ethogram comparison</h3><div id='feedingCompareNote' class='compare-note'>Loads saved eating / not-eating annotations for Cameras C/A/B.</div><canvas id='feedingCompareEthogram' width='1200' height='175' style='height:175px'></canvas></div><div id='comparePanel' class='compare'><h3>C/A/B behavioral-state ethogram comparison</h3><div id='compareNote' class='compare-note'>Loads saved annotations for this trial across all recorded dates.</div><canvas id='compareEthogram' width='1200' height='210'></canvas><div id='compareAnalytics' class='analytics'></div></div><div id='intertrialPanel' class='compare'><h3>Intertrial same-camera ethogram comparison</h3><div id='intertrialNote' class='compare-note'>Loads saved annotation comparisons across discovered trials.</div><div class='annbar' style='align-items:end;flex-wrap:wrap'><div style='min-width:180px;flex:1'><label>Intertrial comparison</label><select id='intertrialMode'><option value='all'>All trials</option><option value='pair'>Two selected trials</option></select></div><div style='min-width:220px;flex:1'><label>Trial 1</label><select id='intertrialTrialA'></select></div><div style='min-width:220px;flex:1'><label>Trial 2</label><select id='intertrialTrialB'></select></div><button class='secondary' onclick='drawIntertrialEthograms()'>Apply intertrial view</button></div><div id='intertrialCanvases'></div><div id='intertrialAnalytics' class='analytics'></div></div></div></main><aside class='side'><h3>Status</h3><div id='status'>Loading recordings...</div><h3>Selection info</h3><div id='summary'></div><h3>Source segments</h3><pre id='segments'></pre></aside></div><script>
let opts=null,currentMeta=null,currentVideoName=null,annEvents=[],feedingEvents=[],undoStack=[],redoStack=[],compareData=null,intertrialData=null,annRevision=0,saveTimer=null,saveInFlight=false,saveQueued=false,saveDirty=false,savePromise=null,saveBlocked=false,compareRefreshTimer=null,compareRefreshInFlight=false,compareRefreshQueued=false,annotationLocked=false,recoveryCandidates=[];
const trialEl=document.getElementById('trial'),dateEl=document.getElementById('date'),camEl=document.getElementById('camera'),video=createAnnotationPlayer(document.getElementById('video')),ph=document.getElementById('placeholder'),ethowrap=document.getElementById('ethowrap'),canvas=document.getElementById('ethogram'),ctx=canvas.getContext('2d'),feedingCanvas=document.getElementById('feedingEthogram'),feedingCtx=feedingCanvas.getContext('2d'),scrubber=document.getElementById('scrubber'),timecode=document.getElementById('timecode'),videoControls=document.getElementById('videoControls'),videoSize=document.getElementById('videoSize'),clockSeek=document.getElementById('clockSeek'),clockStatus=document.getElementById('clockStatus'),activeSwatch=document.getElementById('activeSwatch'),activeBehavior=document.getElementById('activeBehavior'),feedingActiveSwatch=document.getElementById('feedingActiveSwatch'),feedingActiveState=document.getElementById('feedingActiveState'),annstatus=document.getElementById('annstatus'),feedingCompareCanvas=document.getElementById('feedingCompareEthogram'),feedingCompareCtx=feedingCompareCanvas.getContext('2d'),feedingCompareNote=document.getElementById('feedingCompareNote'),compareCanvas=document.getElementById('compareEthogram'),compareCtx=compareCanvas.getContext('2d'),compareNote=document.getElementById('compareNote'),compareAnalytics=document.getElementById('compareAnalytics'),intertrialCanvases=document.getElementById('intertrialCanvases'),intertrialMode=document.getElementById('intertrialMode'),intertrialTrialA=document.getElementById('intertrialTrialA'),intertrialTrialB=document.getElementById('intertrialTrialB'),intertrialNote=document.getElementById('intertrialNote'),intertrialAnalytics=document.getElementById('intertrialAnalytics');
const BEH=[{key:'q',name:'moving',label:'moving',color:'#e53935',y:0},{key:'w',name:'SA',label:'still alert (SA)',color:'#1e88e5',y:1},{key:'e',name:'quiescent',label:'quiescent',color:'#fdd835',y:2},{key:'r',name:'poking',label:'poking',color:'#43a047',y:3}], BEHBY=Object.fromEntries(BEH.map(b=>[b.name,b]));
const FEED=[{key:'z',name:'eating',label:'eating',color:'#f97316',y:0},{key:'x',name:'not_eating',label:'not eating',color:'#06b6d4',y:1}], FEEDBY=Object.fromEntries(FEED.map(s=>[s.name,s]));
function esc(s){return String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]))}
async function api(path,opts){let r=await fetch(path,opts);let t=await r.text();let data=null;try{data=JSON.parse(t)}catch(e){}if(r.status===409||r.status===423){let err=new Error((data&&data.error)||t);err.status=r.status;err.payload=data;throw err}if(!r.ok&&r.status!==202)throw new Error(t||('HTTP '+r.status));if(data==null)throw new Error(t||'invalid JSON');return data}
function toggleStatusPanel(){let hidden=document.body.classList.toggle('status-hidden');document.getElementById('statusToggle').textContent=hidden?'Show status':'Hide status';setTimeout(drawEthogram,50)}
function selectedTrial(){return opts?.trials?.find(t=>t.trial===trialEl.value)}function selectedDate(){return selectedTrial()?.dates?.find(d=>d.date===dateEl.value)}
function selectionKey(){return `${trialEl.value}|${dateEl.value}|${camEl.value}`}
function fillTrials(preserveLoaded=false){let before=selectionKey(),keepTrial=trialEl.value;trialEl.innerHTML=(opts.trials||[]).map(t=>`<option value="${esc(t.trial)}">${esc(t.label)} (${t.segments} segments)</option>`).join('');if(keepTrial&&[...trialEl.options].some(o=>o.value===keepTrial))trialEl.value=keepTrial;fillDates(preserveLoaded,before)}function fillDates(preserveLoaded=false,beforeKey=null){let keepDate=dateEl.value,keepCam=camEl.value;let t=selectedTrial();dateEl.innerHTML=(t?.dates||[]).map(d=>`<option value="${esc(d.date)}">${esc(d.label||d.date)} - ${d.segments} segments</option>`).join('');if(keepDate&&[...dateEl.options].some(o=>o.value===keepDate))dateEl.value=keepDate;fillCameras(keepCam,preserveLoaded,beforeKey)}function fillCameras(preferredCamera,preserveLoaded=false,beforeKey=null){let keep=preferredCamera||camEl.value;let d=selectedDate();let cams=(d?.cameras?.length?d.cameras:(selectedTrial()?.cameras||opts.camera_order||['C','A','B']));camEl.innerHTML=cams.map(c=>`<option value="${c}">Camera ${c}</option>`).join('');if(keep&&cams.includes(keep))camEl.value=keep;let unchanged=preserveLoaded&&beforeKey&&beforeKey===selectionKey();if(!unchanged)clearLoadedVideo();if(!currentMeta)updateSummary();if(opts?.auto_prewarm)setTimeout(prewarmCurrentDate,200)}
let loadedGeneration=0;
function clearLoadedVideo(){if(saveTimer||saveInFlight||saveDirty){return false}loadedGeneration++;video.resetTimeline();video.pause();video.removeAttribute('src');video.load();video.style.display='none';videoControls.style.display='none';ph.style.display='grid';ethowrap.style.display='none';currentMeta=null;currentVideoName=null;annEvents=[];feedingEvents=[];undoStack=[];redoStack=[];compareData=null;intertrialData=null;annRevision=0;saveQueued=false;annotationLocked=false;recoveryCandidates=[];renderRecovery(null);document.getElementById('segments').textContent='';saveBlocked=false;return true}
function updateSummary(extra=''){let t=selectedTrial(),d=selectedDate(),quality=t?.quality?.quality==='poor'?`<span class='pill err'>⚠ poor-quality trial${t.quality.note?': '+esc(t.quality.note):''}</span>`:'',cohort=t?.cohort==='experiment_1'?`<span class='pill'>Experiment 1 · Light 04:45-22:00</span>`:t?.cohort==='experiment_2'?`<span class='pill'>Experiment 2 · Light 06:00-18:00</span>`:'';document.getElementById('summary').innerHTML=`<span class='pill'>root: ${esc(opts?.record_root)}</span><br><span class='pill'>trial: ${esc(t?.trial||'--')}</span>${cohort}${quality}<span class='pill'>date: ${esc(d?.date||'--')}</span><span class='pill'>camera: ${esc(camEl.value||'--')}</span>${extra?'<hr>'+extra:''}`}
async function prewarmCurrentDate(){try{let d=selectedDate(); if(!trialEl.value||!dateEl.value||!d) return; let q=new URLSearchParams({trial:trialEl.value,date:dateEl.value,cameras:(d.cameras||['C','A','B']).join(',')}); let r=await api('/api/prewarm?'+q.toString()); let running=Object.values(r.jobs||{}).filter(j=>j.state==='running').length; if(r.submitted.length||running) document.getElementById('status').innerHTML=`<span class='warn'>Prewarming caches in background: ${r.submitted.join(', ')||running+' running'}.</span>`}catch(e){}}
function orderedSelectedCameras(){let d=selectedDate(),cams=[...(d?.cameras||['C','A','B'])],sel=camEl.value;if(sel&&cams.includes(sel))cams=[sel,...cams.filter(c=>c!==sel)];return cams}
async function prepareSelectedDay(){try{let d=selectedDate();if(!trialEl.value||!dateEl.value||!d){document.getElementById('status').innerHTML='<span class=err>Select a trial and day first.</span>';return}let cams=orderedSelectedCameras().join(',');let q=new URLSearchParams({trial:trialEl.value,date:dateEl.value,cameras:cams,fast:'1'});document.getElementById('status').innerHTML='<span class=warn>Queueing low-priority C/A/B cache preparation, selected camera first...</span>';let r=await api('/api/prewarm?'+q.toString());let running=Object.values(r.jobs||{}).filter(j=>j.trial===trialEl.value&&j.date===dateEl.value&&(j.state==='running'||j.state==='queued')).length;document.getElementById('status').innerHTML=`<span class=ok>Cache preparation queued for ${esc(dateEl.value)}. Selected camera is first. Submitted: ${esc((r.submitted||[]).join(', ')||'none; cached or already queued')}. Active/queued: ${running}.</span>`;setTimeout(pollSelectedDayPrep,4000)}catch(e){document.getElementById('status').innerHTML='<span class=err>'+esc(e.message)+'</span>'}}
async function pollSelectedDayPrep(){try{if(!trialEl.value||!dateEl.value)return;let q=new URLSearchParams({trial:trialEl.value,date:dateEl.value});let r=await api('/api/prewarm-status?'+q.toString());let jobs=Object.values(r.jobs||{}).filter(j=>j.trial===trialEl.value&&j.date===dateEl.value);let active=jobs.filter(j=>j.state==='queued'||j.state==='running'),done=jobs.filter(j=>j.state==='cached'||j.state==='done'),err=jobs.filter(j=>j.state==='error');if(active.length||done.length||err.length){document.getElementById('status').innerHTML=`<span class=${err.length?'err':active.length?'warn':'ok'}>Cache prep ${done.length}/3 ready, ${active.length} active/queued${err.length?', '+err.length+' error(s)':''}. ${active.length?'Checking again...':'Ready for C/A/B loads.'}</span>`}if(active.length)setTimeout(pollSelectedDayPrep,8000)}catch(e){}}
async function refreshOptions(silent=false){if(!silent)document.getElementById('status').textContent='Scanning finalized recorder outputs...';let oldTrials=new Set((opts?.trials||[]).map(t=>t.trial));opts=await api('/api/options'+(silent?'':'?force=1'));fillTrials(true);let m=opts.cache_maintenance_status||{};let ms=m.submitted?.length?` Cache refresh queued: ${m.submitted.map(x=>x.date+' '+x.camera).join(', ')}.`:` Cache maintenance checked ${m.checked||0} camera-day(s); max lag ${(m.max_lag_hours??1)} h.`;let newTrials=(opts.trials||[]).filter(t=>!oldTrials.has(t.trial)).map(t=>t.trial);if(!silent||newTrials.length)document.getElementById('status').innerHTML=`<span class='ok'>Found ${opts.trials.length} trial(s).${newTrials.length?' New trial(s): '+newTrials.map(esc).join(', ')+'.':''} Automatic camera prewarm disabled; A/B/C use the same load path.${ms}</span>`;if(!silent||!currentMeta)updateSummary()}
function videoNameFromUrl(u){return decodeURIComponent(String(u||'').split('/').pop().split('?')[0])}
async function loadTimelapse(){return window.requestSelectedPlayback();}

async function loadAnnotations(){const loadKey=selectionKey(),generation=loadedGeneration;annEvents=[];feedingEvents=[];undoStack=[];redoStack=[];annRevision=0;annotationLocked=true;if(!currentVideoName)return;try{let q=new URLSearchParams({video:currentVideoName,trial:currentMeta?.trial||trialEl.value,date:currentMeta?.date||dateEl.value,camera:currentMeta?.camera||camEl.value});let data=await api('/api/annotations?'+q.toString());if(loadKey!==selectionKey()||generation!==loadedGeneration)return;annEvents=(data.events||[]).sort((a,b)=>a.time-b.time);feedingEvents=(data.feeding_events||[]).sort((a,b)=>a.time-b.time);annRevision=Number(data.revision||0);renderRecovery(data);if(data.recovery){annstatus.textContent=`Read-only recovery: canonical JSON is unreadable. ${data.recovery.error||''}`}else{annstatus.textContent=`Loaded ${annEvents.length} behavioral and ${feedingEvents.length} feeding event(s), revision ${annRevision}. JSON: ${data.json_path||'not saved yet'}`}}catch(e){if(loadKey!==selectionKey()||generation!==loadedGeneration)return;annstatus.textContent='No annotations loaded: '+e.message}}
function renderRecovery(data){let banner=document.getElementById('recoveryBanner');if(!banner)return;if(!data||!data.recovery){banner.style.display='none';banner.innerHTML='';annotationLocked=false;recoveryCandidates=[];return}annotationLocked=true;recoveryCandidates=data.recovery.candidates||[];let buttons=recoveryCandidates.map((c,i)=>`<button class='secondary' onclick='recoverAnnotation(${i})'>Restore ${esc(c.source)}${c.revision!=null?' rev '+c.revision:''} (${c.event_count||0} behavior, ${c.feeding_count||0} feeding)</button>`).join('');banner.style.display='block';banner.innerHTML=`<strong>Annotation file is unreadable.</strong> This selection is read-only until you restore a copy. Canonical was not overwritten. ${esc(data.recovery.error||'')}<div class='annbar'>${buttons||'No valid recovery copies found. Do not annotate this selection.'}</div>`}
async function recoverAnnotation(i){let c=recoveryCandidates[i];if(!c||!currentVideoName||!currentMeta)return;if(!confirm('Restore this copy into the canonical annotation file? Recovery sources will not be overwritten.'))return;try{let body={video:currentVideoName,trial:currentMeta.trial||trialEl.value,date:currentMeta.date||dateEl.value,camera:currentMeta.camera||camEl.value,source:c.source,revision:c.revision,path:c.path,meta:currentMeta};let data=await api('/api/annotations/recover',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});annEvents=(data.events||[]).sort((a,b)=>a.time-b.time);feedingEvents=(data.feeding_events||[]).sort((a,b)=>a.time-b.time);annRevision=Number(data.revision||0);renderRecovery(null);annstatus.textContent=`Restored from ${esc(c.source)} as revision ${annRevision}.`;drawEthogram();await loadCompareEthograms();drawCompareEthograms()}catch(e){annstatus.textContent='Recovery failed: '+e.message}}
function scheduleSave(){if(annotationLocked){annstatus.textContent='Read-only: restore a recovery copy before annotating.';return}saveDirty=true;if(saveTimer){clearTimeout(saveTimer);saveTimer=null}if(saveInFlight){saveQueued=true;return}saveTimer=setTimeout(()=>{saveTimer=null;saveAnnotations()},200)}
function scheduleCompareRefresh(){if(compareRefreshTimer)clearTimeout(compareRefreshTimer);compareRefreshTimer=setTimeout(()=>{compareRefreshTimer=null;refreshCompareAfterSave()},150)}
async function refreshCompareAfterSave(){if(compareRefreshInFlight){compareRefreshQueued=true;return}compareRefreshInFlight=true;try{await loadCompareEthograms();drawCompareEthograms()}finally{compareRefreshInFlight=false;if(compareRefreshQueued){compareRefreshQueued=false;scheduleCompareRefresh()}}}
// Keep the loaded selection alive until this serial drain succeeds. Never replace local edits on a save error.
async function saveAnnotations(){if(saveInFlight)return savePromise;if(!currentVideoName||!currentMeta){annstatus.textContent='Load a timelapse before saving.';return Promise.resolve(false)}if(annotationLocked||saveBlocked){annstatus.textContent='Save blocked: local edits retained. Resolve the conflict/recovery before leaving this selection.';return Promise.resolve(false)}if(saveTimer){clearTimeout(saveTimer);saveTimer=null}const saveGeneration=loadedGeneration;saveDirty=true;saveInFlight=true;savePromise=(async()=>{try{do{saveQueued=false;const body=JSON.stringify({video:currentVideoName,duration:getDuration(),events:annEvents,feeding_events:feedingEvents,meta:currentMeta,base_revision:annRevision});const data=await api('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body});if(saveGeneration!==loadedGeneration)return false;annRevision=Number(data.revision||annRevision);saveDirty=saveQueued;}while(saveQueued);annstatus.textContent=`Saved revision ${annRevision}. All behavioral and feeding edits saved.`;drawEthogram();scheduleCompareRefresh();return true;}catch(e){if(saveGeneration!==loadedGeneration)return false;saveDirty=true;if(e.status===409||e.status===423){saveBlocked=true;annotationLocked=true;}annstatus.textContent=(e.status===409?'Revision conflict':e.status===423?'Recovery lock':'Save failed')+': '+e.message+'. Local edits retained; selection switch blocked. '+(saveBlocked?'Resolve without discarding these edits before reloading.':'Use Save or Load / retry to retry.');return false;}finally{if(saveGeneration===loadedGeneration){saveInFlight=false;savePromise=null;}}})();return savePromise;}
function getDuration(){return video.duration&&isFinite(video.duration)?video.duration:(currentMeta?.play_duration_seconds||currentMeta?.target_duration_seconds||currentMeta?.coverage_hours||0)}
function updateActiveIndicator(){let b=BEHBY[stateAt(video.currentTime||0)],f=FEEDBY[feedingStateAt(video.currentTime||0)];activeSwatch.style.background=b?b.color:'#64748b';activeBehavior.textContent=b?'current: '+b.label:'current: unannotated';feedingActiveSwatch.style.background=f?f.color:'#64748b';feedingActiveState.textContent=f?'current: '+f.label:'current: unannotated'}
function updateTimeUI(){let dur=Math.max(getDuration(),0.001),t=Math.max(0,Math.min(video.currentTime||0,dur));scrubber.max=dur.toFixed(4);scrubber.step=(window.event&&window.event.shiftKey)?'0.01':'0.05';scrubber.value=t.toFixed(4);timecode.textContent=`${t.toFixed(2)} / ${dur.toFixed(2)} s (${formatClockFromPlayback(t)})${video.waitingForCache?' · Waiting for cached footage':''}`;updateActiveIndicator()}
function clampTime(t){return Math.max(0,Math.min(getDuration(),t))}
function stepVideo(delta){video.currentTime=clampTime((video.currentTime||0)+delta);drawEthogram()}
function toggleVideoPlayback(){if(!currentVideoName)return;if(video.paused||video.ended){video.play().catch(e=>{annstatus.textContent='Play failed: '+e.message})}else{video.pause()}drawEthogram()}
function pad2(n){return String(n).padStart(2,'0')}
function metaPieces(){return currentMeta?.pieces||[]}
function clipStartTs(){return currentMeta?.clip_start_ts??null}
function gapPlayIntervals(){return (currentMeta?.gaps||[]).map(g=>[Number(g.play_start)||0,(Number(g.play_start)||0)+(Number(g.play_dur)||0)])}
function inGapAtPlay(t){for(const [a,b] of gapPlayIntervals())if(t>=a-1e-9&&t<=b+1e-9)return true;return false}
// Map a playback-second offset to a real wall-clock unix timestamp (seconds),
// walking the piece timeline so recording gaps are honored.
function playToWallTs(t){let pieces=metaPieces(),cs=clipStartTs();if(cs==null)return null;if(!pieces.length)return cs+t*3600;t=Math.max(0,t);for(const p of pieces){let p0=p.play_start,p1=p0+p.play_dur;if(p.play_dur<=0)continue;if(t<=p1+1e-9){let frac=Math.min(1,Math.max(0,(t-p0)/p.play_dur));if(p.wall_start_ts==null||p.wall_end_ts==null)return cs+t*3600;return p.wall_start_ts+frac*(p.wall_end_ts-p.wall_start_ts)}}let last=pieces[pieces.length-1];return (last&&last.wall_end_ts!=null)?last.wall_end_ts:cs+t*3600}
function wallTsToPlay(ts){let pieces=metaPieces();if(ts==null||!pieces.length)return null;for(const p of pieces){let ws=p.wall_start_ts,we=p.wall_end_ts;if(ws==null||we==null||we<=ws)continue;if(ts<ws)return p.play_start;if(ts<=we){let frac=(ts-ws)/(we-ws);return p.play_start+frac*p.play_dur}}return null}
function formatClockFromPlayback(t){let ts=playToWallTs(t);if(ts==null){let total=Math.max(0,t*3600),hh=Math.floor(total/3600)%24,mm=Math.floor((total%3600)/60),ss=Math.floor(total%60);return `${pad2(hh)}:${pad2(mm)}:${pad2(ss)}`}let d=new Date(ts*1000);return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`}
function dayMidnightTs(){let cs=clipStartTs();if(cs==null)return null;let b=new Date(cs*1000);return new Date(b.getFullYear(),b.getMonth(),b.getDate(),0,0,0,0).getTime()/1000}
function currentLightRegime(){return compareData?.light_dark||selectedTrial()?.light_regime||{light_start_hour:4.75,light_end_hour:22,light_label:'Light 04:45-22:00',dark_label:'Dark 22:00-04:45'}}
function directLightDarkMarks(){let cs=clipStartTs();if(cs==null)return [];let dur=getDuration(),mid=dayMidnightTs(),endTs=playToWallTs(dur)??(cs+dur*3600),marks=[],r=currentLightRegime();for(let day=0;day<=Math.ceil((endTs-mid)/86400)+1;day++){for(const m of [{h:Number(r.light_start_hour),c:'#f97316',lab:'Light '+String(r.light_label||'').replace(/^Light /,'').split('-')[0]},{h:Number(r.light_end_hour),c:'#8b5cf6',lab:'Dark '+String(r.dark_label||'').replace(/^Dark /,'').split('-')[0]}]){let markTs=mid+day*86400+m.h*3600;let t=wallTsToPlay(markTs);if(t!=null&&t>=0&&t<=dur)marks.push({...m,t})}}return marks}
function parseClockInput(v){let s=String(v||'').trim();let m=s.match(/^(\\d{1,2}):(\\d{2})(?::(\\d{2}))?$/);let h,min,sec=0;if(m){h=+m[1];min=+m[2];sec=+(m[3]||0)}else{let digits=s.replace(/\\D/g,'');if(digits.length===3){h=+digits.slice(0,1);min=+digits.slice(1,3)}else if(digits.length===4){h=+digits.slice(0,2);min=+digits.slice(2,4)}else if(digits.length===6){h=+digits.slice(0,2);min=+digits.slice(2,4);sec=+digits.slice(4,6)}else return null}if(h<0||h>23||min<0||min>59||sec<0||sec>59)return null;return (h*3600+min*60+sec)/3600}
function seekClockTime(){let hourFrac=parseClockInput(clockSeek.value);if(hourFrac===null){clockStatus.textContent='Invalid time. Use HHMM or HH:MM[:SS].';return}let dur=getDuration(),cs=clipStartTs(),mid=dayMidnightTs();if(cs==null||mid==null){video.currentTime=clampTime(hourFrac);drawEthogram();return}let targetTs=mid+hourFrac*3600;let t=wallTsToPlay(targetTs),note='';if(t==null){if(targetTs<=cs){t=0;note=' (before available footage)'}else{t=dur;note=' (after available footage)'}}else if(inGapAtPlay(t)){note=' (inside a recording gap - no video)'}video.currentTime=clampTime(t);clockStatus.textContent=`${dateEl.value} ${formatClockFromPlayback(video.currentTime)} → video ${video.currentTime.toFixed(3)} s${note}`;drawEthogram()}
function cloneAnnotationState(){return {behavior:annEvents.map(ev=>({time:ev.time,behavior:ev.behavior})),feeding:feedingEvents.map(ev=>({time:ev.time,state:ev.state}))}}
function pushUndo(){undoStack.push(cloneAnnotationState());if(undoStack.length>200)undoStack.shift();redoStack=[]}
function restoreAnnotationState(snapshot,label){if(annotationLocked){annstatus.textContent='Read-only: restore a recovery copy before annotating.';return}annEvents=(snapshot?.behavior||[]).map(ev=>({time:ev.time,behavior:ev.behavior})).sort((a,b)=>a.time-b.time);feedingEvents=(snapshot?.feeding||[]).map(ev=>({time:ev.time,state:ev.state})).sort((a,b)=>a.time-b.time);annstatus.textContent=label;drawEthogram();scheduleSave()}
function undoAnnotation(){if(!currentVideoName||!undoStack.length){annstatus.textContent='Nothing to undo.';return}redoStack.push(cloneAnnotationState());restoreAnnotationState(undoStack.pop(),'Undo applied to behavioral/feeding annotations. Auto-saving...')}
function redoAnnotation(){if(!currentVideoName||!redoStack.length){annstatus.textContent='Nothing to redo.';return}undoStack.push(cloneAnnotationState());restoreAnnotationState(redoStack.pop(),'Redo applied to behavioral/feeding annotations. Auto-saving...')}
function addAnnotation(behavior){if(!currentVideoName||annotationLocked||video.waitingForCache)return;pushUndo();let dur=getDuration();let t=Math.max(0,Math.min(video.currentTime||0,dur||999999)),rt=+t.toFixed(4);annEvents=annEvents.filter(ev=>Math.abs(ev.time-rt)>0.0001);annEvents.push({time:rt,behavior});annEvents.sort((a,b)=>a.time-b.time);annEvents=annEvents.filter((ev,i,a)=>i===0||!(ev.behavior===a[i-1].behavior&&Math.abs(ev.time-a[i-1].time)<0.0001));annstatus.textContent=`${BEHBY[behavior].label} at ${t.toFixed(2)} s. Auto-saving...`;drawEthogram();scheduleSave()}
function addFeedingAnnotation(state){if(!currentVideoName||annotationLocked||video.waitingForCache)return;pushUndo();let dur=getDuration();let t=Math.max(0,Math.min(video.currentTime||0,dur||999999)),rt=+t.toFixed(4);feedingEvents=feedingEvents.filter(ev=>Math.abs(ev.time-rt)>0.0001);feedingEvents.push({time:rt,state});feedingEvents.sort((a,b)=>a.time-b.time);feedingEvents=feedingEvents.filter((ev,i,a)=>i===0||!(ev.state===a[i-1].state&&Math.abs(ev.time-a[i-1].time)<0.0001));annstatus.textContent=`${FEEDBY[state].label} at ${t.toFixed(2)} s. Auto-saving...`;drawEthogram();scheduleSave()}
function clearAnnotations(){if(annotationLocked){annstatus.textContent='Read-only: restore a recovery copy before annotating.';return}if(!confirm('Clear behavioral and feeding events for the loaded video and overwrite sidecars on next save?'))return;pushUndo();annEvents=[];feedingEvents=[];drawEthogram();scheduleSave()}
function resizeCanvas(){let r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(r.width*dpr)||canvas.height!==Math.round(r.height*dpr)){canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0)}}
function resizeFeedingCanvas(){let r=feedingCanvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;if(feedingCanvas.width!==Math.round(r.width*dpr)||feedingCanvas.height!==Math.round(r.height*dpr)){feedingCanvas.width=Math.round(r.width*dpr);feedingCanvas.height=Math.round(r.height*dpr);feedingCtx.setTransform(dpr,0,0,dpr,0,0)}}
function stateAt(t){let cur=null;for(const ev of annEvents){if(ev.time<=t)cur=ev.behavior;else break}return cur}
function feedingStateAt(t){let cur=null;for(const ev of feedingEvents){if(ev.time<=t)cur=ev.state;else break}return cur}
function resizeCanvasEl(c,context){let r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;if(c.width!==Math.round(r.width*dpr)||c.height!==Math.round(r.height*dpr)){c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);context.setTransform(dpr,0,0,dpr,0,0)}}
async function loadCompareEthograms(){compareData=null;if(!trialEl.value)return;try{let params={trial:trialEl.value};if(['all_first_72h','all_post_observation'].includes(dateEl.value))params.date=dateEl.value;compareData=await api('/api/ethogram-compare?'+new URLSearchParams(params).toString());let range=`${compareData.first_date||'--'} through ${compareData.last_date||'--'} (${(compareData.axis_hours||0).toFixed(1)} h axis)`;feedingCompareNote.textContent=`Trial ${compareData.trial}: ${range}. Saved eating / not-eating annotations shown for C/A/B.`;compareNote.textContent=`Trial ${compareData.trial}: ${range}. Saved behavioral-state annotations shown for C/A/B.`;}catch(e){feedingCompareNote.textContent='Feeding comparison unavailable: '+e.message;compareNote.textContent='Behavioral comparison unavailable: '+e.message}}
function cameraDisplayLabel(cam){let group=compareData?.camera_groups?.[cam]||({C:'nonstress',A:'control',B:'stress'}[cam]||'');return `Camera ${cam}${group?' - '+group:''}`}
function buildCompareCompressedAxis(){
    if(!compareData||!compareData.axis_hours)return null;
    let raw=[];
    for(const row of (compareData.rows||[])){
        for(const d of (row.dates||[])){
            let events=[...(d.events||[])].sort((a,b)=>a.time-b.time);
            if(!events.length)continue;
            let base=(Number(d.offset_hours)||0)+(Number(d.clip_start_hour)||0);
            let dur=Number(d.duration_hours)||0;
            let first=Math.max(0,Math.min(dur,Number(events[0].time)||0));
            if(dur>first)raw.push([base+first,base+dur]);
        }
    }
    for(const m of compareLightDarkMarks()){raw.push([Math.max(0,m.abs-0.015),Math.min(compareData.axis_hours||m.abs+0.015,m.abs+0.015)])}
    raw.sort((a,b)=>a[0]-b[0]);
    let merged=[];
    for(const seg of raw){
        if(!merged.length||seg[0]>merged[merged.length-1][1]+1e-6)merged.push([...seg]);
        else merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],seg[1]);
    }
    let total=0,segs=[];
    for(const seg of merged){let len=Math.max(0,seg[1]-seg[0]);if(len<=0)continue;segs.push({start:seg[0],end:seg[1],compressedStart:total,len});total+=len}
    if(!segs.length)return null;
    return {segs,total};
}
function cmpX(abs,axis,l,plotW){
    if(!axis||!axis.segs)return l+(abs/(compareData.axis_hours||1))*plotW;
    for(const seg of axis.segs){
        if(abs>=seg.start&&abs<=seg.end){return l+((seg.compressedStart+(abs-seg.start))/axis.total)*plotW}
    }
    return null;
}
function compareLightDarkMarks(){let marks=[],hours=Math.max(compareData?.axis_hours||0,(compareData?.dates?.length||0)*24),r=compareData?.light_dark||currentLightRegime();for(let base=0;base<=hours;base+=24){for(const m of [{h:Number(r.light_start_hour),color:'#f97316',label:'L '+String(r.light_label||'').replace(/^Light /,'').split('-')[0]},{h:Number(r.light_end_hour),color:'#8b5cf6',label:'D '+String(r.dark_label||'').replace(/^Dark /,'').split('-')[0]}]){let abs=base+m.h;if(abs<=hours)marks.push({...m,abs})}}return marks}
function buildCompareFeedingCompressedAxis(){if(!compareData||!compareData.axis_hours)return null;let raw=[];for(const row of (compareData.rows||[]))for(const d of (row.dates||[])){let events=[...(d.feeding_events||[])].sort((a,b)=>a.time-b.time);if(!events.length)continue;let base=(Number(d.offset_hours)||0)+(Number(d.clip_start_hour)||0),dur=Number(d.duration_hours)||0,first=Math.max(0,Math.min(dur,Number(events[0].time)||0));if(dur>first)raw.push([base+first,base+dur])}raw.sort((a,b)=>a[0]-b[0]);let merged=[];for(const seg of raw){if(!merged.length||seg[0]>merged[merged.length-1][1]+1e-6)merged.push([...seg]);else merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],seg[1])}let total=0,segs=[];for(const seg of merged){let len=Math.max(0,seg[1]-seg[0]);if(len<=0)continue;segs.push({start:seg[0],end:seg[1],compressedStart:total,len});total+=len}return segs.length?{segs,total}:null}
function drawFeedingCompareEthograms(){if(!feedingCompareCanvas||ethowrap.style.display==='none')return;resizeCanvasEl(feedingCompareCanvas,feedingCompareCtx);let c=feedingCompareCanvas,ctx2=feedingCompareCtx,w=c.clientWidth,h=c.clientHeight,l=116,r=12,top=34,b=34,plotW=w-l-r,plotH=h-top-b;ctx2.clearRect(0,0,w,h);ctx2.fillStyle='#05070a';ctx2.fillRect(0,0,w,h);ctx2.font='12px system-ui';if(!compareData||!compareData.axis_hours){ctx2.fillStyle='#9aa7b7';ctx2.fillText('No feeding comparison data loaded.',l,top+20);return}let axisObj=buildCompareFeedingCompressedAxis();if(!axisObj){ctx2.fillStyle='#9aa7b7';ctx2.fillText('No saved feeding annotations for C/A/B yet.',l,top+20);return}let rows=compareData.rows||[],days=compareData.dates||[],rowH=plotH/3;ctx2.fillStyle='#9aa7b7';ctx2.fillText(`compressed to feeding-annotated spans: ${axisObj.total.toFixed(2)} h shown from ${(compareData.axis_hours||0).toFixed(1)} h trial axis`,l+4,top-20);ctx2.fillStyle=FEEDBY.eating.color;ctx2.fillText('eating',l+4,top-7);ctx2.fillStyle=FEEDBY.not_eating.color;ctx2.fillText('not eating',l+70,top-7);for(let ri=0;ri<3;ri++){let row=rows[ri]||{camera:['C','A','B'][ri],dates:[]},y=top+ri*rowH;ctx2.fillStyle='rgba(255,255,255,.03)';ctx2.fillRect(l,y,plotW,rowH);ctx2.strokeStyle='#2d3748';ctx2.strokeRect(l,y,plotW,rowH);ctx2.fillStyle='#e5edf6';ctx2.fillText(cameraDisplayLabel(row.camera),8,y+rowH/2+4);for(const d of row.dates||[]){let base=(Number(d.offset_hours)||0)+(Number(d.clip_start_hour)||0),dur=Number(d.duration_hours)||0,events=[...(d.feeding_events||[])].sort((a,b)=>a.time-b.time);for(let i=0;i<events.length;i++){let ev=events[i],state=FEEDBY[ev.state];if(!state)continue;let t0=base+Number(ev.time||0),t1=base+(i+1<events.length?Number(events[i+1].time||dur):dur);for(const seg of axisObj.segs){let a=Math.max(t0,seg.start),bb=Math.min(t1,seg.end);if(bb<=a)continue;let x0=cmpX(a,axisObj,l,plotW),x1=cmpX(bb,axisObj,l,plotW);if(x0==null||x1==null)continue;ctx2.fillStyle=state.color;ctx2.globalAlpha=.84;ctx2.fillRect(x0,y+5,Math.max(1,x1-x0),rowH-10);ctx2.globalAlpha=1}}for(const g of (d.gaps||[])){let ga=base+(Number(g.play_start)||0),gb=ga+(Number(g.play_dur)||0);for(const seg of axisObj.segs){let a=Math.max(ga,seg.start),bb=Math.min(gb,seg.end);if(bb<=a)continue;let x0=cmpX(a,axisObj,l,plotW),x1=cmpX(bb,axisObj,l,plotW);if(x0==null||x1==null)continue;ctx2.fillStyle='rgba(8,10,14,.7)';ctx2.fillRect(x0,y+5,Math.max(2,x1-x0),rowH-10);ctx2.strokeStyle='#f59e0b';ctx2.strokeRect(x0+.5,y+5.5,Math.max(2,x1-x0-1),rowH-11)}}}}ctx2.strokeStyle='#2d3748';ctx2.fillStyle='#9aa7b7';for(const seg of axisObj.segs){let x=cmpX(seg.start,axisObj,l,plotW);if(x==null)continue;let dayIdx=Math.floor(seg.start/24),hour=seg.start-dayIdx*24;ctx2.beginPath();ctx2.moveTo(x,top);ctx2.lineTo(x,top+plotH);ctx2.stroke();ctx2.fillText(`${days[dayIdx]?.slice(5)||''} ${String(Math.floor(hour)).padStart(2,'0')}:${String(Math.round((hour%1)*60)).padStart(2,'0')}`,x+3,top+plotH+18)}if(currentMeta&&video.duration){let dIdx=days.indexOf(currentMeta.date);if(dIdx>=0){let absolute=dIdx*24+(Number(video.currentTime)||0),x=cmpX(absolute,axisObj,l,plotW);if(x!=null){ctx2.strokeStyle='white';ctx2.lineWidth=2;ctx2.beginPath();ctx2.moveTo(x,top);ctx2.lineTo(x,top+plotH);ctx2.stroke()}}}}
function drawCompareEthograms(){
    drawFeedingCompareEthograms();
    renderCompareAnalytics();
    if(!compareCanvas||ethowrap.style.display==='none')return;
    resizeCanvasEl(compareCanvas,compareCtx);
    let c=compareCanvas,ctx2=compareCtx,w=c.clientWidth,h=c.clientHeight,l=116,r=12,top=34,b=34,plotW=w-l-r,plotH=h-top-b;
    ctx2.clearRect(0,0,w,h);ctx2.fillStyle='#05070a';ctx2.fillRect(0,0,w,h);ctx2.font='12px system-ui';
    if(!compareData||!compareData.axis_hours){ctx2.fillStyle='#9aa7b7';ctx2.fillText('No comparison data loaded.',l,top+20);return}
    let axisObj=buildCompareCompressedAxis();
    if(!axisObj){ctx2.fillStyle='#9aa7b7';ctx2.fillText('No annotated comparison spans yet.',l,top+20);return}
    let rowH=plotH/3,rows=compareData.rows||[],days=compareData.dates||[],regime=compareData.light_dark||currentLightRegime();
    ctx2.fillStyle='#9aa7b7';ctx2.fillText(`compressed to annotated spans: ${axisObj.total.toFixed(2)} h shown from ${(compareData.axis_hours||0).toFixed(1)} h trial axis`,l+4,top-20);
    ctx2.fillStyle='#f97316';ctx2.fillText(`orange tick = ${regime.light_label}`,l+4,top-8);
    ctx2.fillStyle='#a78bfa';ctx2.fillText(`violet tick = ${regime.dark_label}`,l+250,top-8);
    for(const m of compareLightDarkMarks()){let x=cmpX(m.abs,axisObj,l,plotW);if(x==null)continue;ctx2.strokeStyle=m.color;ctx2.lineWidth=3;ctx2.beginPath();ctx2.moveTo(x,top-16);ctx2.lineTo(x,top-1);ctx2.stroke();ctx2.fillStyle=m.color;ctx2.fillText(m.label,Math.min(w-62,x+4),top-5);ctx2.strokeStyle=m.color;ctx2.globalAlpha=.18;ctx2.lineWidth=1;ctx2.beginPath();ctx2.moveTo(x,top);ctx2.lineTo(x,top+plotH);ctx2.stroke();ctx2.globalAlpha=1}
    for(let ri=0;ri<3;ri++){
        let row=rows[ri]||{camera:['C','A','B'][ri],dates:[]},y=top+ri*rowH;
        ctx2.fillStyle='rgba(255,255,255,.03)';ctx2.fillRect(l,y,plotW,rowH);ctx2.strokeStyle='#2d3748';ctx2.strokeRect(l,y,plotW,rowH);ctx2.fillStyle='#e5edf6';ctx2.fillText(cameraDisplayLabel(row.camera),8,y+rowH/2+4);
        for(const d of row.dates||[]){
            let base=(Number(d.offset_hours)||0)+(Number(d.clip_start_hour)||0),dur=Number(d.duration_hours)||0,events=[...(d.events||[])].sort((a,b)=>a.time-b.time);
            for(let i=0;i<events.length;i++){
                let ev=events[i],beh=BEHBY[ev.behavior];if(!beh)continue;
                let t0=base+Number(ev.time||0),t1=base+(i+1<events.length?Number(events[i+1].time||dur):dur);
                for(const seg of axisObj.segs){
                    let a=Math.max(t0,seg.start),bb=Math.min(t1,seg.end);if(bb<=a)continue;
                    let x0=cmpX(a,axisObj,l,plotW),x1=cmpX(bb,axisObj,l,plotW);if(x0==null||x1==null)continue;
                    ctx2.fillStyle=beh.color;ctx2.globalAlpha=.82;ctx2.fillRect(x0,y+5,Math.max(1,x1-x0),rowH-10);ctx2.globalAlpha=1;
                }
            }
            for(const g of (d.gaps||[])){let ga=base+(Number(g.play_start)||0),gb=ga+(Number(g.play_dur)||0);for(const seg of axisObj.segs){let a=Math.max(ga,seg.start),bb=Math.min(gb,seg.end);if(bb<=a)continue;let x0=cmpX(a,axisObj,l,plotW),x1=cmpX(bb,axisObj,l,plotW);if(x0==null||x1==null)continue;ctx2.save();ctx2.fillStyle='rgba(8,10,14,.7)';ctx2.fillRect(x0,y+5,Math.max(2,x1-x0),rowH-10);ctx2.strokeStyle='#f59e0b';ctx2.lineWidth=1;ctx2.strokeRect(x0+.5,y+5.5,Math.max(2,x1-x0-1),rowH-11);ctx2.restore()}}
        }
    }
    ctx2.strokeStyle='#2d3748';ctx2.fillStyle='#9aa7b7';
    for(const seg of axisObj.segs){
        for(let d=0;d<days.length;d++){let boundary=d*24;if(boundary<seg.start||boundary>seg.end)continue;let x=cmpX(boundary,axisObj,l,plotW);if(x==null)continue;ctx2.beginPath();ctx2.moveTo(x,top);ctx2.lineTo(x,top+plotH);ctx2.stroke();ctx2.fillText(days[d]?.slice(5)||'',x+4,top+plotH+22)}
    }
    // Also label each compressed span start so gaps are explicit.
    for(const seg of axisObj.segs){let x=cmpX(seg.start,axisObj,l,plotW);if(x==null)continue;let dayIdx=Math.floor(seg.start/24),hour=seg.start-dayIdx*24;ctx2.fillStyle='#9aa7b7';ctx2.fillText(`${days[dayIdx]?.slice(5)||''} ${String(Math.floor(hour)).padStart(2,'0')}:${String(Math.round((hour%1)*60)).padStart(2,'0')}`,x+3,top+plotH+10)}
    if(currentMeta&&video.duration){let dIdx=days.indexOf(currentMeta.date);if(dIdx>=0){let absolute=dIdx*24+(Number(video.currentTime)||0);let x=cmpX(absolute,axisObj,l,plotW);if(x!=null){ctx2.strokeStyle='white';ctx2.lineWidth=2;ctx2.beginPath();ctx2.moveTo(x,top);ctx2.lineTo(x,top+plotH);ctx2.stroke();ctx2.fillStyle='white';ctx2.fillText(`${currentMeta.camera} ${formatClockFromPlayback(video.currentTime||0)}`,x+4,top+16)}}}
}
function analyticsPercentCells(a){let s=a.states||{},light=a.periods?.light?.states||{},dark=a.periods?.dark?.states||{},fmt=x=>(Number(x||0)).toFixed(1)+'%';return `<td>${fmt(s.moving?.pct)}</td><td>${fmt(s.SA?.pct)}</td><td>${fmt(s.quiescent?.pct)}</td><td>${fmt(light.moving?.pct)}</td><td>${fmt(light.SA?.pct)}</td><td>${fmt(light.quiescent?.pct)}</td><td>${fmt(dark.moving?.pct)}</td><td>${fmt(dark.SA?.pct)}</td><td>${fmt(dark.quiescent?.pct)}</td>`}
function analyticsHeader(firstCols){let r=compareData?.light_dark||currentLightRegime();return `<thead><tr>${firstCols}<th colspan="3">All recorded</th><th colspan="3">${esc(r.light_label)}</th><th colspan="3">${esc(r.dark_label)}</th></tr><tr><th>moving</th><th>SA</th><th>quiescent</th><th>moving</th><th>SA</th><th>quiescent</th><th>moving</th><th>SA</th><th>quiescent</th></tr></thead>`}
function renderCompareAnalytics(){if(!compareAnalytics)return;if(!compareData||!compareData.analytics){compareAnalytics.innerHTML='';return}let overallRows=compareData.analytics.map(a=>`<tr><td>${esc(cameraDisplayLabel(a.camera))}</td>${analyticsPercentCells(a)}</tr>`).join('');let overall=`<table><caption>Averages within the selected trial by camera</caption>${analyticsHeader('<th rowspan="2">Camera / group</th>')}<tbody>${overallRows}</tbody></table>`;compareAnalytics.innerHTML=overall}
async function loadIntertrialEthograms(){intertrialData=null;try{intertrialData=await api('/api/ethogram-intertrial');populateIntertrialControls();let n=intertrialData.trials?.length||0;intertrialNote.textContent=`Same-camera comparison across ${n} trial(s). Use the selector for all trials or any two selected trials. Each camera is drawn in its own panel.`;}catch(e){intertrialNote.textContent='Intertrial comparison unavailable: '+e.message}}
function intertrialCameraLabel(cam){let group=intertrialData?.camera_groups?.[cam]||({C:'nonstress',A:'control',B:'stress'}[cam]||'');return `Camera ${cam}${group?' - '+group:''}`}
function populateIntertrialControls(){let trials=intertrialData?.trials||[];let optsHtml=trials.map(t=>`<option value="${esc(t.trial)}">${esc(t.label||t.trial)}</option>`).join('');let oldA=intertrialTrialA?.value,oldB=intertrialTrialB?.value;if(intertrialTrialA)intertrialTrialA.innerHTML=optsHtml;if(intertrialTrialB)intertrialTrialB.innerHTML=optsHtml;if(oldA&&[...intertrialTrialA.options].some(o=>o.value===oldA))intertrialTrialA.value=oldA;if(oldB&&[...intertrialTrialB.options].some(o=>o.value===oldB))intertrialTrialB.value=oldB;else if(intertrialTrialB&&intertrialTrialB.options.length>1)intertrialTrialB.selectedIndex=1;updateIntertrialControlState()}
function updateIntertrialControlState(){let pair=intertrialMode?.value==='pair';if(intertrialTrialA)intertrialTrialA.disabled=!pair;if(intertrialTrialB)intertrialTrialB.disabled=!pair}
function selectedIntertrialIds(){let ids=(intertrialData?.trials||[]).map(t=>t.trial);if(intertrialMode?.value==='pair'){let a=intertrialTrialA?.value,b=intertrialTrialB?.value;let out=[a,b].filter(Boolean);return [...new Set(out)]}return ids}
function intertrialRowsForView(){let ids=new Set(selectedIntertrialIds()),rows=[];for(const camRow of (intertrialData?.rows||[])){let selected=(camRow.trials||[]).filter(t=>ids.has(t.trial));for(const cohort of ['experiment_1','experiment_2']){let trials=selected.filter(t=>t.cohort===cohort);if(trials.length)rows.push({camera:camRow.camera,group:camRow.group,cohort,cohortLabel:cohort==='experiment_1'?'Experiment 1 (Trials 1–18 · Light 04:45–22:00)':'Experiment 2 (Trials 21+ · Light 06:00–18:00)',trials})}}return rows}
function intertrialAnalyticsForView(){let ids=new Set(selectedIntertrialIds());return (intertrialData?.analytics||[]).filter(a=>ids.has(a.trial))}
function intertrialAxisHoursForRows(rows){return Math.max(...rows.flatMap(r=>(r.trials||[]).map(t=>Number(t.axis_hours)||0)),0,48)}
function buildIntertrialCompressedAxis(rows){
    rows=rows||intertrialRowsForView();let axisHours=intertrialAxisHoursForRows(rows);if(!intertrialData||!axisHours)return null;
    let raw=[];
    for(const camRow of rows)for(const tr of (camRow.trials||[]))for(const d of (tr.dates||[])){
        let events=[...(d.events||[])].sort((a,b)=>a.time-b.time);if(!events.length)continue;
        let base=(Number(d.offset_hours)||0)+(Number(d.clip_start_hour)||0),dur=Number(d.duration_hours)||0;
        let first=Math.max(0,Math.min(dur,Number(events[0].time)||0));if(dur>first)raw.push([base+first,base+dur]);
    }
    for(const cohort of [...new Set(rows.map(r=>r.cohort))])for(const m of intertrialLightDarkMarks(axisHours,cohort)){raw.push([Math.max(0,m.abs-0.015),Math.min(axisHours,m.abs+0.015)])}
    raw.sort((a,b)=>a[0]-b[0]);let merged=[];
    for(const seg of raw){if(!merged.length||seg[0]>merged[merged.length-1][1]+1e-6)merged.push([...seg]);else merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],seg[1]);}
    let total=0,segs=[];for(const seg of merged){let len=Math.max(0,seg[1]-seg[0]);if(len<=0)continue;segs.push({start:seg[0],end:seg[1],compressedStart:total,len});total+=len}
    return segs.length?{segs,total,axisHours}:null;
}
function intertrialX(abs,axis,l,plotW){if(!axis||!axis.segs)return l+(abs/(axis?.axisHours||intertrialData?.axis_hours||1))*plotW;for(const seg of axis.segs){if(abs>=seg.start&&abs<=seg.end)return l+((seg.compressedStart+(abs-seg.start))/axis.total)*plotW}return null}
function intertrialRegime(cohort){return intertrialData?.light_dark_by_cohort?.[cohort]||(cohort==='experiment_2'?{light_start_hour:6,light_end_hour:18,light_label:'Light 06:00-18:00',dark_label:'Dark 18:00-06:00'}:{light_start_hour:4.75,light_end_hour:22,light_label:'Light 04:45-22:00',dark_label:'Dark 22:00-04:45'})}
function intertrialLightDarkMarks(axisHours,cohort){let marks=[],hours=Math.max(axisHours||intertrialData?.axis_hours||0,48),r=intertrialRegime(cohort);for(let base=0;base<=hours;base+=24){for(const m of [{h:Number(r.light_start_hour),color:'#f97316',label:'L '+String(r.light_label).replace(/^Light /,'').split('-')[0]},{h:Number(r.light_end_hour),color:'#8b5cf6',label:'D '+String(r.dark_label).replace(/^Dark /,'').split('-')[0]}]){let abs=base+m.h;if(abs<=hours)marks.push({...m,abs})}}return marks}
function intertrialRowsFlat(){let out=[];for(const camRow of intertrialRowsForView()){for(const tr of (camRow.trials||[])){out.push({camera:camRow.camera,group:camRow.group,cohort:camRow.cohort,trial:tr.trial,label:tr.label,dates:tr.dates||[],axis_hours:tr.axis_hours,error:tr.error})}}return out}
function drawOneIntertrialCamera(canvas,camRow,axisObj){let ctx3=canvas.getContext('2d');resizeCanvasEl(canvas,ctx3);let w=canvas.clientWidth,h=canvas.clientHeight,l=230,r=12,top=34,b=34,plotW=w-l-r,plotH=h-top-b;ctx3.clearRect(0,0,w,h);ctx3.fillStyle='#05070a';ctx3.fillRect(0,0,w,h);ctx3.font='12px system-ui';let trials=camRow.trials||[];if(!axisObj){ctx3.fillStyle='#9aa7b7';ctx3.fillText('No saved annotation spans for this camera/view yet.',l,top+20);return}ctx3.fillStyle='#9aa7b7';ctx3.fillText(`compressed to annotated spans + light/dark ticks: ${axisObj.total.toFixed(2)} h shown from ${axisObj.axisHours.toFixed(1)} h trial axis`,l+4,top-10);for(const m of intertrialLightDarkMarks(axisObj.axisHours,camRow.cohort)){let x=intertrialX(m.abs,axisObj,l,plotW);if(x==null)continue;ctx3.strokeStyle=m.color;ctx3.lineWidth=3;ctx3.beginPath();ctx3.moveTo(x,top-16);ctx3.lineTo(x,top-1);ctx3.stroke();ctx3.fillStyle=m.color;ctx3.fillText(m.label,Math.min(w-62,x+4),top-5);ctx3.strokeStyle=m.color;ctx3.globalAlpha=.18;ctx3.lineWidth=1;ctx3.beginPath();ctx3.moveTo(x,top);ctx3.lineTo(x,top+plotH);ctx3.stroke();ctx3.globalAlpha=1}let rowH=plotH/Math.max(1,trials.length);trials.forEach((tr,ri)=>{let y=top+ri*rowH;ctx3.fillStyle=ri%2?'rgba(255,255,255,.045)':'rgba(255,255,255,.025)';ctx3.fillRect(l,y,plotW,rowH);ctx3.strokeStyle='#2d3748';ctx3.strokeRect(l,y,plotW,rowH);ctx3.fillStyle='#e5edf6';ctx3.fillText(`${tr.label||tr.trial}`,8,y+rowH/2+4);for(const d of tr.dates||[]){let base=(Number(d.offset_hours)||0)+(Number(d.clip_start_hour)||0),dur=Number(d.duration_hours)||0,events=[...(d.events||[])].sort((a,b)=>a.time-b.time);for(let i=0;i<events.length;i++){let ev=events[i],beh=BEHBY[ev.behavior];if(!beh)continue;let t0=base+Number(ev.time||0),t1=base+(i+1<events.length?Number(events[i+1].time||dur):dur);for(const seg of axisObj.segs){let a=Math.max(t0,seg.start),bb=Math.min(t1,seg.end);if(bb<=a)continue;let x0=intertrialX(a,axisObj,l,plotW),x1=intertrialX(bb,axisObj,l,plotW);if(x0==null||x1==null)continue;ctx3.fillStyle=beh.color;ctx3.globalAlpha=.82;ctx3.fillRect(x0,y+4,Math.max(1,x1-x0),rowH-8);ctx3.globalAlpha=1}}}});ctx3.fillStyle='#9aa7b7';for(const seg of axisObj.segs){let x=intertrialX(seg.start,axisObj,l,plotW);if(x==null)continue;ctx3.fillText(`${seg.start.toFixed(1)} h`,x+3,top+plotH+18)}}
function drawIntertrialEthograms(){
    updateIntertrialControlState();renderIntertrialAnalytics();
    if(!intertrialCanvases||ethowrap.style.display==='none')return;
    intertrialCanvases.innerHTML='';
    if(!intertrialData||!intertrialData.axis_hours){intertrialCanvases.innerHTML='<div class="compare-note">No intertrial comparison data loaded.</div>';return}
    let rows=intertrialRowsForView();let axisObj=buildIntertrialCompressedAxis(rows);let selected=selectedIntertrialIds();let mode=intertrialMode?.value==='pair'?`selected pair (${selected.length} trial${selected.length===1?'':'s'})`:'Experiment 1 versus Experiment 2';intertrialNote.textContent=`Intertrial view: ${mode}. Each camera is split into Experiment 1 (Trials 1–18) and Experiment 2 (Trials 21+) panels, with each experiment's own light/dark regime. Rows are aligned by elapsed trial time.`;
    for(const camRow of rows){let card=document.createElement('div');card.className='compare-camera-card';card.innerHTML=`<h4>${esc(intertrialCameraLabel(camRow.camera))} — ${esc(camRow.cohortLabel)}</h4>`;let c=document.createElement('canvas');c.width=1200;c.height=Math.max(190,(camRow.trials?.length||1)*46+78);c.style.height=Math.max(220,(camRow.trials?.length||1)*46+78)+'px';card.appendChild(c);intertrialCanvases.appendChild(card);drawOneIntertrialCamera(c,camRow,axisObj)}
}
function meanPct(vals){vals=vals.filter(v=>Number.isFinite(v));return vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:0}
function cameraAverageRows(analytics){let grouped={};for(const a of analytics){let cohort=a.cohort,key=`${cohort}|${a.camera}`;(grouped[key]||(grouped[key]=[])).push(a)}let rows=[];for(const cohort of ['experiment_1','experiment_2'])for(const cam of ['C','A','B']){let arr=grouped[`${cohort}|${cam}`];if(!arr?.length)continue;let vals=name=>arr.map(a=>Number(a.states?.[name]?.pct));let cohortLabel=cohort==='experiment_1'?'Experiment 1 (Trials 1–18)':'Experiment 2 (Trials 21+)';rows.push(`<tr><td>${esc(cohortLabel)}</td><td>${esc(intertrialCameraLabel(cam))}</td><td>${arr.length}</td><td>${meanPct(vals('quiescent')).toFixed(1)}%</td><td>${meanPct(vals('SA')).toFixed(1)}%</td><td>${meanPct(vals('moving')).toFixed(1)}%</td></tr>`)}return rows.join('')}
function cameraAveragesTable(analytics){let rows=cameraAverageRows(analytics);let mode=intertrialMode?.value==='pair'?'two selected trials':'all trials';return rows?`<table><caption>Averages between ${esc(mode)} by camera and cohort</caption><thead><tr><th>Cohort</th><th>Camera / group</th><th>n trials</th><th>avg quiescent</th><th>avg SA</th><th>avg moving</th></tr></thead><tbody>${rows}</tbody></table>`:''}
function roundedStateTriplet(stratum){if(!stratum?.available)return ['—','—','—'];let names=['moving','SA','quiescent'],raw=names.map(n=>Math.max(0,Number(stratum.states?.[n]?.pct)||0)*10),units=raw.map(Math.floor),left=1000-units.reduce((a,b)=>a+b,0),order=raw.map((v,i)=>({i,f:v-Math.floor(v)})).sort((a,b)=>b.f-a.f||a.i-b.i);for(let i=0;i<left;i++)units[order[i%3].i]++;return units.map(v=>(v/10).toFixed(1)+'%')}
function experiment2MetricHeader(r,firstCols){return `<thead><tr>${firstCols}<th rowspan="2">recorded h</th><th rowspan="2">classified h</th><th colspan="3">All Recorded</th><th colspan="3">${esc(r.light_label)}</th><th colspan="3">${esc(r.dark_label)}</th></tr><tr><th>moving</th><th>still alert</th><th>quiescent</th><th>moving</th><th>still alert</th><th>quiescent</th><th>moving</th><th>still alert</th><th>quiescent</th></tr></thead>`}
function experiment2MetricCells(w){let all=roundedStateTriplet(w),light=roundedStateTriplet(w.periods?.light),dark=roundedStateTriplet(w.periods?.dark);return `<td>${Number(w.recorded_hours||0).toFixed(1)}</td><td>${Number(w.classified_hours||0).toFixed(1)}</td>${[...all,...light,...dark].map(v=>`<td>${v}</td>`).join('')}`}
function experiment2IndividualTable(analytics){let rows=[];for(const a of analytics.filter(x=>x.cohort==='experiment_2'))for(const w of a.experiment_2_72h_windows||[])rows.push(`<tr><td>${esc(a.trial_label||a.trial)}</td><td>${esc(intertrialCameraLabel(a.camera))}</td><td>${esc(w.label)}</td>${experiment2MetricCells(w)}</tr>`);if(!rows.length)return '';let r=intertrialRegime('experiment_2');return `<table><caption>Experiment 2 — individual trials</caption>${experiment2MetricHeader(r,'<th rowspan="2">Trial</th><th rowspan="2">Camera / group</th><th rowspan="2">Window</th>')}<tbody>${rows.join('')}</tbody></table>`}
function experiment2PooledTable(summary){if(!summary?.cameras?.length)return '';let r=summary.light_regime||intertrialRegime('experiment_2'),rows=[];for(const cam of summary.cameras||[])for(const w of cam.windows||[])rows.push(`<tr><td>${esc(intertrialCameraLabel(cam.camera))}</td><td>${esc(w.label)}</td><td>${w.n_trials_with_recording||0}</td><td>${w.n_trials_with_classified||0}</td>${experiment2MetricCells(w)}</tr>`);return `<table><caption>Experiment 2 — duration-weighted pooled totals across all trials</caption>${experiment2MetricHeader(r,'<th rowspan="2">Camera / group</th><th rowspan="2">Window</th><th rowspan="2">n recorded</th><th rowspan="2">n classified</th>')}<tbody>${rows.join('')}</tbody></table><div class="compare-note">Windows are fixed elapsed trial hours 0–72 and 72–144. Each all/light/dark percentage triplet is normalized only over summed moving, still-alert, and quiescent hours. No-video gaps, unannotated footage, and poking are excluded from that denominator; recorded hours remain visible as coverage. Pooled values sum state hours before calculating percentages. Rounded available triplets total exactly 100.0%; “—” means no classified footage in that stratum.</div>`}
function renderIntertrialAnalytics(){if(!intertrialAnalytics)return;if(!intertrialData||!intertrialData.analytics){intertrialAnalytics.innerHTML='';return}intertrialAnalytics.innerHTML=experiment2IndividualTable(intertrialAnalyticsForView())+experiment2PooledTable(intertrialData.experiment_2_72h_summary)+cameraAveragesTable(intertrialAnalyticsForView())}
function drawGapBands(c,l,top,plotW,plotH,dur){let gaps=gapPlayIntervals();if(!gaps.length||dur<=0)return;c.font='10px system-ui';for(const [a,b] of gaps){let x0=l+(Math.max(0,a)/dur)*plotW,x1=l+(Math.min(dur,b)/dur)*plotW,w=Math.max(3,x1-x0);c.save();c.beginPath();c.rect(x0,top,w,plotH);c.clip();c.fillStyle='rgba(8,10,14,.74)';c.fillRect(x0,top,w,plotH);c.strokeStyle='rgba(245,158,11,.55)';c.lineWidth=1;for(let xx=x0-plotH;xx<x0+w;xx+=7){c.beginPath();c.moveTo(xx,top+plotH);c.lineTo(xx+plotH,top);c.stroke()}c.restore();c.strokeStyle='#f59e0b';c.lineWidth=1;c.strokeRect(x0+.5,top+.5,Math.max(2,w-1),plotH-1);if(w>24){c.fillStyle='#f59e0b';c.fillText('no video',x0+3,top+11)}}}
function drawFeedingEthogram(){if(ethowrap.style.display==='none')return;resizeFeedingCanvas();let w=feedingCanvas.clientWidth,h=feedingCanvas.clientHeight,l=88,r=12,top=8,b=24,plotW=w-l-r,plotH=h-top-b,rowH=plotH/FEED.length,dur=Math.max(getDuration(),0.001);feedingCtx.clearRect(0,0,w,h);feedingCtx.fillStyle='#05070a';feedingCtx.fillRect(0,0,w,h);feedingCtx.lineWidth=1;feedingCtx.font='12px system-ui';FEED.forEach((state,i)=>{let y=top+i*rowH;feedingCtx.fillStyle='rgba(255,255,255,.03)';feedingCtx.fillRect(l,y,plotW,rowH);feedingCtx.strokeStyle='#2d3748';feedingCtx.strokeRect(l,y,plotW,rowH);feedingCtx.fillStyle=state.color;feedingCtx.fillText(state.label,8,y+rowH/2+4)});let evs=[...feedingEvents].sort((a,b)=>a.time-b.time);for(let i=0;i<evs.length;i++){let ev=evs[i],state=FEEDBY[ev.state];if(!state)continue;let t0=Math.max(0,ev.time),t1=i+1<evs.length?evs[i+1].time:dur;if(t1<t0)continue;let x0=l+(t0/dur)*plotW,x1=l+(Math.min(t1,dur)/dur)*plotW,y=top+state.y*rowH;feedingCtx.fillStyle=state.color;feedingCtx.globalAlpha=.82;feedingCtx.fillRect(x0,y+3,Math.max(2,x1-x0),rowH-6);feedingCtx.globalAlpha=1}drawGapBands(feedingCtx,l,top,plotW,plotH,dur);feedingCtx.fillStyle='#9aa7b7';for(let i=0;i<=4;i++){let t=dur*i/4,x=l+(t/dur)*plotW;feedingCtx.strokeStyle='#475569';feedingCtx.beginPath();feedingCtx.moveTo(x,top);feedingCtx.lineTo(x,top+plotH);feedingCtx.stroke();feedingCtx.fillText(t.toFixed(1)+'s',x-12,h-6)}for(const m of directLightDarkMarks()){let mx=l+(m.t/dur)*plotW;feedingCtx.strokeStyle=m.c;feedingCtx.lineWidth=2;feedingCtx.beginPath();feedingCtx.moveTo(mx,top);feedingCtx.lineTo(mx,top+plotH);feedingCtx.stroke()}let x=l+((video.currentTime||0)/dur)*plotW;feedingCtx.strokeStyle='#fff';feedingCtx.lineWidth=2;feedingCtx.beginPath();feedingCtx.moveTo(x,top);feedingCtx.lineTo(x,top+plotH);feedingCtx.stroke()}
function drawEthogram(){if(ethowrap.style.display==='none')return;updateTimeUI();drawFeedingEthogram();resizeCanvas();let w=canvas.clientWidth,h=canvas.clientHeight,l=88,r=12,top=12,b=28,plotW=w-l-r,plotH=h-top-b,rowH=plotH/BEH.length,dur=Math.max(getDuration(),0.001);ctx.clearRect(0,0,w,h);ctx.fillStyle='#05070a';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#2d3748';ctx.lineWidth=1;ctx.font='12px system-ui';BEH.forEach((beh,i)=>{let y=top+i*rowH;ctx.fillStyle='rgba(255,255,255,.03)';ctx.fillRect(l,y,plotW,rowH);ctx.strokeStyle='#2d3748';ctx.strokeRect(l,y,plotW,rowH);ctx.fillStyle=beh.color;ctx.fillText(beh.label,8,y+rowH/2+4)});let evs=[...annEvents].sort((a,b)=>a.time-b.time);for(let i=0;i<evs.length;i++){let ev=evs[i],beh=BEHBY[ev.behavior];if(!beh)continue;let t0=Math.max(0,ev.time),t1=i+1<evs.length?evs[i+1].time:dur;if(t1<t0)continue;let x0=l+(t0/dur)*plotW,x1=l+(Math.min(t1,dur)/dur)*plotW,y=top+beh.y*rowH;ctx.fillStyle=beh.color;ctx.globalAlpha=.82;ctx.fillRect(x0,y+3,Math.max(2,x1-x0),rowH-6);ctx.globalAlpha=1}drawGapBands(ctx,l,top,plotW,plotH,dur);ctx.fillStyle='#9aa7b7';for(let i=0;i<=4;i++){let t=dur*i/4,x=l+(t/dur)*plotW;ctx.strokeStyle='#475569';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+plotH);ctx.stroke();ctx.fillText(t.toFixed(1)+'s',x-12,h-8)}for(const m of directLightDarkMarks()){let mx=l+(m.t/dur)*plotW;ctx.strokeStyle=m.c;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(mx,top);ctx.lineTo(mx,top+plotH);ctx.stroke();ctx.fillStyle=m.c;ctx.font='12px system-ui';ctx.fillText(m.lab,Math.min(w-86,mx+4),top+14)}let x=l+((video.currentTime||0)/dur)*plotW;ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+plotH);ctx.stroke();ctx.fillStyle='#fff';ctx.fillText((video.currentTime||0).toFixed(2)+'s',Math.min(w-55,x+4),top+12)}
function seekFromCanvasEvent(e,targetCanvas=canvas){let rct=targetCanvas.getBoundingClientRect(),x=e.clientX-rct.left,l=88,r=12,plotW=targetCanvas.clientWidth-l-r,dur=getDuration();let t=Math.max(0,Math.min(dur,((x-l)/plotW)*dur));video.currentTime=t;drawEthogram()}
function wheelSeek(e){e.preventDefault();let delta=e.shiftKey?0.05:0.5;video.currentTime=Math.max(0,Math.min(getDuration(),(video.currentTime||0)+(e.deltaY>0?delta:-delta)));drawEthogram()}
let dragging=false;canvas.addEventListener('pointerdown',e=>{dragging=true;canvas.setPointerCapture(e.pointerId);seekFromCanvasEvent(e,canvas)});canvas.addEventListener('pointermove',e=>{if(dragging)seekFromCanvasEvent(e,canvas)});canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('wheel',wheelSeek,{passive:false});
let feedingDragging=false;feedingCanvas.addEventListener('pointerdown',e=>{feedingDragging=true;feedingCanvas.setPointerCapture(e.pointerId);seekFromCanvasEvent(e,feedingCanvas)});feedingCanvas.addEventListener('pointermove',e=>{if(feedingDragging)seekFromCanvasEvent(e,feedingCanvas)});feedingCanvas.addEventListener('pointerup',()=>feedingDragging=false);feedingCanvas.addEventListener('wheel',wheelSeek,{passive:false});
scrubber.addEventListener('input',()=>{video.currentTime=Math.max(0,Math.min(getDuration(),parseFloat(scrubber.value)||0));drawEthogram()});
videoSize.addEventListener('input',()=>{document.documentElement.style.setProperty('--video-height', videoSize.value+'px')});document.documentElement.style.setProperty('--video-height', videoSize.value+'px');
clockSeek.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();seekClockTime()}});
video.addEventListener('timeupdate',()=>{drawEthogram();drawCompareEthograms();drawIntertrialEthograms()});video.addEventListener('loadedmetadata',()=>{drawEthogram();drawCompareEthograms();drawIntertrialEthograms()});window.addEventListener('resize',()=>{drawEthogram();drawCompareEthograms();drawIntertrialEthograms()});
if(intertrialMode)intertrialMode.addEventListener('change',drawIntertrialEthograms);if(intertrialTrialA)intertrialTrialA.addEventListener('change',drawIntertrialEthograms);if(intertrialTrialB)intertrialTrialB.addEventListener('change',drawIntertrialEthograms);
document.addEventListener('keydown',e=>{let tag=(e.target?.tagName||'').toLowerCase(),id=e.target?.id||'',key=e.key.toLowerCase();let hotkeySafeFocus=(id==='scrubber'||id==='videoSize'||tag==='body'||tag==='canvas'||tag==='video');if(['select','textarea','button'].includes(tag)||(tag==='input'&&!hotkeySafeFocus))return;if(e.ctrlKey&&key==='z'){e.preventDefault();undoAnnotation();return}if(e.ctrlKey&&key==='y'){e.preventDefault();redoAnnotation();return}if((key===' '||key==='spacebar'||e.code==='Space')&&currentVideoName){e.preventDefault();toggleVideoPlayback();return}if(key==='arrowleft'&&currentVideoName){e.preventDefault();stepVideo(-0.01);return}if(key==='arrowright'&&currentVideoName){e.preventDefault();stepVideo(0.01);return}let f=FEED.find(x=>x.key===key);if(f&&currentVideoName){e.preventDefault();addFeedingAnnotation(f.name);return}let b=BEH.find(x=>x.key===key);if(b&&currentVideoName){e.preventDefault();addAnnotation(b.name)}});
trialEl.onchange=()=>fillDates(false);dateEl.onchange=()=>fillCameras(camEl.value,false);camEl.onchange=()=>{clearLoadedVideo();updateSummary()};refreshOptions().catch(e=>document.getElementById('status').innerHTML='<span class=err>'+esc(e.message)+'</span>');setInterval(()=>refreshOptions(true).catch(e=>console.warn('auto option refresh failed',e)),900000);
</script></body></html>"""

# On-demand chunks share the canonical annotation player and wall-clock timeline.
import on_demand_cache
import sys
selection_cache = on_demand_cache.SelectionCache(
    lambda key, stop, update: on_demand_cache.build_selection(sys.modules[__name__], key, stop, update))
INDEX = INDEX.replace('<script>', '<script>' + Path(__file__).with_name('canonical_player.js').read_text() + '</script><script>', 1)
INDEX = INDEX.replace('</body>', '<script>' + Path(__file__).with_name('selection_cache.js').read_text() + '</script></body>')

class Handler(BaseHTTPRequestHandler):
    timeout = 60
    def send_json(self, obj, status=200, indent=2):
        body=json.dumps(obj, indent=indent).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p=urllib.parse.urlparse(self.path); qs=urllib.parse.parse_qs(p.query)
        try:
            if p.path in ('/','/index.html'):
                body=INDEX.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body); return
            if p.path=='/api/selection-cache':
                self.send_json(selection_cache.status(qs.get('client',[''])[0], int(qs.get('revision',['0'])[0]))); return
            if p.path.startswith('/selection-chunk/'):
                name=p.path.rsplit('/',1)[-1]
                if not re.fullmatch(r'[a-f0-9]{64}\.mp4', name):
                    self.send_error(404); return
                chosen=CACHE_ROOT/'selection_chunks'/name
                if not chosen.is_file(): self.send_error(404); return
                send_file(self,chosen,'video/mp4'); return
            if p.path=='/api/health':
                self.send_json({'ok': True, 'service': 'basler-annotation-interface', 'cache_maintenance_enabled': CACHE_MAINTENANCE_ENABLED, 'cache_maintenance_cohorts': sorted(CACHE_MAINTENANCE_COHORTS)}); return
            if p.path=='/api/options':
                _, options=scan_recordings(force=bool(qs.get('force')))
                maintenance = dict(cache_maintenance_state)
                maintenance.update({'auto_enabled': CACHE_MAINTENANCE_ENABLED, 'interval_seconds': CACHE_MAINTENANCE_INTERVAL, 'cohorts': sorted(CACHE_MAINTENANCE_COHORTS), 'note': 'automatic cache maintenance refreshes finalized footage only for configured cohorts; generation is serialized and low-priority'})
                options = dict(options); options['current_day_prewarm_status'] = {'submitted': [], 'note': 'automatic C/A/B prewarm disabled; all cameras use the same explicit load path'}; options['cache_maintenance_status'] = maintenance
                self.send_json(options); return
            if p.path=='/api/cache-maintenance':
                _, options=scan_recordings(force=bool(qs.get('force')))
                self.send_json(queue_cache_maintenance(options, reason='manual-api')); return
            if p.path=='/api/timelapse':
                trial=qs.get('trial',[''])[0]; date=qs.get('date',[''])[0]; camera=qs.get('camera',[''])[0].upper()
                if camera not in CAMERA_ORDER or not trial or not date: raise RuntimeError('trial, date, and camera are required')
                stale_ok = qs.get('stale_ok', ['1'])[0].lower() not in ('0','false','no')
                if stale_ok:
                    # Fast path for current/future trials: never make the browser wait
                    # for a full segment scan/key build.  Use any completed cache now;
                    # otherwise queue low-priority C/A/B generation and return 202.
                    stale = latest_cached_timelapse(trial,date,camera)
                    prep = start_prewarm_lazy(trial, date, [camera] if camera in CAMERA_ORDER else [])
                    if stale:
                        stale.update({'refresh_queued': bool(prep.get('submitted')), 'prewarm_submitted': prep.get('submitted', []), 'note': 'Showing latest completed cache immediately; low-priority generation is queued for the selected camera only from finalized MP4s. Use Prepare C/A/B to cache the other cameras.'})
                        self.send_json(stale); return
                    pending = {'trial': trial, 'date': date, 'camera': camera, 'cached': False, 'exact_cache': False, 'stale_cache': False, 'pending': True, 'refresh_queued': True, 'prewarm_submitted': prep.get('submitted', []), 'prewarm_workers': prep.get('workers'), 'video_url': None, 'segments': 0, 'coverage_hours': 0.0, 'target_duration_seconds': 0.0, 'segments_preview': [], 'note': 'No completed cache exists yet. Low-priority generation was queued for the selected camera only from finalized MP4s; poll /api/timelapse for readiness. Use Prepare C/A/B to cache the other cameras.'}
                    self.send_json(pending, status=202); return
                exact=timelapse_status(trial,date,camera)
                if exact.get('cached'):
                    exact.update({'exact_cache': True, 'stale_cache': False})
                    self.send_json(exact); return
                out=generate_timelapse(trial,date,camera); meta=timelapse_status(trial,date,camera); meta.update({'cached':True,'exact_cache':True,'stale_cache':False,'path':str(out)}); self.send_json(meta); return
            if p.path=='/api/prewarm-status':
                trial=qs.get('trial',[''])[0]; date=qs.get('date',[''])[0]; camera=qs.get('camera',[''])[0].upper()
                with prewarm_lock:
                    jobs = dict(prewarm_jobs)
                if trial or date or camera:
                    jobs = {k:v for k,v in jobs.items() if (not trial or v.get('trial')==trial) and (not date or v.get('date')==date) and (not camera or v.get('camera')==camera)}
                self.send_json({'workers': PREWARM_WORKERS, 'jobs': jobs}); return
            if p.path=='/api/prewarm':
                trial=qs.get('trial',[''])[0]; date=qs.get('date',[''])[0]
                cams=[c.strip().upper() for c in qs.get('cameras',[','.join(CAMERA_ORDER)])[0].split(',') if c.strip()]
                if not trial or not date: raise RuntimeError('trial and date are required')
                fast = qs.get('fast', ['0'])[0].lower() in ('1','true','yes','on')
                self.send_json(start_prewarm_lazy(trial,date,cams) if fast else start_prewarm(trial,date,cams)); return
            if p.path=='/api/annotations':
                video=qs.get('video',[''])[0]
                trial=qs.get('trial',[''])[0]; date=qs.get('date',[''])[0]; camera=qs.get('camera',[''])[0].upper()
                self.send_json(load_annotations(video, trial=trial, date=date, camera=camera)); return
            if p.path=='/api/ethogram-compare':
                trial=qs.get('trial',[''])[0]; selected_date=qs.get('date',[''])[0]
                self.send_json(compare_ethograms(trial, selected_date=selected_date), indent=None); return
            if p.path=='/api/ethogram-intertrial':
                self.send_json(compare_intertrial_ethograms(), indent=None); return
            if p.path.startswith('/video/'):
                name=Path(urllib.parse.unquote(Path(p.path).name)).name; chosen=CACHE_ROOT/'timelapses'/name
                if not chosen.exists(): self.send_error(404); return
                send_file(self, chosen, 'video/mp4'); return
            self.send_error(404)
        except Exception as e:
            self.send_json({'error':str(e)}, status=400)
    def log_message(self, fmt, *args):
        # Logging must never abort a response when an old terminal's stdout pipe closes.
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with (STATE_DIR / 'requests.log').open('a') as log:
                log.write('%s - %s\n' % (self.address_string(), fmt % args))
        except Exception:
            pass

    def do_POST(self):
        p=urllib.parse.urlparse(self.path)
        try:
            if p.path == '/api/selection-release':
                length=int(self.headers.get('Content-Length') or 0)
                if not 0 < length <= 4096: raise ValueError('Invalid release payload size')
                payload=json.loads(self.rfile.read(length))
                self.send_json(selection_cache.release(str(payload.get('client','')),int(payload.get('revision',0)))); return
            if p.path == '/api/selection-cache':
                length=int(self.headers.get('Content-Length') or 0)
                if not 0 < length <= 4096: raise ValueError('Invalid selection payload size')
                payload=json.loads(self.rfile.read(length))
                client=str(payload.get('client','')); revision=int(payload.get('revision',0))
                key=tuple(str(payload.get(k,'')) for k in ('trial','date','camera'))
                if not re.fullmatch(r'[a-zA-Z0-9-]{1,80}',client) or revision < 1:
                    raise ValueError('Invalid client/revision')
                if key[2] not in CAMERA_ORDER or not all(key): raise ValueError('Select a trial, date and camera')
                self.send_json(selection_cache.select(client,revision,key)); return
            if p.path in ('/api/annotations', '/api/annotations/recover'):
                length=int(self.headers.get('Content-Length') or 0)
                if length > ANNOTATION_POST_MAX_BYTES:
                    self.send_json({'error':'annotation payload too large','code':'too_large'}, status=413); return
                payload=json.loads(self.rfile.read(length).decode() if length else '{}')
                if p.path=='/api/annotations/recover':
                    self.send_json(recover_annotations(payload)); return
                self.send_json(save_annotations(payload)); return
            self.send_error(404)
        except AnnotationConflict as e:
            self.send_json({'error':str(e),'code':'conflict','current':e.current}, status=409)
        except AnnotationReadOnly as e:
            body=dict(e.payload or {}); body.update({'error':str(e),'code':'recovery'}); self.send_json(body, status=423)
        except Exception as e:
            self.send_json({'error':str(e)}, status=400)

def main():
    CACHE_ROOT.mkdir(parents=True,exist_ok=True); STATE_DIR.mkdir(parents=True,exist_ok=True)
    if CACHE_MAINTENANCE_ENABLED:
        threading.Thread(target=cache_maintenance_loop, daemon=True).start()
    print(f'Basler annotation interface at http://127.0.0.1:{PORT}/ record_root={RECORD_ROOT} cache={CACHE_ROOT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
if __name__=='__main__': main()
