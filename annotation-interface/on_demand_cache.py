"""Selection-only canonical playback cache with pinned timeline manifests."""
import hashlib
import json
import os
import math
import subprocess
import tempfile
import time
from pathlib import Path


class Cancelled(Exception):
    pass


class ChunkRenderer:
    def __init__(self, root, encoder, width, height, fps, command=lambda x: x):
        self.root = Path(root)
        self.encoder, self.width, self.height, self.fps = encoder, width, height, fps
        self.command = command

    def path(self, chunk):
        return self.root / (chunk['id'] + '.mp4')

    def duration(self, path):
        try:
            data = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(path)],stderr=subprocess.DEVNULL,timeout=10))
            value = float(data.get('format',{}).get('duration',0))
            return value if math.isfinite(value) and value > 0 else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def render(self, chunk, cancel):
        out = self.path(chunk)
        if self.duration(out):
            return out
        if cancel.is_set():
            raise Cancelled()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.part.mp4')
        if chunk['kind'] == 'run':
            inputs = ['-ss',str(chunk['source_offset']),'-t',str(chunk['source_duration']),'-i',chunk['source']]
            speed = chunk['source_duration'] / (chunk['end']-chunk['start'])
            vf = f'setpts=(PTS-STARTPTS)/{speed},fps={self.fps}:round=up,scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,setsar=1'
        else:
            inputs = ['-f','lavfi','-i',f'color=c=0x182030:s={self.width}x{self.height}:r={self.fps}:d={max(1/self.fps,chunk["end"]-chunk["start"])}']
            vf = 'setsar=1'
        cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y'] + inputs + ['-filter_threads','1','-vf',vf,'-an'] + self.encoder + ['-movflags','+faststart',str(tmp)]
        try:
            with tempfile.TemporaryFile() as errors:
                proc = subprocess.Popen(self.command(cmd), stdout=subprocess.DEVNULL, stderr=errors)
                while proc.poll() is None:
                    if cancel.wait(.1):
                        proc.terminate()
                        try: proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill(); proc.wait()
                        raise Cancelled()
                if proc.returncode:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors='replace')[-2000:])
            # A successful process can still emit zero frames for tiny sources.
            data = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(tmp)]))
            if float(data.get('format',{}).get('duration',0)) <= 0:
                raise RuntimeError('Chunk contains no playable frames')
            os.replace(tmp, out)
            return out
        finally:
            tmp.unlink(missing_ok=True)


import threading
from concurrent.futures import ThreadPoolExecutor


class SelectionCache:
    """One active generation; newer client revisions cancel obsolete work."""
    def __init__(self, build, lease_seconds=45):
        self.lease_seconds = lease_seconds
        self.build = build
        self.lock = threading.RLock()
        self.clients = {}
        self.pool = ThreadPoolExecutor(max_workers=1)

    def select(self, client, revision, key):
        with self.lock:
            old = self.clients.get(client)
            if old and revision <= old['revision']:
                return self.status(client, revision)
            if old:
                old['stop'].set()
                old['future'].cancel()
            job = dict(key=key, revision=revision, state='queued', ready=[], total=0,
                       eta_seconds=None, stop=threading.Event())
            self.clients[client] = job
            job['future'] = self.pool.submit(self._run, job)
            self._renew(job)
            return self.status(client, revision)

    def _renew(self, job):
        if job.get('timer'): job['timer'].cancel()
        job['lease_token'] = job.get('lease_token', 0) + 1
        job['timer'] = threading.Timer(self.lease_seconds, self._expire, args=(job, job['lease_token']))
        job['timer'].daemon = True
        job['timer'].start()

    def _expire(self, job, token=None):
        with self.lock:
            if token is not None and token != job.get('lease_token'): return
            job['stop'].set()
            job['future'].cancel()
            if job['state'] in ('queued', 'running'):
                job.update(state='cancelled', eta_seconds=None)

    def release(self, client, revision):
        with self.lock:
            job = self.clients.get(client)
            if job and job['revision'] == revision:
                if job.get('timer'): job['timer'].cancel()
                self._expire(job)
            return self.status(client, -1)

    def _run(self, job):
        def update(**fields):
            with self.lock:
                job.update(fields)
        try:
            if job['stop'].is_set(): raise Cancelled()
            update(state='running')
            self.build(job['key'], job['stop'], update)
            update(state='done', eta_seconds=0)
        except Cancelled:
            update(state='cancelled', eta_seconds=None)
        except Exception as exc:
            update(state='error', error=str(exc), eta_seconds=None)

    def status(self, client, revision):
        with self.lock:
            job = self.clients.get(client)
            if job and job['revision'] == revision and job['state'] in ('queued', 'running'):
                self._renew(job)
            return {k:v for k,v in job.items() if k not in ('stop','future','timer','lease_token')} if job else {'state':'idle','eta_seconds':None,'ready':[]}


def selection_manifest(app, key, settings):
    """Pin annotation timing, not frame-rounded per-file media durations."""
    trial, date, camera = key
    saved = app.load_annotations(trial=trial, date=date, camera=camera)
    previous = saved.get('meta') or {}
    if previous.get('playback_mode') == 'chunks-v1':
        name = Path(str(saved.get('video') or '')).name
        manifest = json.loads((app.CACHE_ROOT/'selection_manifests'/(name+'.json')).read_text())
        if tuple(manifest.get(k) for k in ('trial','date','camera')) != tuple(key):
            raise RuntimeError('Pinned manifest selection mismatch')
        return manifest, manifest['chunks']
    # Never replace an annotated monolith with a newly composed timeline.
    if saved.get('video') and (saved.get('events') or saved.get('feeding_events') or saved.get('revision')):
        name = Path(saved['video']).name
        path = app.CACHE_ROOT/'timelapses'/name
        if not path.is_file():
            return reconstruct_legacy_manifest(app, key, saved, settings)
        meta = json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else dict(previous)
        meta.update(video_url='/video/'+name, video_name=name, playback_mode='file')
        return meta, []
    existing = app.latest_cached_timelapse(*key)
    if existing:
        existing.update(video_name=Path(existing['video_url']).name, playback_mode='file')
        return existing, []
    segs, _, _ = app.select_segments(*key)
    if not segs: raise RuntimeError('No finalized segments for this selection')
    segs = sorted(segs, key=lambda s:s['start_ts'])
    timeline = app.compose_timeline(segs)
    chunks = plan_chunks(segs, timeline['gaps'], settings=settings)
    identity = hashlib.sha256(json.dumps([key, chunks, app.timeline_pieces_public(timeline)],sort_keys=True).encode()).hexdigest()
    meta = dict(trial=trial,date=date,camera=camera,playback_mode='chunks-v1',
                video_name='progressive-'+identity+'.mp4',chunks=chunks,
                pieces=app.timeline_pieces_public(timeline),gaps=timeline['gaps'],
                clip_start_ts=timeline['clip_start_ts'],play_duration_seconds=timeline['play_total'],
                target_duration_seconds=timeline['play_total'])
    app.atomic_write_text(app.CACHE_ROOT/'selection_manifests'/(meta['video_name']+'.json'), json.dumps(meta))
    return meta, chunks


def reconstruct_legacy_manifest(app, key, saved, settings):
    """Rebuild media only: the saved piece clock and video identity are immutable.

    Recorder discovery supplies trusted paths, never a replacement timeline.
    Six-decimal public piece boundaries can differ by one microsecond. Reconcile
    only that rounding in playback ranges; retain the original pieces verbatim.
    """
    previous = saved.get('meta') or {}
    pieces = previous.get('pieces')
    def refuse(reason):
        raise RuntimeError('Pinned annotated video is missing; cannot reconstruct: '+reason)
    def number(value):
        try:
            result = float(value)
            if math.isfinite(result): return result
        except (TypeError, ValueError):
            pass
        refuse('insufficient or invalid saved timing metadata')
    if not isinstance(pieces, list) or not pieces:
        refuse('insufficient saved piece metadata')
    if tuple(previous.get(k) for k in ('trial','date','camera')) != tuple(key):
        refuse('saved selection mismatch')
    duration = number(saved.get('duration_seconds') or previous.get('play_duration_seconds') or previous.get('target_duration_seconds'))
    if duration <= 0: refuse('invalid saved duration')
    number(previous.get('clip_start_ts'))
    name = Path(saved['video']).name
    manifest_path = app.CACHE_ROOT/'selection_manifests'/(name+'.json')
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if (tuple(meta.get(k) for k in ('trial','date','camera')) != tuple(key)
                or meta.get('pieces') != pieces or meta.get('canonical_duration_seconds') != duration):
            refuse('pinned reconstruction manifest does not match saved timing')
        return meta, meta['chunks']
    segments, _, _ = app.select_segments(*key)
    segments = sorted(segments, key=lambda s:s['start_ts'])
    chunks = []
    cursor = 0.0
    wall_cursor = None
    def append(kind, start, end, **fields):
        if start >= duration: return
        original_end = end
        end = min(end, duration)
        if end <= start: return
        if kind == 'run' and end < original_end:
            fields['source_duration'] *= (end-start)/(original_end-start)
        item = dict(kind=kind, start=start, end=end, **fields)
        item['id'] = hashlib.sha256(json.dumps(['legacy-pieces-v1', settings, item],sort_keys=True).encode()).hexdigest()
        chunks.append(item)
    for piece in pieces:
        if not isinstance(piece, dict): refuse('invalid saved piece')
        ps, pd, ws, we = [number(piece.get(k)) for k in ('play_start','play_dur','wall_start_ts','wall_end_ts')]
        if piece.get('kind') not in ('run','gap') or pd <= 0 or we <= ws:
            refuse('invalid saved piece interval')
        if abs(ps-cursor) > .000002 or (wall_cursor is not None and abs(ws-wall_cursor) > .000002):
            refuse('non-contiguous saved piece mapping')
        end = ps+pd
        if piece['kind'] == 'gap':
            append('gap', cursor, end)
        else:
            wall = ws
            while wall < we:
                source = next((s for s in segments if float(s['start_ts']) <= wall < float(s['end_ts'])), None)
                if source is None: refuse(f'unavailable source coverage at wall timestamp {wall}')
                path = Path(source['mp4'])
                if not path.is_file(): refuse(f'unavailable source file at wall timestamp {wall}')
                stop = min(we, float(source['end_ts']), wall+1800)
                offset = wall-float(source['start_ts'])
                length = stop-wall
                if number(source.get('duration', float(source['end_ts'])-float(source['start_ts'])))+.000002 < offset+length:
                    refuse('source duration cannot cover saved wall interval')
                start = cursor if wall == ws else ps+(wall-ws)/(we-ws)*pd
                finish = end if stop == we else ps+(stop-ws)/(we-ws)*pd
                append('run', start, finish, source=str(path), source_offset=offset,
                       source_duration=length, source_mtime=source.get('mtime',path.stat().st_mtime_ns))
                wall = stop
        cursor, wall_cursor = end, we
    # A monolith's frame-rounded duration can extend beyond its mapping. Preserve
    # that annotation clock with an explicit neutral tail, never stretch pieces.
    if cursor < duration: append('gap', cursor, duration, rounding_tail=True)
    if not chunks: refuse('no playable saved intervals')
    meta = dict(previous)
    meta.update(video_name=name, playback_mode='chunks-v1', chunks=chunks,
                canonical_duration_seconds=duration, reconstructed_from=name)
    app.atomic_write_text(manifest_path, json.dumps(meta))
    return meta, chunks


def build_selection(app, key, stop, update):
    encoder, encoder_name = app.ffmpeg_encode_args()
    settings = json.dumps(['selection-v1',encoder,encoder_name,app.MAX_WIDTH,app.MAX_HEIGHT,app.TARGET_FPS])
    if hasattr(app, 'load_annotations'):
        meta, chunks = selection_manifest(app, key, settings)
        update(meta=meta)
        if meta['playback_mode'] == 'file':
            update(total=1,ready=[],eta_seconds=0,state='done')
            return
    else:  # Small renderer fixtures do not include annotation storage.
        segs, _, _ = app.select_segments(*key)
        chunks = plan_chunks(segs, app.compose_timeline(segs)['gaps'], settings=settings)
    renderer = ChunkRenderer(app.CACHE_ROOT/'selection_chunks',encoder,app.MAX_WIDTH,app.MAX_HEIGHT,app.TARGET_FPS,app.low_priority_cmd)
    ready = []
    missing = []
    def public(chunk):
        return dict(start=chunk['start'],end=chunk['end'],kind=chunk['kind'],media_duration=renderer.duration(renderer.path(chunk)),url='/selection-chunk/'+renderer.path(chunk).name)
    for chunk in chunks:
        if renderer.duration(renderer.path(chunk)):
            ready.append(public(chunk))
        else:
            missing.append(chunk)
    update(total=len(chunks), duration=chunks[-1]['end'], ready=list(ready), eta_seconds=None,
           state='queued' if missing else 'done')
    if not missing:
        update(eta_seconds=0)
        return
    while not app.generation_semaphore.acquire(timeout=.1):
        if stop.is_set(): raise Cancelled()
    try:
        started=time.monotonic()
        rendered_seconds=0.0
        remaining=sum(c['end']-c['start'] for c in missing)
        update(state='running')
        for chunk in missing:
            if stop.is_set(): raise Cancelled()
            renderer.render(chunk,stop)
            ready.append(public(chunk))
            length=chunk['end']-chunk['start']
            rendered_seconds+=length
            remaining-=length
            estimate=max(0,remaining)*(time.monotonic()-started)/rendered_seconds
            update(ready=sorted(ready,key=lambda r:r['start']),eta_seconds=estimate,
                   estimate_basis='completed playback seconds / elapsed render time; source complexity varies')
        update(eta_seconds=0)
    finally:
        app.generation_semaphore.release()


def plan_chunks(segments, gaps, bound=1800, settings=''):
    chunks = []
    cursor = 0.0
    for seg in segments:
        for gap in gaps:
            if gap['wall_end_ts'] == seg['start_ts']:
                item = dict(kind='gap', start=cursor, end=cursor+gap['play_dur'], piece=gap)
                item['id'] = hashlib.sha256(json.dumps([settings, gap], sort_keys=True).encode()).hexdigest()
                chunks.append(item)
                cursor = item['end']
        duration = float(seg.get('duration') or seg['end_ts']-seg['start_ts'])
        offset = 0.0
        while offset < duration:
            length = min(bound, duration-offset)
            identity = [settings, str(seg['mp4']), seg['mtime'], offset, length]
            item = dict(kind='run', start=cursor, end=cursor+length/3600,
                        source=str(seg['mp4']), source_offset=offset, source_duration=length,
                        id=hashlib.sha256(json.dumps(identity).encode()).hexdigest())
            chunks.append(item)
            cursor = item['end']
            offset += length
    return chunks
