#!/usr/bin/env python3
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

import annotation_interface as ai


def _data(events=None, feeding=None, **extra):
    payload = {
        'events': events or [],
        'feeding_events': feeding or [],
        'behaviors': ai.BEHAVIORS,
        'feeding_states': ai.FEEDING_STATES,
        'note': 'test',
    }
    payload.update(extra)
    return payload


class AnnotationStoreTests(unittest.TestCase):
    def test_annotation_root_lookup_uses_cached_index_without_scanning(self):
        with tempfile.TemporaryDirectory() as td:
            old_root = ai.RECORD_ROOT
            old_index = ai.selection_session_roots
            old_select = ai.select_segments
            try:
                ai.RECORD_ROOT = Path(td)
                first = Path(td) / 'C' / 'recording_start-first' / 'segment' / 'one.mp4'
                second = Path(td) / 'C' / 'recording_start-second' / 'segment' / 'two.mp4'
                segments = [
                    {'camera': 'C', 'trial': 'session-2', 'trial_group': 'logical-trial',
                     'date': '2026-08-28', 'start_ts': 200.0, 'end_ts': 300.0, 'mp4': str(second)},
                    {'camera': 'C', 'trial': 'session-1', 'trial_group': 'logical-trial',
                     'date': '2026-08-28', 'start_ts': 100.0, 'end_ts': 200.0, 'mp4': str(first)},
                ]
                ai.selection_session_roots = ai.build_selection_session_root_index(segments)
                ai.select_segments = lambda *args, **kwargs: self.fail('annotation lookup must not scan/select segments')
                selection_date = ai.segment_calendar_dates(segments[1])[0]
                self.assertEqual(
                    ai.recording_session_root_for_selection('logical-trial', selection_date, 'C'),
                    first.parent.parent.resolve(),
                )
            finally:
                ai.RECORD_ROOT = old_root
                ai.selection_session_roots = old_index
                ai.select_segments = old_select

    def test_annotation_root_falls_back_to_validated_cached_video_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            old_root = ai.RECORD_ROOT
            old_cache = ai.CACHE_ROOT
            old_index = ai.selection_session_roots
            try:
                ai.RECORD_ROOT = td / 'recordings'
                ai.CACHE_ROOT = td / 'cache'
                ai.selection_session_roots = {}
                source = ai.RECORD_ROOT / 'A' / 'recording_start-safe' / 'segment' / 'source.mp4'
                source.parent.mkdir(parents=True)
                source.write_bytes(b'video')
                timelapses = ai.CACHE_ROOT / 'timelapses'
                timelapses.mkdir(parents=True)
                video = timelapses / 'safe.mp4'
                video.write_bytes(b'cache')
                video.with_suffix('.json').write_text(json.dumps({
                    'trial': 'trial-1', 'date': '2026-08-28', 'camera': 'A',
                    'segments_preview': [{'rel': str(source.relative_to(ai.RECORD_ROOT))}],
                }))
                self.assertEqual(
                    ai.recording_session_root_for_selection(
                        'trial-1', '2026-08-28', 'A', video_name=video.name,
                    ),
                    source.parent.parent.resolve(),
                )
                self.assertIsNone(ai.recording_session_root_for_selection(
                    'other-trial', '2026-08-28', 'A', video_name=video.name,
                ))
            finally:
                ai.RECORD_ROOT = old_root
                ai.CACHE_ROOT = old_cache
                ai.selection_session_roots = old_index

    def test_frontend_save_releases_before_comparison_refresh(self):
        source = Path(ai.__file__).read_text(encoding='utf-8')
        save_fn = source.split('async function saveAnnotations(){', 1)[1].split('\n', 1)[0]
        self.assertIn('scheduleCompareRefresh()', save_fn)
        self.assertNotIn('await loadCompareEthograms()', save_fn)
        self.assertIn('if(saveInFlight){saveQueued=true;return}', source)

    def test_comparison_cache_reuses_result_and_save_invalidation_expires_it(self):
        original_compute = ai._compute_compare_ethograms
        original_cache = ai.comparison_cache
        original_generation = ai.comparison_cache_generation
        calls = []
        try:
            ai.comparison_cache = {}
            ai.comparison_cache_generation = 0
            ai._compute_compare_ethograms = lambda trial, date, items, options: calls.append(
                (trial, date, items, options)
            ) or {'trial': trial, 'rows': []}
            snapshot = ([], {'trials': []}, 123.0)
            first = ai.compare_ethograms('trial-1', _snapshot=snapshot)
            second = ai.compare_ethograms('trial-1', _snapshot=snapshot)
            self.assertIs(first, second)
            self.assertEqual(len(calls), 1)
            ai.invalidate_comparison_cache()
            ai.compare_ethograms('trial-1', _snapshot=snapshot)
            self.assertEqual(len(calls), 2)
        finally:
            ai._compute_compare_ethograms = original_compute
            ai.comparison_cache = original_cache
            ai.comparison_cache_generation = original_generation

    def test_comparison_snapshot_reuses_published_catalog_without_scanning(self):
        original_cache = ai.scan_cache
        original_scan = ai.scan_recordings
        try:
            items = [{'trial': 'trial-1'}]
            options = {'trials': [{'trial': 'trial-1'}]}
            ai.scan_cache = {'ts': 789.0, 'items': items, 'options': options}
            ai.scan_recordings = lambda *args, **kwargs: self.fail(
                'published comparison snapshot must not rescan'
            )
            self.assertEqual(ai.comparison_snapshot(), (items, options, 789.0))
        finally:
            ai.scan_cache = original_cache
            ai.scan_recordings = original_scan

    def test_intertrial_comparison_passes_one_snapshot_to_every_trial(self):
        original_compare = ai.compare_ethograms
        snapshots = []
        try:
            snapshot = ([], {'trials': [
                {'trial': 'trial-1', 'label': 'One', 'cohort': 'experiment_1'},
                {'trial': 'trial-2', 'label': 'Two', 'cohort': 'experiment_2'},
            ]}, 456.0)
            def fake_compare(trial, selected_date=None, _snapshot=None):
                snapshots.append(_snapshot)
                return {
                    'trial': trial, 'dates': [], 'axis_hours': 0.0,
                    'rows': [], 'analytics': [], 'light_dark': {},
                }
            ai.compare_ethograms = fake_compare
            result = ai.compare_intertrial_ethograms(_snapshot=snapshot)
            self.assertEqual(len(result['trials']), 2)
            self.assertEqual(snapshots, [snapshot, snapshot])
        finally:
            ai.compare_ethograms = original_compare

    def test_cache_maintenance_only_checks_configured_cohort(self):
        options = {'trials': [
            {'trial': 'exp1', 'cohort': 'experiment_1', 'dates': [{'date': '2026-08-13', 'cameras': ['C']}]},
            {'trial': 'exp2', 'cohort': 'experiment_2', 'dates': [{'date': '2026-08-13', 'cameras': ['C']}]},
        ]}
        checked = []
        original_status = ai.timelapse_status
        original_latest = ai.latest_cached_timelapse
        original_queue = ai.queue_refresh_if_needed
        try:
            ai.timelapse_status = lambda trial, date, camera: checked.append(trial) or {
                'cached': True, 'segments': 1, 'coverage_hours': 1,
            }
            ai.latest_cached_timelapse = lambda *args: None
            ai.queue_refresh_if_needed = lambda *args: False
            result = ai.queue_cache_maintenance(options, reason='test')
        finally:
            ai.timelapse_status = original_status
            ai.latest_cached_timelapse = original_latest
            ai.queue_refresh_if_needed = original_queue
        self.assertEqual(checked, ['exp2'])
        self.assertEqual(result['cohorts'], ['experiment_2'])

    def test_read_missing_vs_malformed(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / 'missing.json'
            ok, data, err = ai.read_annotation_json(missing)
            self.assertFalse(ok)
            self.assertIsNone(data)
            self.assertEqual(err, 'missing')
            bad = Path(td) / 'bad.json'
            bad.write_text('{not json', encoding='utf-8')
            ok, data, err = ai.read_annotation_json(bad)
            self.assertFalse(ok)
            self.assertIsNone(data)
            self.assertIn('malformed JSON', err)

    def test_atomic_write_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'out.json'
            ai.atomic_write_text(path, '{"ok": true}')
            self.assertEqual(json.loads(path.read_text()), {'ok': True})
            ai.atomic_write_text(path, '{"ok": false, "n": 2}')
            self.assertEqual(json.loads(path.read_text()), {'ok': False, 'n': 2})
            self.assertFalse(list(Path(td).glob('*.tmp')))

    def test_corrupt_canonical_is_read_only_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            json_path = td / 'sel.annotations.json'
            csv_path = td / 'sel.ethogram.csv'
            json_path.write_text('{truncated', encoding='utf-8')
            paired = td / 'paired.annotations.json'
            paired.write_text(json.dumps(_data(events=[{'time': 1.0, 'behavior': 'moving'}])), encoding='utf-8')
            view = ai.load_annotations_from_paths('sel', json_path, csv_path, paired=(paired, td / 'paired.ethogram.csv'))
            self.assertTrue(view['read_only'])
            self.assertTrue(view['recovery']['corrupt_canonical'])
            self.assertEqual(view['events'], [])
            self.assertEqual(json_path.read_text(encoding='utf-8'), '{truncated')
            self.assertTrue(any(c['source'] == 'paired' and c['event_count'] == 1 for c in view['recovery']['candidates']))

    def test_revision_conflict_and_history(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            json_path = td / 'sel.annotations.json'
            csv_path = td / 'sel.ethogram.csv'
            first = ai.commit_annotation_revision('sel', json_path, csv_path, _data(events=[{'time': 0.5, 'behavior': 'SA'}]), 'video_time_s\n', 0, lacie_dir=None)
            self.assertEqual(first['revision'], 1)
            with self.assertRaises(ai.AnnotationConflict):
                ai.commit_annotation_revision('sel', json_path, csv_path, _data(events=[{'time': 1.0, 'behavior': 'moving'}]), 'video_time_s\n', 0, lacie_dir=None)
            second = ai.commit_annotation_revision('sel', json_path, csv_path, _data(events=[{'time': 1.0, 'behavior': 'moving'}]), 'video_time_s\n', 1, lacie_dir=None)
            self.assertEqual(second['revision'], 2)
            hist = ai.revision_history_dir(json_path) / '000001.annotations.json'
            self.assertTrue(hist.exists())
            old = json.loads(hist.read_text(encoding='utf-8'))
            self.assertEqual(old['events'][0]['behavior'], 'SA')
            current = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(current['events'][0]['behavior'], 'moving')

    def test_corrupt_save_refused_and_recover_replace(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            json_path = td / 'sel.annotations.json'
            csv_path = td / 'sel.ethogram.csv'
            json_path.write_text('not-json', encoding='utf-8')
            with self.assertRaises(ai.AnnotationReadOnly):
                ai.commit_annotation_revision('sel', json_path, csv_path, _data(), 'video_time_s\n', 0, lacie_dir=None)
            self.assertEqual(json_path.read_text(encoding='utf-8'), 'not-json')
            recovered = ai.commit_annotation_revision(
                'sel', json_path, csv_path,
                _data(events=[{'time': 2.0, 'behavior': 'quiescent'}], revision=4),
                'video_time_s\n', 0, lacie_dir=None, replace_unreadable=True,
            )
            self.assertGreaterEqual(recovered['revision'], 5)
            ok, data, err = ai.read_annotation_json(json_path)
            self.assertTrue(ok, err)
            self.assertEqual(data['events'][0]['behavior'], 'quiescent')

    def test_replica_failure_does_not_undo_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            json_path = td / 'sel.annotations.json'
            csv_path = td / 'sel.ethogram.csv'
            committed = ai.commit_annotation_revision('sel', json_path, csv_path, _data(), 'video_time_s\n', 0, lacie_dir=None)
            blocked = Path('/proc/1/not-a-writable-annotation.json')
            ai.replicate_committed('sel', json.dumps(committed), 'video_time_s\n', json_path, csv_path, paired=(blocked, blocked), lacie_dir=Path('/proc/1'))
            ok, data, err = ai.read_annotation_json(json_path)
            self.assertTrue(ok, err)
            self.assertEqual(data['revision'], committed['revision'])

    def test_experiment_2_half_windows_normalize_only_classified_states(self):
        start = ai.dt.datetime(2026, 8, 1, 0, 0).timestamp()
        pieces = [{
            'play_start': 0.0, 'play_dur': 24.0,
            'wall_start_ts': start, 'wall_end_ts': start + 24 * 3600,
        }]
        events = [
            {'time': 2.0, 'behavior': 'moving'},
            {'time': 8.0, 'behavior': 'poking'},
            {'time': 12.0, 'behavior': 'SA'},
            {'time': 18.0, 'behavior': 'quiescent'},
        ]
        result = ai.state_window_duration_hours(
            events, 24.0, start, start + 24 * 3600,
            pieces=pieces, clip_start_ts=start, light_start=6.0, light_end=18.0,
        )
        self.assertAlmostEqual(result['recorded_hours'], 24.0)
        self.assertAlmostEqual(result['classified_hours'], 18.0)
        self.assertTrue(result['available'])
        for name in ('moving', 'SA', 'quiescent'):
            self.assertAlmostEqual(result['states'][name]['pct'], 100.0 / 3.0)
        self.assertAlmostEqual(sum(v['pct'] for v in result['states'].values()), 100.0)
        light = result['periods']['light']
        self.assertAlmostEqual(light['recorded_hours'], 12.0)
        self.assertAlmostEqual(light['classified_hours'], 8.0)
        self.assertAlmostEqual(light['states']['moving']['pct'], 25.0)
        self.assertAlmostEqual(light['states']['SA']['pct'], 75.0)
        dark = result['periods']['dark']
        self.assertAlmostEqual(dark['recorded_hours'], 12.0)
        self.assertAlmostEqual(dark['classified_hours'], 10.0)
        self.assertAlmostEqual(dark['states']['moving']['pct'], 40.0)
        self.assertAlmostEqual(dark['states']['quiescent']['pct'], 60.0)

    def test_experiment_2_zero_classified_strata_are_unavailable(self):
        start = ai.dt.datetime(2026, 8, 1, 0, 0).timestamp()
        pieces = [{
            'play_start': 0.0, 'play_dur': 24.0,
            'wall_start_ts': start, 'wall_end_ts': start + 24 * 3600,
        }]
        result = ai.state_window_duration_hours(
            [{'time': 0.0, 'behavior': 'poking'}], 24.0,
            start, start + 24 * 3600, pieces=pieces, clip_start_ts=start,
            light_start=6.0, light_end=18.0,
        )
        self.assertAlmostEqual(result['recorded_hours'], 24.0)
        self.assertEqual(result['classified_hours'], 0.0)
        self.assertFalse(result['available'])
        self.assertFalse(result['periods']['light']['available'])
        self.assertFalse(result['periods']['dark']['available'])
        self.assertTrue(all(v['pct'] == 0.0 for v in result['states'].values()))

    def test_experiment_2_half_windows_exclude_no_video_gaps(self):
        start = ai.dt.datetime(2026, 8, 1, 0, 0).timestamp()
        pieces = [{
            'play_start': 0.0, 'play_dur': 72.0,
            'wall_start_ts': start, 'wall_end_ts': start + 72 * 3600,
        }]
        result = ai.state_window_duration_hours(
            [{'time': 0.0, 'behavior': 'moving'}], 72.0,
            start, start + 72 * 3600,
            gaps=[(10.0, 20.0)], pieces=pieces, clip_start_ts=start,
            light_start=6.0, light_end=18.0,
        )
        self.assertAlmostEqual(result['recorded_hours'], 62.0)
        self.assertAlmostEqual(result['states']['moving']['hours'], 62.0)
        self.assertAlmostEqual(result['states']['moving']['pct'], 100.0)

    def test_experiment_2_pooling_weights_underlying_classified_hours(self):
        def window(recorded, moving, sa=0.0, quiescent=0.0, key='first_72h'):
            states = {
                'moving': {'hours': moving}, 'SA': {'hours': sa},
                'quiescent': {'hours': quiescent},
            }
            return {
                'key': key, 'recorded_hours': recorded, 'states': states,
                'periods': {
                    'light': {'recorded_hours': recorded, 'states': states},
                    'dark': {'recorded_hours': 0.0, 'states': {}},
                },
            }
        analytics = [
            {'trial': 't1', 'cohort': 'experiment_2', 'camera': 'C', 'experiment_2_72h_windows': [window(10.0, 10.0)]},
            {'trial': 't2', 'cohort': 'experiment_2', 'camera': 'C', 'experiment_2_72h_windows': [window(30.0, 0.0, sa=30.0)]},
        ]
        pooled = ai.pool_experiment_2_72h_analytics(analytics)
        camera_c = next(c for c in pooled['cameras'] if c['camera'] == 'C')
        first = next(w for w in camera_c['windows'] if w['key'] == 'first_72h')
        self.assertAlmostEqual(first['recorded_hours'], 40.0)
        self.assertAlmostEqual(first['classified_hours'], 40.0)
        self.assertAlmostEqual(first['states']['moving']['pct'], 25.0)
        self.assertAlmostEqual(first['states']['SA']['pct'], 75.0)
        self.assertEqual(first['n_trials_with_recording'], 2)
        self.assertEqual(first['n_trials_with_classified'], 2)
        self.assertAlmostEqual(sum(v['pct'] for v in first['states'].values()), 100.0)

    def test_experiment_2_pooling_uses_classified_hours_and_last_window(self):
        def window(key, recorded, moving=0.0, sa=0.0, quiescent=0.0):
            states = {
                'moving': {'hours': moving}, 'SA': {'hours': sa},
                'quiescent': {'hours': quiescent},
            }
            return {
                'key': key, 'recorded_hours': recorded, 'states': states,
                'periods': {
                    'light': {'recorded_hours': recorded / 2, 'states': states},
                    'dark': {'recorded_hours': recorded / 2, 'states': {}},
                },
            }
        analytics = [
            {'trial': 't1', 'cohort': 'experiment_2', 'camera': 'B',
             'experiment_2_72h_windows': [window('last_72h', 100.0, moving=10.0)]},
            {'trial': 't2', 'cohort': 'experiment_2', 'camera': 'B',
             'experiment_2_72h_windows': [window('last_72h', 30.0, sa=30.0)]},
        ]
        pooled = ai.pool_experiment_2_72h_analytics(analytics)
        camera_b = next(c for c in pooled['cameras'] if c['camera'] == 'B')
        last = next(w for w in camera_b['windows'] if w['key'] == 'last_72h')
        self.assertAlmostEqual(last['recorded_hours'], 130.0)
        self.assertAlmostEqual(last['classified_hours'], 40.0)
        self.assertAlmostEqual(last['states']['moving']['pct'], 25.0)
        self.assertAlmostEqual(last['states']['SA']['pct'], 75.0)
        self.assertAlmostEqual(sum(v['pct'] for v in last['states'].values()), 100.0)
        light = last['periods']['light']
        self.assertTrue(light['available'])
        self.assertAlmostEqual(light['classified_hours'], 40.0)
        self.assertAlmostEqual(light['states']['moving']['pct'], 25.0)
        self.assertAlmostEqual(light['states']['SA']['pct'], 75.0)
        self.assertFalse(last['periods']['dark']['available'])


if __name__ == '__main__':
    unittest.main()
