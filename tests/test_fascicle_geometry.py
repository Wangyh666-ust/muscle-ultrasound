import math
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fascicle_geometry import (  # noqa: E402
    extract_segments, build_tracks, match_segments, summarize_match,
    aggregate_variants,
)


def line_mask(h, w, p0, p1, width=3):
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).line([p0, p1], fill=1, width=width)
    return (np.asarray(im) > 0).astype(np.uint8)


class ExtractSegmentTests(unittest.TestCase):
    def test_evaluable_diagonal_segment(self):
        m = line_mask(400, 600, (100, 100), (500, 300), width=3)
        ext = extract_segments(m)
        self.assertEqual(len(ext['segments']), 1)
        seg = ext['segments'][0]
        self.assertEqual(seg['category'], 'evaluable')
        expected_len = math.hypot(400, 200)
        self.assertAlmostEqual(seg['length_px'], expected_len, delta=8)
        self.assertAlmostEqual(seg['direction_deg'], 26.565, delta=2)
        self.assertGreater(seg['linearity'], 0.9)
        self.assertGreater(seg['elongation'], 20)

    def test_short_component_is_rejected(self):
        m = line_mask(100, 100, (10, 50), (25, 50), width=3)
        ext = extract_segments(m)
        self.assertEqual(len(ext['segments']), 0)
        cats = [r['category'] for r in ext['rejected']]
        self.assertIn('short_segment', cats)

    def test_crossing_component_is_not_evaluable(self):
        m = line_mask(200, 200, (20, 100), (180, 100), width=3)
        m |= line_mask(200, 200, (100, 20), (100, 180), width=3)
        ext = extract_segments(m)
        self.assertEqual(len(ext['segments']), 0)
        cats = [r['category'] for r in ext['rejected']]
        self.assertIn('crossing_or_branched', cats)


class MatchingTests(unittest.TestCase):
    def test_near_line_matches(self):
        human = extract_segments(line_mask(300, 500, (100, 100), (400, 200)))['segments']
        pred = extract_segments(line_mask(300, 500, (100, 103), (400, 203)))['segments']
        m = match_segments(pred, human, perp_tol=10, angle_tol=10, overlap_min=0.2)
        s = summarize_match(pred, human, m)
        self.assertEqual(s['n_matched'], 1)
        self.assertEqual(s['coverage'], 1.0)
        self.assertLess(s['position_perp_median_px'], 5)
        self.assertLess(s['direction_median_deg'], 5)

    def test_far_line_is_extra_candidate_not_auto_false_positive(self):
        human = extract_segments(line_mask(300, 500, (100, 100), (400, 200)))['segments']
        pred = extract_segments(line_mask(300, 500, (100, 200), (400, 300)))['segments']
        m = match_segments(pred, human, perp_tol=10, angle_tol=10, overlap_min=0.2)
        s = summarize_match(pred, human, m)
        self.assertEqual(s['n_matched'], 0)
        self.assertEqual(s['n_unmatched_pred_extra_candidates'], 1)
        self.assertEqual(s['extra_candidate_rate'], 1.0)
        self.assertEqual(s['failure_rate_human'], 1.0)

    def test_collinear_fragments_group_into_track_without_mask_change(self):
        m = np.zeros((200, 400), np.uint8)
        m[98:103, 60:160] = 1
        m[98:103, 170:280] = 1
        ext = extract_segments(m)
        self.assertEqual(len(ext['segments']), 2)
        tracks = build_tracks(ext['segments'], angle_tol=5, perp_tol=5, gap_tol=20)
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0]['fragmented'])
        self.assertEqual(tracks[0]['n_fragments'], 2)

    def test_variant_aggregation_ignores_missing_denominators(self):
        variants = [
            {'coverage': 0.5, 'failure_rate_human': 0.5, 'position_perp_mean_px': 2,
             'position_perp_median_px': 2, 'position_mid_mean_px': 5,
             'position_mid_median_px': 5, 'direction_mean_deg': 2,
             'direction_median_deg': 2, 'extra_candidate_rate': 0.1,
             'position_perp_p90_px': 3, 'direction_p90_deg': 3,
             'n_human_evaluable': 10, 'n_pred_evaluable': 8, 'n_matched': 5,
             'n_unmatched_pred_extra_candidates': 3},
            {'coverage': 1.0, 'failure_rate_human': 0.0, 'position_perp_mean_px': 4,
             'position_perp_median_px': 4, 'position_mid_mean_px': 7,
             'position_mid_median_px': 7, 'direction_mean_deg': 4,
             'direction_median_deg': 4, 'extra_candidate_rate': 0.0,
             'position_perp_p90_px': 6, 'direction_p90_deg': 5,
             'n_human_evaluable': 2, 'n_pred_evaluable': 2, 'n_matched': 2,
             'n_unmatched_pred_extra_candidates': 0},
        ]
        out = aggregate_variants(variants)
        self.assertAlmostEqual(out['coverage'], 0.75)
        self.assertAlmostEqual(out['position_perp_median_px'], 3.0)
        self.assertEqual(out['n_variants'], 2)
        self.assertEqual(out['n_human_evaluable_sum_variants'], 12)


if __name__ == '__main__':
    unittest.main()
