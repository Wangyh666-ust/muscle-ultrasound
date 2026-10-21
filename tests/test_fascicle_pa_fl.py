import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.umud_baseline.apo_baseline import load_frozen_rule, measure_conservative  # noqa: E402
from scripts.fascicle_pa_fl import (  # noqa: E402
    synthetic_case, parallel_case, evaluate_segment, pa_for_segment,
    fl_for_segment, select_candidates, evaluate_mask_pair,
)

RULE, RULE_HASH = load_frozen_rule()


def synthetic_apo(h=512, w=512):
    m = np.zeros((h, w), np.uint8)
    m[98:103, 50:450] = 1
    m[298:303, 50:450] = 1
    out = measure_conservative(m, RULE)
    return m, out['fits']


class SyntheticGeometryTests(unittest.TestCase):
    def test_known_angle_and_length(self):
        r = synthetic_case()
        self.assertEqual(r['status'], 'ok')
        self.assertAlmostEqual(r['pa_deg'], 30.0, delta=0.5)
        self.assertAlmostEqual(r['fl_px'], 400.0, delta=3.0)
        self.assertAlmostEqual(r['extrapolation_ratio'],
                               r['expected_extrapolation_ratio'], delta=0.02)

    def test_parallel_segment_rejected_for_fl(self):
        r = parallel_case()
        self.assertEqual(r['pa_status'], 'ok')
        self.assertAlmostEqual(r['pa_deg'], 0.0, delta=0.1)
        self.assertEqual(r['fl_status'], 'rejected')
        self.assertEqual(r['fl_reason'], 'parallel_to_aponeurosis')


class IndependentSelectionTests(unittest.TestCase):
    def test_pa_can_survive_while_fl_is_rejected(self):
        apo, fits = synthetic_apo()
        self.assertIsNotNone(fits)
        # Shallow segment near the superficial boundary: the deep-boundary
        # intersection is far outside the image. PA does not need that
        # intersection and remains available; FL keeps its raw geometry and
        # all failing conditions.
        seg = {
            'centroid': [250.0, 120.0], 'p0': [225.0, 117.8], 'p1': [275.0, 122.2],
            'length_px': 50.2, 'direction_deg': 5.0,
        }
        pa = pa_for_segment(seg, fits, 512)
        fl = fl_for_segment(seg, fits, 512, 512)
        self.assertEqual(pa['pa_status'], 'ok')
        self.assertAlmostEqual(pa['pa_deg'], abs(5.0 - math.degrees(
            math.atan2(float(fits['deep'][0]), 1.0))), delta=0.1)
        self.assertEqual(fl['fl_status'], 'rejected')
        self.assertIn('intersection_outside_image', fl['fl_failure_conditions'])
        self.assertIn('intersection_outside_image', fl['fl_failure_conditions_pure'])
        self.assertIn('local_support_missing', fl['fl_failure_conditions'])
        self.assertNotIn('local_support_missing', fl['fl_failure_conditions_pure'])
        self.assertIsNotNone(fl['fl_px_raw'])
        self.assertIsNotNone(fl['extrapolation_ratio'])
        self.assertIsNotNone(fl['intersection_sup'])
        self.assertIsNotNone(fl['intersection_deep'])

        rec = {'segment': seg, **pa, **fl}
        sel = select_candidates([rec])
        self.assertEqual(sel['n_pa_selected'], 1)
        self.assertIsNotNone(sel['pa_mean_deg'])
        self.assertEqual(sel['n_fl_selected'], 0)
        self.assertIsNone(sel['fl_mean_px'])

    def test_selection_is_independent(self):
        apo, fits = synthetic_apo()
        long_seg = {
            'centroid': [250.0, 200.0], 'p0': [150.0, 142.3], 'p1': [350.0, 257.7],
            'length_px': 230.9, 'direction_deg': 30.0,
        }
        shallow_seg = {
            'centroid': [250.0, 120.0], 'p0': [225.0, 117.8], 'p1': [275.0, 122.2],
            'length_px': 50.2, 'direction_deg': 5.0,
        }
        cands = [evaluate_segment(long_seg, fits, 512, 512),
                 evaluate_segment(shallow_seg, fits, 512, 512)]
        sel = select_candidates(cands)
        self.assertEqual(sel['n_pa_selected'], 2)
        self.assertEqual(sel['n_fl_selected'], 1)
        self.assertAlmostEqual(sel['fl_mean_px'], cands[0]['fl_px'])

    def test_all_fl_conditions_retained(self):
        fits = {
            'sup': np.array([0.0, 100.0]), 'deep': np.array([0.0, 300.0]),
            'support': (0, 511), 'paired_x': np.array([0.0, 1.0, 2.0]),
            'max_sample_gap': 5.0, 'image_shape': (512, 512),
        }
        seg = {
            'centroid': [50.0, 120.0], 'p0': [25.0, 117.8], 'p1': [75.0, 122.2],
            'length_px': 50.2, 'direction_deg': 5.0,
        }
        fl = fl_for_segment(seg, fits, 512, 512, apo_ok=True)
        self.assertEqual(fl['fl_status'], 'rejected')
        self.assertEqual(set(fl['fl_failure_conditions']), {
            'intersection_outside_image', 'intersection_outside_apo_support',
            'local_support_missing', 'excessive_extrapolation',
        })
        self.assertEqual(set(fl['fl_failure_conditions_pure']), {
            'intersection_outside_image', 'intersection_outside_apo_support',
            'excessive_extrapolation',
        })
        self.assertEqual(set(fl['fl_failure_conditions_strict']), {
            'intersection_outside_image', 'intersection_outside_apo_support',
            'local_support_missing', 'excessive_extrapolation',
        })
        self.assertFalse(fl['fl_accept_pure'])
        self.assertFalse(fl['fl_accept_strict'])
        self.assertIsNotNone(fl['fl_px_raw'])
        self.assertGreater(fl['extrapolation_ratio'], 0.8)

    def test_pure_acceptance_does_not_require_local_support(self):
        fits = {
            'sup': np.array([0.0, 100.0]), 'deep': np.array([0.0, 300.0]),
            'support': (0, 511), 'paired_x': np.array([0.0, 1.0, 2.0]),
            'max_sample_gap': 5.0, 'image_shape': (512, 512),
        }
        seg = {
            'centroid': [250.0, 200.0], 'p0': [150.0, 142.3], 'p1': [350.0, 257.7],
            'length_px': 230.9, 'direction_deg': 30.0,
        }
        fl = fl_for_segment(seg, fits, 512, 512, apo_ok=True)
        self.assertTrue(fl['fl_accept_pure'])
        self.assertFalse(fl['fl_accept_strict'])
        self.assertEqual(fl['fl_status'], 'ok')
        self.assertEqual(fl['fl_status_strict'], 'rejected')
        self.assertIsNotNone(fl['fl_px_pure'])
        self.assertIsNone(fl['fl_px_strict'])
        self.assertIn('local_support_missing', fl['fl_failure_conditions'])
        self.assertEqual(fl['fl_failure_conditions_pure'], [])

    def test_rejected_apo_status_keeps_raw_geometry_but_no_selection(self):
        apo, fits = synthetic_apo()
        long_seg = {
            'centroid': [250.0, 200.0], 'p0': [150.0, 142.3], 'p1': [350.0, 257.7],
            'length_px': 230.9, 'direction_deg': 30.0,
        }
        fl = fl_for_segment(long_seg, fits, 512, 512, apo_ok=False)
        self.assertEqual(fl['fl_status'], 'rejected')
        self.assertIn('apo_status_rejected', fl['fl_failure_conditions'])
        self.assertIsNotNone(fl['fl_px_raw'])
        self.assertIsNotNone(fl['intersection_deep'])

    def test_apo_rejection_propagates_to_both_metrics(self):
        fasc = np.zeros((100, 100), np.uint8)
        fasc[48:52, 20:80] = 1
        apo = np.zeros((100, 100), np.uint8)
        out = evaluate_mask_pair(fasc, apo, 100, 100, RULE, RULE_HASH)
        self.assertEqual(out['apo_status'], 'rejected')
        self.assertEqual(out['apo_reason'], 'no_mask')
        self.assertEqual(out['apo_source'], 'baseline_rejected')
        self.assertFalse(out['apo_diagnostic_fits_available'])
        for c in out['candidates']:
            self.assertEqual(c['pa_status'], 'rejected')
            self.assertEqual(c['fl_status'], 'rejected')
            self.assertIn('no_apo_boundaries', c['pa_failure_conditions'])
            self.assertIn('apo_status_rejected', c['pa_failure_conditions'])
            self.assertIn('no_apo_boundaries', c['fl_failure_conditions'])
        self.assertIsNone(out['selection']['pa_mean_deg'])
        self.assertIsNone(out['selection']['fl_mean_px'])


if __name__ == '__main__':
    unittest.main()
