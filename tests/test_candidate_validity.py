import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.candidate_validity import check_candidate_region  # noqa: E402


def apo_fits(paired_x=None, max_gap=8.0):
    paired_x = np.arange(90, 331, dtype=float) \
        if paired_x is None else np.asarray(paired_x, dtype=float)
    return {
        'sup': np.array([0.0, 100.0]),
        'deep': np.array([0.0, 300.0]),
        'support': (0, 511),
        'paired_x': paired_x,
        'max_sample_gap': max_gap,
        'image_shape': (512, 512),
    }


def segment(p0, p1, direction=0.0):
    p0 = np.asarray(p0, dtype=float); p1 = np.asarray(p1, dtype=float)
    return {
        'p0': p0.tolist(), 'p1': p1.tolist(),
        'centroid': ((p0 + p1) / 2).tolist(),
        'direction_deg': direction,
        'length_px': float(np.linalg.norm(p1 - p0)),
    }


class CandidateValidityTests(unittest.TestCase):
    def test_inside_candidate_with_local_support_is_valid(self):
        seg = segment((100, 150), (300, 150))
        r = check_candidate_region(seg, apo_fits())
        self.assertEqual(r['region_status'], 'inside')
        self.assertAlmostEqual(r['region_inside_fraction'], 1.0)
        self.assertEqual(r['local_support_status'], 'supported')
        self.assertEqual(r['candidate_validity_status'], 'valid')
        self.assertEqual(r['candidate_validity_conditions'], [])

    def test_candidate_outside_band_region_is_invalid(self):
        # Above the superficial boundary.
        seg = segment((100, 50), (300, 50))
        r = check_candidate_region(seg, apo_fits())
        self.assertEqual(r['region_status'], 'outside')
        self.assertAlmostEqual(r['region_inside_fraction'], 0.0)
        self.assertIn('candidate_outside_band_region',
                      r['candidate_validity_conditions'])
        self.assertEqual(r['candidate_validity_status'], 'invalid')

    def test_candidate_in_support_gap_is_invalid(self):
        # Candidate is inside the band, but the apo paired columns are far
        # from its x range (gap > max_sample_gap).
        seg = segment((190, 150), (210, 150))
        r = check_candidate_region(
            seg, apo_fits(paired_x=[100, 110, 120, 300, 310, 320]))
        self.assertEqual(r['region_status'], 'inside')
        self.assertEqual(r['local_support_status'], 'missing')
        self.assertIn('candidate_local_support_gap',
                      r['candidate_validity_conditions'])
        self.assertEqual(r['candidate_validity_status'], 'invalid')

    def test_no_apo_boundaries_is_unknown(self):
        r = check_candidate_region(segment((100, 150), (300, 150)), None)
        self.assertEqual(r['region_status'], 'unknown')
        self.assertEqual(r['candidate_validity_status'], 'unknown')
        self.assertIn('no_apo_boundaries',
                      r['candidate_validity_conditions'])


if __name__ == '__main__':
    unittest.main()
