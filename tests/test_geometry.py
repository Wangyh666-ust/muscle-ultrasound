import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.umud_baseline.geometry import measure_mt,extract_apo_bands,mt_at_positions
class GeometryTests(unittest.TestCase):
    def mask(self):
        m=np.zeros((200,300),bool);m[20:23,40:260]=True;m[50:53,40:260]=True;return m
    def test_both_above_midline(self):
        r,_=measure_mt(self.mask(),px_per_mm=2)
        self.assertEqual(r['status'],'candidate');np.testing.assert_allclose(r['mt_px'],30)
        self.assertAlmostEqual(r['mt_mean_mm'],15);self.assertIsNone(r['fl_mm'])
    def test_single(self):
        m=self.mask();m[50:53]=False;self.assertEqual(measure_mt(m)[0]['status'],'rejected')
    def test_third_band(self):
        m=self.mask();m[80:83,40:260]=True
        self.assertEqual(measure_mt(m)[0]['reason'],'ambiguous_multiple_bands')
    def test_no_extrapolation(self):
        self.assertEqual(measure_mt(self.mask(),[10,150,280])[0]['reason'],'measurement_outside_support')
    def test_gap(self):
        m=self.mask();m[:,140:170]=False
        self.assertEqual(measure_mt(m)[0]['reason'],'measurement_in_support_gap')
    def test_vertical_not_normal(self):
        m=np.zeros((200,300),bool)
        for x in range(40,260):
            y=round(.25*x+10);m[y:y+3,x]=True;m[y+40:y+43,x]=True
        np.testing.assert_allclose(measure_mt(m)[0]['mt_px'],40,atol=1e-6)
    def test_negative(self):
        f,_=extract_apo_bands(self.mask());f['deep']=f['sup']-np.array([0,2])
        self.assertEqual(mt_at_positions(f,[70,150,230])[1],'nonpositive_thickness')
    def test_bad_scale(self):
        self.assertEqual(measure_mt(self.mask(),px_per_mm=0)[0]['reason'],'invalid_scale')
    def test_missing_scale(self):self.assertIsNone(measure_mt(self.mask())[0]['mt_mm'])
    def test_empty(self):self.assertEqual(measure_mt(np.zeros((100,100)))[0]['reason'],'no_mask')
if __name__=='__main__':unittest.main()
