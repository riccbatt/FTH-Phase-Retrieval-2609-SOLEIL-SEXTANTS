"""Exercise the actual alternating notebook cells on small synthetic sums."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class AlternatingNotebookTests(unittest.TestCase):
    def test_physical_then_retrieval_cycle_and_diagnostics(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from library import phase_retrieval_geometry as geometry
        from library import phase_retrieval_universal as universal
        from library import phase_retrieval_universal as pr
        from scipy.fft import set_workers
        from library.retrieval_progress import retrieval_progress
        import h5py
        for name in ['05_maxiv_hyperspectral_phase_retrieval_linear_2modes.ipynb',
                     '06_maxiv_summed_helicities_saturated_reference.ipynb']:
            is05=name.startswith('05')
            nb=json.loads((ROOT/'maxiv_phase_test'/name).read_text())
            for count in [2,3]:
                with self.subTest(notebook=name,modes=count), tempfile.TemporaryDirectory() as folder:
                    support=np.zeros((16,16));support[6:10,6:10]=1;support[3,3]=1
                    material=support.copy();material[3,3]=0
                    prepared=np.ones((2,2,16,16));prepared[:,0]*=1.1
                    ns=dict(np=np,plt=plt,json=json,h5py=h5py,pr=pr,geometry=geometry,universal=universal,
                        set_workers=set_workers,retrieval_progress=retrieval_progress,
                        MODES=[1,1] if count==2 else [1,1,2],support=support,material_thickness=material,
                        modal_supports=pr.mode_supports(support,[1,1] if count==2 else [1,1,2],support.shape),
                        OUTER_ITERATIONS=2,INNER_ITERATIONS=1,ALGORITHMS=['HAPRE','ER'],ITERATIONS=[2,1],
                        INITIALIZATION_SEED=0,PROJECTION_RELAXATION=.3,FFT_WORKERS=1,
                        USE_PHYSICAL_PROJECTION=True,USE_SATURATION_CONSTRAINT=True,SATURATED_SIGN=-1,
                        COMMON_BACKGROUND=True,energies=np.array([775.,776.]),ENERGY_EV=776.,
                        state_labels=['state_0','state_1'],FOCUS_PROP_UM=0.,FOCUS_PHASE_RAD=0.,FOCUS_SETUP={},
                        summed_intensity=prepared.sum(axis=1),prepared=prepared,mask_pixel=np.zeros((2,16,16),bool),
                        OUTPUT_FILE=Path(folder)/'result.h5',STATE_PAIRS=[],PAIRS=[],metadata=[],scan_metadata={},
                        CROP=0,BINNING=1,FRAME_NORMALIZATION=1,reference_energy=775.5,
                        SUPPORT_FILE=Path(folder)/'support.h5',DETECTOR_CENTER=(8,8),SATURATION=7500,
                        USE_SAVED_PIXEL_MASK=False,BEAMSTOP_RADIUS=0,CACHE_REDUCTION='mean')
                    code=''.join(nb['cells'][6]['source'])
                    boundary=code.index('history, retrieval_history, components')
                    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                        exec(code[:boundary],ns)
                        if is05:
                            plus, minus = ns['helicity_seed']
                            charge = (plus + minus)/np.sqrt(2.)
                            magnetic = (plus - minus)/np.sqrt(2.)
                            np.testing.assert_allclose(charge.imag, 0., atol=1e-14)
                            self.assertTrue(np.all(charge.real >= 0))
                            np.testing.assert_allclose(magnetic[material == 0], 0., atol=1e-14)
                            self.assertAlmostEqual(np.linalg.norm(magnetic)/np.linalg.norm(charge), .1)
                            np.testing.assert_allclose(abs(plus)**2+abs(minus)**2,
                                                       abs(charge)**2+abs(magnetic)**2)
                        events=[]
                        project,update=ns['physical_project'],ns['update_pair']
                        def project_record(stack):
                            events.append('physical')
                            return project(stack)
                        def update_record(f,i,warmup=False):
                            events.append('warmup' if warmup else 'retrieval')
                            return update(f,i,warmup)
                        ns.update(physical_project=project_record,update_pair=update_record)
                        exec(code[boundary:],ns)
                        for i in [8,9]:exec(''.join(nb['cells'][i]['source']),ns)
                    self.assertEqual(events,['warmup']*2+['physical','retrieval','retrieval']*2)
                    self.assertEqual(ns['fields'].shape,(2,count,16,16))
                    self.assertEqual(np.shape(ns['history']),(2,2))
                    self.assertLess(np.max(ns['sum_errors'](ns['fields'])),1e-10)
                    if not is05:
                        c=ns['components'];index=list(c['state_names']).index('state_0')
                        np.testing.assert_allclose(c['magnetization'][index][np.fft.fftshift(material)>0],-1)
                    with h5py.File(ns['OUTPUT_FILE']) as h:
                        self.assertEqual(h.attrs['solver'],'alternating_physical_projection_then_phase_retrieval')
                        self.assertIn('retrieval_history',h)
                        self.assertIn('separation_summary_json',h.attrs)
                    plt.close('all')

if __name__=='__main__':unittest.main()
