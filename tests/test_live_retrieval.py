import sys,contextlib,io,json,ast
import matplotlib
matplotlib.use('Agg')
import numpy as np
from library import phase_retrieval_universal as u
from library.retrieval_live import LiveReconstruction
from unittest.mock import patch,Mock
from IPython.core.inputtransformer2 import TransformerManager
import unittest

class LiveRetrievalTests(unittest.TestCase):
    def test_live_observer_preserves_fields(self):
        shape=(8,8); images=np.full((2,*shape),4.)
        r=u.default_universal_phase_retrieval_recipe()
        r.update(modes=[1,2],mode_initialization='support_fft',warmup_mode=['ER'],warmup_Nit=[1],
         inner_mode=['ER'],inner_Nit=[1],outer_iterations=2,physical_iterations=1,
         projection_model='physical_factorized',projection_every=2,final_projection_relaxation=0,
         average_img=1,shuffle_observations=False,constrain_nonphysical_modes_common=True)
        def kernel(**kw):return kw['Phase'].copy(),np.zeros(1),np.zeros(1),kw.get('gamma')
        def run(callback=None):
         with contextlib.redirect_stdout(io.StringIO()):
          return u.universal_phase_retrieval_algorithm(images,np.zeros_like(images),np.ones(shape),
           ['a','b'],[783]*2,[1,-1],['beam']*2,universal_recipe=r,
           phase_retrieval_kernel=kernel,progress_callback=callback)
        baseline=run();events=[];view=LiveReconstruction([0,1],min_seconds=0)
        def observe(event):
         events.append((event['stage'],event['outer_round']))
         view(event)
        with patch('IPython.display.display',return_value=Mock()) as display:
         result=run(observe)
         assert display.call_count==1
         assert view.handle.update.call_count==2
        np.testing.assert_array_equal(result[0],baseline[0])
        assert events==[('physical_projection',1),('final',2)],events
