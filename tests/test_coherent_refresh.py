"""Coherent refresh releases masked pixels and replaces subsequent PC targets."""
import unittest
import numpy as np
import test_parallel_retrieval as parallel_tests
from library import phase_retrieval_universal as u


class CoherentRefreshTests(unittest.TestCase):
    def test_repeated_refresh_replaces_fills_and_retains_gamma(self):
        for strategy in ("average", "pooled", "sequential"):
            for modes in ([1], [1,2]):
                with self.subTest(strategy=strategy, modes=modes):
                    self.check_refresh(strategy, modes)

    def check_refresh(self, strategy, modes):
        harness = parallel_tests.ParallelRetrievalTests()
        recipe = harness.recipe(strategy, 2, modes)
        recipe.update(outer_iterations=3, coherent_refresh_rounds=[2,3],
                      coherent_refresh_iterations=3)
        calls=[]
        # Kernel mask spelling follows the public core contract.
        def checked_kernel(**kw):
            calls.append({key: value.copy() if isinstance(value, np.ndarray) else value
                          for key, value in kw.items()})
            field=kw['Phase'].copy()
            if kw['RL_it']==0:
                self.assertTrue(np.any(kw['bsmask']))
                field[...,1,2] += 1
            else:
                self.assertFalse(np.any(kw['bsmask']))
            return field, np.zeros(1), np.zeros(1), kw.get('gamma')
        result=harness.run_recipe(recipe,checked_kernel)
        errors=result[-1]
        self.assertEqual(len(errors['coherent_refresh_steps']),6)
        refresh_calls=[c for c in calls if c['Nit']==3 and c['RL_it']==0]
        self.assertEqual(len(refresh_calls),6)
        for call in refresh_calls:
            self.assertIsNone(call['gamma'])
        partial=[c for c in calls if c['RL_it']>0]
        for call in partial:
            self.assertIsNotNone(call["gamma"])
            np.testing.assert_allclose(call["diffract"][1,2] ** 2,
                                       np.sum(abs(call["Phase"][...,1,2]) ** 2))
        fills=[float(c['diffract'][1,2]) for c in partial]
        self.assertGreater(max(fills),min(fills))

if __name__=='__main__': unittest.main()
