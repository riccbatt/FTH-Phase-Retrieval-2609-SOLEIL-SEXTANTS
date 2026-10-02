"""Check the notebook's acquisition scaling and saturated two-helicity projection."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from library import phase_retrieval_universal as universal

NOTEBOOK = Path(__file__).resolve().parents[1] / 'maxiv_phase_test/06_maxiv_summed_helicities_saturated_reference.ipynb'


def functions_from_notebook(namespace, *names):
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        tree = ast.parse(source, filename=f'cell_{index}')
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), f'cell_{index}', 'exec'), namespace)
    return namespace


class SaturatedNotebookTests(unittest.TestCase):
    def test_mean_and_sum_caches_have_same_count_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            ns = functions_from_notebook(dict(np=np, h5py=h5py, AVERAGE_FOLDER=Path(folder),
                support_source=np.ones((4,4)), scan_metadata={1: {'frames':25}},
                CACHE_REDUCTION='mean'), 'load_mean')
            path = Path(folder)/'scan_0001_avg.h5'
            mean_counts = np.arange(16).reshape(4,4).astype(float)
            with h5py.File(path,'w') as h: h['image'] = mean_counts
            mean_rate = ns['load_mean'](1)/.6
            with h5py.File(path,'w') as h: h['image'] = 25*mean_counts
            ns['CACHE_REDUCTION'] = 'sum'
            np.testing.assert_allclose(ns['load_mean'](1)/.6, mean_rate)



if __name__ == '__main__':
    unittest.main()
