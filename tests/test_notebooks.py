import json
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from library import phase_retrieval_universal as u
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("dataset",["hyperspectral","2601_XPCS"])
@pytest.mark.parametrize("modes",[[1],[1,2]])
@pytest.mark.parametrize("crop,binning",[(0,1),(2,2)])
def test_shared_notebook_cells(tmp_path,monkeypatch,dataset,modes,crop,binning):
    monkeypatch.chdir(ROOT)
    cells=json.loads((ROOT/f"notebooks/{dataset}/01_reconstruct.ipynb").read_text())["cells"]
    ns={}
    exec("".join(cells[1]["source"]).split("CACHE =")[0],ns)
    support=np.zeros((32,32));support[12:20,12:20]=1
    support[6:8,6:8]=1;support[6:8,24:26]=1;support[24:26,6:8]=1
    if binning>1:
        support[:]=0; support[14:18,14:18]=1
    states=["a","a","b","b"] if dataset=="hyperspectral" else ["p","mid","n"]
    energy=[700.,710.,700.,710.] if dataset=="hyperspectral" else [710.]*3
    ns.update(holograms=np.random.default_rng(1).uniform(.5,2,(len(states),32,32)),
        mask_pixel=np.zeros((len(states),32,32)),support=support,
        material=u.largest_support_component(support).astype(float),
        state_labels=states,energy_labels=energy,polarizations=[1.]*len(states),
        saturated={} if dataset=="hyperspectral" else {"p":1.,"n":-1.},
        current=None if dataset=="hyperspectral" else np.array([1.,0.,-1.]),
        crop=crop,binning=binning,THICKNESS_NM=None,
        config={"test":"synthetic"},OUTPUT=tmp_path/"result.h5")
    for i in [3,5]:exec("".join(cells[i]["source"]),ns)
    ns["MODES"]=modes
    if binning>1:
        ns["recipe"].update(fit_aperture_illumination=False,fit_illumination_intensity=False,align_global_phase=False)
    ns["recipe"].update(modes=modes,mode_support_center="components",outer_iterations=2,
        inner_Nit=1,physical_iterations=2,TV_freq=1000000)
    for i in [7,9,11,13,15]:exec("".join(cells[i]["source"]),ns)
    assert np.all(np.isfinite(ns["fields"]))
    assert ns["fields"].shape[-2:]==((32-2*crop)//binning,)*2
    assert ns["OUTPUT"].is_file()
    plt.close("all")

def test_streaming_saturation_and_mean(tmp_path):
    import h5py
    from library.scan_io import frame_mean, DATASET
    path=tmp_path/"frames.nx"
    frames=np.array([[[1,2],[3,4]],[[5,65535],[7,8]]],dtype=np.uint16)
    with h5py.File(path,"w") as h:h.create_dataset(DATASET,data=frames)
    mean,invalid,count=frame_mean(path)
    np.testing.assert_allclose(mean,frames.astype(float).mean(axis=0))
    assert invalid[0,1] and invalid.sum()==1 and count==2
