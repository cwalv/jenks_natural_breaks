import os.path as osp
import numpy as np

from jenks_natural_breaks import classify

def test_main():
    data = np.load(osp.join(osp.dirname(__file__), "test_data.npy"))
    expected = [
        0.0005421961588045754,
        0.20570633478192457,
        0.4197343611487738,
        0.6230659944190866,
        0.8134908717021093,
        0.9995623760709071
    ]

    result = classify(data, 5)
    r = lambda v: round(v, 6)
    assert map(r, result) == map(r, expected)
