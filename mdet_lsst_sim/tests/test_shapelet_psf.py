import os
import numpy as np
import pytest
import galsim
from ..shapelet_psf import make_shapelet_psf


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_shapelet_psf_smoke():
    rng = np.random.RandomState(2332)

    psf = make_shapelet_psf(rng=rng)
    assert psf.threshold == 0.0
    assert psf.nepoch == 1

    pos = galsim.PositionD(1.5, 2.2)
    _ = psf.getPSF(pos)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('nepoch', [1, 15])
@pytest.mark.parametrize('threshold', [0, -1.0e-10])
def test_shapelet_psf(nepoch, threshold):
    rng = np.random.RandomState(2332)

    psf = make_shapelet_psf(
        rng=rng,
        nepoch=nepoch,
        threshold=threshold,
    )

    assert psf.nepoch == nepoch
    assert psf.threshold == threshold

    pos = galsim.PositionD(1.5, 2.2)
    _ = psf.getPSF(pos)
