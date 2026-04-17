import os
import numpy as np
import pytest
import galsim
from ..gmix_psf import (
    make_gmix_psf,
    DEFAULT_MAX_NONGAUSS_FRAC,
    DEFAULT_THRESHOLD,
)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_gmix_psf_smoke():
    rng = np.random.RandomState(8221)

    psf = make_gmix_psf(rng=rng)
    assert psf.gmix_lib.max_nongauss_frac == DEFAULT_MAX_NONGAUSS_FRAC
    assert psf.gmix_lib.threshold == DEFAULT_THRESHOLD
    assert psf.nepoch == 1

    pos = galsim.PositionD(1.5, 2.2)
    opsf = psf.getPSF(pos)
    assert np.allclose(opsf.flux, 1.0)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('nepoch', [1, 15])
@pytest.mark.parametrize('threshold', [0, -1.0e-10])
@pytest.mark.parametrize('max_nongauss_frac', [0.005, 0.01])
def test_shapelet_psf(nepoch, threshold, max_nongauss_frac):
    rng = np.random.RandomState(7843)

    psf = make_gmix_psf(
        rng=rng,
        nepoch=nepoch,
        threshold=threshold,
        max_nongauss_frac=max_nongauss_frac,
    )
    assert psf.gmix_lib.max_nongauss_frac == max_nongauss_frac
    assert psf.gmix_lib.threshold == threshold
    assert psf.nepoch == nepoch

    pos = galsim.PositionD(1.5, 2.2)
    opsf = psf.getPSF(pos)
    assert np.allclose(opsf.flux, 1.0)
