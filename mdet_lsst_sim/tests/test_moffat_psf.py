import os
import numpy as np
import pytest
import galsim
from ..moffat_psf import (
    make_moffat_psf,
    DEFAULT_MAX_NONGAUSS_FRAC,
)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_moffat_psf_smoke():
    rng = np.random.RandomState(8221)

    psf = make_moffat_psf(rng=rng)
    assert psf.moffat_lib.max_nongauss_frac == DEFAULT_MAX_NONGAUSS_FRAC
    assert psf.nepoch == 1

    pos = galsim.PositionD(1.5, 2.2)
    opsf = psf.getPSF(pos)
    assert np.allclose(opsf.flux, 1.0)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('nepoch', [1, 15])
@pytest.mark.parametrize('max_nongauss_frac', [0.005, 0.01])
def test_moffat_psf(nepoch, max_nongauss_frac):
    rng = np.random.RandomState(7843)

    psf = make_moffat_psf(
        rng=rng,
        nepoch=nepoch,
        max_nongauss_frac=max_nongauss_frac,
    )
    assert psf.moffat_lib.max_nongauss_frac == max_nongauss_frac
    assert psf.nepoch == nepoch

    pos = galsim.PositionD(1.5, 2.2)
    opsf = psf.getPSF(pos)
    assert np.allclose(opsf.flux, 1.0)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('fwhm_fac', [0.8, 1.0])
def test_moffat_psf_fwhm_fac(fwhm_fac):

    seed = 21

    psf0 = make_moffat_psf(
        rng=np.random.RandomState(seed)
    )

    psf_rescaled = make_moffat_psf(
        rng=np.random.RandomState(seed),
        fwhm_fac=fwhm_fac,
    )

    # same rng seeds, so should give same object
    psf = psf0.moffat_lib.get_psf(3)
    psf_rescaled = psf_rescaled.moffat_lib.get_psf(3)
