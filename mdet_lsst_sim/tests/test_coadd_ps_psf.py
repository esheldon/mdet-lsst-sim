import numpy as np
import pytest
import galsim
from ..coadd_ps_psf import make_coadd_ps_psf


def test_gmix_psf_smoke():
    rng = np.random.RandomState(4522)
    dim = 100
    nepoch = 3

    psf = make_coadd_ps_psf(rng=rng, dim=dim, nepoch=nepoch)
    assert len(psf.psfs) == nepoch

    pos = galsim.PositionD(1.5, 2.2)
    opsf = psf.getPSF(pos)
    assert np.allclose(opsf.flux, 1.0)


@pytest.mark.parametrize('nepoch', [1, 15])
def test_shapelet_psf(nepoch):
    rng = np.random.RandomState(7843)

    dim = 100
    psf = make_coadd_ps_psf(
        rng=rng,
        dim=dim,
        nepoch=nepoch,
    )
    assert len(psf.psfs) == nepoch

    pos = galsim.PositionD(1.5, 2.2)
    opsf = psf.getPSF(pos)
    assert np.allclose(opsf.flux, 1.0)
