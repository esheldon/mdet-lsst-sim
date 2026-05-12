import functools

DEFAULT_MAX_NONGAUSS_FRAC = 0.005


def make_moffat_psf(
    rng,
    nepoch=1,
    rotate=True,
    max_nongauss_frac=DEFAULT_MAX_NONGAUSS_FRAC,
    fwhm_fac=None,
):
    """
    Load Moffat fits.

    Parameters
    ----------
    rng: np.random.RandomState
        The random state
    nepoch: int, optional
        If set greater than 1, stack that many PSFs
    rotate: bool, optional
        Whether to randomly rotate PSFs, default True
    max_nongauss_frac: float
        Maximum allowed fraction of the power in non-gaussian parts of
        a shapelets fit.  Default is 0.005
    fwhm_fac: float, optional
        A factor to scale all gaussian sizes.  Default is None, meaning
        do not scale.
    """
    import os

    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'moffat-psfs',
        'moffat-dp2-i.fits',
    )
    moffat_lib = MoffatLibrary(
        fname=fname,
        rng=rng,
        rotate=rotate,
        max_nongauss_frac=max_nongauss_frac,
        fwhm_fac=fwhm_fac,
    )

    return MoffatPSF(moffat_lib=moffat_lib, nepoch=nepoch)


class MoffatPSF:
    """
    Moffat based PSF objects

    Parameters
    ----------
    moffat_lib: MoffatLibrary
        The psf library.
    nepoch: int, optional
        If set greater than 1, stack that many PSFs
    """

    def __init__(self, moffat_lib, nepoch=1):
        self.moffat_lib = moffat_lib
        self.nepoch = int(nepoch)

        if self.nepoch < 1:
            raise ValueError(f'expected nepoch >= 1, got {nepoch}')

    def getPSF(self, pos):  # noqa: N802
        """
        Get a PSF as a galsim.GSObject.  Note the position is only
        used to get the local wcs

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.  This is only used to get the wcs.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism GSObject
        """

        for i in range(self.nepoch):
            tmp = self._get_one_psf(pos)

            if i == 0:
                psf = tmp
            else:
                psf += tmp

        return psf.withFlux(1.0)

    def _get_one_psf(self, pos):
        return self.moffat_lib.sample_psf()


class MoffatLibrary:
    """
    A library of Moffat objects.

    Parameters
    ----------
    fname: str
        Path to the library
    rng: np.random.RandomState
        The random state
    rotate: bool, optional
        Whether to randomly rotate PSFs, default True
    max_nongauss_frac: float
        Maximum allowed fraction of the power in non-gaussian parts of
        a shapelets fit.  Default is None, meaning do not make any cuts.
    fwhm_fac: float, optional
        A factor to scale all gaussian sizes.  Default is None, meaning
        do not scale.
    """

    def __init__(
        self,
        fname,
        rng,
        rotate=True,
        max_nongauss_frac=DEFAULT_MAX_NONGAUSS_FRAC,
        fwhm_fac=None,
    ):

        self.fname = fname
        self.rng = rng
        self.rotate = rotate
        self.max_nongauss_frac = max_nongauss_frac
        self.fwhm_fac = fwhm_fac

        self.data = cached_moffat_read(
            fname=self.fname,
            max_nongauss_frac=self.max_nongauss_frac,
        )

    def get_psf(self, idx):
        """
        Get the galsim GSObject for the specified index.

        Parameters
        ----------
        idx: int
            Integer of shapelet

        Returns
        -------
        galsim.GSObject
        """
        import galsim

        psf = self._get_gsobj(idx)

        if self.rotate:
            angle = self.rng.uniform(low=0, high=360)
            psf = psf.rotate(angle * galsim.degrees)

        return psf

    def _get_gsobj(self, idx):
        import galsim

        pars = self.data['moffat_pars'][idx]

        g1 = pars[2]
        g2 = pars[3]
        fwhm = pars[4]
        beta = pars[5]

        if self.fwhm_fac is not None:
            fwhm = fwhm * self.fwhm_fac

        gsobj = galsim.Moffat(
            beta=beta,
            fwhm=fwhm,
            flux=1.0,
        ).shear(
            g1=g1,
            g2=g2,
        )

        return gsobj

    def sample_psf(self):
        """
        Returns a randomly sampled PSF from the library.

        Returns
        -------
        galsim.GSObject
        """
        idx = self.rng.choice(len(self))
        return self.get_psf(idx)

    def __len__(self):
        return self.data.size

    def __repr__(self):
        return (
            f"GMixLibrary(n={len(self)}, "
            f"rotate={self.rotate} "
            f"source={self.fname})"
        )


@functools.lru_cache(maxsize=8)
def cached_moffat_read(fname, max_nongauss_frac):
    """
    Load the gmix library, possibly limiting the amount of non gaussian power
    """
    import numpy as np
    import fitsio
    from esutil.numpy_util import between

    date_min = 2026010100000
    date_max = 2026012700000

    print(f'loading {fname}')
    print(f'    max_nongauss_frac < {max_nongauss_frac}')
    print(f'    {date_min} <= date <= {date_max}')

    with fitsio.FITS(fname) as fits:
        data = fits[1].read()

    logic = between(
        data['visit'],
        date_min,
        date_max,
        type='[]',
    )

    logic &= (data['flags'] == 0)

    if max_nongauss_frac is not None:
        logic &= data['nongaussian_frac'] < max_nongauss_frac

    w, = np.where(logic)
    print(f'    keeping {w.size}/{data.size}')
    data = data[w]

    return data
