import functools

DEFAULT_MAX_NONGAUSS_FRAC = 0.005
DEFAULT_THRESHOLD = 0.0


def make_gmix_psf(
    rng,
    nepoch=1,
    rotate=True,
    max_nongauss_frac=DEFAULT_MAX_NONGAUSS_FRAC,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Load a compiled shapelet bvec library and reconstruct GalSim PSF objects.

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
        original shapelets.  Default is 0.005
    threshold: float, optional
        Lowest allowed value in origial shapelets reconstruction image.
        Default is None, meaning do not make any cuts.
    """
    import os

    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'shapelet_bvec_stacked',
        'gmix-5gauss-i.fits',
    )
    gmix_lib = GMixLibrary(
        fname=fname,
        rng=rng,
        rotate=rotate,
        max_nongauss_frac=max_nongauss_frac,
        threshold=threshold,
    )

    return GMixPSF(gmix_lib=gmix_lib, nepoch=nepoch)


class GMixPSF:
    """
    A gmix based PSF objects

    Parameters
    ----------
    fname:
        The psf object
    nepoch: int, optional
        If set greater than 1, stack that many PSFs
    """

    def __init__(self, gmix_lib, nepoch=1):
        self.gmix_lib = gmix_lib
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
        return self.gmix_lib.sample_psf()


class GMixLibrary:
    """
    A library of ngmix GMix objects.

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
        original shapelets.  Default is None, meaning do not make any cuts.
    threshold: float, optional
        Lowest allowed value in origial shapelets reconstruction image.
        Default is None, meaning do not make any cuts.
    """

    def __init__(
        self,
        fname,
        rng,
        rotate=True,
        max_nongauss_frac=DEFAULT_MAX_NONGAUSS_FRAC,
        threshold=DEFAULT_THRESHOLD,
    ):

        self.fname = fname
        self.rng = rng
        self.rotate = rotate
        self.max_nongauss_frac = max_nongauss_frac
        self.threshold = threshold

        self.data = cached_gmix_read(
            fname=self.fname,
            max_nongauss_frac=self.max_nongauss_frac,
            threshold=self.threshold,
        )

    def get_psf(self, idx):
        """
        Get the galsim GSObject for the specified index.
        This converts from the ngmix representation

        Parameters
        ----------
        idx: int
            Integer of shapelet

        Returns
        -------
        galsim.GSObject
        """
        import galsim
        import ngmix

        pars = self.data['pars'][idx]

        gmix = ngmix.GMix(pars=pars)
        psf = gmix.make_galsim_object()

        if self.rotate:
            angle = self.rng.uniform(low=0, high=360)
            psf = psf.rotate(angle * galsim.degrees)

        psf = psf.withFlux(1.0)
        return psf

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
def cached_gmix_read(fname, max_nongauss_frac, threshold):
    """
    Load the gmix library, possibly limiting the amount of non gaussian power
    and putting a threshold on the minval of the original shapelets
    reconstruction image
    """
    import numpy as np
    import fitsio

    print(
        f'loading {fname} with '
        f'max_nongauss_frac={max_nongauss_frac} threshold={threshold}'
    )

    with fitsio.FITS(fname) as fits:
        data = fits['fits'].read()

    if max_nongauss_frac is not None or threshold is not None:
        logic = np.ones(data.size, dtype=bool)

        if max_nongauss_frac is not None:
            logic &= (data['nongaussian_frac'] < max_nongauss_frac)

        if threshold is not None:
            logic &= (data['minval'] > threshold)

        data = data[logic]

    return data
