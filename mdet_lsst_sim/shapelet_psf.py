import os
import functools
import numpy as np
import galsim


def make_shapelet_psf(
    rng, nepoch=1, threshold=0.0, rotate=True, max_nongauss_frac=None,
):
    """
    Load a compiled shapelet bvec library and reconstruct GalSim PSF objects.

    Parameters
    ----------
    rng: np.random.RandomState
        The random state
    rotate: bool, optional
        Whether to randomly rotate PSFs, default True
    nepoch: int, optional
        If set greater than 1, stack that many PSFs
    threshold: float, optional
        Only keep psfs that evaluate positive
    max_nongauss_frac: float
        Maximum allowed fraction of the power in non-gaussian parts of
        shapelets.  power.  Default is None, meaning do not make any cuts.
    """
    npz_path = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'shapelet_bvec_stacked',
        'shapelet_i_all.npz',
    )
    shlib = ShapeletPSFLibrary(
        npz_path=npz_path,
        rng=rng,
        rotate=rotate,
        max_nongauss_frac=max_nongauss_frac,
    )

    return ShapeletPSF(
        shlib=shlib,
        threshold=threshold,
        nepoch=nepoch,
    )


class ShapeletPSFLibrary:
    """
    Load a compiled shapelet bvec library and reconstruct GalSim PSF objects.

    Based on code by Tianqing Zhang

    Parameters
    ----------
    npz_path: str
        Path to a .npz file produced by ShapeletFitter
    rng: np.random.RandomState
        The random state
    rotate: bool, optional
        Whether to randomly rotate PSFs, default True
    max_nongauss_frac: float
        Maximum allowed fraction of the power in non-gaussian parts of
        shapelets.  power.  Default is None, meaning do not make any cuts.
    """

    def __init__(self, npz_path, rng, rotate=True, max_nongauss_frac=None):
        self.rotate = rotate
        self.max_nongauss_frac = max_nongauss_frac
        self.rng = rng

        data = cached_bvec_read(npz_path, max_nongauss_frac=max_nongauss_frac)

        self.bvec_all = data['bvec']  # (N, n_coeffs) float64
        self.sigma_all = data['sigma']  # (N,) float64
        self.bmax = int(data['bmax'])
        self.band = str(data['band'])
        self.visit = data['visit']  # (N,) int64
        self.detector = data['detector']  # (N,) int32
        self.npz_path = npz_path
        self._n = len(self.sigma_all)

    def get_psf(self, idx):
        """
        Get the galsim Shapelet specified index.

        Parameters
        ----------
        idx: int
            Integer of shapelet

        Returns
        -------
        galsim.Shapelet
        """

        bvec = self.bvec_all[idx]
        # bvec = self.bvec_all[idx].copy()
        # bvec[15:] = 0.0
        psf = galsim.Shapelet(
            float(self.sigma_all[idx]),
            self.bmax,
            bvec,
        )

        if self.rotate:
            angle = self.rng.uniform(low=0, high=360)
            psf = psf.rotate(angle * galsim.degrees)

        psf = psf.withFlux(1.0)
        return psf

    def sample_psf(self):
        """
        Returns a randomly sampled Shapelet from the library.

        Returns
        -------
        galsim.Shapelet
        """
        idx = self.rng.choice(self._n)
        return self.get_psf(idx)

    def draw(self, psf, n, pixel_scale=None, wcs=None):
        """
        Draw psf onto an ``n×n`` image and return the pixel array.
        """

        if wcs is None and pixel_scale is None:
            raise ValueError('send pixel_scale= or wcs=')

        gsimage = psf.drawImage(
            nx=n,
            ny=n,
            method='no_pixel',
            scale=pixel_scale,
            wcs=wcs,
        )
        gsimage.array[:, :] *= 1.0 / gsimage.array[:, :].sum()
        return gsimage

    def __len__(self):
        return self._n

    def __repr__(self):
        return (
            f"ShapeletPSFLibrary(band={self.band}, n={self._n}, "
            f"rotate={self.rotate} "
            f"bmax={self.bmax}, source={self.npz_path})"
        )


class ShapeletPSF(object):
    """
    A simple fixed PSF object

    Parameters
    ----------
    shlib: ShapletPSFLibrary
        The psf object
    nepoch: int, optional
        If set greater than 1, stack that many PSFs
    threshold: float, optional
        Only keep psfs that evaluate positive
    """

    def __init__(self, shlib, nepoch=1, threshold=0.0):
        self.shlib = shlib
        self.nepoch = int(nepoch)
        self.threshold = threshold

        if self.nepoch < 1:
            raise ValueError(f'expected nepoch >= 1, got {nepoch}')

        self.ntry = 0
        self.nskipped = 0

    def getPSF(self, pos):  # noqa: N802
        """
        Get a PSF as a galsim.Shapelets.  Note the position is only
        used to get the local wcs

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.  This is only used to get the wcs.

        Returns
        -------
        psf : galsim.Shapelets
            A representation of the PSF as a galism object.
        """

        while True:
            self.ntry += 1

            psf = self._get_psf(pos)

            gsimage = self.shlib.draw(
                psf,
                n=51,
                pixel_scale=0.2,
            )

            gsimage *= 1.0 / gsimage.array.sum()

            if gsimage.array.min() > self.threshold:
                break

            self.nskipped += 1

        return psf

    def _get_psf(self, pos):
        for i in range(self.nepoch):
            tmp = self._get_one_psf(pos)

            if i == 0:
                psf = tmp
            else:
                psf += tmp

        return psf.withFlux(1.0)

    def _get_one_psf(self, pos):
        return self.shlib.sample_psf()


@functools.lru_cache(maxsize=8)
def cached_bvec_read(fname, max_nongauss_frac):
    """
    Load the shapelets, possibly limiting the amount of non gaussian power
    """
    data = np.load(fname)

    output = {
        name: data[name].copy() for name in data.keys()
    }

    if max_nongauss_frac is not None:
        # the shapelet coefficient terms that couple with optics

        bvec_all = output['bvec']

        bvec_mask = list(range(6, 10)) + list(range(15, 24))

        # fractional power of these coefficients
        allpower = np.sum(bvec_all[:, :] ** 2, axis=1)
        nongaussian_power = np.sum(bvec_all[:, bvec_mask] ** 2, axis=1)

        nongaussian_frac = nongaussian_power / allpower

        ind, = np.where(nongaussian_frac < max_nongauss_frac)

        for name in ['bvec', 'sigma', 'visit', 'detector']:
            output[name] = output[name][ind]

    return output
