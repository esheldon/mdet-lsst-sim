from descwl_shear_sims.constants import SCALE, FIXED_PSF_FWHM


def make_coadd_ps_psf(
    *,
    rng,
    dim,
    pixel_scale=SCALE,
    psf_fwhm=FIXED_PSF_FWHM,
    variation_factor=1,
    nepoch=1,
):
    """
    get a power spectrum psf

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    dim: int
        Dimensions of image
    pixel_scale: float
        pixel scale
    psf_fwhm: float
       Median FWHM of PSF in units of arcsec
    variation_factor : float, optional
        This factor is used internally to scale the overall variance in the
        PSF shape power spectra and the change in the PSF size across the
        image. Setting this factor greater than 1 results in more variation
        and less than 1 results in less variation.
    nepoch: int, optional
        Number of epochs to stack

    Returns
    -------
    PowerSpectrumPSF
    """
    return CoaddPSPSF(
        rng=rng,
        im_width=dim,
        buff=dim/2,
        scale=pixel_scale,
        median_seeing=psf_fwhm,
        variation_factor=variation_factor,
        nepoch=nepoch,
    )


class CoaddPSPSF:
    """
    Coadded PS PSF

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance.
    im_width : float
        The width of the image in pixels.
    buff : int
        An extra buffer of pixels for things near the edge.
    scale : float
        The pixel scale of the image
    trunc : float
        The truncation scale for the shape/magnification power spectrum
        used to generate the PSF variation.
    noise_level : float, optional
        If not `None`, generate a noise field to add to the PSF images with
        desired noise. A value of 1e-2 generates a PSF image with an
        effective signal-to-noise of ~250.
    variation_factor : float, optional
        This factor is used internally to scale the overall variance in the
        PSF shape power spectra and the change in the PSF size across the
        image. Setting this factor greater than 1 results in more variation
        and less than 1 results in less variation.
    median_seeing : float, optional
        The approximate median seeing for the PSF.
    nepoch: int, optional
        Number of epochs to stack

    Methods
    -------
    getPSF(pos)
        Get a PSF model at a given position.
    """

    def __init__(
        self,
        rng,
        im_width,
        buff,
        scale,
        trunc=1,
        noise_level=None,
        variation_factor=1,
        median_seeing=0.8,
        nepoch=1,
    ):
        from descwl_shear_sims.psfs import PowerSpectrumPSF

        self.nepoch = int(nepoch)
        self.psfs = [
            PowerSpectrumPSF(
                rng=rng,
                im_width=im_width,
                buff=buff,
                scale=scale,
                trunc=trunc,
                variation_factor=variation_factor,
                median_seeing=median_seeing,
            )
            for i in range(nepoch)
        ]

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
            tmp_ps_psf = self.psfs[i].getPSF(pos)

            if i == 0:
                psf = tmp_ps_psf
            else:
                psf += tmp_ps_psf

        return psf.withFlux(1.0)
