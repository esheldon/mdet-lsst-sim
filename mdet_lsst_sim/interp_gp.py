from descwl_coadd.defaults import FLAGS2INTERP

# we probably want to get this as a multiple of the PSF FWHM
DEFAULT_GP_FWHM = 15
DEFAULT_GP_METHOD = 'jax'
DEFAULT_GP_BIN_SPACING = 25
DEFAULT_GP_THRESHOLD_DYNAMIC_BINNING = 2000
DEFAULT_GP_THRESHOLD_SUBDIVIDE = 20000


class GPInterpolator(object):
    """
    A class wrapping the gprubin interpolator function that
    interpolates inplace and sets the INTRP bit

    This conforms to the interface required for the interpolator
    sent to descwl_coadd.coadd.warp_exposures

    This is a "functor" meaning the object can be called
        interpolator(exposure)
    """

    def __init__(
        self,
        bad_mask_planes=FLAGS2INTERP,
        fwhm=DEFAULT_GP_FWHM,
        method=DEFAULT_GP_METHOD,
        bin_spacing=DEFAULT_GP_BIN_SPACING,
        threshold_dynamic_binning=DEFAULT_GP_THRESHOLD_DYNAMIC_BINNING,
        threshold_subdivide=DEFAULT_GP_THRESHOLD_SUBDIVIDE,
    ):
        self.bad_mask_planes = bad_mask_planes
        self.fwhm = fwhm
        self.method = method
        self.bin_spacing = bin_spacing
        self.threshold_dynamic_binning = threshold_dynamic_binning
        self.threshold_subdivide = threshold_subdivide

    def __call__(self, exp):
        """
        Interpolate the exposure in place

        Parameters
        ----------
        exp: ExposureF
            The exposure object to interpolate.  The INTRP flag will be
            set for any pixels that are interpolated
        """
        from gprubin import interpolateOverDefectsGP

        interpolateOverDefectsGP(
            exp.maskedImage,
            fwhm=self.fwhm,
            badList=self.bad_mask_planes,
            method=self.method,
            bin_spacing=self.bin_spacing,
            threshold_dynamic_binning=self.threshold_dynamic_binning,
            threshold_subdivide=self.threshold_subdivide,
        )
