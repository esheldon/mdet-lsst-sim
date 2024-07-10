from numba import njit
import esutil as eu
import numpy as np
from lsst.meas.algorithms import AccumulatorMeanStack
import lsst.afw.image as afw_image

from descwl_coadd.procflags import HIGH_MASKFRAC
from descwl_coadd.defaults import (
    FLAGS2INTERP,
    MAX_MASKFRAC,
)
from descwl_coadd.coadd import (
    check_max_maskfrac,
    check_psf_dims,
    get_coadd_center,
    get_coadd_psf_bbox,
    make_coadd_exposure,
    get_pbar,
    warp_exposures,
    get_bad_mask,
    warp_psf,
    add_all,
    extract_coadd_psf,
)
import logging

LOG = logging.getLogger('mdet_lsst_sim.coadd_missing')


def make_coadd_fill(
    exps, coadd_wcs, coadd_bbox, psf_dims, rng, remove_poisson, psfs=None,
    wcss=None, max_maskfrac=MAX_MASKFRAC, bad_mask_planes=FLAGS2INTERP,
    is_warps=False, interpolator=None, warper=None, mfrac_warper=None,
):
    """
    make a coadd from the input exposures, working in "online mode",
    adding each exposure separately.  This saves memory when
    the exposures are being read from disk

    Parameters
    ----------
    exps: list
        Either a list of exposures (if `is_warps` is True) or a list of
        PackedExposure objects (if `is_warps` is False).
    coadd_wcs: DM wcs object
        The target wcs
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system
    psf_dims: tuple
        The dimensions of the psf
    rng: np.random.RandomState
        The random number generator for making noise images (used only if
        `is_warps` is False.)
    remove_poisson: bool, optional
        If set to True, remove the poisson noise from the variance
        estimate.
    psfs: list of PSF objects, optional
        List of PSF objects. If None, then the PSFs will be extracted from the
        exposures provided in ``exps``.
    wcss: list of DM wcs objects, optional
        List of DM wcs objects. If None, then the WCSs will be extracted from
        the exposures provided in ``exps``.
    max_maskfrac: float, optional
        Maximum allowed masked fraction.  Images masked more than
        this will not be included in the coadd.  Must be in range
        [0, 1]
    bad_mask_planes: list of str, optional
        List of mask plane names to be considered as bad pixels. Pixels with
        these masks set will be interpolated over (if `is_warps` is True) and
        will count towards the calculation of masked fraction.
    is_warps: bool, optional
        If set to True the input argument ``exps`` are list of PackedExposure
        objects containing the warped exposure, noise, masked fraction and
        an exposure info table. If False, ``exps`` is a list of exposures.
    interpolator: interpolator object, optional
        An object or function used to interpolate pixels.
        Must be callable as interpolator(exposure)
    warper: afw_math.Warper, optional
        The warper to use for the PSF, and for image and noise if ``is_warps``
        is False.
    mfrac_warper: afw_math.Warper, optional
        The warper to use for the masked fraction image. Used only if
        ``is_warps`` is False.

    Returns
    -------
    coadd_data : dict
        A dict with keys and values:

            nkept: int
                Number of exposures deemed valid for coadding
            coadd_exp : ExposureF
                The coadded image.
            coadd_noise_exp : ExposureF
                The coadded noise image.
            coadd_psf_exp : ExposureF
                The coadded PSF image.
            coadd_mfrac_exp : ExposureF
                The fraction of SE images interpolated in each coadd pixel.
    """

    interp_flag = afw_image.Mask.getPlaneBitMask('INTRP')
    if True:
        afw_image.Mask.addMaskPlane('ALL_INTRP')
        bad_flag = afw_image.Mask.getPlaneBitMask('ALL_INTRP')
    else:
        bad_flag = 0

    check_max_maskfrac(max_maskfrac)

    filter_label = exps[0].getFilter()

    # this is the requested coadd psf dims
    check_psf_dims(psf_dims)

    # Get integer center of coadd and corresponding sky center.  This is used
    # to construct the coadd psf bounding box and to reconstruct the psfs
    coadd_cen_integer, coadd_cen_skypos = get_coadd_center(
        coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
    )

    coadd_psf_bbox = get_coadd_psf_bbox(cen=coadd_cen_integer, dim=psf_dims[0])
    coadd_psf_wcs = coadd_wcs

    # separately stack data, noise, and psf
    coadd_exp = make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label)
    coadd_noise_exp = make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label)
    coadd_psf_exp = make_coadd_exposure(
        coadd_psf_bbox, coadd_psf_wcs,
        filter_label,
    )
    coadd_mfrac_exp = make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label)

    coadd_dims = coadd_exp.image.array.shape
    stacker = make_stacker(
        coadd_dims=coadd_dims, bit_mask_value=bad_flag,
    )
    noise_stacker = make_stacker(
        coadd_dims=coadd_dims, bit_mask_value=bad_flag,
    )
    psf_stacker = make_stacker(
        coadd_dims=psf_dims, bit_mask_value=bad_flag,
    )
    mfrac_stacker = make_stacker(
        coadd_dims=coadd_dims, bit_mask_value=bad_flag,
    )

    # will zip these with the exposures to warp and add
    stackers = [stacker, noise_stacker, psf_stacker, mfrac_stacker]

    dlist, exp_infos = _get_warplists(
        exps=exps,
        psfs=psfs,
        wcss=wcss,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=rng,
        remove_poisson=remove_poisson,

        max_maskfrac=max_maskfrac,
        bad_mask_planes=bad_mask_planes,
        is_warps=is_warps,
        interpolator=interpolator,
        warper=warper,
        mfrac_warper=mfrac_warper,
    )

    exp_info = eu.numpy_util.combine_arrlist(exp_infos)

    if True:
        flagged_count = set_bad_bit(
            dlist=dlist, flags=interp_flag, bad_flag=bad_flag
        )
        weightlist = exp_info['weight']
    else:
        weightlist, flagged_count = get_weightlist(dlist, interp_flag)

    for data, weight in zip(dlist, weightlist):
        warps = [
            data['warp'], data['noise_warp'],
            data['psf_warp'], data['mfrac_warp'],
        ]
        add_all(stackers, warps, weight)

    wkept, = np.where(exp_info['flags'] == 0)
    nkept = wkept.size
    result = {'nkept': nkept, 'exp_info': exp_info}

    if nkept > 0:

        stacker.fill_stacked_masked_image(coadd_exp.maskedImage)
        noise_stacker.fill_stacked_masked_image(coadd_noise_exp.maskedImage)
        psf_stacker.fill_stacked_masked_image(coadd_psf_exp.maskedImage)
        mfrac_stacker.fill_stacked_masked_image(coadd_mfrac_exp.maskedImage)

        wall = np.where(flagged_count == nkept)
        coadd_exp.variance.array[wall] = np.inf

        LOG.info('making psf')
        psf = extract_coadd_psf(coadd_psf_exp)
        coadd_exp.setPsf(psf)
        coadd_noise_exp.setPsf(psf)
        coadd_mfrac_exp.setPsf(psf)

        result.update({
            'coadd_exp': coadd_exp,
            'coadd_noise_exp': coadd_noise_exp,
            'coadd_psf_exp': coadd_psf_exp,
            'coadd_mfrac_exp': coadd_mfrac_exp,
        })

    return result


def _get_warplists(
    exps,
    coadd_wcs,
    coadd_bbox,
    psf_dims,
    rng,
    remove_poisson,
    # below are optional make_coadd
    psfs,
    wcss,
    max_maskfrac,
    bad_mask_planes,
    is_warps,
    interpolator,
    warper,
    mfrac_warper,
):

    filter_label = exps[0].getFilter()

    if psfs is None:
        psfs = (exp.getPsf() for exp in exps)

    if wcss is None:
        wcss = (exp.getWcs() for exp in exps)

    exp_infos = []
    # warplist = []
    # noise_warplist = []
    # psf_warplist = []
    # mfrac_warplist = []
    dlist = []

    for iexp, (exp, psf, wcs) in enumerate(
        get_pbar(list(zip(exps, psfs, wcss)))
    ):

        if is_warps:
            # Load the individual warps from the PackedExposure object.
            warp = exp.warp
            noise_warp = exp.noise_warp
            mfrac_warp = exp.mfrac_warp
            this_exp_info = exp.exp_info
        else:
            warp, noise_warp, mfrac_warp, this_exp_info = warp_exposures(
                exp=exp, coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox, rng=rng,
                remove_poisson=remove_poisson, bad_mask_planes=bad_mask_planes,
                interpolator=interpolator, warper=warper,
                mfrac_warper=mfrac_warper,
            )

        if this_exp_info['exp_id'] == -9999:
            this_exp_info['exp_id'] = iexp

        # Append ``this_exp_info`` to ``exp_infos`` regardless of whether we
        # use this exposure or not.
        exp_infos.append(this_exp_info)

        # If WarpBoundaryError was raised, ``warp`` will be None and we move
        # on.
        if not warp:
            continue

        # Compute the maskfrac from the warp, so as to be consistent with
        # the implementation on the actual data. This is necessary to do here
        # because the warp is limited to the coadd bounding box (cell), but the
        # exp maskfrac is computed over the entire detector.
        _, maskfrac = get_bad_mask(warp, bad_mask_planes=bad_mask_planes)

        if maskfrac >= max_maskfrac:
            LOG.info("skipping %d maskfrac %f >= %f",
                     this_exp_info['exp_id'], maskfrac, max_maskfrac)
            this_exp_info['flags'][0] |= HIGH_MASKFRAC
            continue

        if this_exp_info['flags'][0] != 0:
            continue

        # Read in the ``medvar`` stored in the variance plane of
        # ``noise_warp`` and restore it to the ``variance`` plane of
        # ``warp``.
        medvar = noise_warp.variance.array[0, 0]
        noise_warp.variance.array[:, :] = warp.variance.array[:, :]

        psf_warp = warp_psf(
            psf=psf, wcs=wcs, coadd_wcs=coadd_wcs,
            coadd_bbox=coadd_bbox, warper=warper,
            psf_dims=psf_dims, var=medvar, filter_label=filter_label,
        )

        # warplist.append(warp)
        # noise_warplist.append(noise_warp)
        # psf_warplist.append(psf_warp)
        # mfrac_warplist.append(mfrac_warp)

        data = {
            'warp': warp,
            'noise_warp': noise_warp,
            'psf_warp': psf_warp,
            'mfrac_warp': mfrac_warp,
        }
        dlist.append(data)

    # return warplist, noise_warplist, psf_warplist, mfrac_warplist, exp_infos
    return dlist, exp_infos


def get_weightlist(dlist, flags):
    weightlist = []

    flagged_count = np.zeros(dlist[0]['warp'].image.array.shape, dtype='i2')

    for data in dlist:
        warp = data['warp']
        var = warp.variance.array

        weight = np.zeros(var.shape)
        wpos = np.where(var > 0)
        weight[wpos] = 1/var[wpos]
        weightlist.append(weight)

        wflagged = np.where(warp.mask.array & flags != 0)
        flagged_count[wflagged] += 1

    for data, weight in zip(dlist, weightlist):
        mask = data['warp'].mask.array
        _set_zero_weights(
            weight=weight,
            mask=mask,
            flags=flags,
            flagged_count=flagged_count,
            num=len(dlist),
        )

    return weightlist, flagged_count


@njit
def _set_zero_weights(weight, mask, flags, flagged_count, num):
    ny, nx = weight.shape

    for iy in range(ny):
        for ix in range(nx):
            if mask[iy, ix] & flags != 0:
                # if we have some other data to fill in the value, set weight
                # to zero for this image
                if flagged_count[iy, ix] < num:
                    weight[iy, ix] = 0


def set_bad_bit(dlist, flags, bad_flag):

    flagged_count = np.zeros(dlist[0]['warp'].image.array.shape, dtype='i2')

    for data in dlist:
        warp = data['warp']
        wflagged = np.where(warp.mask.array & flags != 0)
        flagged_count[wflagged] += 1

    for data in dlist:
        mask = data['warp'].mask.array
        _set_one_bad_bit(
            mask=mask,
            flags=flags,
            flagged_count=flagged_count,
            num=len(dlist),
            bad_flag=bad_flag,
        )

    return flagged_count


@njit
def _set_one_bad_bit(mask, flags, flagged_count, num, bad_flag):
    ny, nx = mask.shape

    for iy in range(ny):
        for ix in range(nx):
            if mask[iy, ix] & flags != 0:
                # if we have some other data to fill in the value, set
                # the bad flag
                if flagged_count[iy, ix] < num:
                    mask[iy, ix] |= bad_flag


def make_stacker(coadd_dims, bit_mask_value=0, stats_ctrl=None):
    """
    make an AccumulatorMeanStack to do online coadding

    Parameters
    ----------
    coadd_dims: tuple/list
        The coadd dimensions

    Returns
    -------
    lsst.meas.algorithms.AccumulatorMeanStack

    Notes
    -----
    bit_mask_value = 0 says no filtering on mask plane
    mask_threshold_dict={} says propagate all mask plane bits to the
        coadd mask plane
    mask_map=[] says do not remap any mask plane bits to new values; negotiable
    no_good_pixels_mask=None says use default NO_DATA mask plane in coadds for
        areas that have no inputs; shouldn't matter since no filtering
    calc_error_from_input_variance=True says use the individual
        variances planes to predict coadd variance
    compute_n_image=False says do not compute number of images input to
        each pixel
    """
    from lsst.meas.algorithms import AccumulatorMeanStack
    import lsst.afw.math as afw_math

    if stats_ctrl is None:
        stats_ctrl = afw_math.StatisticsControl()

    # TODO after sprint
    # Eli will fix bug and will set no_good_pixels_mask to None
    # return AccumulatorMeanStackUnbugged(
    return AccumulatorMeanStack(
        shape=coadd_dims,
        # bit_mask_value=0,
        bit_mask_value=bit_mask_value,
        mask_threshold_dict={},
        mask_map=[],
        # no_good_pixels_mask=None,
        no_good_pixels_mask=stats_ctrl.getNoGoodPixelsMask(),
        calc_error_from_input_variance=True,
        compute_n_image=False,
    )


class AccumulatorMeanStackUnbugged(AccumulatorMeanStack):
    def add_masked_image(self, masked_image, weight=1.0):
        """Add a masked image to the stack.

        Parameters
        ----------
        masked_image : `lsst.afw.image.MaskedImage`
            Masked image to add to the stack.
        weight : `float` or `np.ndarray`, optional
            Weight to apply for weighted mean.  If an array,
            must be same size and shape as input masked_image.
        """
        good_pixels = np.where(
            ((masked_image.mask.array & self.bit_mask_value) == 0)
            & np.isfinite(masked_image.mask.array)
        )

        if isinstance(weight, np.ndarray):
            uweight = weight[good_pixels]
        else:
            uweight = weight

        uweight2 = uweight ** 2

        self.sum_weight[good_pixels] += uweight
        self.sum_wdata[good_pixels] += (
            uweight*masked_image.image.array[good_pixels]
        )

        if self.compute_n_image:
            self.n_image[good_pixels] += 1

        if self.calc_error_from_input_variance:
            self.sum_w2var[good_pixels] += (
                uweight2*masked_image.variance.array[good_pixels]
            )
        else:
            self.sum_weight2[good_pixels] += uweight2
            self.sum_wdata2[good_pixels] += (
                uweight*(masked_image.image.array[good_pixels]**2.)
            )

        # Mask bits are propagated for good pixels
        self.or_mask[good_pixels] |= masked_image.mask.array[good_pixels]

        # Bad pixels are only tracked if they cross a threshold
        for bit in self.mask_threshold_dict:
            bad_pixels = ((masked_image.mask.array & 2**bit) > 0)
            self.rejected_weights_by_bit[bit][bad_pixels] += uweight
            self.masked_pixels_mask[bad_pixels] |= 2**bit
