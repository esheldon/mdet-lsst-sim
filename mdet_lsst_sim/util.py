import logging
from copy import deepcopy
import numpy as np
import esutil as eu
import ngmix

logger = logging.getLogger('mdet_lsst.util')

DEFAULT_COADD_CONFIG = {'nowarp': False, 'remove_poisson': False}

DEFAULT_MDET_CONFIG_WITH_SX = {
    "model": "wmom",

    "weight": {
        "fwhm": 1.2,  # arcsec
    },

    "metacal": {
        "psf": "fitgauss",
        "types": ["noshear", "1p", "1m", "2p", "2m"],
    },

    "sx": {
        # in sky sigma
        # DETECT_THRESH
        "detect_thresh": 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        "deblend_cont": 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        "minarea": 4,

        "filter_type": "conv",

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        "filter_kernel": [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
        ]
    },

    "meds": {
        "min_box_size": 48,
        "max_box_size": 48,

        "box_type": "iso_radius",

        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },

    # needed for PSF symmetrization
    "psf": {
        "model": "am",
        "ntry": 2,
    },

    # check for an edge hit
    "bmask_flags": 2**30,

    "maskflags": 2**0,
}


def get_coadd_config(config=None):
    """
    metadetect configuration
    """
    if config is None:
        config_in = {}
    else:
        config_in = deepcopy(config)

    config = deepcopy(DEFAULT_COADD_CONFIG)
    config.update(config_in)
    return config


def get_mdet_config(config=None, sx=False):
    """
    metadetect configuration
    """
    from metadetect.lsst.metadetect import get_config
    if config is None:
        config_in = {}
    else:
        config_in = deepcopy(config)

    use_sx = config_in.pop('use_sx', False)
    trim_pixels = config_in.pop('trim_pixels', 0)

    if use_sx:
        config_out = deepcopy(DEFAULT_MDET_CONFIG_WITH_SX)
        config_out.update(config_in)
        config_out['meas_type'] = config_out['model']
    else:
        config_out = get_config(config_in)

    return config_out, use_sx, trim_pixels


def trim_output(data, meas_type):
    if meas_type == 'admom':
        meas_type = 'am'

    cols2keep_orig = [
        'flags',
        'row_noshear',
        'col_noshear',
        # 'row', 'row0',
        # 'col', 'col0',
        'mfrac',
        '%s_s2n' % meas_type,
        '%s_T_ratio' % meas_type,
        '%s_g' % meas_type,
        '%s_g_cov' % meas_type,
    ]

    cols2keep = []
    for col in cols2keep_orig:
        if col in data.dtype.names:
            cols2keep.append(col)

    return eu.numpy_util.extract_fields(data, cols2keep)


def make_comb_data(
    res,
    meas_type,
    star_catalog,
    mask_frac,
    full_output=False,
):
    add_dt = [
        ('shear_type', 'U7'),
        ('true_star_density', 'f4'),
        ('mask_frac', 'f4'),
    ]

    if not hasattr(res, 'keys'):
        res = {'noshear': res}

    dlist = []
    for stype in res.keys():
        data = res[stype]
        if data is not None:

            if not full_output:
                data = trim_output(data, meas_type)

            newdata = eu.numpy_util.add_fields(data, add_dt)
            newdata['shear_type'] = stype
            add_truth_summary(
                data=newdata,
                star_catalog=star_catalog,
                mask_frac=mask_frac,
            )
            dlist.append(newdata)

    if len(dlist) > 0:
        return eu.numpy_util.combine_arrlist(dlist)
    else:
        return []


def make_truth_data_full(object_data):

    nobj = len(object_data)

    obj0 = object_data[0]
    bands = list(obj0.keys())

    nband = len(bands)

    dt = [
        ('type', 'U6'),
        ('mag', 'f4', nband),
    ]
    data = np.zeros(nobj, dtype=dt)
    data['mag'] = 9999.0

    for i, obj_data in enumerate(object_data):

        otype = obj_data[bands[0]]['type']
        data['type'][i] = otype

        if otype == 'star':
            # ordered dict
            for iband, band in enumerate(obj_data):
                data['mag'][i, iband] = obj_data[band]['mag']

    return data


def add_truth_summary(data, star_catalog, mask_frac):
    """
    get field stats and assign for each object
    """

    if star_catalog is not None:
        data['true_star_density'] = star_catalog.density

    data['mask_frac'] = mask_frac

    return data


def make_mbobs(obs):
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)
    return mbobs


def get_command_from_config_file(config_file):
    import yaml
    with open(config_file) as fobj:
        config = yaml.load(fobj, Loader=yaml.SafeLoader)

    if config.get('do_photometry', False):
        command = 'mdet-lsst-sim-phot'
    else:
        command = 'mdet-lsst-sim'

    return command


def coadd_sim_data(rng, sim_data, nowarp, remove_poisson):
    from descwl_coadd.coadd import make_coadd
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp
    from metadetect.lsst.util import extract_multiband_coadd_data

    bands = list(sim_data['band_data'].keys())

    if nowarp:
        exps = sim_data['band_data'][bands[0]]

        if len(exps) > 1:
            raise ValueError('only one epoch for nowarp')

        coadd_data_list = [
            make_coadd_nowarp(
                exp=exps[0],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=remove_poisson,
            )
            for band in bands
        ]
    else:
        coadd_data_list = [
            make_coadd(
                exps=sim_data['band_data'][band],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                remove_poisson=remove_poisson,
            )
            for band in bands
        ]
    return extract_multiband_coadd_data(coadd_data_list)


def get_mask_frac(mfrac_mbexp, stamp_size, trim_pixels=0):
    """
    get the average mask frac for each band and then return the max of those
    """

    trim = trim_pixels + stamp_size // 2

    mask_fracs = []
    for mfrac_exp in mfrac_mbexp:
        mfrac = mfrac_exp.image.array
        dim = mfrac.shape[0]
        mfrac = mfrac[
            trim:dim - trim - 1,
            trim:dim - trim - 1,
        ]
        mask_fracs.append(mfrac.mean())

    return max(mask_fracs)


def trim_catalog_boundary(data, dim, trim_pixels, show=False):

    row = data['row_noshear']
    col = data['col_noshear']
    # row = data['row'] - data['row0']
    # col = data['col'] - data['col0']

    w, = np.where(
        (row > trim_pixels) &
        (col > trim_pixels) &
        (row < (dim - trim_pixels - 1)) &
        (col < (dim - trim_pixels - 1))
    )

    if show:
        import matplotlib.pyplot as mplt
        from matplotlib.patches import Rectangle

        fig, ax = mplt.subplots()
        logger.info('kept: %d/%d', w.size, data.size)
        alpha = 0.1
        ax.add_patch(
            Rectangle(
                [trim_pixels]*2,
                dim-2*trim_pixels,
                dim-2*trim_pixels,
                fill=False,
            ),
        )
        ax.scatter(col, row, color='red', alpha=alpha)
        ax.scatter(col[w], row[w], color='blue', alpha=alpha)
        mplt.show()

    return data[w]


def get_sim_shear(rng, shear, randomize_shear):
    if randomize_shear:
        theta = rng.uniform(low=0, high=np.pi)

        g1, g2 = ngmix.shape.rotate_shape(g1=shear, g2=0, theta=theta)
    else:
        g1 = shear
        g2 = 0.0
        theta = None

    return g1, g2, theta


def unrotate_noshear_shear(data, meas_type, theta):
    """
    only unrotate the noshear version
    """
    w, = np.where((data['flags'] == 0) & (data['shear_type'] == 'noshear'))

    if w.size > 0:

        gname = f'{meas_type}_g'

        g1, g2 = ngmix.shape.rotate_shape(
            g1=data[gname][w, 0],
            g2=data[gname][w, 1],
            theta=-theta,
        )

        data[gname][w, 0] = g1
        data[gname][w, 1] = g2


def extract_cell_coadd_data(
    coadd_data, cell_size, cell_buff, cell_ix, cell_iy,
):
    """
    Parameters
    ----------
    coadd_data: dict
        outpout of metadetect.lsst.util.extract_multiband_coadd_data
    cell_size: int
        Size of the cell in pixels
    cell_buff: int
        Overlap buffer for cells
    cell_ix: int
        The index of the cell for x
    cell_iy: int
        The index of the cell for y

    Returns
    --------
    dict
    """

    output = {}

    start_x, start_y = get_cell_start(
        cell_size=cell_size, cell_buff=cell_buff,
        cell_ix=cell_ix, cell_iy=cell_iy,
    )

    for key, item in coadd_data.items():
        if key == 'ormasks':
            new_item = [
                ormask[
                    start_y:start_y+cell_size,
                    start_x:start_x+cell_size
                ].copy()
                for ormask in item
            ]
        else:
            new_item = extract_cell_mbexp(
                mbexp=item, cell_size=cell_size, cell_buff=cell_buff,
                cell_ix=cell_ix,
                cell_iy=cell_iy,
            )
        output[key] = new_item

    return output


def extract_cell_mbexp(mbexp, cell_size, cell_buff, cell_ix, cell_iy):
    """
    extract a sub mbexp for the specified cell

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The image data
    cell_size: int
        Size of the cell in pixels
    cell_buff: int
        Overlap buffer for cells
    cell_ix: int
        The index of the cell for x
    cell_iy: int
        The index of the cell for y

    Returns
    --------
    mbexp: lsst.afw.image.MultibandExposure
        The sub-mbexp for the cell
    """
    import lsst.geom as geom
    from metadetect.lsst.util import get_mbexp, copy_mbexp

    start_x, start_y = get_cell_start(
        cell_size=cell_size, cell_buff=cell_buff,
        cell_ix=cell_ix, cell_iy=cell_iy,
    )

    bbox_begin = mbexp.getBBox().getBegin()

    new_begin = geom.Point2I(
        x=bbox_begin.getX() + start_x,
        y=bbox_begin.getY() + start_y,
    )
    extent = geom.Extent2I(cell_size)
    new_bbox = geom.Box2I(
        new_begin,
        extent,
    )

    subexps = []
    for band in mbexp.filters:
        exp = mbexp[band]
        # we need to make a copy of it
        subexp = exp[new_bbox]
        assert np.all(
            exp.image.array[
                start_y:start_y+cell_size,
                start_x:start_x+cell_size
            ] == subexp.image.array[:, :]
        )

        subexps.append(subexp)

    # subexps = [mbexp[band][new_bbox] for band in mbexp.filters]
    return copy_mbexp(get_mbexp(subexps))


def get_cell_start(cell_size, cell_buff, cell_ix, cell_iy):
    start_x = cell_ix*(cell_size - 2*cell_buff)
    start_y = cell_iy*(cell_size - 2*cell_buff)
    return start_x, start_y
