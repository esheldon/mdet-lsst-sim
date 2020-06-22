from copy import deepcopy
import numpy as np
import esutil as eu
import ngmix

DEFAULT_MDET_CONFIG = {
    "model": "wmom",
    "bmask_flags": 0,
    "metacal": {
        "use_noise_image": True,
        "psf": "fitgauss",
    },
    "psf": {
        "model": "gauss",
        "lm_pars": {},
        "ntry": 2,
    },
    "weight": {
        "fwhm": 1.2,
    },
    "detect": {
        "thresh": 10.0,
    },
    "meds": {},
}


def get_mdet_config(config=None, nostack=False, use_sx=False):
    """
    metadetect configuration
    """
    if config is None:
        config_in = {}
    else:
        config_in = deepcopy(config)

    config = deepcopy(DEFAULT_MDET_CONFIG)
    config.update(config_in)

    if nostack or use_sx:
        config['sx'] = {
            # in sky sigma
            # DETECT_THRESH
            'detect_thresh': 0.8,

            # Minimum contrast parameter for deblending
            # DEBLEND_MINCONT
            'deblend_cont': 0.00001,

            # minimum number of pixels above threshold
            # DETECT_MINAREA: 6
            'minarea': 4,

            'filter_type': 'conv',

            # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
            'filter_kernel': [
                [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
                [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
                [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
                [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
                [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
                [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
                [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            ]
        }

        config['meds'] = {
            'min_box_size': 32,
            'max_box_size': 256,

            'box_type': 'iso_radius',

            'rad_min': 4,
            'rad_fac': 2,
            'box_padding': 2,
        }
        # fraction of slice where STAR or TRAIL was set.  We may cut objects
        # detected there
        config['star_flags'] = 96

        # we don't interpolate over tapebumps
        config['tapebump_flags'] = 16384

        # things interpolated using the spline
        config['spline_interp_flags'] = 3155

        # replaced with noise
        config['noise_interp_flags'] = 908

        # pixels will have these flag set in the ormask if they were
        # interpolated plus adding in tapebump and star
        config['imperfect_flags'] = 20479

    return config


def trim_output(data, model):
    cols2keep_orig = [
        'flags',
        'row',
        'col',
        'ormask',
        '%s_s2n' % model,
        '%s_T_ratio' % model,
        '%s_g' % model,
        '%s_g_cov' % model,
    ]

    cols2keep = []
    for col in cols2keep_orig:
        if col in data.dtype.names:
            cols2keep.append(col)

    return eu.numpy_util.extract_fields(data, cols2keep)


def make_comb_data(res, model, full_output=False):
    add_dt = [
        ('shear_type', 'S7'),
        ('star_density', 'f4'),
        ('min_star_mag', 'f4'),
        ('mask_frac', 'f4'),
    ]

    dlist = []
    for stype in res.keys():
        data = res[stype]
        if data is not None:

            if not full_output:
                data = trim_output(data, model)

            newdata = eu.numpy_util.add_fields(data, add_dt)
            newdata['shear_type'] = stype
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
        ('type', 'S6'),
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


def make_truth_summary(object_data):
    nobj = len(object_data)

    obj0 = object_data[0]
    bands = list(obj0.keys())

    dt = [
        ('nobj', 'i8'),
        ('ngal', 'i8'),
        ('nstar', 'i8'),
        ('min_star_mag', 'f4'),
    ]

    data = np.zeros(1, dtype=dt)
    data['nobj'] = nobj
    data['nstar'] = 0
    data['ngal'] = 0
    data['min_star_mag'] = 9999.0

    for i, obj_data in enumerate(object_data):

        otype = obj_data[bands[0]]['type']

        if otype == 'galaxy':
            data['ngal'][0] += 1
        else:
            data['nstar'][0] += 1

            # ordered dict
            for iband, band in enumerate(obj_data):

                if otype == 'star':
                    mag = obj_data[band]['mag']
                    if mag < data['min_star_mag'][0]:
                        data['min_star_mag'][0] = mag

    return data


def make_mbobs(obs):
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)
    return mbobs
