import esutil as eu
import ngmix


def get_config(nostack=False, use_sx=False):
    """
    metadetect configuration
    """
    config = {
        'bmask_flags': 0,
        'metacal': {
            'use_noise_image': True,
            'psf': 'fitgauss',
        },
        'psf': {
            'model': 'gauss',
            'lm_pars': {},
            'ntry': 2,
        },
        'weight': {
            'fwhm': 1.2,
        },
        'detect': {
            'thresh': 5.0,
        },
        'meds': {},
    }

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
            'filter_kernel':  [
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


def trim_output(data):
    cols2keep = [
        'flags',
        'row',
        'col',
        'wmom_s2n',
        'wmom_T_ratio',
        'wmom_g',
    ]

    return eu.numpy_util.extract_fields(data, cols2keep)


def make_comb_data(res, full_output=False):
    add_dt = [
        ('shear_type', 'S7'),
        ('star_density', 'f4'),
    ]

    dlist = []
    for stype in res.keys():
        data = res[stype]
        if data is not None:

            if not full_output:
                data = trim_output(data)

            newdata = eu.numpy_util.add_fields(data, add_dt)
            newdata['shear_type'] = stype
            dlist.append(newdata)

    if len(dlist) > 0:
        return eu.numpy_util.combine_arrlist(dlist)
    else:
        return []


def make_mbobs(obs):
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)
    return mbobs
