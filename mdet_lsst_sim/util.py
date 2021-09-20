from copy import deepcopy
import numpy as np
import esutil as eu
import ngmix
import metadetect

DEFAULT_COADD_CONFIG = {'nowarp': False, 'remove_poisson': False}


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


def get_mdet_config(config=None):
    """
    metadetect configuration
    """
    if config is None:
        config = {}

    return metadetect.lsst_metadetect.get_config(config)


def trim_output(data, meas_type):
    cols2keep_orig = [
        'flags',
        'row',
        'col',
        'mfrac',
        'ormask',
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
    meta,
    full_output=False,
):
    add_dt = [
        ('shear_type', 'S7'),
        ('true_star_density', 'f4'),
        ('mask_frac', 'f4'),
    ]

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
                meta=meta,
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


def add_truth_summary(data, star_catalog, meta):
    """
    get field stats and assign for each object
    """

    if star_catalog is not None:
        data['true_star_density'] = star_catalog.density

    data['mask_frac'] = meta['mask_frac']

    return data


def make_mbobs(obs):
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)
    return mbobs
