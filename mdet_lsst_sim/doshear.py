#!/usr/bin/env python
import os
import numpy as np
import fitsio
from esutil.numpy_util import between, combine_arrlist
import yaml

OUTDIR = 'mc-results'

NSIGMA = 3
perc = '99.7'


def get_flist(directory, limit=None):

    flist = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for basename in files:
            if '.fits' in basename:
                fname = os.path.join(root, basename)
                fname = os.path.abspath(fname)
                flist.append(fname)

    nf = len(flist)
    if nf == 0:
        raise RuntimeError('no files found')

    if limit is not None:
        flist = flist[:limit]

    return flist


def read_config(fname):
    print('reading config:', fname)
    with open(fname) as fobj:
        return yaml.safe_load(fobj)


def get_summed(data):
    sdata = data[0].copy()

    for n in sdata.dtype.names:
        sdata[n] = data[n].sum(axis=0)

    return sdata


def sub1(data, subdata):
    odata = data.copy()

    for n in data.dtype.names:
        odata[n] -= subdata[n]

    return odata


def get_m1_c1(data, nocancel):
    g1_1p = data['g_ns_sum_1p'][0]/data['wsum_ns_1p']
    g1_1p_1p = data['g_1p_sum_1p'][0]/data['wsum_1p_1p']
    g1_1m_1p = data['g_1m_sum_1p'][0]/data['wsum_1m_1p']

    R11_1p = (g1_1p_1p - g1_1m_1p)/0.02  # noqa

    s1_1p = g1_1p/R11_1p

    g1_1m = data['g_ns_sum_1m'][0]/data['wsum_ns_1m']
    g1_1p_1m = data['g_1p_sum_1m'][0]/data['wsum_1p_1m']
    g1_1m_1m = data['g_1m_sum_1m'][0]/data['wsum_1m_1m']

    R11_1m = (g1_1p_1m - g1_1m_1m)/0.02  # noqa

    s1_1m = g1_1m/R11_1m

    if nocancel:
        R11 = R11_1p
        m1 = s1_1p / 0.02 - 1
    else:
        R11 = 0.5*(R11_1p + R11_1m)  # noqa
        m1 = (s1_1p - s1_1m)/0.04 - 1

    c1 = (g1_1p + g1_1m)/2/R11

    return m1, c1, R11


def get_c2(data):
    g2_1p = data['g_ns_sum_1p'][1]/data['wsum_ns_1p']
    g2_1m = data['g_ns_sum_1m'][1]/data['wsum_ns_1m']

    c2 = (g2_1p + g2_1m)/2
    return c2


def get_jackknife_struct(data, err_err=False):
    dt = [
        ('s2n_min', 'f8'),
        ('s2n_max', 'f8'),
        ('max_mask_frac', 'f8'),
        ('max_mfrac', 'f8'),
        ('Tratio_min', 'f8'),
        ('max_star_density', 'f8'),

        ('R11', 'f8'),
        ('m1', 'f8'),
        ('m1err', 'f8'),
        ('c1', 'f8'),
        ('c1err', 'f8'),
        ('c2', 'f8'),
        ('c2err', 'f8'),
    ]
    if err_err:
        dt += [('m1err_err', 'f8')]

    jdata = np.zeros(1, dtype=dt)
    jdata['s2n_min'][0] = data['s2n_min'][0]
    jdata['s2n_max'][0] = data['s2n_max'][0]
    jdata['max_mask_frac'][0] = data['max_mask_frac'][0]
    jdata['max_mfrac'][0] = data['max_mfrac'][0]
    jdata['Tratio_min'][0] = data['Tratio_min'][0]
    jdata['max_star_density'][0] = data['max_star_density'][0]
    return jdata


def jackknife(data, nocancel, err_err=True):
    """
    jackknife the errors

    Parameters
    ----------
    data: array
        The data to be jackknifed
    nocancel: bool
        Don't use the noise cancelling trick
    err_err: bool
        Don't jackknife to get the error on the error.  Internal use only.
    """
    st = get_jackknife_struct(data, err_err=err_err)

    sdata = get_summed(data)
    m1, c1, R11 = get_m1_c1(sdata, nocancel=nocancel)  # noqa
    print('R11:', R11)

    c2 = get_c2(sdata)
    c2 /= R11

    m1vals = np.zeros(data.size)
    c1vals = np.zeros(data.size)
    c2vals = np.zeros(data.size)

    for i in range(m1vals.size):
        subdata = sub1(sdata, data[i])
        tm1, tc1, tR11 = get_m1_c1(subdata, nocancel=nocancel)
        tc2 = get_c2(subdata)
        m1vals[i] = tm1
        c1vals[i] = tc1
        c2vals[i] = tc2/tR11

    nchunks = m1vals.size
    fac = (nchunks-1)/float(nchunks)

    m1cov = fac*((m1 - m1vals)**2).sum()
    m1err = np.sqrt(m1cov)

    c1cov = fac*((c1 - c1vals)**2).sum()
    c1err = np.sqrt(c1cov)

    c2cov = fac*((c2 - c2vals)**2).sum()
    c2err = np.sqrt(c2cov)

    st['R11'] = R11
    st['m1'] = m1
    st['m1err'] = m1err
    st['c1'] = c1
    st['c1err'] = c1err
    st['c2'] = c2
    st['c2err'] = c2err

    if err_err:
        err_stlist = []
        for i in range(data.size):
            indices = np.array(
                [ii for ii in range(data.size) if ii != i]
            )
            tmpst = jackknife(data[indices], nocancel, err_err=False)
            err_stlist.append(tmpst)

        err_st = combine_arrlist(err_stlist)

        m1err_vals = err_st['m1err']
        m1err_cov = fac * ((m1err - m1err_vals)**2).sum()
        st['m1err_err'] = np.sqrt(m1err_cov)

    return st


def get_weights(data, ind, model, weight_type, sn, get_cov_weights=False):
    g_cov = data['%s_g_cov' % model]
    err_term = 0.5 * (g_cov[ind, 0, 0] + g_cov[ind, 1, 1])

    weights = 1.0/(sn**2 + err_term)
    if get_cov_weights:
        cov_weights = weights.copy()

    if weight_type == 'g':
        import ngmix
        prior = ngmix.priors.GPriorBA(0.3, rng=np.random.RandomState())
        g = data['%s_g' % model]
        pvals = prior.get_prob_array2d(g[ind, 0], g[ind, 1])
        weights *= pvals
    elif weight_type == 'covar':
        pass
    else:
        raise ValueError(f'bad weight_type {weight_type}')

    if get_cov_weights:
        return weights, cov_weights
    else:
        return weights


def get_sums(
    stype,
    s2n_min,
    s2n_max,
    Tratio_min,
    max_mask_frac,
    max_mfrac,
    max_star_density,
    require_primary,
    use_weights,
    weight_type,
    shapenoise,
    data,
):

    if 'wmom_g' in data.dtype.names:
        model = 'wmom'
    elif 'ksigma_g' in data.dtype.names:
        model = 'ksigma'
    elif 'am_g' in data.dtype.names:
        model = 'am'
    elif 'gap_g' in data.dtype.names:
        model = 'gap'
    elif 'pgauss_g' in data.dtype.names:
        model = 'pgauss'
    else:
        model = 'gauss'

    s2n = data['%s_s2n' % model]
    T_ratio = data['%s_T_ratio' % model]
    gvals = data['%s_g' % model]
    g = np.sqrt(gvals[:, 0]**2 + gvals[:, 1]**2)

    logic = (
        (data['shear_type'] == stype) &
        ((data['flags'] == 0) | (data['flags'] == 2**19)) &
        # (data['flags'] == 0) &
        between(s2n, s2n_min, s2n_max) &
        (T_ratio > Tratio_min) &
        # (g < 1) &
        (g < 3)
    )

    if 'bmask' in data.dtype.names:
        logic &= (data['bmask'] == 0)

    if 'primary' in data.dtype.names and require_primary:
        logic &= data['primary']

    logic &= (data['mask_frac'] < max_mask_frac)
    logic &= (data['mfrac'] < max_mfrac)
    logic &= (data['true_star_density'] < max_star_density)

    w, = np.where(logic)

    g_sum = np.zeros(2)
    wsum = 0.0

    if w.size > 0:

        if use_weights:
            wts = get_weights(
                data, w, model, weight_type=weight_type, sn=shapenoise,
            )
            g_sum[0] = (wts * gvals[w, 0]).sum()
            g_sum[1] = (wts * gvals[w, 1]).sum()
            wsum = wts.sum()
        else:
            g_sum[0] = gvals[w, 0].sum()
            g_sum[1] = gvals[w, 1].sum()
            wsum = w.size

    return g_sum, wsum


def get_key(
    s2n_min,
    s2n_max,
    Tratio_min,
    max_mask_frac,
    max_mfrac,
    max_star_density,
    use_weights,
):

    klist = []
    if use_weights:
        klist += ['weighted']

    klist += [
        f's2n-{s2n_min}-{s2n_max}',
        f'mask_frac-{max_mask_frac:.2f}',
        f'mfrac-{max_mfrac:.2f}',
        f'Tratio-{Tratio_min:.1f}',
        f'sdens-{max_star_density:03d}',
    ]
    return '-'.join(klist)


def get_mc_file(key, nocancel, weight_type, require_primary=True):
    nlist = []

    if nocancel:
        nlist += ['nocancel']

    if not require_primary:
        nlist += ['noprimary']

    if weight_type is not None:
        nlist += [weight_type+'weight']

    nlist += [
        'mc',
        key,
    ]
    fname = '-'.join(nlist) + '.fits'
    return os.path.join(OUTDIR, fname)


def get_sums_file(chunk):
    fname = f'sums-{chunk:06d}.fits'
    return fname


def get_sums_flist_file(chunk):
    fname = f'flist-{chunk:06d}.txt'
    return fname


def get_sums_script_file():
    return 'run.sh'


def get_doshear_condor_file():
    return 'doshear.condor'


def chunk_flist(flist, nchunks):

    nf = len(flist)
    chunksize = nf // nchunks
    extra_items = nf % nchunks

    flists = []

    start = 0
    for i in range(nchunks):

        this_chunksize = chunksize
        if i <= extra_items:
            this_chunksize += 1

        end = start + this_chunksize

        chunk = flist[start:end]
        flists.append(chunk)

        start = start + this_chunksize

    flists = [
        flist for flist in flists if len(flist) > 0
    ]
    return flists


def get_struct(
    s2n_min,
    s2n_max,
    max_mask_frac,
    max_mfrac,
    Tratio_min,
    max_star_density,
):
    dt = [
        ('s2n_min', 'f8'),
        ('s2n_max', 'f8'),
        ('max_mask_frac', 'f8'),
        ('max_mfrac', 'f8'),
        ('Tratio_min', 'f8'),
        ('max_star_density', 'f8'),

        ('g_ns_sum_1p', ('f8', 2)),
        ('g_1p_sum_1p', ('f8', 2)),
        ('g_1m_sum_1p', ('f8', 2)),

        ('wsum_ns_1p', 'f8'),
        ('wsum_1p_1p', 'f8'),
        ('wsum_1m_1p', 'f8'),

        ('g_ns_sum_1m', ('f8', 2)),
        ('g_1p_sum_1m', ('f8', 2)),
        ('g_1m_sum_1m', ('f8', 2)),

        ('wsum_ns_1m', 'f8'),
        ('wsum_1p_1m', 'f8'),
        ('wsum_1m_1m', 'f8'),
    ]

    data = np.zeros(1, dtype=dt)
    data['s2n_min'] = s2n_min
    data['s2n_max'] = s2n_max
    data['max_mask_frac'] = max_mask_frac
    data['max_mfrac'] = max_mfrac
    data['Tratio_min'] = Tratio_min
    data['max_star_density'] = max_star_density
    return data


def add_more_sums(sums, tsums):
    for key in tsums:
        if key not in sums:
            sums[key] = tsums[key]
        else:
            for datakey in tsums[key].dtype.names:
                if 'sum' in datakey:
                    # only need to add the sum columns, others indicate the
                    # cut which is constant
                    sums[key][datakey] += tsums[key][datakey]


def process_set(inputs):
    config, flist = inputs

    nf = len(flist)
    sums = {}
    for i, fname in enumerate(flist):
        print(f'{i+1}/{nf} {fname}')
        tsums = process_one(config, fname)
        if tsums is not None:
            add_more_sums(sums, tsums)

    return sums


def process_one(config, fname):

    # print(fname)

    try:
        with fitsio.FITS(fname) as fobj:
            data_1p = fobj['1p'].read()
            data_1m = fobj['1m'].read()
    except OSError as err:
        print(err)
        return None

    if data_1p is None or data_1m is None:
        return None

    use_weights = config['use_weights']
    weight_type = config.get('weight_type', 'g')

    data_dict = {}
    for s2n_min in config['s2n_min']:
        for s2n_max in config['s2n_max']:
            for Tratio_min in config['Tratio_min']:
                for max_mask_frac in config['max_mask_frac']:
                    for max_mfrac in config['max_mfrac']:
                        for max_star_density in config['max_star_density']:

                            d = get_struct(
                                s2n_min=s2n_min,
                                s2n_max=s2n_max,
                                max_mask_frac=max_mask_frac,
                                max_mfrac=max_mfrac,
                                Tratio_min=Tratio_min,
                                max_star_density=max_star_density,
                            )
                            for stype in ['noshear', '1p', '1m']:
                                for data, ext in [(data_1p, '1p'), (data_1m, '1m')]:  # noqa
                                    g_sum, wsum = get_sums(
                                        stype=stype,
                                        s2n_min=s2n_min,
                                        s2n_max=s2n_max,
                                        Tratio_min=Tratio_min,
                                        max_mask_frac=max_mask_frac,
                                        max_mfrac=max_mfrac,
                                        max_star_density=max_star_density,
                                        use_weights=use_weights,
                                        weight_type=weight_type,
                                        shapenoise=config['shapenoise'],
                                        require_primary=config['require_primary'],  # noqa
                                        data=data,
                                    )
                                    if stype == 'noshear':
                                        sn = 'ns'
                                    else:
                                        sn = stype

                                    d[f'g_{sn}_sum_{ext}'] = g_sum
                                    d[f'wsum_{sn}_{ext}'] = wsum

                            key = get_key(
                                s2n_min=s2n_min,
                                s2n_max=s2n_max,
                                Tratio_min=Tratio_min,
                                max_mask_frac=max_mask_frac,
                                max_mfrac=max_mfrac,
                                max_star_density=max_star_density,
                                use_weights=use_weights,
                            )
                            data_dict[key] = d

    return data_dict


def print_stats(st):
    print('m1err: %g +/- %g (%s%%)' % (
        st['m1err'][0]*NSIGMA, st['m1err_err'][0]*NSIGMA, perc)
    )

    print_range(st['m1'][0], st['m1err'][0], 'm1')
    print_range(st['c1'][0], st['c1err'][0], 'c1')
    print_range(st['c2'][0], st['c2err'][0], 'c2')


def print_range(val, err, name):
    """
    print the confidence range 99.7% confidence
    """

    tup = (
        val - err*NSIGMA,
        name,
        val + err*NSIGMA,
        perc,
    )

    print('%g < %s < %g (%s%%)' % tup)
