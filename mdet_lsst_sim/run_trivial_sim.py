import sys
import logging
import numpy as np

from descwl_shear_sims import make_trivial_sim, get_trivial_sim_config
from descwl_coadd.coadd import MultiBandCoadds
from metadetect.lsst_metadetect import LSSTMetadetect
import fitsio
import esutil as eu

from . import util, vis


def run_trivial_sim(
    *,
    sim_config,
    seed,
    ntrial,
    output,
    mdet_config=None,
    full_output=False,
    show=False,
    show_sheared=False,
    show_sim=False,
    deblend=False,
    interp_bright=False,
    replace_bright=False,
    loglevel='info',
):
    """
    sim_config: dict
        Dict with sim configuration.
    mdet_config: dict
        Dict with mdet configuration.
    seed: int
        Seed for a random number generator
    ntrial: int
        Number of trials to run, paired by simulation plus and minus shear
    output: string
        Output file path.  If output is None, this is a dry
        run and no output is written.
    full_output: bool
        If True, write full output rather than trimming.  Default False
    show: bool
        If True, show some images.  Default False
    show_sheared: bool
        If True, show the sheared images, default False
    show_sim: bool
        If True, show the sims.  default False
    deblend: bool
        If True, run the lsst deblender, default False
    interp_bright: bool
        If True, interpolate regions marked BRIGHT, default False.
    replace_bright: bool
        If True, replace regions marked BRIGHT with noise, default False.
    loglevel: string
        Log level, default 'info'
    """

    assert deblend is False

    sim_type = sim_config.pop('type', 'trivial')
    assert sim_type == 'trivial'

    rng = np.random.RandomState(seed)

    mdet_config = util.get_mdet_config(config=mdet_config)
    sim_config = get_trivial_sim_config(config=sim_config)

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger('mdet_lsst_sim')
    logger.setLevel(getattr(logging, loglevel.upper()))

    logger.info(str(mdet_config))

    dlist_p = []
    dlist_m = []

    g2 = 0.0
    for trial in range(ntrial):
        logger.info('-'*70)
        logger.info('trial: %d/%d' % (trial+1, ntrial))

        trial_seed = rng.randint(0, 2**30)

        for shear_type in ('1p', '1m'):

            logger.info(str(shear_type))

            if shear_type == '1p':
                send_show = show
            else:
                send_show = False

            if shear_type == '1p':
                g1 = 0.02
            else:
                g1 = -0.02

            trial_rng = np.random.RandomState(trial_seed)
            sim_data = make_trivial_sim(
                rng=trial_rng,
                g1=g1,
                g2=g2,
                **sim_config
            )

            if show_sim:
                vis.show_sim(sim_data['band_data'])

            mbc = MultiBandCoadds(
                rng=trial_rng,
                interp_bright=interp_bright,
                replace_bright=replace_bright,
                data=sim_data['band_data'],
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_dims=sim_data['coadd_dims'],
                psf_dims=sim_data['psf_dims'],
                byband=False,
                show=send_show,
                loglevel=loglevel,
            )

            coadd_obs = mbc.coadds['all']

            logger.info('mask_frac: %g' % coadd_obs.meta['mask_frac'])

            coadd_mbobs = util.make_mbobs(coadd_obs)

            md = LSSTMetadetect(
                mdet_config,
                coadd_mbobs,
                trial_rng,
                show=show_sheared,
                loglevel=loglevel,
            )

            md.go()
            res = md.result

            comb_data = util.make_comb_data(res, full_output=full_output)

            if len(comb_data) > 0:
                if shear_type == '1p':
                    dlist_p.append(comb_data)
                else:
                    dlist_m.append(comb_data)

    data_1p = eu.numpy_util.combine_arrlist(dlist_p)
    data_1m = eu.numpy_util.combine_arrlist(dlist_m)

    if output is None:
        logger.info('doing dry run, not writing')
    else:
        logger.info('writing: %s' % output)
        with fitsio.FITS(output, 'rw', clobber=True) as fits:
            fits.write(data_1p, extname='1p')
            fits.write(data_1m, extname='1m')