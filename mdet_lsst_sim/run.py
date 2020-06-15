import sys
import copy
import logging
import numpy as np

from descwl_shear_sims import Sim
import descwl_coadd.vis
from descwl_coadd.coadd import MultiBandCoadds
from descwl_coadd.coadd_simple import MultiBandCoaddsSimple
from metadetect.lsst_metadetect import LSSTMetadetect, LSSTDeblendMetadetect
from metadetect.metadetect import Metadetect
import fitsio
import esutil as eu

from . import util, vis


def run(
    *,
    sim_config,
    seed,
    ntrial,
    output,
    mdet_config=None,
    full_output=False,
    show=False,
    show_sheared=False,
    show_masks=False,
    show_sim=False,
    nostack=False,
    trim_psf=False,
    use_sx=False,
    deblend=False,
    interp_bright=False,
    replace_bright=False,
    loglevel='info',
    max_mask_frac=0.1,
):
    """
    sim_config: dict
        Dict with configuration for the simulation
    mdet_config: dict
        Dict with mdet configuration, default None
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
    show_masks: bool
        If True, show the masks, default False
    show_sim: bool
        If True, show the sims.  default False
    nostack: bool
        If True, don't use any lsst stack code, default False
    trim_psf: bool
        If True, trim the psf to psf_dim/np.sqrt(3) to avoid bad pixels
        in coadded psf
    use_sx: bool
        If True, us sx for detection, default False
    deblend: bool
        If True, run the lsst deblender, default False
    interp_bright: bool
        If True, interpolate regions marked BRIGHT, default False.
    replace_bright: bool
        If True, replace regions marked BRIGHT with noise, default False.
    loglevel: string
        Log level, default 'info'
    max_mask_frac: float
        Maximum allowed masked fraction.  If the masked fraction
        exceeds this amount, the exposure is rejected.  The masked
        fraction is calculed base on the part of the images that
        are interpolated.
    """

    if sim_config is None:
        sim_config = {}

    sim_type = sim_config.pop('type', 'simple')
    assert sim_type == 'simple'

    rng = np.random.RandomState(seed)
    mdet_config = util.get_mdet_config(
        config=mdet_config,
        nostack=nostack,
        use_sx=use_sx,
    )
    print(mdet_config)

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger('mdet_lsst_sim')
    logger.setLevel(getattr(logging, loglevel.upper()))

    dlist_p = []
    dlist_m = []
    truth_summary_list = []

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

            sim_kw = copy.deepcopy(sim_config)
            sim_kw['g2'] = 0.0

            trial_rng = np.random.RandomState(trial_seed)

            if shear_type == '1p':
                sim_kw['g1'] = 0.02
            else:
                sim_kw['g1'] = -0.02

            sim = Sim(rng=trial_rng, **sim_kw)

            data = sim.gen_sim()

            if show_sim:
                vis.show_sim(data)

            if nostack:
                coadd_obs = MultiBandCoaddsSimple(data=data)

                coadd_mbobs = util.make_mbobs(coadd_obs)
                md = Metadetect(
                    mdet_config,
                    coadd_mbobs,
                    trial_rng,
                    show=send_show,
                )
            else:

                psf_dim = sim.psf_dim
                if trim_psf:
                    psf_dim = int(psf_dim/np.sqrt(3))
                    if psf_dim % 2 == 0:
                        psf_dim -= 1
                    logger.info(
                        "trimming psf %d -> %d" % (sim.psf_dim, psf_dim)
                    )

                mbc = MultiBandCoadds(
                    rng=trial_rng,
                    interp_bright=interp_bright,
                    replace_bright=replace_bright,
                    data=data,
                    coadd_wcs=sim.coadd_wcs,
                    coadd_dims=[sim.coadd_dim]*2,
                    psf_dims=[psf_dim]*2,
                    byband=False,
                    show=send_show,
                    loglevel=loglevel,
                )
                if show_masks and shear_type == '1p':
                    show_all_masks(mbc.exps)

                coadd_obs = mbc.coadds['all']

                logger.info('mask_frac: %g' % coadd_obs.meta['mask_frac'])

                coadd_mbobs = util.make_mbobs(coadd_obs)

                if use_sx:
                    md = Metadetect(
                        mdet_config,
                        coadd_mbobs,
                        trial_rng,
                        show=send_show,
                    )
                elif deblend:
                    md = LSSTDeblendMetadetect(
                        mdet_config,
                        coadd_mbobs,
                        trial_rng,
                        show=show_sheared,
                        loglevel=loglevel,
                    )
                else:
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

            if sim.object_data is not None:
                truth_summary = util.make_truth_summary(sim.object_data)

                if shear_type == '1p':
                    truth_summary_list.append(truth_summary)

            if len(comb_data) > 0:
                comb_data['star_density'] = sim.star_density
                if 'mask_frac' in coadd_obs.meta:
                    comb_data['mask_frac'] = coadd_obs.meta['mask_frac']

                if sim.object_data is not None:
                    comb_data['min_star_mag'] = (
                        truth_summary['min_star_mag'][0]
                    )

                if shear_type == '1p':
                    dlist_p.append(comb_data)
                else:
                    dlist_m.append(comb_data)

    data_1p = eu.numpy_util.combine_arrlist(dlist_p)
    data_1m = eu.numpy_util.combine_arrlist(dlist_m)
    if len(truth_summary_list) > 0:
        truth_summary = eu.numpy_util.combine_arrlist(truth_summary_list)

    if output is None:
        logger.info('doing dry run, not writing')
    else:
        logger.info('writing: %s' % output)
        with fitsio.FITS(output, 'rw', clobber=True) as fits:
            fits.write(data_1p, extname='1p')
            fits.write(data_1m, extname='1m')
            if len(truth_summary_list) > 0:
                fits.write(truth_summary, extname='truth_summary')


def show_all_masks(exps):
    for exp in exps:
        descwl_coadd.vis.show_image_and_mask(exp)
        if 'q' == input('hit a key (q to quit) '):
            raise KeyboardInterrupt('halting at user request')
