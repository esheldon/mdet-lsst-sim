"""
TODO

  - adapt to exposure based metadetect
  - use new bright info to mask before sending data to metadetect
  - make sure overall mask frac gets passed on; this should combine
    bands; maybe do in metadetect.
  - masking is different; make sure doshear is not bothering with mask/ormask
    just with mask_frac and mfrac
"""

import sys
import time
import logging
import numpy as np

from descwl_shear_sims.sim import (
    make_sim,
    get_sim_config,
    get_se_dim,
)
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import make_star_catalog
import metadetect.lsst.vis as lsst_vis
from metadetect.lsst.metadetect import run_metadetect
from metadetect.lsst.masking import (
    apply_apodized_edge_masks_mbexp,
    apply_apodized_bright_masks_mbexp,
    AP_RAD,
)
from metadetect.masking import get_ap_range
import fitsio
import esutil as eu

from . import util, vis


def run_sim(
    *,
    seed,
    ntrial,
    output,
    sim_config=None,
    mdet_config=None,
    coadd_config=None,
    shear=0.02,
    randomize_shear=True,
    nocancel=False,
    full_output=False,
    show=False,
    show_sheared=False,
    show_sim=False,
    loglevel='info',
):
    """
    seed: int
        Seed for a random number generator
    ntrial: int
        Number of trials to run, paired by simulation plus and minus shear
    output: string
        Output file path.  If output is None, this is a dry
        run and no output is written.
    sim_config: dict, optional
        Dict with sim configuration.
    mdet_config: dict, optional
        Dict with mdet configuration.
    coadd_config: dict, optional
        Dict with coadd configuration.
    shear: float, optional
        Magnitude of the shear.  Shears +/- shear will be applied
    nocancel: bool, optional
        If True, don't run -shear
    full_output: bool, optional
        If True, write full output rather than trimming.  Default False
    show: bool, optional
        If True, show some images.  Default False
    show_sheared: bool, optional
        If True, show the sheared images, default False
    show_sim: bool, optional
        If True, show the sims.  default False
    loglevel: str
        e.g. 'info'
    """
    tm0 = time.time()
    tmsim = 0.0
    tmcoadd = 0.0
    tmmeas = 0.0

    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, loglevel.upper()),
    )
    ap_padding = get_ap_range(AP_RAD)

    logger = logging.getLogger('mdet_lsst_sim')
    logger.info(f"seed: {seed}")

    rng = np.random.RandomState(seed)

    mdet_config, use_sx, trim_pixels = util.get_mdet_config(config=mdet_config)

    coadd_config = util.get_coadd_config(config=coadd_config)
    assert not coadd_config['remove_poisson'], (
        'Do not set remove_poisson=True in the coadd config; '
        'there is no poisson noise in the sim'
    )

    sim_config = get_sim_config(config=sim_config)

    if sim_config['se_dim'] is None:
        sim_config['se_dim'] = get_se_dim(
            coadd_dim=sim_config['coadd_dim'],
            dither=sim_config['dither'],
            rotate=sim_config['rotate'],
        )

    if sim_config['gal_type'] != 'wldeblend':
        gal_config = sim_config.get('gal_config', None)
    else:
        logger.info("setting wldeblend layout to None")
        sim_config["layout"] = None
        gal_config = None

    star_config = sim_config.get('star_config', None)

    logger.info(str(sim_config))
    logger.info(str(mdet_config))

    dlist_p = []
    dlist_m = []

    if nocancel:
        shear_types = ('1p', )
    else:
        shear_types = ('1p', '1m')

    infolist = []

    for trial in range(ntrial):
        logger.info('-'*70)
        logger.info('trial: %d/%d' % (trial+1, ntrial))

        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type=sim_config['gal_type'],
            coadd_dim=sim_config['coadd_dim'],
            buff=sim_config['buff'],
            layout=sim_config['layout'],
            sep=sim_config['sep'],  # for layout='pair'
            gal_config=gal_config,
        )

        if sim_config['stars']:
            star_catalog = make_star_catalog(
                rng=rng,
                coadd_dim=sim_config['coadd_dim'],
                buff=sim_config['buff'],
                star_config=star_config,
            )
            star_density = star_catalog.density
            logger.info('star_density: %g' % star_catalog.density)
        else:
            star_density = 0
            star_catalog = None

        trial_seed = rng.randint(0, 2**30)

        for shear_type in shear_types:

            logger.info(str(shear_type))

            trial_rng = np.random.RandomState(trial_seed)

            if shear_type == '1p':
                this_shear = shear
            else:
                this_shear = -shear

            g1, g2, theta = util.get_sim_shear(
                rng=trial_rng, shear=this_shear,
                randomize_shear=randomize_shear,
            )
            logger.info('shear: %.3f, %.3f theta: %s', g1, g2, theta)

            if sim_config['psf_type'] == 'ps':
                psf = make_ps_psf(rng=trial_rng, dim=sim_config['se_dim'])
            else:
                psf = make_fixed_psf(psf_type=sim_config["psf_type"])

            tmsim0 = time.time()
            sim_data = make_sim(
                rng=trial_rng,
                galaxy_catalog=galaxy_catalog,
                coadd_dim=sim_config['coadd_dim'],
                se_dim=sim_config['se_dim'],
                g1=g1,
                g2=g2,
                psf=psf,
                star_catalog=star_catalog,
                draw_stars=sim_config['draw_stars'],
                psf_dim=sim_config['psf_dim'],
                dither=sim_config['dither'],
                rotate=sim_config['rotate'],
                bands=sim_config['bands'],
                epochs_per_band=sim_config['epochs_per_band'],
                noise_factor=sim_config['noise_factor'],
                cosmic_rays=sim_config['cosmic_rays'],
                bad_columns=sim_config['bad_columns'],
                star_bleeds=sim_config['star_bleeds'],
                sky_n_sigma=sim_config['sky_n_sigma'],
            )
            tmsim += time.time() - tmsim0

            if show_sim:
                vis.show_sim(sim_data['band_data'])

            tmcoadd0 = time.time()
            coadd_data = util.coadd_sim_data(
                rng=trial_rng, sim_data=sim_data,
                **coadd_config
            )

            if show:
                if len(coadd_data['mbexp']) >= 3:
                    # from metadetect.lsst.util import get_mbexp
                    # exps = [bd[0] for band, bd in sim_data['band_data'].items()]
                    # sim_mbexp = get_mbexp(exps[:3])
                    # lsst_vis.show_mbexp(sim_mbexp)
                    lsst_vis.show_mbexp(
                        coadd_data['mbexp'],
                        stretch=3,
                        # q=20,
                    )
                else:
                    lsst_vis.show_multi_mbexp(coadd_data['mbexp'])

            apply_apodized_edge_masks_mbexp(**coadd_data)
            if len(sim_data['bright_info']) > 0:
                # Note padding due to apodization, otherwise we get donuts the
                # radii coming out of the sim code are not super conservative,
                # just going to the noise level
                sim_data['bright_info']['radius_pixels'] += ap_padding
                apply_apodized_bright_masks_mbexp(
                    bright_info=sim_data['bright_info'],
                    **coadd_data
                )
                if show:
                    lsst_vis.show_multi_mbexp(coadd_data['mbexp'])

            mask_frac = util.get_mask_frac(
                coadd_data['mfrac_mbexp'],
                trim_pixels=trim_pixels,
            )
            if shear_type == '1p':
                info = util.make_info()
                info['star_density'] = star_density
                info['mask_frac'] = mask_frac
                infolist.append(info)

            tmcoadd += time.time() - tmcoadd0

            logger.info('mask_frac: %g' % mask_frac)

            if mask_frac == 1:
                continue

            tmmeas0 = time.time()

            if use_sx:
                raise RuntimeError("adapt sx run to exposures")
                # res = run_metadetect_sx(
                #     config=mdet_config,
                #     mbobs=coadd_mbobs,
                #     rng=trial_rng,
                # )
            else:
                res = run_metadetect(
                    rng=trial_rng, config=mdet_config, show=show_sheared,
                    **coadd_data,
                )
            tmmeas += time.time() - tmmeas0

            # res is a dict, so len(res) means no keys
            if res is None or len(res) == 0:
                continue

            comb_data = util.make_comb_data(
                res=res,
                meas_type=mdet_config['meas_type'],
                star_catalog=star_catalog,
                mask_frac=mask_frac,
                full_output=full_output,
            )

            if len(comb_data) > 0:
                if theta is not None:
                    util.unrotate_noshear_shear(
                        comb_data, meas_type=mdet_config['meas_type'],
                        theta=theta,
                    )

                if trim_pixels > 0:
                    good = util.trim_catalog_boundary_strict(
                        data=comb_data,
                        dim=sim_config['coadd_dim'],
                        trim_pixels=trim_pixels,
                        checks=['l', 'r', 'u', 'd'],
                        show=show,
                    )
                    comb_data['primary'][good] = True
                else:
                    comb_data['primary'] = True

                if shear_type == '1p':
                    dlist_p.append(comb_data)
                else:
                    dlist_m.append(comb_data)

    tm_seconds = time.time()-tm0
    tm_minutes = tm_seconds/60.0
    tm_per_trial = tm_seconds/ntrial
    logger.info('time: %g minutes' % tm_minutes)
    logger.info('time sim: %g minutes' % (tmsim / 60))
    logger.info('time coadd: %g minutes' % (tmcoadd / 60))
    logger.info('time meas: %g minutes' % (tmmeas / 60))
    logger.info('time per trial: %g seconds' % tm_per_trial)

    info = eu.numpy_util.combine_arrlist(infolist)
    data_1p = eu.numpy_util.combine_arrlist(dlist_p)
    if not nocancel:
        data_1m = eu.numpy_util.combine_arrlist(dlist_m)

    if output is None:
        logger.info('doing dry run, not writing')
    else:
        logger.info('writing: %s' % output)
        with fitsio.FITS(output, 'rw', clobber=True) as fits:
            fits.write(data_1p, extname='1p')

            if not nocancel:
                fits.write(data_1m, extname='1m')

            fits.write(info, extname='info')
