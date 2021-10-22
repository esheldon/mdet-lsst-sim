import time
import logging
import numpy as np

import ngmix
from descwl_shear_sims.sim import (
    make_sim,
    get_sim_config,
    get_se_dim,
)
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog
from descwl_coadd import make_coadd_obs, make_coadd_obs_nowarp
from metadetect.lsst.photometry import run_photometry
import fitsio
import esutil as eu

from . import util, vis


def run_sim_phot(
    *,
    seed,
    ntrial,
    output,
    sim_config=None,
    mdet_config=None,
    coadd_config=None,
    full_output=False,
    show=False,
    show_sim=False,
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
    full_output: bool, optional
        If True, write full output rather than trimming.  Default False
    show: bool, optional
        If True, show some images.  Default False
    show_sim: bool, optional
        If True, show the sims.  default False
    """

    tm0 = time.time()
    tmsim = 0.0
    tmcoadd = 0.0
    tmmeas = 0.0

    logger = logging.getLogger('mdet_lsst_sim')

    logger.info(f"seed: {seed}")

    rng = np.random.RandomState(seed)

    mdet_config, use_sx = util.get_mdet_config(config=mdet_config)

    coadd_config = util.get_coadd_config(config=coadd_config)
    assert not coadd_config['remove_poisson'], (
        'Do not set remove_poisson=True in the coadd config; '
        'there is no poisson noise in the sim'
    )

    sim_config = get_sim_config(config=sim_config)

    if sim_config['se_dim'] is None:
        sim_config['se_dim'] = get_se_dim(coadd_dim=sim_config['coadd_dim'])

    if sim_config['gal_type'] != 'wldeblend':
        gal_config = sim_config.get('gal_config', None)
    else:
        logger.info("setting wldeblend layout to None")
        sim_config["layout"] = None
        gal_config = None

    if sim_config['stars']:
        star_config = sim_config.get('star_config', {})

    logger.info(str(sim_config))
    logger.info(str(mdet_config))

    dlist = []

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
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=sim_config['coadd_dim'],
                buff=sim_config['buff'],
                density=star_config.get('density'),
            )
            logger.info('star_density: %g' % star_catalog.density)
        else:
            star_catalog = None

        if sim_config['psf_type'] == 'ps':
            psf = make_ps_psf(rng=rng, dim=sim_config['se_dim'])
        else:
            psf = make_fixed_psf(psf_type=sim_config["psf_type"])

        tmsim0 = time.time()
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=sim_config['coadd_dim'],
            se_dim=sim_config['se_dim'],
            g1=0.0,
            g2=0.0,
            psf=psf,
            star_catalog=star_catalog,
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

        coadd_mbobs = ngmix.MultiBandObsList()

        tmcoadd0 = time.time()
        for band, band_exps in sim_data['band_data'].items():
            obslist = ngmix.ObsList()

            if coadd_config['nowarp']:
                if len(band_exps) > 1:
                    raise ValueError('only one exp allowed for nowarp')

                coadd_obs = make_coadd_obs_nowarp(
                    exp=band_exps[0],
                    psf_dims=sim_data['psf_dims'],
                    rng=rng,
                    remove_poisson=coadd_config['remove_poisson'],
                )

            else:

                coadd_obs = make_coadd_obs(
                    exps=band_exps,
                    coadd_wcs=sim_data['coadd_wcs'],
                    coadd_bbox=sim_data['coadd_bbox'],
                    psf_dims=sim_data['psf_dims'],
                    rng=rng,
                    remove_poisson=coadd_config['remove_poisson'],
                )

            if coadd_obs is None:
                break

            coadd_obs.meta['band'] = band
            obslist.append(coadd_obs)
            coadd_mbobs.append(obslist)

        tmcoadd += time.time() - tmcoadd0

        if coadd_obs is None:
            continue

        # if show:
        #     coadd_obs.show()

        logger.info('mask_frac: %g' % coadd_obs.meta['mask_frac'])

        tmmeas0 = time.time()

        res = run_photometry(
            config=mdet_config,
            mbobs=coadd_mbobs,
            rng=rng,
            show=show,
        )
        tmmeas += time.time() - tmmeas0

        if res is None:
            continue

        comb_data = util.make_comb_data(
            res=res,
            meas_type=mdet_config['meas_type'],
            star_catalog=star_catalog,
            meta=coadd_obs.meta,
            full_output=full_output,
        )

        if len(comb_data) > 0:
            dlist.append(comb_data)

    tm_seconds = time.time()-tm0
    tm_minutes = tm_seconds/60.0
    tm_per_trial = tm_seconds/ntrial
    print('time: %g minutes' % tm_minutes)
    print('time sim: %g minutes' % (tmsim / 60))
    print('time coadd: %g minutes' % (tmcoadd / 60))
    print('time meas: %g minutes' % (tmmeas / 60))
    print('time per trial: %g seconds' % tm_per_trial)

    data = eu.numpy_util.combine_arrlist(dlist)

    if output is None:
        logger.info('doing dry run, not writing')
    else:
        logger.info('writing: %s' % output)
        with fitsio.FITS(output, 'rw', clobber=True) as fits:
            fits.write(data, extname='1p')
