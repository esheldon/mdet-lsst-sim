import time
import logging
import numpy as np

from descwl_shear_sims.sim import (
    make_sim,
    get_sim_config,
    get_se_dim,
)
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf, make_rand_psf
from descwl_shear_sims.stars import make_star_catalog
from metadetect.lsst.photometry import run_photometry
import metadetect.lsst.vis as lsst_vis
from metadetect.lsst.masking import (
    apply_apodized_edge_masks_mbexp,
    apply_apodized_bright_masks_mbexp,
    AP_RAD,
)
from metadetect.masking import get_ap_range
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

    ap_padding = get_ap_range(AP_RAD)

    rng = np.random.RandomState(seed)

    mdet_config, extra = util.get_mdet_config(config=mdet_config)

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

    dlist = []
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

        if sim_config['psf_type'] == 'ps':
            psf = make_ps_psf(
                rng=rng,
                dim=sim_config['se_dim'],
                variation_factor=sim_config['psf_variation_factor'],
            )
        elif sim_config['randomize_psf']:
            psf = make_rand_psf(
                psf_type=sim_config["psf_type"], rng=rng,
            )
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
            rng=rng, sim_data=sim_data,
            **coadd_config
        )

        if show:
            if len(coadd_data['mbexp']) >= 3:
                lsst_vis.show_mbexp(
                    coadd_data['mbexp'],
                    stretch=3,
                    # q=20,
                )
            else:
                lsst_vis.show_multi_mbexp(coadd_data['mbexp'])

        apply_apodized_edge_masks_mbexp(**coadd_data)
        if extra['mask_bright'] and len(sim_data['bright_info']) > 0:
            # Note padding due to apodization, otherwise we get donuts the
            # radii coming out of the sim code are not super conservative,
            # just going to the noise level
            sim_data['bright_info']['radius_pixels'] += ap_padding
            apply_apodized_bright_masks_mbexp(
                bright_info=sim_data['bright_info'],
                **coadd_data
            )
            if show:
                # lsst_vis.show_image_and_mask(coadd_data['mbexp'])
                lsst_vis.show_multi_mbexp(coadd_data['mbexp'])

        mask_frac = util.get_mask_frac(
            coadd_data['mfrac_mbexp'],
            trim_pixels=extra['trim_pixels'],
        )

        info = util.make_info()
        info['star_density'] = star_density
        info['mask_frac'] = mask_frac
        infolist.append(info)

        tmcoadd += time.time() - tmcoadd0

        logger.info('mask_frac: %g' % mask_frac)

        if mask_frac == 1:
            continue

        tmmeas0 = time.time()

        res = run_photometry(
            rng=rng, config=mdet_config, show=show,
            mbexp=coadd_data['mbexp'],
            mfrac_mbexp=coadd_data['mfrac_mbexp'],
            ormasks=coadd_data['ormasks'],
        )

        tmmeas += time.time() - tmmeas0

        if res is None:
            continue

        comb_data = util.make_comb_data(
            res=res,
            meas_type=mdet_config['meas_type'],
            star_catalog=star_catalog,
            mask_frac=mask_frac,
            full_output=full_output,
        )

        if len(comb_data) > 0:
            dlist.append(comb_data)
            if extra['trim_pixels'] > 0:
                good = util.trim_catalog_boundary_strict(
                    data=comb_data,
                    dim=sim_config['coadd_dim'],
                    trim_pixels=extra['trim_pixels'],
                    checks=['l', 'r', 'u', 'd'],
                    show=show,
                )
                comb_data['primary'][good] = True
            else:
                comb_data['primary'] = True

    tm_seconds = time.time()-tm0
    tm_minutes = tm_seconds/60.0
    tm_per_trial = tm_seconds/ntrial
    print('time: %g minutes' % tm_minutes)
    print('time sim: %g minutes' % (tmsim / 60))
    print('time coadd: %g minutes' % (tmcoadd / 60))
    print('time meas: %g minutes' % (tmmeas / 60))
    print('time per trial: %g seconds' % tm_per_trial)

    info = eu.numpy_util.combine_arrlist(infolist)
    data = eu.numpy_util.combine_arrlist(dlist)

    if output is None:
        logger.info('doing dry run, not writing')
    else:
        logger.info('writing: %s' % output)
        with fitsio.FITS(output, 'rw', clobber=True) as fits:
            fits.write(data, extname='1p')
            fits.write(info, extname='info')
