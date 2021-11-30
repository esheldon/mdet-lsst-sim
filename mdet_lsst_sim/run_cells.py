"""
TODO

  - adapt to exposure based metadetect
  - use new bright info to mask before sending data to metadetect
  - make sure overall mask frac gets passed on; this should combine
    bands; maybe do in metadetect.
  - masking is different; make sure doshear is not bothering with mask/ormask
    just with mask_frac and mfrac
"""

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
)
# from metadetect.metadetect import do_metadetect as run_metadetect_sx
import fitsio
import esutil as eu

from . import util, vis


def run_cells(
    *,
    seed,
    ntrial,
    output,
    cell_size,
    cell_buff,
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
    """

    tm0 = time.time()
    tmsim = 0.0
    tmcoadd = 0.0
    tmmeas = 0.0

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

    assert sim_config['coadd_dim'] % cell_size == 2*cell_buff
    ncells = sim_config['coadd_dim'] // (cell_size - 2*cell_buff)

    dlist_p = []
    dlist_m = []

    if nocancel:
        shear_types = ('1p', )
    else:
        shear_types = ('1p', '1m')

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
            logger.info('star_density: %g' % star_catalog.density)
        else:
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
            orig_coadd_data = util.coadd_sim_data(
                rng=trial_rng, sim_data=sim_data,
                **coadd_config
            )

            for cell_ix in range(ncells):
                for cell_iy in range(ncells):
                    logger.info('cell {%s, %s}' % (cell_ix, cell_iy))
                    cell_coadd_data = util.extract_cell_coadd_data(
                        coadd_data=orig_coadd_data,
                        cell_size=cell_size,
                        cell_buff=cell_buff,
                        cell_ix=cell_ix,
                        cell_iy=cell_iy,
                    )

                    apply_apodized_edge_masks_mbexp(**cell_coadd_data)

                    if len(sim_data['bright_info']) > 0:
                        apply_apodized_bright_masks_mbexp(
                            bright_info=sim_data['bright_info'],
                            **cell_coadd_data
                        )

                    mask_frac = util.get_mask_frac(
                        cell_coadd_data['mfrac_mbexp'],
                        trim_pixels=trim_pixels,
                        stamp_size=mdet_config['stamp_size'],
                    )

                    tmcoadd += time.time() - tmcoadd0

                    logger.info('mask_frac: %g' % mask_frac)

                    if show:
                        lsst_vis.show_mbexp(cell_coadd_data['mbexp'])

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
                            rng=trial_rng, config=mdet_config,
                            show=show_sheared, **cell_coadd_data,
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

                    if theta is not None:
                        util.unrotate_noshear_shear(
                            comb_data, meas_type=mdet_config['meas_type'],
                            theta=theta,
                        )

                    if trim_pixels > 0:
                        checks = get_cell_checks(
                            ncells=ncells, cell_ix=cell_ix, cell_iy=cell_iy,
                        )
                        comb_data = util.trim_catalog_boundary_strict(
                            data=comb_data,
                            dim=cell_size,
                            trim_pixels=trim_pixels,
                            checks=checks,
                            show=show,
                        )

                    if len(comb_data) > 0:
                        if shear_type == '1p':
                            dlist_p.append(comb_data)
                        else:
                            dlist_m.append(comb_data)

    tm_seconds = time.time()-tm0
    tm_minutes = tm_seconds/60.0
    tm_per_trial = tm_seconds/ntrial
    print('time: %g minutes' % tm_minutes)
    print('time sim: %g minutes' % (tmsim / 60))
    print('time coadd: %g minutes' % (tmcoadd / 60))
    print('time meas: %g minutes' % (tmmeas / 60))
    print('time per trial: %g seconds' % tm_per_trial)

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


def get_cell_checks(ncells, cell_ix, cell_iy):
    checks = ['l', 'r', 'u', 'd']
    if cell_ix == 0:
        checks.remove('l')

    if cell_ix == (ncells-1):
        checks.remove('r')

    if cell_iy == 0:
        checks.remove('d')

    if cell_iy == (ncells-1):
        checks.remove('u')

    return checks
