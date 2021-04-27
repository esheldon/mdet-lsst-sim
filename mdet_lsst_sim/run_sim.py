import sys
import logging
import numpy as np

from descwl_shear_sims.sim import (
    make_sim,
    make_dmsim,
    get_sim_config,
    make_galaxy_catalog,
    make_psf,
    make_ps_psf,
    get_se_dim,
    StarCatalog,
)
from descwl_coadd.coadd import MultiBandCoadds, MultiBandCoaddsDM
from metadetect.lsst_metadetect import LSSTMetadetect
import fitsio
import esutil as eu

from . import util, vis


def run_sim(
    *,
    sim_config,
    seed,
    ntrial,
    output,
    shear=0.02,
    nocancel=False,
    mdet_config=None,
    full_output=False,
    show=False,
    show_sheared=False,
    show_sim=False,
    deblend=False,
    interp_bright=False,
    replace_bright=False,
    use_dmsim=False,
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
    shear: float
        Magnitude of the shear.  Shears +/- shear will be applied
    nocancel: bool
        If True, don't run -shear
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

    sim_type = sim_config.pop('type', 'lsst')
    assert sim_type == 'lsst'

    rng = np.random.RandomState(seed)

    mdet_config = util.get_mdet_config(config=mdet_config)
    sim_config = get_sim_config(config=sim_config)

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger('mdet_lsst_sim')
    logger.setLevel(getattr(logging, loglevel.upper()))

    if sim_config['gal_type'] != 'wldeblend':
        gal_config = sim_config.get('gal_config', None)
    else:
        logger.info("setting wldeblend layout to None")
        sim_config["layout"] = None
        gal_config = None

    logger.info(str(sim_config))
    logger.info(str(mdet_config))

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
            gal_config=gal_config,
        )
        if sim_config['stars']:
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=sim_config['coadd_dim'],
                buff=sim_config['buff'],
            )
        else:
            star_catalog = None

        trial_seed = rng.randint(0, 2**30)

        for shear_type in shear_types:

            logger.info(str(shear_type))

            trial_rng = np.random.RandomState(trial_seed)

            if shear_type == '1p':
                send_show = show
            else:
                send_show = False

            if shear_type == '1p':
                g1 = shear
            else:
                g1 = -shear

            if sim_config['psf_type'] == 'ps':
                se_dim = get_se_dim(coadd_dim=sim_config['coadd_dim'])
                psf = make_ps_psf(rng=trial_rng, dim=se_dim)
            else:
                psf = make_psf(psf_type=sim_config["psf_type"])

            if use_dmsim:
                sim_data = make_dmsim(
                    rng=trial_rng,
                    galaxy_catalog=galaxy_catalog,
                    coadd_dim=sim_config['coadd_dim'],
                    g1=g1,
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
                )
            else:
                sim_data = make_sim(
                    rng=trial_rng,
                    galaxy_catalog=galaxy_catalog,
                    coadd_dim=sim_config['coadd_dim'],
                    g1=g1,
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
                )

            if show_sim:
                vis.show_sim(sim_data['band_data'])

            if use_dmsim:
                mbc = MultiBandCoaddsDM(
                    interp_bright=interp_bright,
                    data=sim_data['band_data'],
                    coadd_wcs=sim_data['coadd_wcs'],
                    coadd_bbox=sim_data['coadd_bbox'],
                    psf_dims=sim_data['psf_dims'],
                    byband=False,
                    show=send_show,
                    loglevel=loglevel,
                )
            else:
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
            if coadd_obs is None:
                continue

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

            comb_data = util.make_comb_data(
                res,
                mdet_config["model"],
                full_output=full_output,
            )

            if len(comb_data) > 0:
                if sim_config['stars']:
                    comb_data['star_density'] = star_catalog.density
                    logger.info('star_density: %g' % star_catalog.density)

                comb_data['mask_frac'] = coadd_obs.meta['mask_frac']

                if shear_type == '1p':
                    dlist_p.append(comb_data)
                else:
                    dlist_m.append(comb_data)

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
