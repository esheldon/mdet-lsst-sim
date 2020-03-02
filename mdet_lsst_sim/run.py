import sys
import copy
import logging
import numpy as np

from descwl_shear_sims import Sim
import descwl_coadd.vis
from descwl_coadd.coadd import MultiBandCoadds
from descwl_coadd.coadd_simple import MultiBandCoaddsSimple
from metadetect.lsst_metadetect import LSSTMetadetect
from metadetect.metadetect import Metadetect
import fitsio
import esutil as eu

from . import util, vis


def run(*,
        sim_config,
        seed,
        ntrial,
        output,
        full_output=False,
        show=False,
        show_sheared=False,
        show_masks=False,
        show_sim=False,
        nostack=False,
        loglevel='info'):

    rng = np.random.RandomState(seed)
    mdet_config = util.get_config(nostack=nostack)

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger('mdet_lsst_sim')
    logger.setLevel(getattr(logging, loglevel.upper()))

    dlist_p = []
    dlist_m = []

    for trial in range(ntrial):
        logger.info('-'*70)
        logger.info('trial: %d/%d' % (trial+1, ntrial))

        trial_seed = rng.randint(0, 2**30)

        for shear_type in ('1p', '1m'):
            logger.info(str(shear_type))

            if shear_type == '1p':
                send_show=show
            else:
                send_show=False

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

                psf_dim = int(sim.psf_dim/np.sqrt(3))
                if psf_dim % 2 == 0:
                    psf_dim -= 1

                mbc = MultiBandCoadds(
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
                comb_data['star_density'] = sim.star_density
                if shear_type == '1p':
                    dlist_p.append(comb_data)
                else:
                    dlist_m.append(comb_data)

    data_1p = eu.numpy_util.combine_arrlist(dlist_p)
    data_1m = eu.numpy_util.combine_arrlist(dlist_m)

    logger.info('writing: %s' % output)
    with fitsio.FITS(output, 'rw', clobber=True) as fits:
        fits.write(data_1p, extname='1p')
        fits.write(data_1m, extname='1m')

def show_all_masks(exps):
    for exp in exps:
        descwl_coadd.vis.show_image_and_mask(exp)
        if 'q' == input('hit a key (q to quit) '):
            raise KeyboardInterrupt('halting at user request')
