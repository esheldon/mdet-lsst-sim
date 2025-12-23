import os
import pytest
from ..run_sim import run_sim


def test_run_smoke():
    run_sim(
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize('layout', ['grid', 'random'])
def test_run_layout(layout):

    sim_config = {'layout': layout}
    run_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize('shear', (0.01, 0.02))
def test_run_shear(shear):

    sim_config = {'layout': 'random'}
    run_sim(
        sim_config=sim_config,
        shear=shear,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize('nocancel', (False, True))
def test_run_nocancel(nocancel):

    run_sim(
        nocancel=nocancel,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize(
    'cosmic_rays, bad_columns',
    [(True, False),
     (False, True)],
)
def test_run_artifacts(cosmic_rays, bad_columns):

    sim_config = {
        'layout': 'grid',
        'cosmic_rays': cosmic_rays,
        'bad_columns': bad_columns,
    }
    run_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.skipif(
    'CATSIM_DIR' not in os.environ,
    reason='simulation input data is not present',
)
def test_run_gal_wldeblend():

    sim_config = {
        'gal_type': 'wldeblend',
        'coadd_dim': 101,
        'buff': 5,
    }
    run_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.skipif(
    'CATSIM_DIR' not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize(
    'gal_type,stars',
    [('exp', True),
     ('exp', False),
     ('wldeblend', True),
     ('wldeblend', False)]
)
def test_run_stars(gal_type, stars):

    sim_config = {
        'gal_type': gal_type,
        'layout': 'random',
        'coadd_dim': 101,
        'buff': 5,
        'stars': stars,
    }
    run_sim(
        sim_config=sim_config,
        seed=125,
        ntrial=1,
        output=None,
    )


@pytest.mark.skipif(
    'CATSIM_DIR' not in os.environ,
    reason='simulation input data is not present',
)
def test_run_star_bleeds():

    sim_config = {
        'gal_type': 'wldeblend',
        'layout': 'random',
        'coadd_dim': 101,
        'buff': 5,
        'stars': True,
        'star_bleeds': True,
    }
    run_sim(
        sim_config=sim_config,
        seed=125,
        ntrial=1,
        output=None,
    )
