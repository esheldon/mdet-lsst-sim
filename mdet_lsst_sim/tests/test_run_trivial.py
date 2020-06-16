import pytest
from ..run_trivial_sim import run_trivial_sim


def test_run_trivial_smoke():
    sim_config = {
        'coadd_dim': 101,
        'buff': 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize("layout", ("grid", "random"))
def test_run_trivial_layout(layout):

    sim_config = {
        'layout': layout,
        'coadd_dim': 101,
        'buff': 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize("gal_type", ("exp", "wldeblend"))
def test_run_trivial_gal_type(gal_type):

    sim_config = {
        'gal_type': gal_type,
        'coadd_dim': 101,
        'buff': 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )
