import os
import pytest
from ..run_trivial_sim import run_trivial_sim


def test_run_trivial_smoke():
    sim_config = {
        "coadd_dim": 101,
        "buff": 5,
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
        "layout": layout,
        "coadd_dim": 101,
        "buff": 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize("shear", (0.01, 0.02))
def test_run_trivial_shear(shear):

    sim_config = {
        "layout": "random",
        "coadd_dim": 101,
        "buff": 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        shear=shear,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize("nocancel", (False, True))
def test_run_trivial_nocancel(nocancel):

    sim_config = {
        "layout": "random",
        "coadd_dim": 101,
        "buff": 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        nocancel=nocancel,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize("model", ["wmom", "gauss"])
def test_run_trivial_model(model):

    sim_config = {
        "layout": "random",
        "coadd_dim": 101,
        "buff": 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        mdet_config={"model": model},
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize(
    "cosmic_rays, bad_columns",
    [(True, True),
     (True, False),
     (False, True),
     (True, True)],
)
def test_run_trivial_defects(cosmic_rays, bad_columns):

    sim_config = {
        "layout": "grid",
        "coadd_dim": 101,
        "buff": 5,
        "cosmic_rays": cosmic_rays,
        "bad_columns": bad_columns,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason="simulation input data is not present",
)
def test_run_trivial_gal_wldeblend():

    sim_config = {
        "gal_type": "wldeblend",
        "coadd_dim": 101,
        "buff": 5,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason="simulation input data is not present",
)
@pytest.mark.parametrize(
    "gal_type,stars",
    [("exp", True),
     ("exp", False),
     ("wldeblend", True),
     ("wldeblend", False)]
)
def test_run_trivial_stars(gal_type, stars):

    sim_config = {
        "gal_type": gal_type,
        "layout": "random",
        "coadd_dim": 101,
        "buff": 5,
        "stars": stars,
    }
    run_trivial_sim(
        sim_config=sim_config,
        seed=125,
        ntrial=1,
        output=None,
    )
