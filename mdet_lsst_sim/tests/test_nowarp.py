import pytest
from ..run_sim import run_sim


@pytest.mark.parametrize('nowarp', [False, True])
def test_nowarp(nowarp):

    sim_config = {
        'gal_type': 'fixed',
        'layout': 'random',
        'coadd_dim': 101,
        'buff': 5,
    }
    coadd_config = {'nowarp': nowarp}
    run_sim(
        sim_config=sim_config,
        coadd_config=coadd_config,
        seed=3125,
        ntrial=1,
        output=None,
    )


@pytest.mark.parametrize(
    'inputs',
    [{'epochs_per_band': 2, 'nband': 1},
     {'epochs_per_band': 1, 'nband': 2}]
)
def test_nowarp_errors(inputs):

    sim_config = {
        'gal_type': 'fixed',
        'layout': 'random',
        'coadd_dim': 101,
        'buff': 5,
    }
    sim_config.update(inputs)
    coadd_config = {'nowarp': True}

    with pytest.raises(ValueError):
        run_sim(
            sim_config=sim_config,
            coadd_config=coadd_config,
            seed=3125,
            ntrial=1,
            output=None,
        )
