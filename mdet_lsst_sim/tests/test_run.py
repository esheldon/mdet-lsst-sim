from ..run import run


def test_nothing():
    pass


def test_run_smoke():
    run(
        sim_config={},
        seed=3125,
        ntrial=1,
        output=None,
    )
