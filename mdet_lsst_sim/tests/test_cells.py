import numpy as np
import pytest
import os
import tempfile
import yaml
from mdet_lsst_sim.run_sim import run_sim
from mdet_lsst_sim.run_cells import run_cells
import fitsio

config_string_rand = """
mls:
    shear: 0.02
    randomize_shear: true

    cell_size: 250
    cell_buff: 50

sim:
    gal_type: fixed
    gal_config:
        mag: 22
    layout: random

    # we will do 5 cells
    coadd_dim: 850
    se_dim: 850

    # no objects drawn here
    buff: 50

coadd:
    nowarp: true

mdet:
    meas_type: wmom
    trim_pixels: 50
"""


config_string_wldeblend = """
mls:
    shear: 0.02
    randomize_shear: true

    cell_size: 250
    cell_buff: 50

sim:
    gal_type: wldeblend

    # bring noise down to approximately 3 band level
    noise_factor: 0.58

    # we will do 5 cells
    coadd_dim: 850
    se_dim: 850

    # no objects drawn here
    buff: 50

coadd:
    nowarp: true

mdet:
    meas_type: wmom
    trim_pixels: 50
"""


def get_config_wldeblend():
    return yaml.safe_load(config_string_wldeblend)


def get_config_rand():
    return yaml.safe_load(config_string_rand)


def test_cells_rand():
    config = get_config_rand()
    seed = 1010
    ntrial = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'nocells.fits.gz')
        cells_fname = os.path.join(tmpdir, 'cells.fits.gz')

        run_sim(
            seed=seed,
            ntrial=ntrial,
            output=fname,
            sim_config=config['sim'],
            mdet_config=config['mdet'],
            coadd_config=config['coadd'],
            shear=config['mls']['shear'],
            randomize_shear=config['mls']['randomize_shear'],
        )
        run_cells(
            seed=seed,
            ntrial=ntrial,
            output=cells_fname,
            sim_config=config['sim'],
            mdet_config=config['mdet'],
            coadd_config=config['coadd'],
            **config['mls']
        )

        output = fitsio.read(fname)
        wp, = np.where(output['primary'])

        cells_output = fitsio.read(cells_fname)
        wpc, = np.where(cells_output['primary'])
        assert wp.size < output.size

        # make sure the number of detections is close
        print('ncells:', wp.size)
        print('nnocells:', wpc.size)

        # they are less discrepant for this setup than wldeblend
        assert abs(wpc.size / wp.size - 1) < 0.05


@pytest.mark.skipif(
    'CATSIM_DIR' not in os.environ,
    reason='simulation input data is not present',
)
def test_cells_wldeblend():
    config = get_config_wldeblend()
    seed = 101
    ntrial = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'nocells.fits.gz')
        cells_fname = os.path.join(tmpdir, 'cells.fits.gz')

        run_sim(
            seed=seed,
            ntrial=ntrial,
            output=fname,
            sim_config=config['sim'],
            mdet_config=config['mdet'],
            coadd_config=config['coadd'],
            shear=config['mls']['shear'],
            randomize_shear=config['mls']['randomize_shear'],
        )
        run_cells(
            seed=seed,
            ntrial=ntrial,
            output=cells_fname,
            sim_config=config['sim'],
            mdet_config=config['mdet'],
            coadd_config=config['coadd'],
            **config['mls']
        )

        output = fitsio.read(fname)
        wp, = np.where(output['primary'])

        cells_output = fitsio.read(cells_fname)
        wpc, = np.where(cells_output['primary'])
        assert wpc.size < cells_output.size

        # make sure the number of detections is close
        print('ncells:', wp.size)
        print('nnocells:', wpc.size)

        # Can't put this lower than 10%. I've seen it vary from 2.5% to 8%. The
        # sign is that the cells tends to have fewer detections.
        #
        # I find this surprising.
        #
        # Could this be related to our issue with DM detection code, which
        # re-determins the sky noise, works poorly for crowded images?
        # also the detection can't detect near an edge, but I thought the 50
        # pixel boundary would be good enough, so I doubt that is it

        assert abs(wpc.size / wp.size - 1) < 0.10


if __name__ == '__main__':
    test_cells_rand()
