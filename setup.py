import os
import glob
from setuptools import setup, find_packages

__version__ = None

pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "mdet_lsst_sim",
    "version.py"
)
with open(pth, 'r') as fp:
    exec(fp.read())

scripts = glob.glob('bin/*')
scripts = [s for s in scripts if '~' not in s]

setup(
    name="mdet_lsst_sim",
    version=__version__,
    packages=find_packages(),
    scripts=scripts,
    # install_requires=['numpy', 'galsim'],
    # include_package_data=True,
    author='Erin Sheldon',
    author_email='erin.sheldon@gmail.com',
    url='https://github.com/esheldon/mdet-lsst-sim',
)
