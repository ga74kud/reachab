# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Reachability Analysis
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='reachab',
      version='0.4.10',
      description='Reachability Analysis with zonotypes',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ga74kud/reachab',
      author='Michael Hartmann',
      author_email='michael.hartmann@v2c2.at',
      license='GNU GENERAL PUBLIC LICENSE',
      packages=setuptools.find_packages(),
      install_requires=[
          "scipy",
          "numpy",
          "matplotlib",
          "argparse",
          "logging",
        ],
      zip_safe=False)
