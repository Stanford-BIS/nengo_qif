#!/usr/bin/env python
import imp
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup

root = os.path.dirname(os.path.realpath(__file__))
description = "QIF neuron models for Nengo" 
with open(os.path.join(root, 'README.md')) as readme:
    long_description = readme.read()

setup(
    name="nengo_qif",
    version=1.0,
    author="Sam Fok",
    author_email="samfok@stanford.edu",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://github.com/Stanford-BIS/nengo_qif.git",
    license="https://github.com/Stanford-BIS/nengo_qif/blob/master/LICENSE",
    description=description,
    install_requires=[
        "nengo",
    ],
)
