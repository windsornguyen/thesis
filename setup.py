# Copyright (c) Windsor Nguyen.
# This software may be used and distributed in accordance with the terms of the Sage Community License Agreement.

from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="sage",
    version="1.0.0",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
