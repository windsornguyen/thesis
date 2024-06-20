# =============================================================================#
# Authors: Windsor Nguyen
# File: setup.py
# 
# This software may be used and distributed in accordance 
# with the terms of the Apache 2.0 License.
# =============================================================================#

"""Princeton Senior Thesis."""

import setuptools


setuptools.setup(
    name='sage',
    version='1.0',
    description='Setup manager for Windsor Nguyen\'s Princeton Senior Thesis',
    long_description="""
        Spectral State Space Models. See more details in the
        [`README.md`](https://github.com/windsornguyen/thesis).
        """,
    long_description_content_type="text/markdown",
    author='Windsor Nguyen',
    author_email='mn4560@princeton.edu',
    url='https://github.com/windsornguyen/thesis',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch==2.3.1',
        'tqdm==4.66.4',
        'numpy==1.26.4',
    ],
    python_requires='>=3.9, <3.12', # Dynamo is not supported on Python 3.12+
    extras_require={
        'dev': ['ipykernel>=6.29.3', 'ruff>=0.3.7']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='pytorch machine learning spectral state space models',
)