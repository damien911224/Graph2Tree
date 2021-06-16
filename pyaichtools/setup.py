#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List

setup(
    name="pyaichtools",
    version="0.0.1",
    author="VCLAB",
    url="https://github.com/JunhoPark0314/pyaichtools.git",
    description="AI challenge python source code encoder decoder",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.6",
    install_requires=[
        "yacs>=0.1.8",
        "libcst>=0.3.19",  # or use pillow-simd for better performance
        "treelib>=1.6.1",
    ],
)