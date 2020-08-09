#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from setuptools import setup, find_packages

setup(
    name="few_shot_learning",
    version="0.1.0",
    license="SalesforceIQ internal",
    author="yue.li",
    author_email="yue.li@salesforce.com",
    description="a collection of meta learning algorithms for few shot learning",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires=">=3.7.3"
)
