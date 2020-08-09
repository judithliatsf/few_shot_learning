#!/bin/bash

# Set up conda environment for databricks ML runtime
conda env create -f environment.yml
conda install --yes setuptools

# Build
conda run -n few-shot-learning python setup.py install --force

# Run tests with xml reports and code coverage
conda run -n few-shot-learning coverage run -m xmlrunner discover -s ./tests/ -o ./target/test-reports/

# Report code coverage
conda run -n few-shot-learning coverage html
