#!/bin/bash

# List of packages to be installed first
pip install jsonlines
pip install datasets
pip install bitsandbytes
pip install autoawq
pip install optimum
pip install auto-gptq

# Packages to be installed after the above packages
pip install packaging
pip install ninja
pip install wheel
pip install flash-attn
