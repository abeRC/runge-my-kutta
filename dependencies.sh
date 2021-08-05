#!/bin/bash

# install dependencies with conda
conda create --name pyqt
conda activate pyqt
conda install matplotlib pynamical pillow numba qt pyqt pyqtgraph pyopengl
pip install celluloid

# install dependencies with pip
#pip install matplotlib cellulloid pynamical pillow numba qt pyqt pyqtgraph pyopengl
