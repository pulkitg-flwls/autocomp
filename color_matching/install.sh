#!/bin/bash

cd face_parsing
cd face_detection
pip install -e .
cd ..
cd roi_tanh_warping
pip install -e .
cd ..
pip install -e .
cd ..
pip install psutil