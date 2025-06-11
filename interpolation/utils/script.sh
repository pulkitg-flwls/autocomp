#!/bin/bash

DIR="vmdf02"
EP="vmdf02_ep01_pt03_0050"

declare -a pairs=(
    "vmdf02 vmdf02_ep01_pt03_0050"
    "vmdf02 vmdf02_ep01_pt37_0320"
    "vmdf02 vmdf02_ep01_pt22_0080"
    "vmdf02 vmdf02_ep01_pt26_0250"
    "vmdf02 vmdf02_ep01_pt45_0090"
    "book01 book01_ep01_pt13_0090"
    "book01 book01_ep01_pt10_0050"
    "ufof01 ufof01_ep01_pt15_0030"
    "ufof01 ufof01_ep01_pt33_0240"
    "ufof01 ufof01_ep01_pt34_0100"
)
# Function to clear Python memory
clear_python_memory() {
    # Run a small Python script that cleans up memory
    python -c "
import gc
import sys
gc.collect()
sys.stdout.write('Memory cleared\n')
"
}

for pair in "${pairs[@]}"; do
    # Split the pair into DIR and EP
    DIR=$(echo $pair | cut -d' ' -f1)
    EP=$(echo $pair | cut -d' ' -f2)
    
    echo "Processing $EP"
    python interpolate.py --og /app/data/compositing_workflow/interpolation/$EP/dns/ --nr /app/data/compositing_workflow/interpolation/$EP/nr/ --interpolation /app/data/compositing_workflow/interpolation/$EP/interpolation_values/ --output_dir ../../data/interpolation/$EP/flowformerpp/
    clear_python_memory
    python vis_error.py --dir1 /app/data/compositing_workflow/interpolation/$EP/result/ --dir2 ../../data/interpolation/$EP/flowformerpp/ --output_dir ../vis/interpolation/$EP/flowformerpp/
    clear_python_memory
    mkdir -p ../vis/noise_vids/interpolation/$EP
    ffmpeg -framerate 15 -i ../vis/interpolation/$EP/flowformerpp/%06d.png -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -y ../vis/noise_vids/interpolation/$EP/flowformerpp.mp4
    echo "Completed $EP"
done