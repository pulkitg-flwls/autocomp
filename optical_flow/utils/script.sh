#!/bin/bash

DIR="vmdf02"
EP="vmdf02_ep01_pt03_0050"

declare -a pairs=(
    # "vmdf02 vmdf02_ep01_pt03_0050"
    # "vmdf02 vmdf02_ep01_pt37_0320"
    # "vmdf02 vmdf02_ep01_pt22_0080"
    # "vmdf02 vmdf02_ep01_pt26_0250"
    # "vmdf02 vmdf02_ep01_pt45_0090"
    # "book01 book01_ep01_pt13_0090"
    # "book01 book01_ep01_pt10_0050"
    # "ufof01 ufof01_ep01_pt15_0030"
    "ufof01 ufof01_ep01_pt33_0240"
    # "ufof01 ufof01_ep01_pt34_0100"
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
Algo="gmflow"
Data="things"
# Data="mixed"
for pair in "${pairs[@]}"; do
    # Split the pair into DIR and EP
    DIR=$(echo $pair | cut -d' ' -f1)
    EP=$(echo $pair | cut -d' ' -f2)
    
    echo "Processing $EP"
    # python warp_img.py --og /app/data/compositing_workflow/color_correct/$EP/dns --nr /app/data/compositing_workflow/color_correct/$EP/nr --algo $Algo --output_dir ../../data/optical_flow/$EP/$Algo/
    python warp_flow.py --plate_a /app/data/compositing_workflow/color_correct/$EP/dns --plate_b /app/data/compositing_workflow/color_correct/$EP/nr --warped_dir ../../data/optical_flow/$EP/$Algo/warp/ --flow_viz_dir ../../data/optical_flow/$EP/$Algo/flow/ --model $Algo --ckpt_path $Data
    clear_python_memory
    python vis_optical_flow.py --plate_a /app/data/compositing_workflow/color_correct/$EP/dns --plate_b /app/data/compositing_workflow/color_correct/$EP/nr --warp_img ../../data/optical_flow/$EP/$Algo/warp/ --flow_viz_dir ../../data/optical_flow/$EP/$Algo/flow/ --gt /app/data/compositing_workflow/color_correct/$EP/dns_morph --out_path ../../vis/optical_flow/$EP/$Algo/ --title "$Algo"
    # python vis_error.py --dir1 /app/data/compositing_workflow/interpolation/$EP/result/ --dir2 ../../data/interpolation/$EP/flowformerpp/ --output_dir ../vis/interpolation/$EP/flowformerpp/
    clear_python_memory
    mkdir -p ../../vis/noise_vids/color_correct/$EP
    ffmpeg -framerate 15 -i ../../vis/optical_flow/$EP/$Algo/%06d.png -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -y ../../vis/noise_vids/color_correct/$EP/"$Algo".mp4
    echo "Completed $EP"
done