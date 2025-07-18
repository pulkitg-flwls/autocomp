#!/bin/bash

# =========================
# Configuration
# =========================
# 1.  Add / remove model names below to run different optical-flow networks.
# 2.  Map each model to its checkpoint folder inside get_ckpt().
# 3.  Add or edit dataset/episode pairs in the pairs array.
# =========================

# ---- Models to test ---------------------------------------
declare -a algos=(
    "dpflow"
    "memflow"
    "flowformer_pp"
    "ms_raft_p"
    "gmflow"
    # "gmflow"           # <- example: uncomment to enable
)

# ---- Model-to-checkpoint mapping --------------------------
get_ckpt () {
    # Usage: Data=$(get_ckpt "$1")
    local algo="$1"
    case "$algo" in
        dpflow)        echo "spring"  ;;
        memflow)       echo "spring"  ;;
        flowformer_pp) echo "things"  ;;
        ms_raft_p)     echo "mixed"  ;;
        gmflow)        echo "things"  ;;
        *)             echo "things"  ;;
    esac
}

# ---- Dataset / episode pairs ------------------------------
# Format: "<DIR> <EP>"
# Add more pairs as needed.
declare -a pairs=(
    # "ufof01 ufof01_ep01_pt33_0240"
    # "ufof01 ufof01_ep01_pt34_0100"
    # "vmdf02 vmdf02_ep01_pt03_0050"
    # "vmdf02 vmdf02_ep01_pt37_0320"
    "vmdf02 vmdf02_ep01_pt22_0080"
    # "vmdf02 vmdf02_ep01_pt26_0250"
    # "vmdf02 vmdf02_ep01_pt45_0090"
)

# =========================
# Helper functions
# =========================
clear_python_memory() {
    # Quickly free Python memory between runs
    python - <<'PY'
import gc, sys
gc.collect()
sys.stdout.write('Memory cleared\n')
PY
}

# =========================
# Main processing loops
# =========================
for Algo in "${algos[@]}"; do
    Data="$(get_ckpt "$Algo")"
    echo "=== Running model: $Algo (ckpt: $Data) ==="
    echo "$Algo"
    for pair in "${pairs[@]}"; do
        DIR="$(echo $pair | cut -d' ' -f1)"
        EP="$(echo $pair | cut -d' ' -f2)"

        echo "Processing $EP"
        python metrics.py --dir1 ../../data/optical_flow/$EP/$Algo/warp/ --dir2 /app/data/compositing_workflow/color_correct/$EP/dns_morph --interpolation /app/data/compositing_workflow/interpolation/$EP/interpolation_values/
        clear_python_memory
        # # ---- Compute optical flow & warp images -------------
        # python warp_flow.py \
        #     --plate_a /app/data/compositing_workflow/color_correct/$EP/dns \
        #     --plate_b /app/data/compositing_workflow/color_correct/$EP/nr \
        #     --warped_dir ../../data/optical_flow/$EP/$Algo/warp/ \
        #     --flow_viz_dir ../../data/optical_flow/$EP/$Algo/flow/ \
        #     --model $Algo \
        #     --ckpt_path $Data

        # clear_python_memory

        # ---- Visualise results -----------------------------
        # python vis_optical_flow.py \
        #     --plate_a /app/data/compositing_workflow/color_correct/$EP/dns \
        #     --plate_b /app/data/compositing_workflow/color_correct/$EP/nr \
        #     --warp_img ../../data/optical_flow/$EP/$Algo/warp/ \
        #     --flow_viz_dir ../../data/optical_flow/$EP/$Algo/flow/ \
        #     --gt /app/data/compositing_workflow/color_correct/$EP/dns_morph \
        #     --out_path ../../vis/optical_flow/$EP/$Algo/ \
        #     --title "$Algo"

        # clear_python_memory

        # ---- Encode MP4 preview ----------------------------
        # mkdir -p ../../vis/noise_vids/color_correct/$EP
        # ffmpeg -framerate 15 \
        #     -i ../../vis/optical_flow/$EP/$Algo/%06d.png \
        #     -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" \
        #     -c:v libx264 -pix_fmt yuv420p -y \
        #     ../../vis/noise_vids/color_correct/$EP/"$Algo".mp4

        echo "Completed $EP"
    done

    echo "=== Finished model: $Algo ==="
done
