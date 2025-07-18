"""warp_flow.py

Utility script that:
  1. Takes two folders of PNGs: plate‑A (img_a) and plate‑B (img_b).
  2. Computes forward optical flow from each img_a → img_b pair.
  3. Forward‑warps img_a into img_b’s frame.
  4. Saves the warped image      to <warped_dir>
     and the flow visualisation  to <flow_viz_dir>.
"""
from pathlib import Path
from typing import List, Tuple, Dict

import argparse
import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

import ptlflow
from ptlflow.utils.flow_utils import flow_to_rgb, flow_write
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import tensor_dict_to_numpy
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel

# Re-use helpers from the original inference script
# from ptlflow_scripts.infer import init_input, _read_image   # type: ignore  # noqa: F401


def compute_flow(prev_img: np.ndarray,
                 img: np.ndarray,
                 model,
                 io_adapter: IOAdapter,
                 fp16: bool = False) -> Dict[str, np.ndarray]:
    """Compute optical flow between two RGB frames.

    Returns a dict with raw flow (H×W×2, float32) and 'flows_viz' (H×W×3, uint8).
    """
    inputs = io_adapter.prepare_inputs([prev_img, img])
    with torch.no_grad():
        preds = model(inputs)
    preds["images"] = inputs["images"]
    preds = io_adapter.unscale(preds)
    preds_npy = tensor_dict_to_numpy(preds)

    # Convert flow to an easy-to-view RGB image (in BGR order for OpenCV)
    preds_npy["flows_viz"] = flow_to_rgb(preds_npy["flows"])[..., ::-1]
    return preds_npy


def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward-warp *img* into the next frame using the forward flow (H×W×2)."""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)  # forward warp
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    warped = cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR,
                      borderMode=cv.BORDER_CONSTANT)
    return warped


def _sorted_pngs(folder: str) -> List[Path]:
    """Return a lexicographically‑sorted list of *.png files in *folder*."""
    return sorted([p for p in Path(folder).glob("*.png")])

def save_pair_results(flow_viz: np.ndarray,
                      warped_img: np.ndarray,
                      flow_dir: str,
                      warped_dir: str,
                      stem: str) -> None:
    """Write flow visualisation and warped frame to their respective dirs."""
    Path(flow_dir).mkdir(parents=True, exist_ok=True)
    Path(warped_dir).mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(Path(flow_dir) / f"{stem}.png"), flow_viz.astype(np.uint8))
    cv.imwrite(str(Path(warped_dir) / f"{stem}.png"), warped_img)


def save_results(preds_npy: Dict[str, np.ndarray],
                 warped_img: np.ndarray,
                 output_dir: Path,
                 img_name: str,
                 flow_format: str = "flo") -> None:
    """Save flow field (+ visualisation) and warped frame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw flow
    flow_write(output_dir.joinpath(f"{img_name}.{flow_format}"), preds_npy["flows"])

    # 2. Flow visualisation
    cv.imwrite(str(output_dir.joinpath(f"{img_name}_flow.png")),
               preds_npy["flows_viz"].astype(np.uint8))

    # 3. Warped image
    cv.imwrite(str(output_dir.joinpath(f"{img_name}_warped.png")),
               warped_img)


def build_io_adapter(model, first_img: np.ndarray,
                     input_size: Tuple[int, int],
                     scale_factor: float,
                     fp16: bool) -> IOAdapter:
    """Mirror the logic from *infer.py* for resizing / scaling."""
    if scale_factor is not None:
        return IOAdapter(output_stride=model.output_stride,
                         input_size=first_img.shape[:2],
                         target_scale_factor=scale_factor,
                         cuda=torch.cuda.is_available(),
                         fp16=fp16)
    return IOAdapter(output_stride=model.output_stride,
                     input_size=first_img.shape[:2],
                     target_size=input_size,
                     cuda=torch.cuda.is_available(),
                     fp16=fp16)


def process_plates(args, model) -> None:
    """Iterate over paired frames from plate‑A and plate‑B, compute flow, warp, save."""
    a_paths = _sorted_pngs(args.plate_a)
    b_paths = _sorted_pngs(args.plate_b)
    assert len(a_paths) == len(b_paths) > 0, "Folders must have the same number of PNGs."

    first_img = cv.imread(str(a_paths[0]))
    io_adapter = build_io_adapter(model, first_img,
                                  args.input_size, args.scale_factor, args.fp16)

    for a_path, b_path in zip(a_paths, b_paths):
        img_a = cv.imread(str(a_path))
        img_b = cv.imread(str(b_path))

        preds = compute_flow(img_a, img_b, model, io_adapter, args.fp16)
        warped = warp_image(img_a, preds["flows"])
        save_pair_results(preds["flows_viz"], warped,
                          args.flow_viz_dir, args.warped_dir, a_path.stem)


# --------------------------------------------------------------------------- #
# Main entry-point
# --------------------------------------------------------------------------- #

def args_parse():
    parser = argparse.ArgumentParser(
        "Warp images using PTLFlow optical flow",
        add_help=False,
        conflict_handler="resolve"
    )
    parser.add_argument("--plate_a", type=str, required=True,
                        help="Folder with img_a PNGs")
    parser.add_argument("--plate_b", type=str, required=True,
                        help="Folder with img_b PNGs (same count & order)")
    parser.add_argument("--warped_dir", type=str, default="outputs/warped",
                        help="Directory to save warped img_a → b results")
    parser.add_argument("--flow_viz_dir", type=str, default="outputs/flow_viz",
                        help="Directory to save flow visualisations")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=("Path to a ckpt file for the chosen model."),
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[0, 0],
        help="If larger than zero, resize the input image before forwarding.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help=("Multiply the input image by this scale factor before forwarding."),
    )
    return parser
if __name__ == "__main__":
    
    base_parser = args_parse()

    # Let PTLFlowCLI inject its rich model / trainer options
    cli = PTLFlowCLI(model_class=RegisteredModel,
                     subclass_mode_model=True,
                     parser_kwargs={"parents": [base_parser]},
                     run=False,
                     parse_only=False,
                     auto_configure_optimizers=False)

    cfg = cli.config
    model = cli.model
    model = ptlflow.restore_model(model, cfg.ckpt_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        if cfg.fp16:
            model.half()

    # Create root output dir with model identifier
    cfg.model_name = cfg.model.class_path.split(".")[-1]
    tag = cfg.model_name + (f"_{Path(cfg.ckpt_path).stem}" if cfg.ckpt_path else "")
    # cfg.output_path = str(Path(cfg.output_path) / tag)

    process_plates(cfg, model)

    # print(f"All done. Results are under: {cfg.output_path}")