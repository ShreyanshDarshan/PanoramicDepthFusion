import numpy as np
import torch
import cv2
from equilib import equi2pers, pers2equi
import argparse
import tifffile
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from utils import (
    load_color,
    resize_color,
    load_depth,
    equi2cube_pad,
    cube2equi_pad,
    generate_mask,
    depth_to_distance_fac,
    merge_cube2equi,
    dict_to_tensor,
)
from align import Scaler, Aligner


def panoramic_da(color_tensor, model, device="cuda"):
    """
    Process the panorama and depth images using the PromptDA library.

    Args:
        color_tensor (torch.Tensor): The panorama image tensor.
        depth_tensor (torch.Tensor): The depth map tensor.
        model (DepthAnythingV2): The DepthAnythingV2 model for processing.
        device (str): Device to run the processing on ('cuda' or 'cpu').
    """
    # Ensure tensors are on the correct device
    color = color_tensor.to(device)
    color = resize_color(color)  # 1, 3, H, W

    color_cube = equi2cube_pad(color[0], 120)
    # for face in color_cube:
    #     plt.imshow(color_cube[face][0].permute(1, 2, 0).cpu())
    #     plt.show()
    # depth_cube = equi2cube_pad(depth, image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")120)

    _, _, Wc, _ = color.shape
    mask = generate_mask(cube_size=Wc, fov=120, device=color.device)  # H, W
    d2d_fac = depth_to_distance_fac(cube_size=Wc, fov=120, device=color.device)  # H, W
    disp_sharp_cube = {}
    for face in color_cube:
        inputs = {
            "pixel_values": color_cube[face][None, ...],
        }
        with torch.no_grad():
            output = model(**inputs, return_dict=True)
        disp_sharp = output.predicted_depth / d2d_fac[None, ...]
        disp_sharp_cube[face] = disp_sharp
        # plt.imshow(disp_sharp[0].cpu(), cmap="turbo")
        # plt.show()
    disp_sharp_unmerged = cube2equi_pad(disp_sharp_cube, 120)

    aligner = Aligner()
    disps_aligned = aligner.align(dict_to_tensor(disp_sharp_cube), fov=120)

    disp_aligned_unmerged = cube2equi_pad(disps_aligned, 120, "tensor")
    disp_sharp = merge_cube2equi(disp_aligned_unmerged, fov=120, type="tensor")

    plt.imshow(color[0].permute(1, 2, 0).cpu())
    plt.show()
    plt.imshow(disp_sharp.cpu()[0], cmap="turbo")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load panorama and depth images.")
    parser.add_argument(
        "--color", type=str, required=True, help="Path to the panorama image."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to load images onto.",
    )

    args = parser.parse_args()

    color_tensor = load_color(args.color, device=args.device)
    # depth_tensor = load_depth(args.depth, device=args.device)

    print(f"Color tensor shape: {color_tensor.shape}")
    # print(f"Depth tensor shape: {depth_tensor.shape}")

    # image_processor = AutoImageProcessor.from_pretrained(
    #     "depth-anything/Depth-Anything-V2-Small-hf"
    # )
    model = (
        AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        )
        .to(args.device)
        .eval()
    )

    panoramic_da(color_tensor, model, device=args.device)
