import cv2
import numpy as np
import torch
import torch.nn.functional as F
from equilib import equi2pers, pers2equi
import open3d as o3d
import tifffile


def load_color(path, device="cuda"):
    """
    Load a panorama image from the specified path and convert it to a tensor.

    Args:
        path (str): Path to the panorama image.
        device (str): Device to load the image onto ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The panorama image as a tensor.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_LINEAR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0).to(device)


def resize_color(img_tensor):
    B, C, H, W = img_tensor.shape
    patch_y, patch_x = H // 14, W // 14
    H_new, W_new = patch_y * 14, patch_x * 14
    return F.interpolate(img_tensor, (H_new, W_new), mode="bilinear")


def load_depth(path, device="cuda"):
    """
    Load a depth map from the specified path and convert it to a tensor.

    Args:
        path (str): Path to the depth map image.
        device (str): Device to load the image onto ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The depth map as a tensor.
    """
    depth = tifffile.imread(path)
    if depth is None:
        raise FileNotFoundError(f"Depth map not found at {path}")

    # Normalize depth values
    depth = depth.astype(np.float32)  # Assuming depth is in mm
    depth_tensor = (
        torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )
    return depth_tensor


EQUI2CUBE_ROTS = {
    "F": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "L": {"yaw": np.deg2rad(90), "pitch": 0.0, "roll": 0.0},
    "R": {"yaw": np.deg2rad(-90), "pitch": 0.0, "roll": 0.0},
    "U": {"yaw": 0.0, "pitch": np.deg2rad(-90), "roll": 0.0},
    "D": {"yaw": 0.0, "pitch": np.deg2rad(90), "roll": 0.0},
    "B": {"yaw": np.deg2rad(180), "pitch": 0.0, "roll": 0.0},
}

FACE_LIST = ["F", "L", "R", "U", "D", "B"]


def tensor_to_dict(tensor):
    F, C, H, W = tensor.shape
    assert F == len(FACE_LIST), f"Expected {len(FACE_LIST)} faces, but got {F}"
    return {FACE_LIST[i]: tensor[i] for i in range(F)}


def dict_to_tensor(tensor_dict):
    faces = list(tensor_dict.keys())
    assert len(faces) == len(
        FACE_LIST
    ), f"Expected {len(FACE_LIST)} faces, but got {len(faces)}"
    return torch.stack([tensor_dict[face] for face in FACE_LIST], dim=0)


def equi2cube_pad(equi, fov, type="dict"):
    (
        _,
        h,
        w,
    ) = equi.shape
    equi = equi[None, ...].repeat(len(FACE_LIST), 1, 1, 1)  # Repeat for each face
    rots = [EQUI2CUBE_ROTS[face] for face in FACE_LIST]
    cube_t = equi2pers(equi, rots, h, h, fov)  # F, c, h, w
    if type == "dict":
        cube_dict = {face: cube_t[i] for i, face in enumerate(FACE_LIST)}
        return cube_dict
    elif type == "tensor":
        return cube_t
    else:
        raise ValueError("type must be 'dict' or 'tensor', got {type}")


def cube2equi_pad(cube, fov, type="dict"):
    equi_dict = {}
    if type == "dict":
        C, Wc, _ = cube["F"].shape
        device = cube["F"].device
    else:
        F, C, Wc, _ = cube.shape
        device = cube.device
    rot = [{k: -v for k, v in EQUI2CUBE_ROTS[face].items()} for face in FACE_LIST]
    if type == "dict":
        for i, face in enumerate(FACE_LIST):
            equi_dict[face] = pers2equi(
                cube[face][None, ...], [rot[i]], Wc, Wc * 2, fov
            )[0]
        return equi_dict
    elif type == "tensor":
        equi_t = pers2equi(cube, rot, Wc, Wc * 2, fov)
        return equi_t


def merge_cube2equi(equi, fov, type="dict"):
    if type == "dict":
        C, H, W = equi["F"].shape
        device = equi["F"].device
        F = len(equi)
    elif type == "tensor":
        F, C, H, W = equi.shape
        device = equi.device

    mask_cube = generate_mask(cube_size=H, fov=fov, device=device)  # H, H
    mask_cube = mask_cube[None, None, ...].repeat(F, 1, 1, 1)  # F, 1, H, H
    mask_equi = cube2equi_pad(mask_cube, fov=fov, type="tensor")  # F, 1, H, W

    if type == "dict":
        equi_final = torch.zeros((C, H, W), dtype=torch.float32, device=device)
        weights = torch.zeros((1, H, W), dtype=torch.float32, device=device)
        for i, face in enumerate(FACE_LIST):
            equi_final += equi[face] * mask_equi[i]
            weights += mask_equi[i]
    elif type == "tensor":
        equi_final = (equi * mask_equi).sum(dim=0)
        weights = mask_equi.sum(dim=0)
    else:
        raise ValueError("type must be 'dict' or 'tensor', got {type}")
    equi_final = equi_final / weights
    return equi_final


def generate_mask(
    cube_size,
    fov,
    device,
    cube_fov=90,
):
    f = cube_size / 2 / np.tan(np.deg2rad(fov / 2))
    # Create a mask with a white square in the center fading to black at the borders using Manhattan distance
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, cube_size), np.linspace(-1, 1, cube_size), indexing="ij"
    )
    # Compute Chebyshev (max) distance from center
    dist = np.maximum(np.abs(xx), np.abs(yy))
    # Normalize so that center is 1, borders are 0 (fade radius is 0.5)
    fade = np.clip(1 - dist, 0, 1)
    # Get fade value at inner square border
    tan_fov = np.tan(np.deg2rad(cube_fov / 2))
    fade_border = fade[int(cube_size / 2 + f * tan_fov), cube_size // 2]
    fade = np.clip(fade, 0, fade_border)
    fade = fade / fade_border
    mask = torch.tensor(fade, dtype=torch.float32, device=device)
    return mask


def depth_to_distance_fac(cube_size, fov, device):
    f = cube_size / 2 / np.tan(np.deg2rad(fov))
    scene = o3d.t.geometry.RaycastingScene()
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(
            [[f, 0.0, cube_size / 2 - 0.5], [0.0, f, cube_size / 2 - 0.5], [0, 0, 1]],
            dtype=o3d.core.Dtype.Float32,
        ),
        o3d.core.Tensor(np.eye(4), dtype=o3d.core.Dtype.Float32),
        cube_size,
        cube_size,
    )
    rays = rays.numpy()[:, :, 3:]
    rays = np.linalg.norm(rays, axis=-1)
    rays_t = torch.tensor(rays, dtype=torch.float32, device=device)
    return rays_t


def color_tensor_to_img(color_tensor):
    color_img = color_tensor * 255.0
    color_img = color_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return color_img


def save_point_cloud(rgb, depth, path, mask=None):
    ## rgb: (H, W, 3), depth: (H, W)
    h, w = depth.shape
    rgb = rgb / 255.0
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    # mask = np.ones_like(depth, dtype=bool)
    # mask[:600, :] = False

    X = depth * np.sin(Theta) * np.sin(Phi)
    Y = depth * np.cos(Theta)
    Z = depth * np.sin(Theta) * np.cos(Phi)

    if mask is None:
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        R = rgb[:, :, 0].flatten()
        G = rgb[:, :, 1].flatten()
        B = rgb[:, :, 2].flatten()
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        R = rgb[:, :, 0][mask]
        G = rgb[:, :, 1][mask]
        B = rgb[:, :, 2][mask]

    XYZ = np.stack([X, Y, Z], axis=1)
    RGB = np.stack([R, G, B], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ)
    pcd.colors = o3d.utility.Vector3dVector(RGB)
    o3d.io.write_point_cloud(path, pcd)
