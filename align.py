import matplotlib.pyplot
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import (
    equi2cube_pad,
    cube2equi_pad,
    FACE_LIST,
    tensor_to_dict,
    merge_cube2equi,
    depth_to_distance_fac,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib


class Scaler(nn.Module):
    def __init__(self, grid_size=5, num_disps=6):
        super(Scaler, self).__init__()
        self.grid_size = grid_size
        self.T = num_disps
        self.scale_grid = nn.Parameter(
            torch.ones((self.T, grid_size, grid_size), requires_grad=True)
        )
        self.offset_grid = nn.Parameter(
            torch.zeros((self.T, grid_size, grid_size), requires_grad=True)
        )

    def forward(self, disps, sky_masks):
        T, _, H, W = disps.shape
        assert T == self.T, f"Expected {self.T} disparity samples, but got {T}"
        scale_high = F.interpolate(
            self.scale_grid[:, None, ...],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        offset_high = F.interpolate(
            self.offset_grid[:, None, ...],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        disps = disps * scale_high + offset_high
        disps[sky_masks] = 0.0  # Set sky pixels to 0
        return disps


class Aligner:
    def __init__(
        self,
        disps,
        depth_equi=None,
        fov=120,
        grid_size=[1, 4, 8, 16],
        lr=[5e-2, 5e-2, 2e-2, 1e-2],
        align_iters=[100, 150, 100, 80],
        px_ratio=0.01,
        num_disps=6,
        device="cuda",
    ):
        self.grid_size = grid_size
        self.num_stages = len(grid_size)
        self.lr = lr
        self.align_iters = align_iters
        self.num_disps = num_disps
        self.scalers = [
            Scaler(grid_size=g, num_disps=num_disps).to(device) for g in grid_size
        ]
        self.px_ratio = px_ratio
        self.writer = SummaryWriter("runs/")
        self.disps = disps
        self.depths_cube = None
        self.fov = fov
        if depth_equi is not None:
            self.depths_cube = equi2cube_pad(depth_equi, fov, type="tensor")
        F, C, H, W = disps.shape
        self.d2d_fac = depth_to_distance_fac(
            cube_size=H, fov=self.fov, device=device
        )  # H, W

    def align(self):
        """
        Align disparity maps across multiple stages.
        """
        depths_cube = None

        sky_masks = self.disps == 0
        self.disps = self.normalize_disps(self.disps, depths_cube=depths_cube)
        pair_masks = self.get_masks(self.disps)

        for stage in range(self.num_stages):
            self.align_stage(
                self.disps,
                pair_masks,
                sky_masks,
                stage,
                depths_cube=depths_cube,
            )
            self.scalers[stage].eval()
            with torch.no_grad():
                self.disps = self.scalers[stage](self.disps, sky_masks)
            self.scalers[stage].train()

        # disps -= disps.min() - 1
        self.disps[sky_masks] = 0
        self.disps /= self.d2d_fac[None, ...]  # Normalize by distance factor
        return self.disps

    def align_stage(self, disps, pair_masks, sky_masks, stage, depths_cube=None):
        optimizer = torch.optim.Adam(
            self.scalers[stage].parameters(), lr=self.lr[stage]
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=self.align_iters[stage],
        )
        self.scalers[stage].train()
        for step in tqdm(
            range(self.align_iters[stage])
        ):  # Number of iterations per stage
            optimizer.zero_grad()

            scaled_disps = self.scalers[stage](disps, sky_masks)
            scaled_disps /= self.d2d_fac[None, ...]

            loss_dict = self.loss(
                scaled_disps, pair_masks, stage, depths_cube=depths_cube
            )
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            self.writer.add_scalars(f"aligner/stage_{stage}/loss", loss_dict, step)
            if step % 10 == 0 or step == self.align_iters[stage] - 1:
                eval_out = self.eval_iter(
                    disps, sky_masks, stage, depths_cube=depths_cube
                )
                self.writer.add_image(
                    f"aligner/stage_{stage}/equi_disp",
                    eval_out["equi_disp_img"],
                    step,
                    dataformats="HWC",
                )
                self.writer.add_image(
                    f"aligner/stage_{stage}/equi_depth",
                    eval_out["equi_depth_img"],
                    step,
                    dataformats="HWC",
                )
                self.writer.add_image(
                    f"aligner/stage_{stage}/scalar_img",
                    eval_out["scalar_img"],
                    step,
                    dataformats="HWC",
                )

    def loss(self, disps, pair_masks, stage, depths_cube=None):
        loss_dict = {
            "alignment_loss": self.alignment_loss(disps, pair_masks),
            "smoothness_loss": 40 * self.smoothness_loss(stage),
            "scale_loss": 0.007 * self.scale_loss(stage, depths_cube=depths_cube),
            "metric_loss": 2 * self.metric_loss(disps, depth_cubes=depths_cube),
            "metric_loss_grad": 20
            * self.metric_loss_grad(disps, depth_cubes=depths_cube),
            "sign_loss": 0.1 * self.sign_loss(disps),
        }
        loss_dict["total_loss"] = sum(loss_dict.values())
        return loss_dict

    def sign_loss(self, disps):
        """
        Compute the sign loss between disparity maps.
        Args:
            disps (torch.Tensor): Disparity maps of shape (F, C, H, W).
        """
        F, C, H, W = disps.shape
        sign_loss = torch.mean((disps[disps < 0]) ** 2)
        return sign_loss

    def metric_loss(self, disps, depth_cubes=None):
        """
        Compute the metric loss between disparity maps and depth maps.
        Args:
            disps (torch.Tensor): Disparity maps of shape (F, C, H, W).
            depth_cubes (torch.Tensor): Depth maps of shape (F, C, H, W).
        """
        if depth_cubes is None:
            return 0.0

        disps_metric = 1 / depth_cubes
        mask = depth_cubes > 0
        loss = torch.mean((disps[mask] - disps_metric[mask]) ** 2)
        return loss

    def metric_loss_grad(self, disps, depth_cubes=None):
        """
        Compute the gradient of the metric loss between disparity maps and depth maps.
        Args:
            disps (torch.Tensor): Disparity maps of shape (F, C, H, W).
            depth_cubes (torch.Tensor): Depth maps of shape (F, C, H, W).
        """
        if depth_cubes is None:
            return 0.0

        disps_metric = 1 / depth_cubes
        mask = depth_cubes > 0
        xgrad_disp_cube = disps[..., :, 1:] - disps[..., :, :-1]
        ygrad_disp_cube = disps[..., 1:, :] - disps[..., :-1, :]
        xgrad_disps = disps_metric[..., :, 1:] - disps_metric[..., :, :-1]
        ygrad_disps = disps_metric[..., 1:, :] - disps_metric[..., :-1, :]
        maskx = mask[..., :, 1:] & mask[..., :, :-1]
        masky = mask[..., 1:, :] & mask[..., :-1, :]

        grad_loss_x = torch.mean(((xgrad_disp_cube - xgrad_disps))[maskx].abs())
        grad_loss_y = torch.mean(((ygrad_disp_cube - ygrad_disps))[masky].abs())
        grad_loss = grad_loss_x + grad_loss_y

        return grad_loss

    def alignment_loss(self, disps, pair_masks):
        loss = 0.0
        equi_disps = cube2equi_pad(disps, fov=self.fov, type="tensor")
        equi_disps_dict = tensor_to_dict(equi_disps)
        for face1, face2 in pair_masks:
            mask = pair_masks[(face1, face2)]
            disp1 = equi_disps_dict[face1]
            disp2 = equi_disps_dict[face2]
            diff = disp1[mask] - disp2[mask]
            # px_mask = torch.rand(diff.numel(), device=diff.device) < self.px_ratio
            # diff_masked = diff.view(-1)[px_mask]
            # select random pixels for loss calculation
            loss += (diff**2).mean()
        loss /= len(pair_masks)
        return loss

    def smoothness_loss(self, stage):
        scaler = self.scalers[stage]
        scale_grid = scaler.scale_grid
        offset_grid = scaler.offset_grid
        scale_xgrad = (scale_grid[:, 1:, :] - scale_grid[:, :-1, :]) ** 2
        scale_ygrad = (scale_grid[:, :, 1:] - scale_grid[:, :, :-1]) ** 2
        offset_xgrad = (offset_grid[:, 1:, :] - offset_grid[:, :-1, :]) ** 2
        offset_ygrad = (offset_grid[:, :, 1:] - offset_grid[:, :, :-1]) ** 2
        loss = (
            scale_xgrad.mean()
            + scale_ygrad.mean()
            + offset_xgrad.mean()
            + offset_ygrad.mean()
        )
        return loss

    def scale_loss(self, stage, depths_cube=None):
        if depths_cube is not None:
            return 0.0
        scaler = self.scalers[stage]
        scale_grid = scaler.scale_grid
        loss = (1 / scale_grid).mean()
        return loss

    def get_masks(self, disps):
        masks = torch.ones_like(disps, dtype=torch.uint8).to(disps.device)
        masks_equi = cube2equi_pad(masks, fov=self.fov, type="tensor").to(torch.bool)
        masks_equi = tensor_to_dict(masks_equi)

        pair_masks = {}
        for face1 in FACE_LIST:
            for face2 in FACE_LIST:
                key = (face1, face2)
                mask = masks_equi[face1] & masks_equi[face2]
                if face1 != face2 and key not in pair_masks and mask.any():
                    pair_masks[key] = mask
        return pair_masks

    def eval_iter(self, disps, sky_masks, stage, depths_cube=None):
        self.scalers[stage].eval()
        with torch.no_grad():
            scaled_disps = self.scalers[stage](disps, sky_masks)
            scaled_disps /= self.d2d_fac[None, ...]

            # scaled_disps -= scaled_disps.min() - 1
            scaled_disps[sky_masks] = 0
            equi_disps = cube2equi_pad(scaled_disps, fov=self.fov, type="tensor")
            equi_disp_merged = merge_cube2equi(equi_disps, fov=self.fov, type="tensor")
            equi_disp_for_later = equi_disp_merged.clone()
            if depths_cube is not None:
                disps_cube = 1 / depths_cube
                disps_cube[depths_cube == 0] = 0
                metric_disps_equi = cube2equi_pad(
                    disps_cube, fov=self.fov, type="tensor"
                )
                metric_disps_equi = merge_cube2equi(
                    metric_disps_equi, fov=self.fov, type="tensor"
                )

                equi_disp_merged = torch.cat(
                    [equi_disp_merged, metric_disps_equi], dim=1
                )

            # colormap for visualization
            equi_disp = (equi_disp_merged - equi_disp_merged.min()) / (
                equi_disp_merged.max() - equi_disp_merged.min()
            )
            equi_disp = (equi_disp * 255).byte().cpu().numpy()
            equi_disp = np.transpose(equi_disp, (1, 2, 0))[..., 0]  # H, W
            equi_disp = matplotlib.colormaps.get_cmap("turbo")(equi_disp)[:, :, :3]
            equi_disp = (equi_disp * 255).astype(np.uint8)

            equi_depth = 1 / equi_disp_for_later
            equi_depth[equi_disp_for_later == 0] = 0
            if depths_cube is not None:
                metric_depths_equi = cube2equi_pad(
                    depths_cube, fov=self.fov, type="tensor"
                )
                metric_depths_equi = merge_cube2equi(
                    metric_depths_equi, fov=self.fov, type="tensor"
                )
                equi_depth = torch.cat([equi_depth, metric_depths_equi], dim=1)

            equi_depth = torch.clip(
                equi_depth,
                torch.quantile(equi_depth, 0.03),
                torch.quantile(equi_depth, 0.97),
            )
            equi_depth = (equi_depth - equi_depth.min()) / (
                equi_depth.max() - equi_depth.min()
            )
            equi_depth = (equi_depth * 255).byte().cpu().numpy()
            equi_depth = np.transpose(equi_depth, (1, 2, 0))[..., 0]  # H, W
            equi_depth = matplotlib.colormaps.get_cmap("turbo")(equi_depth)[:, :, :3]
            equi_depth = (equi_depth * 255).astype(np.uint8)

        scale_imgs = self.scalers[stage].scale_grid.detach().cpu().numpy()
        scale_imgs = (scale_imgs - scale_imgs.min()) / (
            scale_imgs.max() - scale_imgs.min()
        )
        scale_imgs = (scale_imgs * 255).astype(np.uint8)
        offset_imgs = self.scalers[stage].offset_grid.detach().cpu().numpy()
        offset_imgs = (offset_imgs - offset_imgs.min()) / (
            offset_imgs.max() - offset_imgs.min()
        )
        offset_imgs = (offset_imgs * 255).astype(np.uint8)

        F, sh, sw = scale_imgs.shape
        scalar_img = np.zeros((sh, F * sw, 3), dtype=np.uint8)
        for i, (scale_img, offset_img) in enumerate(zip(scale_imgs, offset_imgs)):
            scalar_img[:, i * sw : (i + 1) * sw, 0] = scale_img
            scalar_img[:, i * sw : (i + 1) * sw, 1] = offset_img

        self.scalers[stage].train()
        eval_out = {
            "equi_disp_img": equi_disp,
            "equi_depth_img": equi_depth,
            "scalar_img": scalar_img,
        }
        return eval_out

    def normalize_disps(self, disps, depths_cube=None):
        """
        Normalize the disparity maps to the range [0, 1].
        """
        F, C, H, W = disps.shape
        disps_median = torch.median(disps.view(F, -1), dim=1, keepdim=True).values
        disps_dev = disps - disps_median[..., None, None]
        disps_scale = torch.mean(disps_dev.abs().view(F, -1), dim=1, keepdim=True)
        disps = disps_dev / (disps_scale[..., None, None] + 1e-6)

        if depths_cube is not None:
            disps_metric_cube = 1 / depths_cube
            disps_metric_cube[depths_cube == 0] = 0

            F, _, _, _ = disps_metric_cube.shape
            disps_metric_median = torch.median(
                disps_metric_cube.view(F, -1), dim=1, keepdim=True
            ).values
            disps_metric_dev = disps_metric_cube - disps_metric_median[..., None, None]
            disps_metric_scale = torch.mean(
                disps_metric_dev.abs().view(F, -1), dim=1, keepdim=True
            )

            disps = (
                disps * disps_metric_scale[:, None, None]
                + disps_metric_median[:, None, None]
            )

        return disps
