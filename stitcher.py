import torch
import numpy as np
import cv2
import torch.nn as nn
from utils import equi2cube_pad, cube2equi_pad, merge_cube2equi, generate_mask
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from tqdm import tqdm


class Disparity(nn.Module):
    def __init__(self, size=[1024, 2048], fov=120, device="cuda"):
        super(Disparity, self).__init__()
        self.fov = fov
        self.device = device
        self.disp = nn.Parameter(torch.zeros((1, size[0], size[1]), device=device))

    def forward(self):
        disp_cube = equi2cube_pad(self.disp, self.fov, type="tensor")
        return disp_cube


class Stitcher:
    def __init__(
        self, input_disps, disp_stitched_init=None, iters=200, fov=120, device="cuda"
    ):
        super(Stitcher, self).__init__()
        self.fov = fov
        self.device = device
        self.input_disps = input_disps
        self.in_disps_equi = cube2equi_pad(input_disps, fov, type="tensor")
        self.iters = iters
        F, C, H, W = input_disps.shape
        self.disp_stitched = Disparity([H, 2 * H], fov=self.fov, device=self.device)
        if disp_stitched_init is not None:
            self.disp_stitched.disp.data.copy_(disp_stitched_init)

        weight = generate_mask(cube_size=H, fov=fov, device=device, cube_fov=60)  # H, H
        self.weights = weight[None, None, ...].repeat(F, 1, 1, 1)  # F, 1, H, H
        self.wts_equi = cube2equi_pad(
            self.weights, fov=fov, type="tensor"
        )  # F, 1, H, W

        self.optimizer = torch.optim.Adam(self.disp_stitched.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=1.0, total_iters=self.iters
        )

        self.writer = SummaryWriter("runs/stitcher")
        self.disp_stitched.to(device)

    def stitch(self):
        for step in tqdm(range(self.iters)):
            self.optimizer.zero_grad()
            disp_cube = self.disp_stitched()
            loss_dict = self.loss(disp_cube, self.input_disps)
            loss = loss_dict["total_loss"]
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.writer.add_scalars("stitcher/loss", loss_dict, step)
            if step % 10 == 0:
                eval_out = self.eval_iter()
                self.writer.add_image(
                    "stitcher/equi_disp",
                    eval_out["equi_disp_img"],
                    step,
                    dataformats="HWC",
                )

        return self.disp_stitched.disp.data.detach()

    def loss(self, disp_cube, disps):
        loss_dict = {}
        loss_dict["fidelity_loss"] = self.fidelity_loss(disp_cube, disps)
        loss_dict["grad_loss"] = 100 * self.grad_loss(self.disp_stitched.disp)
        loss_dict["total_loss"] = sum(loss_dict.values())
        return loss_dict

    def fidelity_loss(self, disp_cube, disps):
        # Compute the fidelity loss between the stitched disparity and individual disparities
        mask = self.weights > 0.9999
        fidelity_loss = torch.mean((disp_cube[mask] - disps[mask]) ** 2)
        return fidelity_loss

    def grad_loss(self, disp_equi):
        # Compute the gradient loss between the stitched disparity and individual disparities
        xgrad_disp_cube = disp_equi[..., :, 1:] - disp_equi[..., :, :-1]
        ygrad_disp_cube = disp_equi[..., 1:, :] - disp_equi[..., :-1, :]
        xgrad_disps = self.in_disps_equi[..., :, 1:] - self.in_disps_equi[..., :, :-1]
        ygrad_disps = self.in_disps_equi[..., 1:, :] - self.in_disps_equi[..., :-1, :]

        wts = self.wts_equi > 0
        mask_x = wts[..., :, 1:] & wts[..., :, :-1]
        mask_y = wts[..., 1:, :] & wts[..., :-1, :]
        weights_x = self.wts_equi[..., :, 1:] * self.wts_equi[..., :, :-1]
        weights_y = self.wts_equi[..., 1:, :] * self.wts_equi[..., :-1, :]

        xgrad_loss = torch.mean(
            ((weights_x * (xgrad_disp_cube - xgrad_disps) ** 2))[mask_x]
        )
        ygrad_loss = torch.mean(
            ((weights_y * (ygrad_disp_cube - ygrad_disps) ** 2))[mask_y]
        )
        grad_loss = xgrad_loss + ygrad_loss
        return grad_loss

    def eval_iter(self):
        self.disp_stitched.eval()
        with torch.no_grad():
            disp_stitched = self.disp_stitched.disp.data.detach().cpu()
            # colormap for visualization
            equi_disp = (disp_stitched - disp_stitched.min()) / (
                disp_stitched.max() - disp_stitched.min()
            )
            equi_disp = (equi_disp * 255).byte().cpu().numpy()
            equi_disp = np.transpose(equi_disp, (1, 2, 0))[..., 0]  # H, W
            equi_disp = matplotlib.colormaps.get_cmap("turbo")(equi_disp)[:, :, :3]
            equi_disp = (equi_disp * 255).astype(np.uint8)

        eval_out = {
            "equi_disp_img": equi_disp,
        }
        return eval_out
