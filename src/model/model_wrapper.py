from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.nn import functional as F

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import (
    prep_image, 
    save_image, 
    visualize_depth,
)
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder

from ..geometry.ssl import  fig2img
from ..geometry.inspect_epipolar_geometry import inspect


# validate, test and visualization
from evo.core import metrics
from evo.core.trajectory import PosePath3D
from evo.main_ape import ape
from evo.tools import plot
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int


@dataclass
class TestCfg:
    output_path: Path


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    
    theta = torch.acos(cos)
    return theta * 180 / np.pi

def check_invalid_gradients( model: torch.nn.Module):
        encoder_param = []
        flag = True
        for _, param in model.named_parameters():
            encoder_param.append(param)
        for param in encoder_param:
            if getattr(param, 'grad', None) is not None and torch.isnan(param.grad).any():
                print('NaN in gradients.')
                flag = False
                break
            if getattr(param, 'grad', None) is not None and torch.isinf(param.grad).any():
                print('Inf in gradients.')
                flag = False
                break
        return flag

class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        ref_pose = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer("ref_pose", ref_pose, persistent=False)

        self.pose_relation = metrics.PoseRelation.translation_part
        self.plot_mode = plot.PlotMode.xyz        

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()        
        batch: BatchedExample = self.data_shim(batch)
        b, _, _, h, w = batch["target"]["image"].shape
        
        # Run the model.
        gaussians, pose, pose_rev, depths, matching_prob = self.encoder(    # pose = rel_pose, pose_inv = extrinsic (c2w)
            batch["context"], 
            batch["target"], 
            self.global_step, 
            val_or_test=False, 
            supervised=False
        )
        
        extrinsics = torch.cat([pose[:, :1], 
                                repeat(self.ref_pose, "i j -> b () i j", b=b), 
                                pose[:, -1:]], dim=1)   # (b, v+n, 4, 4)
        intrinsics = torch.stack([batch["context"]["intrinsics"][:, 0], 
                                  batch["target"]["intrinsics"][:, 0], 
                                  batch["context"]["intrinsics"][:, -1]], dim=1)
    
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            if loss_fn.name == 'repro':
                loss = loss_fn.forward(output, batch, pose, pose_rev, depths, self.global_step)
            else:
                loss = loss_fn.forward(output, batch, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        self.manual_backward(total_loss)
        do_backprop = check_invalid_gradients(self.encoder)
        if do_backprop:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            print('Invalid gradients. Skip backpropagation.')
            optimizer.zero_grad()

        if self.global_rank == 0:
            print(
                f"train step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)


    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, n, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_ssl, pose, _, depths, _ = self.encoder(
            batch["context"],
            batch["target"],
            self.global_step,
            val_or_test=True,
            supervised=False,
        )

        extrinsics = torch.cat([pose[:, :1], 
                                repeat(self.ref_pose, "i j -> b () i j", b=b), 
                                pose[:, -1:],], dim=1)   # (b, v+n, 4, 4)

        intrinsics = torch.stack([batch["context"]["intrinsics"][:, 0], 
                                  batch["target"]["intrinsics"][:, 0], 
                                  batch["context"]["intrinsics"][:, -1]], dim=1) # (b, v+n-1, 3, 3)

        output_ssl = self.decoder.forward(
            gaussians_ssl,
            extrinsics,
            intrinsics,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        rgb_ssl = output_ssl.color[0]   # (v, 3, h, w)
        depth_ssl = output_ssl.depth[0] # (v, 1, h, w)

        psnr = compute_psnr(batch["target"]["image"][0], rgb_ssl[1:-1])
        ssim = compute_ssim(batch["target"]["image"][0], rgb_ssl[1:-1])
        lpips = compute_lpips(batch["target"]["image"][0], rgb_ssl[1:-1])
        self.log("metric/psnr", psnr)
        self.log("metric/ssim", ssim)
        self.log("metric/lpips", lpips)

        t2c1_pred = pose[0, :1] # w2c
        t2c1_gt  = torch.matmul(batch["context"]["extrinsics"][:, 0].inverse(), batch["target"]["extrinsics"][:, 0])
        norm_pred = t2c1_pred[:,:3,3] / torch.linalg.norm(t2c1_pred[:,:3,3], dim = -1).unsqueeze(-1)
        norm_gt = t2c1_gt[:,:3,3] / torch.linalg.norm(t2c1_gt[:,:3,3], dim =-1).unsqueeze(-1)
        cosine_similarity_0 = torch.dot(norm_pred[0], norm_gt[0])
        angle_degree_0 = torch.arccos(torch.clip(cosine_similarity_0, -1.0,1.0)) * 180 / np.pi
        rot = compute_geodesic_distance_from_two_matrices(t2c1_pred[:, :3, :3], 
                                                          torch.matmul(batch["context"]["extrinsics"][:, 0].inverse(), batch["target"]["extrinsics"][:, 0])[:, :3, :3]).cpu().item()
        trans_angle = angle_degree_0.item()
        self.log("metric/geodesic_rot", rot)
        self.log("metric/translation", trans_angle)

        depths_img = []
        for depth in depths[0]: # (v, 1, h, w)
            depths_img.append(visualize_depth(depth[0])) # (h, w)
        depths_img = torch.stack(depths_img) # (v, 3, h, w)
        
        depths_rendering = []
        for depth in depth_ssl: # (v, 1, h, w)
            depths_rendering.append(visualize_depth(depth[0])) # (h, w)
        depths_rendering = torch.stack(depths_rendering) # (v, 3, h, w)
            
        comparison = hcat(
            add_label(vcat(*torch.cat([batch["context"]["image"][0], batch["target"]["image"][0]])), "Context"),
            add_label(vcat(*rgb_ssl), "Target (Self-Supervised)"),
            add_label(vcat(*depths_img), "Depth"),
            add_label(vcat(*depths_rendering), "Depth Rendering"),
        )
        self.logger.log_image(
            f"{batch['scene'][0]}/comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )



    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, n, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_ssl, pose, pose_rev, depths, matching_prob = self.encoder(
            batch["context"],
            batch["target"],
            self.global_step,
            val_or_test=True,
            supervised=False,
        )

        intrinsics = torch.stack([batch["context"]["intrinsics"][:, 0], 
                                  batch["target"]["intrinsics"][:, 0],
                                  batch["context"]["intrinsics"][:, -1],], dim=1) # (b, v+n, 3, 3)

        extrinsics = torch.cat([pose[:, :1], 
                                repeat(self.ref_pose, "i j -> b () i j", b=b), 
                                pose[:, -1:]], dim=1)   # (b, v+n, 4, 4)

        output_ssl = self.decoder.forward(
            gaussians_ssl,
            extrinsics,
            intrinsics, 
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        rgb_ssl = output_ssl.color[0]   # (v, 3, h, w)
        depth_ssl = output_ssl.depth[0] # (v, 1, h, w)

        ### PNSR ###
        psnr = compute_psnr(batch["target"]["image"][0], rgb_ssl[1:-1])
        self.log("metric/psnr", psnr)

        depths_img = []
        for depth in depths[0]: # (v, 1, h, w)
            depths_img.append(visualize_depth(depth[0])) # (h, w)
        depths_img = torch.stack(depths_img) # (v, 3, h, w)
        
        depths_rendering = []
        for depth in depth_ssl: # (v, 1, h, w)
            depths_rendering.append(visualize_depth(depth[0])) # (h, w)
        depths_rendering = torch.stack(depths_rendering) # (v, 3, h, w)

        matching_vis = []
        for match in matching_prob[0]:
            matching_vis.append(visualize_depth(torch.mean(match, dim=0))) # (h, w)
        matching_vis = torch.stack(matching_vis)

        comparison = hcat(
            add_label(vcat(*torch.cat([batch["context"]["image"][0], batch["target"]["image"][0]])), "Context"),
            add_label(vcat(*rgb_ssl[[0, 2, 1], ...]), "Target (Self-Supervised)"),
            add_label(vcat(*depths_img), "Depth"),
            add_label(vcat(*depths_rendering[[0, 2, 1], ...]), "Depth Rendering"),
            add_label(vcat(*matching_vis), "Matching prob"),
        )
        self.logger.log_image(
            f"{batch['scene'][0]}/comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )


    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_ssl, pose, _, _, _ = self.encoder(batch["context"], batch["target"], self.global_step, 
                                                        val_or_test=True, supervised=False)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        def trajectory_fn_ssl(t, pose):
            extrinsics = interpolate_extrinsics(
                pose[0, 0],
                pose[0, -1],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1],
                t,
            )
            return extrinsics[None], intrinsics[None]

        extrinsics_ssl, intrinsics_ssl = trajectory_fn_ssl(t, pose)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_ssl = self.decoder.forward(
            gaussians_ssl, extrinsics_ssl, intrinsics_ssl, (h, w), None
        )
        images_ssl = [
            vcat(rgb, visualize_depth(depth[0]))
            for rgb, depth in zip(output_ssl.color[0], output_ssl.depth[0])
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_ssl, "Self-Supervised"),
                )
            )
            for image_ssl in images_ssl
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()


        # from moviepy.editor import ImageSequenceClip
        # frames = [np.transpose(frame, (1, 2, 0)) for frame in video]

        # # Create a video clip
        # clip = ImageSequenceClip(frames, fps=8)

        # # Save the video
        # output_path = "output_video.mp4"
        # clip.write_videofile(output_path, codec="libx264")
        # print()
        # if loop_reverse:
        #     video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=8, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
