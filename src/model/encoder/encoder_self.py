from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ...geometry.ssl import (
    pose_vec2mat,
    homogenize_matrices,
)
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .croco.croco_model import CrocoModel, CrocoModelCfg
from .pose_backbone.posenet import PoseNet
from .ldm_unet.unet import UNetModel


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderSelfCfg:
    name: Literal["self"]
    d_feature: int
    num_surfaces: int
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    downscale_factor: int
    shim_patch_size: int
    croco_backbone: CrocoModelCfg
    gaussians_per_pixel: int
    
    using_matching_net: bool
    using_depth_refine: bool


class EncoderSelf(Encoder[EncoderSelfCfg]):
    croco_backbone: CrocoModel
    pose_backbone: PoseNet
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderSelfCfg) -> None:
        super().__init__(cfg)

        self.croco_backbone = CrocoModel(cfg.croco_backbone)
        self.pose_backbone = PoseNet(self.cfg)
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(cfg.d_feature // 4 + 1 + 3, 
                      cfg.num_surfaces * (3 + self.gaussian_adapter.d_in) * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(cfg.num_surfaces * (3 + self.gaussian_adapter.d_in) * 2, 
                      cfg.num_surfaces * (3 + self.gaussian_adapter.d_in), 3, 1, 1)
        )

        nn.init.zeros_(self.gaussian_head[-1].weight[10:])
        nn.init.zeros_(self.gaussian_head[-1].bias[10:])
        
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, cfg.d_feature // 4, 7, 1, 3),
            nn.ReLU(),
        )

        self.refine_unet = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=32,
                model_channels=32,
                out_channels=32,
                num_res_blocks=1, 
                attention_resolutions=[8],
                channel_mult=[1, 1, 1, 1],
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=2,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(32, 32, 3, 1, 1),
        )

        self.depth_refine_unet = nn.Sequential(
            nn.Conv2d(10, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=32,
                model_channels=32,
                out_channels=8,
                num_res_blocks=1, 
                attention_resolutions=[8],
                channel_mult=[1, 1, 1, 1],
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=2,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(1, 1, 3, 1, 1),
        )

        nn.init.zeros_(self.depth_refine_unet[-1].weight)
        nn.init.zeros_(self.depth_refine_unet[-1].bias)        

        self.matching_unet = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=32,
                model_channels=32,
                out_channels=32,
                num_res_blocks=1,
                attention_resolutions=[8],
                channel_mult=[1, 1, 2, 4],  # [1, 1, 2, 4]
                num_head_channels=32,
                dims=2,
                postnorm=False,
                num_frames=3,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        ref_pose = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer("ref_pose", ref_pose, persistent=False) # (4, 4)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))    

    def forward(
        self,
        context: dict,
        target: dict,
        global_step: int,
        val_or_test: bool = False,
        supervised: bool = False,
        visualization_dump: Optional[dict] = None,
    ) -> tuple[
        Gaussians,
        Float[Tensor, "b p 4 4"] | None,
        Float[Tensor, "b p 4 4"] | None, 
        Float[Tensor, "b v 1 h w"],
        Float[Tensor, "b v+1 3 h w"] | None]:        
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        # Get the depth from the context images
        croco_img1= self.normalize_image(context["image"][:, 0])    # (B, 3, H, W)
        croco_img2= self.normalize_image(context["image"][:, -1])   # (B, 3, H, W)
        trgt_img = self.normalize_image(target["image"][:, 0])            # (B, 3, H, W)

        intrinsics = context["intrinsics"]

        dc_pred, gauss_pred = self.croco_backbone(croco_img1, croco_img2)

        feat_unet = rearrange(gauss_pred, "(b v) c h w -> (v b) c h w", b=b, v=v)
        feat_unet = self.refine_unet(feat_unet) # ((v, b), 34, H, W)
        features = rearrange(feat_unet, "(v b) c h w -> (b v) c h w", b=b, v=v)

        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        x = (x.to(context["extrinsics"].dtype) + 0.5).view(1, 1, -1).expand(b, v+1, -1).to(device).contiguous()
        y = (y.to(context["extrinsics"].dtype) + 0.5).view(1, 1, -1).expand(b, v+1, -1).to(device).contiguous()
        intr_stack = torch.stack([context["intrinsics"][:, 0], context["intrinsics"][:, 1], target["intrinsics"][:, 0]], dim=1).clone()
        intr_stack[:, :, 0] *= float(w)
        intr_stack[:, :, 1] *= float(h)
        x = (x-intr_stack[:, :, 0:1, 2]) / intr_stack[:, :, 0:1, 0] # (x-cx)/fx
        y = (y-intr_stack[:, :, 1:2, 2]) / intr_stack[:, :, 1:2, 1] # (y-cy)/fy
        ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (B, V+1, H*W, 3)
        ray_d = F.normalize(ray_d, p=2, dim=-1) # (B, V+1, H*W, 3)
        ray_d = rearrange(ray_d, "b v (h w) c -> b v c h w", h=h, w=w)
        
        imgs = rearrange(torch.stack([croco_img1, croco_img2, trgt_img], dim=1), "b v c h w -> (v b) c h w")
        matching_prob = self.matching_unet(imgs) # ((v, b), 3, H, W)
        # matching_prob = matching_prob / (matching_prob.norm(dim=1, keepdim=True) + 1e-8)
        matching_prob = rearrange(matching_prob, "(v b) c h w -> b v c h w", b=b, v=v+1)
        croco_img1_pose = torch.cat([croco_img1, matching_prob[:, 0], ray_d[:, 0]], dim=1) # (B, 9, H, W)
        croco_img2_pose = torch.cat([croco_img2, matching_prob[:, 1], ray_d[:, 1]], dim=1) # (B, 9, H, W)
        trgt_img_pose = torch.cat([trgt_img, matching_prob[:, 2], ray_d[:, 2]], dim=1) # (B, 12, H, W)
        
        # Get forward pose and reverse pose
        if val_or_test:
            poses1 = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([trgt_img_pose, croco_img1_pose], dim=1)))) # (B, 4, 4)
            poses2 = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([trgt_img_pose, croco_img2_pose], dim=1)))) # (B, 4, 4)
            pose1_rev = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([croco_img1_pose, trgt_img_pose], dim=1))))
            pose2_rev = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([croco_img2_pose, trgt_img_pose], dim=1))))            
            poses = torch.stack([poses1, poses2], dim=1) # (B, 2, 4, 4)                        
            poses_rev = torch.stack([pose1_rev, pose2_rev], dim=1) # (B, 2, 4, 4)
            # inv_poses = pose_inverse_4x4(poses) # (B, 3, 4, 4)
        else:
            poses1 = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([trgt_img_pose, croco_img1_pose], dim=1)))) # (B, 4, 4)
            poses2 = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([trgt_img_pose, croco_img2_pose], dim=1)))) # (B, 4, 4)
            pose1_rev = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([croco_img1_pose, trgt_img_pose], dim=1))))
            pose2_rev = homogenize_matrices(pose_vec2mat(self.pose_backbone(
                torch.cat([croco_img2_pose, trgt_img_pose], dim=1))))
            poses = torch.stack([poses1, poses2], dim=1) # (B, 2, 4, 4)
            poses_rev = torch.stack([pose1_rev, pose2_rev], dim=1) # (B, 2, 4, 4)
            # inv_poses = pose_inverse_4x4(poses) # (B, 1, 4, 4)

        disps = 0.01 + dc_pred.sigmoid() * 0.99
        disps = rearrange(disps, "(b v) c h w -> b v c h w", b=b, v=v)
        
        extr_clone = poses_rev.clone().detach()
        x_r, y_r = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        x_r = (x_r.to(context["extrinsics"].dtype) + 0.5).view(1, 1, -1).expand(b, v, -1).to(device).contiguous()
        y_r = (y_r.to(context["extrinsics"].dtype) + 0.5).view(1, 1, -1).expand(b, v, -1).to(device).contiguous()
        intr_stack_r = torch.stack([context["intrinsics"][:, 0], context["intrinsics"][:, 1]], dim=1).clone()
        intr_stack_r[:, :, 0] *= float(w)
        intr_stack_r[:, :, 1] *= float(h)
        x_r = (x_r-intr_stack_r[:, :, 0:1, 2]) / intr_stack_r[:, :, 0:1, 0] # (x-cx)/fx
        y_r = (y_r-intr_stack_r[:, :, 1:2, 2]) / intr_stack_r[:, :, 1:2, 1] # (y-cy)/fy
        ray_d_r = torch.stack([x_r, y_r, torch.ones_like(x_r)], dim=-1).float() # (B, V, H*W, 3)
        ray_d_r = F.normalize(ray_d_r, p=2, dim=-1) # (B, V, H*W, 3)
        ray_d_r = ray_d_r @ extr_clone[:, :, :3, :3].transpose(-1, -2).contiguous() # (B, V, H*W, 3)
        ray_o_r = extr_clone[:, :, :3, 3].unsqueeze(2).expand(-1, -1, h*w, -1) # (B, V, H*W, 3)
        pluc_r = torch.cat([torch.cross(ray_o_r, ray_d_r, dim=-1), ray_d_r], dim=-1) # (B, V, H*W, 6)
        pluc_r = rearrange(pluc_r, "b v (h w) c -> b v c h w", h=h, w=w)

        depth_refine_input = torch.cat([context["image"], disps, pluc_r], dim=2) # (B, V, 10, H, W)
        depth_refine_input = rearrange(depth_refine_input, "b v c h w -> (v b) c h w")
        depth_refine_output = self.depth_refine_unet(depth_refine_input) # ((v, b), 10, H, W)
        depth_refine_output = rearrange(depth_refine_output, "(v b) c h w -> b v c h w", v=v)

        disps = (disps + depth_refine_output).clamp(
            1.0 / rearrange(context["far"], "b v -> b v () () ()"),
            1.0 / rearrange(context["near"], "b v -> b v () () ()"),)
            
        skip = rearrange(context["image"], "b v c h w -> (b v) c h w")
        features = features + self.high_resolution_skip(skip)
        gaussians = rearrange(
            self.gaussian_head(torch.cat((rearrange(context["image"], "b v c h w -> (b v) c h w"), 
                                          features, 
                                          rearrange(disps, "b v c h w -> (b v) c h w")), dim=1)),
            "(b v) c h w -> b v c h w",
            b=b, v=v,
        )
        gaussians = rearrange(
            gaussians,
            "b v (srf c) h w -> b v (h w) srf c",
            srf=self.cfg.num_surfaces,
        )
            
        depths = 1.0 / disps

        depths = repeat(depths, "b v s h w -> b v (h w) srf s",
                        b=b, v=v, srf=self.cfg.num_surfaces)

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        densities = gaussians[..., 2:3].sigmoid()
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size        
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(poses_rev, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians[..., 3:], "b v r srf c -> b v r srf () c"),
            (h, w),
            # input_images = None,
            input_images=context["image"],
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )

        return (Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        ), poses, poses_rev, rearrange(depths, "b v (h w) srf s -> b v (srf s) h w", h=h, w=w), matching_prob,)


    def normalize_image(self, images):
        '''Normalize image to match the pretrained Croco backbone.
            images: (B, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.45, 0.45, 0.45]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.225, 0.225, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std
    
    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler
