from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from .loss import Loss
from .ssim import SSIM
from kornia.geometry.depth import depth_to_3d


@dataclass
class LossReproCfg:
    weight: float
    geo_weight: float


@dataclass
class LossReproCfgWrapper:
    repro: LossReproCfg


def warp_image(img, depth, ref_depth, pose, intrinsic):

    B, _, H, W = img.shape
    P = torch.matmul(intrinsic, pose[:, :3])[:, :3]

    # cam_points = torch.matmul(P, world_points) # w.r.t color camera
    world_points = depth_to_3d(depth, intrinsic, normalize_points=True)
    world_points = torch.cat([world_points, torch.ones(B,1,H,W).type_as(img)], 1)
    cam_points = torch.matmul(P, world_points.view(B, 4, -1)) # w.r.t color camera    
    pix_coords = cam_points[:, :2] / (cam_points[:, 2].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    computed_depth = cam_points[:, 2].unsqueeze(1).view(B, 1, H, W)

    projected_img = torch.nn.functional.grid_sample(img, pix_coords, 
                                                    padding_mode='border',   # 'zeros'
                                                    align_corners=True)    # False
    projected_depth = torch.nn.functional.grid_sample(ref_depth, pix_coords,
                                                      padding_mode='border', 
                                                      align_corners=True)

    return projected_img, projected_depth, computed_depth


class LossRepro(Loss[LossReproCfg, LossReproCfgWrapper]):
    def __init__(self, cfg: LossReproCfgWrapper) -> None:
        super().__init__(cfg)

        self.ssim = SSIM()

    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic):

        ref_img_warped, projected_depth, computed_depth = warp_image(
            ref_img, tgt_depth, ref_depth, pose, intrinsic)

        diff_depth = (computed_depth - projected_depth).abs() / \
            (computed_depth + projected_depth)
        

        valid_mask_ref = (ref_img_warped.abs().mean(
            dim=1, keepdim=True) > 1e-3).float()
        valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask = valid_mask_tgt * valid_mask_ref

        diff_color = (tgt_img-ref_img_warped).abs().mean(dim=1, keepdim=True)        
        # auto masking
        identity_warp_err = (tgt_img-ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

        diff_img = (tgt_img-ref_img_warped).abs().clamp(0, 1)
        ssim_map = self.ssim(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        diff_img = torch.mean(diff_img, dim=1, keepdim=True)

        weight_mask = (1-diff_depth).detach()

        diff_img = diff_img * weight_mask

        return diff_img, diff_color, diff_depth, valid_mask

    def mean_on_mask(self, tensor, mask):
        mask = mask.expand_as(tensor)
        if mask.sum() > 100:
            return (tensor * mask).sum() / mask.sum()
        else:
            return torch.tensor(0, dtype=torch.float32, device=tensor.device)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        pose: Float[Tensor, "b n 4 4"],
        pose_rev: Float[Tensor, "b n 4 4"],
        depths: Float[Tensor, "b v 1 h w"],
        global_step: int,
        val_mode: bool=False,
    ) -> Float[Tensor, ""] | Float[Tensor, "b d 1 h w"]:

        H, W = depths.shape[-2:]
        n = pose.shape[1]

        
        intrinsics = batch["context"]["intrinsics"][:, 0]
        intrinsics_rev = batch["context"]["intrinsics"][:, -1]
        intrinsics[:, 0, :] *= float(W)
        intrinsics[:, 1, :] *= float(H)
        intrinsics_rev[:, 0, :] *= float(W)
        intrinsics_rev[:, 1, :] *= float(H)
        diff_img_list = []
        diff_color_list = []
        diff_depth_list = []
        valid_mask_list = []

        tgt_img = batch["target"]["image"][:, 0]
        ctxt_img = batch["context"]["image"]

        ctx_depths = [prediction.depth[:, 0], prediction.depth[:, -1]]
        
        for i in range(n):
            diff_img1, diff_color1, diff_depth1, valid_mask1 = self.compute_pairwise_loss(
                tgt_img, ctxt_img[:, i], prediction.depth[:, 1], ctx_depths[i], pose[:, i], intrinsics)
            diff_img2, diff_color2, diff_depth2, valid_mask2 = self.compute_pairwise_loss(
                ctxt_img[:, i], tgt_img, ctx_depths[i], prediction.depth[:, 1], pose_rev[:,i], intrinsics_rev)
            diff_img_list += [diff_img1, diff_img2]
            diff_color_list += [diff_color1, diff_color2]
            diff_depth_list += [diff_depth1, diff_depth2]
            valid_mask_list += [valid_mask1, valid_mask2]

        diff_img = torch.cat(diff_img_list, dim=1)
        diff_color = torch.cat(diff_color_list, dim=1)
        diff_depth = torch.cat(diff_depth_list, dim=1)
        valid_mask = torch.cat(valid_mask_list, dim=1)
        indices = torch.argmin(diff_color, dim=1, keepdim=True)

        diff_img = torch.gather(diff_img, 1, indices)
        diff_depth = torch.gather(diff_depth, 1, indices)
        valid_mask = torch.gather(valid_mask, 1, indices)

        photo_loss = self.mean_on_mask(diff_img, valid_mask)
        geometry_loss = self.mean_on_mask(diff_depth, valid_mask)
        repro_loss = photo_loss + self.cfg.geo_weight*geometry_loss

        return self.cfg.weight * repro_loss
