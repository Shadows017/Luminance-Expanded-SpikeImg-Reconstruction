import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import AutoImageProcessor, AutoModel

class DINOv3PerceptualLoss(nn.Module):
    """
    L = λg*MSE(CLS) + λp*MSE(Patches) + λGram*MSE(Gram)
    Inputs: (B,C,H,W) or (C,H,W), RGB in [0,1] or [0,255]
    """
    def __init__(
        self,
        ckpt: str,
        input_size: int = 512,
        lambda_global: float = 1.0,
        lambda_patch: float  = 1.0,
        lambda_gram: float   = 0.5,
        device: Optional[str] = None,
        freeze_backbone: bool = True,
        detach_gt: bool = True,
        use_fast_processor: bool = True,
        dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None,           # for newer transformers
        use_auth_token: Optional[bool] = None, # for older transformers
        gram_max_patches: Optional[int] = None # cap tokens for Gram to save mem
    ):
        super().__init__()
        self.ckpt = ckpt
        self.input_size = int(input_size)
        self.lambda_global = float(lambda_global)
        self.lambda_patch  = float(lambda_patch)
        self.lambda_gram   = float(lambda_gram)
        self.detach_gt = bool(detach_gt)
        self.gram_max_patches = gram_max_patches

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                dtype = torch.bfloat16 if major >= 8 else torch.float32
            else:
                dtype = torch.float32
        self.model_dtype = dtype

        proc_kwargs = {"use_fast": use_fast_processor}
        model_kwargs = {}
        if token is not None:
            proc_kwargs["token"] = token
            model_kwargs["token"] = token
        if use_auth_token is not None:
            proc_kwargs["use_auth_token"] = use_auth_token
            model_kwargs["use_auth_token"] = use_auth_token

        self.processor = AutoImageProcessor.from_pretrained(self.ckpt, **proc_kwargs)
        self.backbone  = AutoModel.from_pretrained(self.ckpt, **model_kwargs).to(self.device).eval()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        std  = getattr(self.processor, "image_std",  [0.229, 0.224, 0.225])
        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float32).view(1,3,1,1), persistent=False)
        self.register_buffer("_std",  torch.tensor(std,  dtype=torch.float32).view(1,3,1,1), persistent=False)

        self.register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (C,H,W)
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected 3D/4D tensor, got {tuple(x.shape)}")
        return x

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bchw(x).to(self.device).float()
        aaa=x.max()
        if x.max() > 1.5:  # [0,255] -> [0,1]
            x = x / 255.0
        # ensure 3-channel RGB for ViT
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]   # keep RGB
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)
        return x.to(self.model_dtype)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    def _gram(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D] -> [B, N, N]
        if self.gram_max_patches is not None and tokens.shape[1] > self.gram_max_patches:
            idx = torch.linspace(0, tokens.shape[1]-1, self.gram_max_patches,
                                 device=tokens.device).long()
            tokens = tokens[:, idx, :]
        x = self._l2_normalize(tokens, dim=-1)
        return x @ x.transpose(-1, -2)

    def forward(self, pred_img: torch.Tensor, gt_img: torch.Tensor, return_parts: bool = False):
        px = self._prep(pred_img)
        gx = self._prep(gt_img)

        out_pred = self.backbone(px)
        if self.detach_gt:
            with torch.no_grad():
                out_gt = self.backbone(gx)
        else:
            out_gt = self.backbone(gx)

        # CLS
        po = getattr(out_pred, "pooler_output", None)
        cls_pred = po if po is not None else out_pred.last_hidden_state[:, 0, :]
        po = getattr(out_gt, "pooler_output", None)
        cls_gt = po if po is not None else out_gt.last_hidden_state[:, 0, :]

        # cls_pred = getattr(out_pred, "pooler_output", None) or out_pred.last_hidden_state[:, 0, :]
        # cls_gt   = getattr(out_gt,   "pooler_output", None) or out_gt.last_hidden_state[:, 0, :]

        # Patches (strip CLS + register tokens)
        r = self.register_tokens
        patch_pred = out_pred.last_hidden_state[:, 1 + r:, :]
        patch_gt   = out_gt.last_hidden_state[:, 1 + r:, :]

        Lg = F.mse_loss(cls_pred, cls_gt)
        Lp = F.mse_loss(patch_pred, patch_gt)
        if self.lambda_gram > 0.0:
            Lgram = F.mse_loss(self._gram(patch_pred), self._gram(patch_gt))
        else:
            Lgram = torch.zeros((), device=px.device, dtype=px.dtype)

        loss = self.lambda_global * Lg + self.lambda_patch * Lp + self.lambda_gram * Lgram

        if return_parts:
            parts = {
                "L_global": float(Lg.detach().cpu()),
                "L_patch":  float(Lp.detach().cpu()),
                "L_gram":   float(Lgram.detach().cpu()),
                "L_total":  float(loss.detach().cpu()),
                "num_patches": int(patch_pred.shape[1]),
            }
            return loss, parts
        return loss


class CompositeReconstructionLoss(nn.Module):
    """
    α * DINOv3 perceptual loss
    """
    def __init__(
        self,
        dino_ckpt: str,
        input_size: int = 512,
        w_dino: float = 0.2,             # tune this!
        dino_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.w_dino = float(w_dino)
        self.dino = DINOv3PerceptualLoss(
            ckpt=dino_ckpt,
            input_size=input_size,
            **(dino_kwargs or {})
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_parts: bool = False):
        # DINO expects RGB; the class handles scaling/3ch conversion internally
        dino_loss, parts = self.dino(pred, target, return_parts=True)
        total = self.w_dino * dino_loss
        if return_parts:
            return total, parts
        return total
