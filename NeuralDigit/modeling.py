from __future__ import annotations
import logging
import random
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import ModelConfig


# Get logger from engine (will be configured there)
log = logging.getLogger("ocr_hybrid")


# UTILITY MODULES
class DropPath(nn.Module):
    # Stochastic depth / drop path for residual connections.
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SEBlock(nn.Module):
    # Squeeze-and-Excitation block for channel attention.
    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        hidden = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class ResBlock(nn.Module):
    # Residual block with SE attention and drop path.
    def __init__(self, c: int, k: int = 3, dropout: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        p = k // 2
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(c, c, k, padding=p, bias=False)

        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = nn.SiLU(inplace=True)
        self.drop2d = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, k, padding=p, bias=False)

        self.se = SEBlock(c, r=8)
        self.gamma = nn.Parameter(torch.ones(c) * 1e-4)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.conv1(self.act1(self.bn1(x)))
        r = self.conv2(self.drop2d(self.act2(self.bn2(r))))
        r = self.se(r)
        r = r * self.gamma.view(1, -1, 1, 1)
        r = self.drop_path(r)
        return x + r



# SPATIAL TRANSFORMER NETWORK (RECTIFIER)
class STNLocalization(nn.Module):
    # Localization network for STN..
    def __init__(self, in_ch: int = 1, hidden_ch: int = 32) -> None:
        super().__init__()
        
        # Localization CNN - lightweight for efficiency
        self.localization = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch * 2, 5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_ch * 2),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Affine parameter regressor: 6 parameters for 2x3 affine matrix
        self.fc = nn.Sequential(
            nn.Linear(hidden_ch * 2, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_ch, 6),
        )
        
        # Initialize to identity transform
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Args: - x: Input image [B, C, H, W], Returns: - theta: Affine parameters [B, 2, 3]
        feat = self.localization(x)
        theta = self.fc(feat)
        theta = theta.view(-1, 2, 3)
        return theta


class SpatialTransformerNetwork(nn.Module):

    def __init__(
        self,
        in_ch: int = 1,
        hidden_ch: int = 32,
        output_size: Tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()
        self.localization = STNLocalization(in_ch, hidden_ch)
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Args: - x: Input image [B, C, H, W], Returns: - x_rect: Rectified image [B, C, H, W] - theta: Applied transformation [B, 2, 3]
        theta = self.localization(x)
        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        # Sample with bilinear interpolation
        x_rect = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return x_rect, theta


class TPSTransformer(nn.Module):
    # Thin Plate Spline (TPS) transformer
    def __init__(
        self,
        in_ch: int = 1,
        hidden_ch: int = 32,
        num_control_points: int = 16,
        output_size: Tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()
        self.num_control_points = num_control_points
        self.output_size = output_size
        
        # For now, fall back to affine (TPS is extension)
        log.warning("TPS transformer requested but not fully implemented. Using affine fallback.")
        self.affine_fallback = SpatialTransformerNetwork(in_ch, hidden_ch, output_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extension point: implement TPS here
        return self.affine_fallback(x)


class Rectifier(nn.Module):
    # Geometric rectification module using STN. Can be enabled/disabled via config.
    def __init__(self, cfg: ModelConfig, in_ch: int = 1) -> None:
        super().__init__()
        self.enabled = cfg.use_stn
        self.mode = cfg.stn_mode
        
        if self.enabled:
            if self.mode == "affine":
                self.transformer = SpatialTransformerNetwork(
                    in_ch=in_ch,
                    hidden_ch=cfg.stn_localization_channels,
                    output_size=(28, 28),
                )
            elif self.mode == "tps":
                self.transformer = TPSTransformer(
                    in_ch=in_ch,
                    hidden_ch=cfg.stn_localization_channels,
                    output_size=(28, 28),
                )
            else:
                raise ValueError(f"Unknown STN mode: {self.mode}")
        else:
            self.transformer = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Args: - x: Input image [B, C, H, W], Returns: - x_rect: Rectified image (or original if disabled) - theta: Transformation parameters (or None if disabled)
        if self.enabled and self.transformer is not None:
            return self.transformer(x)
        return x, None


# CNN TRUNK
class CNNTrunk(nn.Module):
    """ResNet-style CNN trunk producing 7x7 feature maps.
    Architecture:
        Stem: 28x28 -> 28x28 (width channels)
        Stage1: 28x28 (width channels)
        Down1: 28x28 -> 14x14 (width*2 channels)
        Stage2: 14x14 (width*2 channels)
        Down2: 14x14 -> 7x7 (width*4 channels)
        Stage3: 7x7 (width*4 channels)
    Output: [B, width*4, 7, 7]"""
    def __init__(
        self,
        in_ch: int = 1,
        width: int = 64,
        dropout: float = 0.05,
        drop_path_rate: float = 0.10,
        depths: Tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        
        self.out_channels = width * 4
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(depths)
        dp = [drop_path_rate * (i / max(1, total_blocks - 1)) for i in range(total_blocks)]
        dp_i = 0

        def make_stage(c: int, n: int) -> nn.Sequential:
            nonlocal dp_i
            blocks = []
            for _ in range(n):
                blocks.append(ResBlock(c, dropout=dropout, drop_path=dp[dp_i]))
                dp_i += 1
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(width, depths[0])

        self.down1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
        )
        self.stage2 = make_stage(width * 2, depths[1])

        self.down2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
        )
        self.stage3 = make_stage(width * 4, depths[2])

        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Args: - x: Input [B, C, 28, 28], Returns: - feat: Features [B, width*4, 7, 7]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return x


# LOCAL HEAD
class LocalHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
        
        # Store feature for gating
        self._feat = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        feat = self.flatten(self.pool(x))
        self._feat = feat
        logits = self.fc(self.dropout(feat))
        return logits, feat


# GLOBAL HEAD
class PositionalEmbedding2D(nn.Module):
    # Learned embeddings for 2d spatial tokens
    def __init__(self, h: int, w: int, d: int) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.randn(1, h * w, d) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embed


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    
        B, M, D = q.shape
        N = kv.shape[1]
        
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        
        # Project
        q = self.q_proj(q).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine
        out = (attn @ v).transpose(1, 2).reshape(B, M, D)
        out = self.out_proj(out)
        
        return out


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x, x)


class PerceiverBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        token_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(latent_dim, token_dim, num_heads, dropout)
        self.use_self_attn = use_self_attn
        
        if use_self_attn:
            self.self_attn = SelfAttention(latent_dim, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        latents = latents + self.cross_attn(latents, tokens)
        
        # Self-attention (optional)
        if self.use_self_attn:
            latents = latents + self.self_attn(latents)
        
        # FFN
        latents = latents + self.ffn(latents)
        
        return latents


class GlobalHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        num_latents: int = 8,
        num_heads: int = 4,
        use_self_attn: bool = True,
        dropout: float = 0.1,
        spatial_size: int = 7,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.spatial_size = spatial_size
        
        # Project CNN features to token dimension
        self.input_proj = nn.Linear(in_channels, embed_dim)
        
        # Positional embeddings for 7x7 tokens
        self.pos_embed = PositionalEmbedding2D(spatial_size, spatial_size, embed_dim)
        
        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        
        # Perceiver block
        self.perceiver = PerceiverBlock(
            latent_dim=embed_dim,
            token_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_self_attn=use_self_attn,
        )
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        
        # Store feature for gating
        self._feat = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # Tokenize: [B, C, H, W] -> [B, H*W, C] -> [B, H*W, D]
        tokens = x.flatten(2).transpose(1, 2)  # [B, 49, C]
        tokens = self.input_proj(tokens)  # [B, 49, D]
        
        # Add positional embeddings
        tokens = self.pos_embed(tokens)
        
        # Expand latents for batch
        latents = self.latents.expand(B, -1, -1)  # [B, M, D]
        
        # Perceiver processing
        latents = self.perceiver(latents, tokens)  # [B, M, D]
        
        # Pool latents (mean pooling)
        feat = latents.mean(dim=1)  # [B, D]
        self._feat = feat
        
        # Output
        feat_norm = self.norm(feat)
        logits = self.fc(feat_norm)
        
        return logits, feat


# FUSION MODULE
def compute_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    # Compute entropy of probability distribution.
    log_probs = torch.log(probs + eps)
    return -(probs * log_probs).sum(dim=dim)

def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    # Compute margin between top-2 predictions.
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
    if top2.size(-1) < 2:
        return top2[..., 0]
    return top2[..., 0] - top2[..., 1]


class LearnedGatingNetwork(nn.Module):
    # Learned gating network for adaptive fusion. Takes features from both heads plus confidence signals and produces fusion weights.
    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        
        # Input: local_feat, global_feat, local_entropy, global_entropy, margin_diff, agreement
        input_dim = local_dim + global_dim + 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # weights for [local, global]
        )
    
    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
        local_entropy: torch.Tensor,
        global_entropy: torch.Tensor,
        local_margin: torch.Tensor,
        global_margin: torch.Tensor,
        agreement: torch.Tensor,
    ) -> torch.Tensor:
        
        # Normalize features
        local_feat = F.normalize(local_feat, dim=-1)
        global_feat = F.normalize(global_feat, dim=-1)
        
        # Stack scalar features
        margin_diff = (local_margin - global_margin).unsqueeze(-1)
        
        x = torch.cat([
            local_feat,
            global_feat,
            local_entropy.unsqueeze(-1),
            global_entropy.unsqueeze(-1),
            margin_diff,
            agreement.unsqueeze(-1),
        ], dim=-1)
        
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        
        return weights


class FusionModule(nn.Module):
    """
    Probabilistic fusion of local and global expert predictions. - My personal favorite part of the model!
    Gating modes:
        - fixed: Always 0.5/0.5 weights
        - entropy: Weights based on prediction entropy (lower = more confident)
        - learned: Learned gating network
    Features:
        - Temperature scaling for calibration
        - Weight floor to prevent head collapse
        - Optional entropy regularization on weights
        - Warmup support (fixed weights during warmup)
    Failure modes:
        - One head can dominate if not regularized
        - Learned gating can overfit
    Mitigation:
        - Weight floor (w_min)
        - Auxiliary losses on both heads
        - Gating warmup
    """
    def __init__(
        self,
        num_classes: int,
        local_feat_dim: int,
        global_feat_dim: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        
        self.gating_mode = cfg.gating_mode
        self.weight_floor = cfg.gating_weight_floor
        self.entropy_reg = cfg.gating_entropy_reg
        self.gating_temperature = cfg.gating_temperature
        
        # Temperature scaling for calibration
        if cfg.use_calibration:
            if cfg.trainable_temps:
                self.local_temp = nn.Parameter(torch.tensor(cfg.local_temp_init))
                self.global_temp = nn.Parameter(torch.tensor(cfg.global_temp_init))
            else:
                self.register_buffer('local_temp', torch.tensor(cfg.local_temp_init))
                self.register_buffer('global_temp', torch.tensor(cfg.global_temp_init))
        else:
            self.register_buffer('local_temp', torch.tensor(1.0))
            self.register_buffer('global_temp', torch.tensor(1.0))
        
        # Learned gating network
        if cfg.gating_mode == "learned":
            self.gating_net = LearnedGatingNetwork(local_feat_dim, global_feat_dim)
        else:
            self.gating_net = None
        
        # State
        self._warmup_active = True
        self._last_weights = None
        self._local_entropy = None
        self._global_entropy = None
    
    def set_warmup(self, active: bool) -> None:
        # Enable/disable warmup mode (fixed 0.5/0.5 weights).
        self._warmup_active = active
    
    def forward(
        self,
        local_logits: torch.Tensor,
        global_logits: torch.Tensor,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        B = local_logits.size(0)
        
        # Apply temperature scaling
        local_scaled = local_logits / self.local_temp.clamp(min=0.1)
        global_scaled = global_logits / self.global_temp.clamp(min=0.1)
        
        # Convert to probabilities
        local_probs = F.softmax(local_scaled, dim=-1)
        global_probs = F.softmax(global_scaled, dim=-1)
        
        # Compute confidence signals
        local_entropy = compute_entropy(local_probs)
        global_entropy = compute_entropy(global_probs)
        local_margin = compute_margin(local_probs)
        global_margin = compute_margin(global_probs)
        
        # Agreement
        local_pred = local_probs.argmax(dim=-1)
        global_pred = global_probs.argmax(dim=-1)
        agreement = (local_pred == global_pred).float()
        
        # Store for logging
        self._local_entropy = local_entropy
        self._global_entropy = global_entropy
        
        # Compute gating weights
        if self._warmup_active or self.gating_mode == "fixed":
            # Fixed 0.5/0.5 weights
            weights = torch.full((B, 2), 0.5, device=local_logits.device)
        elif self.gating_mode == "entropy":
            # Entropy-based: lower entropy = higher weight
            # Softmax over negative entropies
            neg_entropies = torch.stack([-local_entropy, -global_entropy], dim=-1)
            weights = F.softmax(neg_entropies / self.gating_temperature, dim=-1)
        elif self.gating_mode == "learned" and self.gating_net is not None:
            weights = self.gating_net(
                local_feat, global_feat,
                local_entropy, global_entropy,
                local_margin, global_margin,
                agreement,
            )
        else:
            weights = torch.full((B, 2), 0.5, device=local_logits.device)
        
        # Apply weight floor
        weights = weights.clamp(min=self.weight_floor)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize
        
        self._last_weights = weights
        
        # Fuse in probability space
        w_local = weights[:, 0:1]  # [B, 1]
        w_global = weights[:, 1:2]  # [B, 1]
        
        fused_probs = w_local * local_probs + w_global * global_probs
        
        # Convert to log probs for stable NLL
        fused_logprobs = torch.log(fused_probs.clamp(min=1e-8))
        
        # Compute statistics
        stats = {
            'gating_weight_local_mean': weights[:, 0].mean(),
            'gating_weight_local_std': weights[:, 0].std(),
            'gating_weight_global_mean': weights[:, 1].mean(),
            'gating_weight_global_std': weights[:, 1].std(),
            'disagreement_rate': 1.0 - agreement.mean(),
            'local_entropy_mean': local_entropy.mean(),
            'global_entropy_mean': global_entropy.mean(),
            'local_temp': self.local_temp.detach(),
            'global_temp': self.global_temp.detach(),
        }
        
        # Gating entropy regularization (encourage diverse weights)
        if self.entropy_reg > 0 and not self._warmup_active:
            gating_entropy = compute_entropy(weights, dim=-1).mean()
            stats['gating_entropy_loss'] = -self.entropy_reg * gating_entropy
        else:
            stats['gating_entropy_loss'] = torch.tensor(0.0, device=local_logits.device)
        
        return fused_logprobs, fused_probs, stats


# AUXILIARY TOPOLOGY TASK
class AuxiliaryTopologyHead(nn.Module):
    """
    Auxiliary task for topology learning.
    
    Predicts structural maps from CNN features to force
    the network to learn stroke topology.
    
    Task types:
        - distance_transform: Predict distance to nearest stroke
        - skeleton: Predict medial axis / skeleton
        - endpoints: Predict stroke endpoints and junctions
    
    This is optional and can be disabled for ablation.
    """
    def __init__(
        self,
        in_channels: int,
        task_type: str = "distance_transform",
        output_size: int = 28,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.output_size = output_size
        
        # Upsampling decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
        )
        
        if task_type in ["distance_transform", "skeleton"]:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Sigmoid()
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pred = self.decoder(feat)
        pred = self.output_activation(pred)
        return pred
    
    @staticmethod
    def compute_target(
        x: torch.Tensor,
        task_type: str,
    ) -> torch.Tensor:
        # Denormalize to [0, 1] approximately
        x_01 = x.clamp(-3, 3)
        x_01 = (x_01 - x_01.min()) / (x_01.max() - x_01.min() + 1e-8)
        
        if task_type == "distance_transform":
            # Approximate distance transform with blur
            # More blur = further from stroke
            kernel_size = 5
            blurred = F.avg_pool2d(
                F.pad(x_01, (kernel_size//2,)*4, mode='replicate'),
                kernel_size, stride=1
            )
            # Invert: high value = close to stroke
            target = blurred
        elif task_type == "skeleton":
            # Approximate skeleton with Laplacian-like operation
            # Detect ridges (local maxima along stroke)
            laplacian = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            edge = F.conv2d(x_01, laplacian, padding=1).abs()
            target = (edge / (edge.max() + 1e-8)).clamp(0, 1)
        elif task_type == "endpoints":
            # Approximate endpoint detection with gradient magnitude
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            gx = F.conv2d(x_01, sobel_x, padding=1)
            gy = F.conv2d(x_01, sobel_y, padding=1)
            grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            target = (grad_mag / (grad_mag.max() + 1e-8)).clamp(0, 1)
        else:
            target = x_01
        
        return target


# COMPLETE OCR MODEL
class OCRHybridExpert(nn.Module):
    # Complete OCR-grade hybrid expert model with multi-scale specialization.
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 10,
        cfg: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__()
        
        if cfg is None:
            cfg = ModelConfig()
        
        self.cfg = cfg
        self.num_classes = num_classes
        
        # Apply ablation preset
        self._apply_ablation_preset()
        
        # Rectifier
        self.rectifier = Rectifier(cfg, in_ch)
        
        # Trunk - we'll access intermediate stages
        self.trunk = CNNTrunk(
            in_ch=in_ch,
            width=cfg.width,
            dropout=cfg.dropout,
            drop_path_rate=cfg.drop_path_rate,
            depths=cfg.depths,
        )
        
        trunk_out_ch = self.trunk.out_channels  # 256 (width * 4)
        stage2_ch = cfg.width * 2  # 128
        
        # Multi-scale vs Standard 
        self.use_multiscale = cfg.use_multiscale_heads and cfg.ablation not in ["baseline_cnn", "global_only"]
        
        # Head adapters
        self.use_adapters = cfg.use_head_adapters
        
        # Local head
        self.use_local = cfg.ablation != "global_only"
        if self.use_local:
            if self.use_multiscale:
                # Local head from stage2 (14×14, 128ch)
                if self.use_adapters:
                    self.local_adapter = HeadAdapter(stage2_ch, cfg.adapter_hidden_dim, stage2_ch)
                self.local_head = MultiScaleLocalHead(
                    in_channels=stage2_ch,
                    num_classes=num_classes,
                    hidden_channels=trunk_out_ch,  # Match global capacity
                    dropout=cfg.local_head_dropout,
                )
                self.local_feat_dim = trunk_out_ch
            else:
                # Standard local head from stage3
                if self.use_adapters:
                    self.local_adapter = HeadAdapter(trunk_out_ch, cfg.adapter_hidden_dim)
                self.local_head = LocalHead(
                    in_channels=trunk_out_ch,
                    num_classes=num_classes,
                    dropout=cfg.local_head_dropout,
                )
                self.local_feat_dim = trunk_out_ch
        else:
            self.local_head = None
            self.local_feat_dim = 0
        
        # Global head
        self.use_global = cfg.ablation != "baseline_cnn"
        if self.use_global:
            if self.use_adapters:
                self.global_adapter = HeadAdapter(trunk_out_ch, cfg.adapter_hidden_dim)
            self.global_head = GlobalHead(
                in_channels=trunk_out_ch,
                num_classes=num_classes,
                embed_dim=cfg.global_embed_dim,
                num_latents=cfg.global_num_latents,
                num_heads=cfg.global_num_heads,
                use_self_attn=cfg.global_use_self_attn,
                dropout=cfg.global_dropout,
                spatial_size=7,
            )
            self.global_feat_dim = cfg.global_embed_dim
        else:
            self.global_head = None
            self.global_feat_dim = 0
        
        # Fusion
        self.use_fusion = self.use_local and self.use_global
        if self.use_fusion:
            self.fusion = FusionModule(
                num_classes=num_classes,
                local_feat_dim=self.local_feat_dim,
                global_feat_dim=self.global_feat_dim,
                cfg=cfg,
            )
        else:
            self.fusion = None
        
        # Auxiliary topology head (from stage3 features)
        self.use_aux_topology = cfg.use_aux_topology
        if self.use_aux_topology:
            self.aux_topology = AuxiliaryTopologyHead(
                in_channels=trunk_out_ch,
                task_type=cfg.aux_topology_type,
            )
        else:
            self.aux_topology = None
        
        # head dropout for regularization
        self.head_dropout_prob = cfg.head_dropout_prob
        
        # For ONNX export mode
        self._onnx_mode = False
    
    def _apply_ablation_preset(self) -> None:
        # Apply ablation preset to config.
        cfg = self.cfg
        
        if cfg.ablation == "baseline_cnn":
            pass
        elif cfg.ablation == "global_only":
            pass
        elif cfg.ablation == "hybrid_fixed":
            cfg.gating_mode = "fixed"
        elif cfg.ablation == "hybrid_entropy":
            cfg.gating_mode = "entropy"
        elif cfg.ablation == "hybrid_learned":
            cfg.gating_mode = "learned"
    
    def set_onnx_mode(self, mode: bool) -> None:
        self._onnx_mode = mode
    
    def set_gating_warmup(self, active: bool) -> None:
        if self.fusion is not None:
            self.fusion.set_warmup(active)
    
    def _trunk_forward_multiscale(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward through trunk, returning both stage2 and stage3 features.

        x = self.trunk.stem(x)
        x = self.trunk.stage1(x)
        x = self.trunk.down1(x)
        feat_stage2 = self.trunk.stage2(x)  # 14×14, 128ch
        
        x = self.trunk.down2(feat_stage2)
        feat_stage3 = self.trunk.stage3(x)  # 7×7, 256ch
        
        return feat_stage2, feat_stage3
    
    def forward(
        self,
        x: torch.Tensor,
        return_parts: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # ONNX mode: simple forward
        if self._onnx_mode:
            if self.cfg.use_stn:
                x, _ = self.rectifier(x)
            feat = self.trunk(x)
            if self.local_head is not None:
                if self.use_adapters and hasattr(self, 'local_adapter'):
                    feat = self.local_adapter(feat)
                logits, _ = self.local_head(feat)
                return logits
            elif self.global_head is not None:
                if self.use_adapters and hasattr(self, 'global_adapter'):
                    feat = self.global_adapter(feat)
                logits, _ = self.global_head(feat)
                return logits
            else:
                raise RuntimeError("No head available in ONNX mode")
        
        # Store original for aux task
        x_orig = x
        x_rect, theta = self.rectifier(x)

        if self.use_multiscale and self.use_local:
            feat_stage2, feat_stage3 = self._trunk_forward_multiscale(x_rect)
            feat_local_input = feat_stage2
            feat_global_input = feat_stage3
        else:
            feat_stage3 = self.trunk(x_rect)
            feat_local_input = feat_stage3
            feat_global_input = feat_stage3
        
        results = {
            'theta': theta,
            'feat': feat_stage3,  # For topology head
        }
        
        # Head dropout during training
        drop_local = False
        drop_global = False
        if self.training and self.head_dropout_prob > 0 and self.use_fusion:
            if random.random() < self.head_dropout_prob:
                # Randomly drop one head
                if random.random() < 0.5:
                    drop_local = True
                else:
                    drop_global = True
        
        # Local head
        local_logits = None
        local_feat = None
        if self.use_local and self.local_head is not None and not drop_local:
            feat_l = feat_local_input
            if self.use_adapters and hasattr(self, 'local_adapter'):
                feat_l = self.local_adapter(feat_l)
            local_logits, local_feat = self.local_head(feat_l)
            results['local_logits'] = local_logits
            results['local_feat'] = local_feat
            results['local_probs'] = F.softmax(local_logits, dim=-1)
        
        # Global head
        global_logits = None
        global_feat = None
        if self.use_global and self.global_head is not None and not drop_global:
            feat_g = feat_global_input
            if self.use_adapters and hasattr(self, 'global_adapter'):
                feat_g = self.global_adapter(feat_g)
            global_logits, global_feat = self.global_head(feat_g)
            results['global_logits'] = global_logits
            results['global_feat'] = global_feat
            results['global_probs'] = F.softmax(global_logits, dim=-1)
        
        # Fusion
        if self.use_fusion and self.fusion is not None:
            if local_logits is not None and global_logits is not None:
                # Normal fusion
                fused_logprobs, fused_probs, fusion_stats = self.fusion(
                    local_logits, global_logits, local_feat, global_feat
                )
                results['fused_logprobs'] = fused_logprobs
                results['fused_probs'] = fused_probs
                results['fused_logits'] = fused_logprobs
                results.update(fusion_stats)
            elif local_logits is not None:
                # Only local (global dropped)
                results['fused_logits'] = local_logits
                results['fused_probs'] = F.softmax(local_logits, dim=-1)
                results['fused_logprobs'] = F.log_softmax(local_logits, dim=-1)
                results['head_dropped'] = 'global'
            elif global_logits is not None:
                # Only global (local dropped)
                results['fused_logits'] = global_logits
                results['fused_probs'] = F.softmax(global_logits, dim=-1)
                results['fused_logprobs'] = F.log_softmax(global_logits, dim=-1)
                results['head_dropped'] = 'local'
        else:
            # Single head mode
            if local_logits is not None:
                results['fused_logits'] = local_logits
                results['fused_probs'] = F.softmax(local_logits, dim=-1)
                results['fused_logprobs'] = F.log_softmax(local_logits, dim=-1)
            elif global_logits is not None:
                results['fused_logits'] = global_logits
                results['fused_probs'] = F.softmax(global_logits, dim=-1)
                results['fused_logprobs'] = F.log_softmax(global_logits, dim=-1)
        
        # Auxiliary topology
        if self.use_aux_topology and self.aux_topology is not None:
            aux_pred = self.aux_topology(feat_stage3)
            results['aux_topology_pred'] = aux_pred
        
        if return_parts:
            return results
        else:
            return results.get('fused_logits', results.get('local_logits', results.get('global_logits')))



# ONNX EXPORT WRAPPER
class ONNXExportWrapper(nn.Module):
    #ONNX-friendly wrapper that exports only the local head path.
    def __init__(self, model: OCRHybridExpert) -> None:
        super().__init__()
        
        # Copy necessary components
        self.use_stn = model.cfg.use_stn
        if self.use_stn:
            self.rectifier = model.rectifier
        self.trunk = model.trunk
        
        if model.local_head is not None:
            self.head = model.local_head
        elif model.global_head is not None:
            log.warning("ONNX export: No local head, using global head (may have compatibility issues)")
            self.head = model.global_head
        else:
            raise RuntimeError("No head available for ONNX export")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_stn:
            x, _ = self.rectifier(x)
        feat = self.trunk(x)
        logits, _ = self.head(feat)
        return logits
    

# HEAD SPECIFIC CLASSES
class HeadAdapter(nn.Module):
    """
    Head-specific feature adapter.
    Transforms shared trunk features into head-specific representations.
    This allows each head to "see" different aspects of the same features.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        out_channels = out_channels or in_channels
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Residual scaling - start small
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.adapter(x)


class MultiScaleLocalHead(nn.Module):
    #Captures fine details
    def __init__(
        self,
        in_channels: int,  # 128 from stage2
        num_classes: int,
        hidden_channels: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        
        # Refinement to match global head capacity
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, num_classes)
        
        self.out_features = hidden_channels
        self._feat = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.refine(x)
        feat = self.flatten(self.pool(x))
        self._feat = feat
        logits = self.fc(self.dropout(feat))
        return logits, feat