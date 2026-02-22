from __future__ import annotations
import contextlib
import copy
import logging
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
#from .adaptive_head_controller import AdaptiveHeadController, HeadControllerConfig

from .configs import (
    AugmentConfig,
    EMADecayConfig,
    ModelConfig,
    RuntimeToggles,
    TrainConfig,
)
from .dataio import _dataset_specs, build_loaders
from .modeling import AuxiliaryTopologyHead, OCRHybridExpert

#try:                   -NOT INCLUDED IN THIS COMMIT - 
    #from NeuralDigit    .dashboard import TrainingDashboard
    #DASHBOARD_AVAILABLE = True
#except ImportError as e:
    #print(f"Dashboard not available: {e}")
    #DASHBOARD_AVAILABLE = False


# LOGGING CONFIGURATION
def _configure_logging() -> logging.Logger:
    # Configure logging with console output.
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(formatter)
    
    log = logging.getLogger("ocr_hybrid")
    log.handlers.clear()
    log.addHandler(console_handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    
    return log


# Initialize the logger
log = _configure_logging()

# LOGGING HELPERS 
ACCURACY_EMOJI = {
    0.99: "✨🌟💫",  # - legendary
    0.98: "✨🌟",  # - incredible  
    0.97: "✨",  #  - excellent
    0.96: "🌳",  # Matured Tree - great
    0.95: "🌲",  # Tree - solid
    0.90: "🪴",  # Baby Tree - progressing
    0.80: "🌿",  # Big Sprout - growing
    0.00: "🌱",  # Sprout - starting
}

def get_accuracy_emoji(acc: float) -> str:
    # Get emoji indicator for accuracy level.
    for threshold, emoji in sorted(ACCURACY_EMOJI.items(), reverse=True):
        if acc >= threshold:
            return emoji
    return "🌱"


def format_gating_bar(local_w: float, global_w: float, width: int = 20) -> str:
    # Create visual gating balance bar.
    # Normalize to ensure they sum to 1
    total = local_w + global_w
    if total > 0:
        local_w = local_w / total
        global_w = global_w / total
    else:
        local_w = global_w = 0.5
    
    local_chars = int(local_w * width)
    bar = '▓' * local_chars + '░' * (width - local_chars)
    return f"[{bar}] L:{local_w:.2f}|G:{global_w:.2f}"


def format_time(seconds: float) -> str:
    # Format seconds into human readable string.
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_eta(avg_epoch_time: float, current_epoch: int, total_epochs: int) -> str:
    # Calculate and format ETA.
    remaining = total_epochs - current_epoch
    eta_seconds = avg_epoch_time * remaining
    return format_time(eta_seconds)


# LOSS FUNCTIONS
class HybridLoss(nn.Module):
    """
    Enhanced hybrid loss with specialization-inducing components.
    Components:
        1. Primary: NLL on fused predictions
        2. Aux local: CE on local head (asymmetric weight)
        3. Aux global: CE on global head (asymmetric weight)
        4. Aux topology: MSE on topology prediction
        5. Diversity: JS divergence between heads (maximize)
        6. NCL: Negative Correlation Learning (penalize confident agreement)
        7. Orthogonal: Feature orthogonality between heads
    """
    def __init__(
        self,
        cfg: ModelConfig,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.label_smoothing = label_smoothing
        
        # Auxiliary weights (asymmetric)
        self.aux_local_weight = cfg.aux_local_weight
        self.aux_global_weight = cfg.aux_global_weight
        self.aux_topology_weight = cfg.aux_topology_weight
        
        # Specialization losses
        self.diversity_weight = cfg.diversity_weight
        self.ncl_weight = cfg.ncl_weight
        self.orthogonal_weight = cfg.orthogonal_weight

        # NEW: Adaptive Head Controller
        self.use_adaptive_controller = True  # true to test
        if self.use_adaptive_controller:
            from .adaptive_head_controller import AdaptiveHeadController, HeadControllerConfig
            self.head_controller = AdaptiveHeadController(HeadControllerConfig(
                error_diversity_weight=0.15,
                rescue_bonus_weight=0.10,
                enable_adaptive_aux=True,
                aux_weight_min=0.15,
                aux_weight_max=0.60,
                aux_adaptation_rate=0.02,
                confidence_shaping_weight=0.05,
                hard_sample_diversity_bonus=2.0,
            ))
        else:
            self.head_controller = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mixup_targets: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
        x_orig: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute total loss with all components.
        loss_dict = {}
        device = targets.device
        total_loss = torch.tensor(0.0, device=device)
        
        # Check if head was dropped
        head_dropped = outputs.get('head_dropped', None)
        
        # STEP 0: Run Adaptive Controller FIRST to get dynamic weights
        adaptive_local_w = self.aux_local_weight   # Default to static
        adaptive_global_w = self.aux_global_weight  # Default to static
        
        if (self.head_controller is not None and 
            'local_logits' in outputs and 
            'global_logits' in outputs and
            head_dropped is None):  # Only when both heads active
            
            # Get the target for adaptive losses (use primary target for mixup)
            adaptive_target = mixup_targets[0] if mixup_targets is not None else targets
            
            # Compute adaptive losses AND update internal weight trackers
            adaptive_loss, adaptive_dict = self.head_controller.compute_losses(
                local_logits=outputs['local_logits'],
                global_logits=outputs['global_logits'],
                fused_logits=outputs.get('fused_logits', outputs.get('fused_logprobs', outputs['local_logits'])),
                targets=adaptive_target,
                local_probs=outputs.get('local_probs'),
                global_probs=outputs.get('global_probs'),
            )
            
            # Add adaptive losses to dict and total
            for k, v in adaptive_dict.items():
                loss_dict[k] = v
            total_loss = total_loss + adaptive_loss
            
            # Get the adaptive weights for aux losses 
            if self.head_controller.cfg.enable_adaptive_aux:
                adaptive_local_w, adaptive_global_w = self.head_controller.get_adaptive_aux_weights()
                loss_dict['adaptive_local_aux_w'] = torch.tensor(adaptive_local_w)
                loss_dict['adaptive_global_aux_w'] = torch.tensor(adaptive_global_w)
        
        # STEP 1: Primary loss on fused predictions
        if 'fused_logprobs' in outputs:
            if mixup_targets is not None:
                y_a, y_b, lam = mixup_targets
                loss_fused = lam * F.nll_loss(outputs['fused_logprobs'], y_a) + \
                             (1 - lam) * F.nll_loss(outputs['fused_logprobs'], y_b)
            else:
                loss_fused = F.nll_loss(outputs['fused_logprobs'], targets)
            loss_dict['loss_fused'] = loss_fused
            total_loss = total_loss + loss_fused
        
        # STEP 2: Auxiliary local head loss (NOW USES ADAPTIVE WEIGHT)
        if 'local_logits' in outputs and adaptive_local_w > 0:
            if mixup_targets is not None:
                y_a, y_b, lam = mixup_targets
                loss_local = lam * F.cross_entropy(outputs['local_logits'], y_a, label_smoothing=self.label_smoothing) + \
                             (1 - lam) * F.cross_entropy(outputs['local_logits'], y_b, label_smoothing=self.label_smoothing)
            else:
                loss_local = F.cross_entropy(outputs['local_logits'], targets, label_smoothing=self.label_smoothing)
            loss_dict['loss_local'] = loss_local
            
            # Apply weight: use adaptive weight, with boost if other head was dropped
            weight = adaptive_local_w * (2.0 if head_dropped == 'global' else 1.0)
            loss_dict['loss_local_weighted'] = weight * loss_local
            total_loss = total_loss + weight * loss_local
        
        # STEP 3: Auxiliary global head loss  (USES ADAPTIVE WEIGHT)
        if 'global_logits' in outputs and adaptive_global_w > 0:
            if mixup_targets is not None:
                y_a, y_b, lam = mixup_targets
                loss_global = lam * F.cross_entropy(outputs['global_logits'], y_a, label_smoothing=self.label_smoothing) + \
                              (1 - lam) * F.cross_entropy(outputs['global_logits'], y_b, label_smoothing=self.label_smoothing)
            else:
                loss_global = F.cross_entropy(outputs['global_logits'], targets, label_smoothing=self.label_smoothing)
            loss_dict['loss_global'] = loss_global
            
            # Apply weight: use adaptive weight, with boost if other head was dropped
            weight = adaptive_global_w * (2.0 if head_dropped == 'local' else 1.0)
            loss_dict['loss_global_weighted'] = weight * loss_global
            total_loss = total_loss + weight * loss_global
        
        # STEP 4: Auxiliary topology loss (unchanged)
        if 'aux_topology_pred' in outputs and self.aux_topology_weight > 0 and x_orig is not None:
            target_map = AuxiliaryTopologyHead.compute_target(x_orig, self.cfg.aux_topology_type)
            loss_topology = F.mse_loss(outputs['aux_topology_pred'], target_map)
            loss_dict['loss_topology'] = loss_topology
            total_loss = total_loss + self.aux_topology_weight * loss_topology
        
        # STEP 5: Gating entropy regularization (unchanged)
        if 'gating_entropy_loss' in outputs:
            total_loss = total_loss + outputs['gating_entropy_loss']
            loss_dict['loss_gating_entropy'] = outputs['gating_entropy_loss']
        
        # STEP 6: Legacy diversity losses (KEEP but remember bro, you can disable)
        # Only run if adaptive controller is NOT handling diversity
        use_legacy_diversity = (self.head_controller is None) or (not self.head_controller.cfg.error_diversity_weight > 0)
        
        if use_legacy_diversity and 'local_probs' in outputs and 'global_probs' in outputs:
            local_p = outputs['local_probs']
            global_p = outputs['global_probs']
            
            # JS Divergence diversity
            if self.diversity_weight > 0:
                m = 0.5 * (local_p + global_p)
                eps = 1e-8
                kl_lm = (local_p * (torch.log(local_p + eps) - torch.log(m + eps))).sum(dim=-1)
                kl_gm = (global_p * (torch.log(global_p + eps) - torch.log(m + eps))).sum(dim=-1)
                js_div = 0.5 * (kl_lm + kl_gm)
                
                diversity_loss = -self.diversity_weight * js_div.mean()
                loss_dict['loss_diversity'] = diversity_loss
                total_loss = total_loss + diversity_loss
            
            # NCL (negative correlation learning)
            if self.ncl_weight > 0:
                local_conf = local_p.max(dim=-1).values
                global_conf = global_p.max(dim=-1).values
                
                local_pred = local_p.argmax(dim=-1)
                global_pred = global_p.argmax(dim=-1)
                agreement = (local_pred == global_pred).float()
                
                ncl = local_conf * global_conf * agreement
                ncl_loss = self.ncl_weight * ncl.mean()
                loss_dict['loss_ncl'] = ncl_loss
                total_loss = total_loss + ncl_loss
            
            # Feature orthogonality
            if self.orthogonal_weight > 0 and 'local_feat' in outputs and 'global_feat' in outputs:
                local_feat = outputs['local_feat']
                global_feat = outputs['global_feat']
                
                min_dim = min(local_feat.size(-1), global_feat.size(-1))
                local_feat_proj = F.normalize(local_feat[..., :min_dim], dim=-1)
                global_feat_proj = F.normalize(global_feat[..., :min_dim], dim=-1)
                
                cos_sim = (local_feat_proj * global_feat_proj).sum(dim=-1)
                ortho_loss = self.orthogonal_weight * cos_sim.abs().mean()
                loss_dict['loss_orthogonal'] = ortho_loss
                total_loss = total_loss + ortho_loss
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


# EMA (Exponential Moving Average) Utilites
class ModelEMA:
    # Exponential Moving Average of model parameters.
    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    def set_decay(self, decay: float) -> None:
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        esd = self.module.state_dict()
        for k in esd.keys():
            if not torch.is_floating_point(esd[k]):
                esd[k].copy_(msd[k])
            else:
                esd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)


class EMADecayController:
    # Adaptive EMA decay controller.
    def __init__(
        self,
        total_ema_updates: int,
        cfg: EMADecayConfig = EMADecayConfig(),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.log = logger
        self.T = max(1, int(total_ema_updates))
        self.step_idx = 0

        h_final = int(round(self.cfg.h_final_frac * self.T))
        self.h_final = int(max(self.cfg.h_min, min(self.cfg.h_max, h_final)))

        ramp = int(round(self.cfg.ramp_frac * self.T))
        self.ramp = int(max(self.cfg.ramp_min, min(self.cfg.ramp_max, ramp)))

        self.h_start = int(max(1, self.cfg.h_start))

        self.r_ema = None
        self.r_baseline = None
        self._baseline_count = 0

    def set_step(self, ema_step_idx: int) -> None:
        self.step_idx = int(max(0, ema_step_idx))

    @staticmethod
    def _half_life_to_decay(H: float) -> float:
        return float(math.exp(math.log(0.5) / max(1e-8, float(H))))

    @staticmethod
    def _smooth_ramp(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def _scheduled_half_life(self) -> float:
        if self.step_idx <= 0:
            return float(self.h_start)
        if self.step_idx >= self.ramp:
            return float(self.h_final)
        u = self._smooth_ramp(self.step_idx / float(self.ramp))
        return float(self.h_start + u * (self.h_final - self.h_start))

    def _noise_adjust_half_life(self, H: float, r_signal: Optional[float]) -> float:
        if self.cfg.noise_strength <= 0.0 or r_signal is None:
            return H

        r = float(max(0.0, r_signal))

        if self.r_ema is None:
            self.r_ema = r
        else:
            b = float(self.cfg.noise_ema_beta)
            self.r_ema = b * self.r_ema + (1.0 - b) * r

        if self._baseline_count < int(self.cfg.baseline_warmup_updates):
            if self.r_baseline is None:
                self.r_baseline = self.r_ema
            else:
                self.r_baseline = 0.995 * self.r_baseline + 0.005 * self.r_ema
            self._baseline_count += 1

        if self.r_baseline is None or self.r_baseline <= 1e-12:
            return H

        ratio = self.r_ema / self.r_baseline
        ratio = max(0.05, min(20.0, ratio))

        if ratio >= 1.0:
            strength = min(1.0, (ratio - 1.0) / 3.0)
            mult = 1.0 / (1.0 + strength * (self.cfg.shrink_max - 1.0))
        else:
            strength = min(1.0, (1.0 - ratio) / 0.7)
            mult = 1.0 + strength * (self.cfg.grow_max - 1.0)

        mult = (1.0 - self.cfg.noise_strength) * 1.0 + self.cfg.noise_strength * mult
        return float(H * mult)

    def step(self, r_signal: Optional[float]) -> float:
        H = self._scheduled_half_life()
        H = self._noise_adjust_half_life(H, r_signal)

        d = self._half_life_to_decay(H)
        d = max(self.cfg.decay_min, min(self.cfg.decay_max, d))

        self.step_idx += 1
        return float(d)


# TRAINING UTILITIES
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    # Apply MixUp augmentation.
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def seed_everything(seed: int) -> None:
    # Set random seeds for reproducibility.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # Get available compute device.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_use_amp(toggles: RuntimeToggles, device: torch.device) -> bool:
    # Check if AMP should be used.
    return bool(toggles.use_amp and device.type == "cuda")


def apply_runtime_toggles(t: RuntimeToggles, device: torch.device) -> None:
    # Apply runtime optimization toggles.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(t.cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(t.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(t.allow_tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# CHECKPOINT UTILITIES
def save_checkpoint(
    path: Path,
    student: nn.Module,
    ema: Optional[nn.Module],
    optimizer: Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    epoch: int,
    best_acc: float,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    toggles: RuntimeToggles,
    model_meta: Dict[str, object],
) -> None:
    # Save training checkpoint. 
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg_dict = asdict(cfg)
    cfg_dict["data_dir"] = str(cfg_dict["data_dir"])
    cfg_dict["out_dir"] = str(cfg_dict["out_dir"])

    payload: Dict[str, object] = {
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "student": student.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg_dict,
        "model_cfg": asdict(model_cfg),
        "toggles": asdict(toggles),
        "model_meta": model_meta,
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    student: nn.Module,
    ema: Optional[nn.Module],
    optimizer: Optional[Optimizer],
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Tuple[int, float, Dict[str, object]]:
    # Load training checkpoint.
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    meta: Dict[str, object] = {}
    if isinstance(ckpt, dict):
        meta = dict(ckpt.get("model_meta", {}) or {})

    if isinstance(ckpt, dict) and "student" in ckpt:
        student.load_state_dict(ckpt["student"])
        if ema is not None and ckpt.get("ema", None) is not None:
            ema.load_state_dict(ckpt["ema"])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        student.load_state_dict(ckpt["model"])
    else:
        student.load_state_dict(ckpt)

    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler", None) is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    best_acc = float(ckpt.get("best_acc", -1.0)) if isinstance(ckpt, dict) else -1.0
    return epoch, best_acc, meta


def make_warmup_cosine_scheduler(
    optimizer: Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> LambdaLR:
    # Create warmup + cosine decay learning rate scheduler.
    warmup_epochs = max(0, int(warmup_epochs))
    epochs = max(1, int(epochs))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch_idx: int) -> float:
        e = int(epoch_idx)
        if warmup_epochs > 0 and e < warmup_epochs:
            return float(e + 1) / float(warmup_epochs)
        t = max(0, e - warmup_epochs)
        T = max(1, epochs - warmup_epochs)
        cos = 0.5 * (1.0 + math.cos(math.pi * float(t) / float(T)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# TRAINER
class Trainer:
    """
    Training orchestrator for hybrid expert OCR model.
    Features:
        - AMP support (BF16/FP16)
        - EMA with adaptive decay
        - MixUp augmentation
        - Gating warmup schedule
        - Per-head metrics logging
        - Gradient clipping
    """
    _OPTIMIZER_BETAS: Tuple[float, float] = (0.9, 0.99)

    def __init__(
        self,
        cfg: TrainConfig,
        model_cfg: ModelConfig,
        toggles: RuntimeToggles,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.toggles = toggles
        self.device = get_device()

        log.info("=" * 80)
        log.info("OCR HYBRID - INITIALIZATION")
        log.info(f"Device: {self.device}")
        log.info(f"Dataset: {cfg.dataset.upper()}")
        log.info(f"Ablation mode: {model_cfg.ablation}")
        log.info("=" * 80)

        # Validation
        if not (0.0 <= float(cfg.mixup_p) <= 1.0):
            raise ValueError(f"mixup_p must be in [0,1], got {cfg.mixup_p}")
        if cfg.mixup_p > 0.0 and float(cfg.mixup_alpha) <= 0.0:
            raise ValueError("mixup_p > 0 but mixup_alpha <= 0")

        apply_runtime_toggles(toggles, self.device)
        seed_everything(cfg.seed)

        # Augmentation config
        aug_cfg = AugmentConfig(
            use_elastic=cfg.use_elastic_aug,
            elastic_alpha=cfg.elastic_alpha,
            elastic_sigma=cfg.elastic_sigma,
            use_local_erasure=cfg.use_local_erasure,
        )

        # Data
        log.info("Loading datasets...")
        self.train_loader, self.test_loader, in_ch, num_classes, self.mean, self.std = build_loaders(
            cfg, self.device, aug_cfg
        )
        log.info(f"✓ Dataset loaded: {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} test")

        # Model metadata
        self.model_args = dict(
            in_ch=in_ch,
            num_classes=num_classes,
            cfg=model_cfg,
        )
        self.model_meta = {
            "in_ch": in_ch,
            "num_classes": num_classes,
            "ablation": model_cfg.ablation,
            "gating_mode": model_cfg.gating_mode,
        }

        # Build model
        log.info(f"Building model: {in_ch}-channel, {num_classes} classes")
        log.info(f"  STN: {model_cfg.use_stn}")
        log.info(f"  Local head: {model_cfg.ablation != 'global_only'}")
        log.info(f"  Global head: {model_cfg.ablation != 'baseline_cnn'}")
        log.info(f"  Gating: {model_cfg.gating_mode}")
        log.info(f"  Aux topology: {model_cfg.use_aux_topology}")
        
        base_model = OCRHybridExpert(**self.model_args).to(self.device)

        if self.device.type == "cuda" and toggles.channels_last:
            base_model = base_model.to(memory_format=torch.channels_last)

        # AMP setup
        self.use_amp = should_use_amp(toggles, self.device)
        self.amp_dtype: Optional[torch.dtype] = None
        self.scaler: Optional[GradScaler] = None

        if self.use_amp and self.device.type == "cuda":
            bf16_ok = False
            try:
                bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            except Exception:
                bf16_ok = False

            if bf16_ok:
                self.amp_dtype = torch.bfloat16
                self.scaler = None
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler("cuda")

        amp_str = "FP32"
        if self.use_amp and self.amp_dtype is not None:
            amp_str = "BF16" if self.amp_dtype == torch.bfloat16 else "FP16"
        log.info(f"AMP: {self.use_amp} | dtype: {amp_str}")

        # EMA
        steps_per_epoch = len(self.train_loader)
        total_opt_steps = int(steps_per_epoch * max(1, int(self.cfg.epochs)))

        self.ema_ctrl = EMADecayController(
            total_ema_updates=total_opt_steps,
            cfg=EMADecayConfig(),
            logger=log,
        )

        initial_decay = self.ema_ctrl._half_life_to_decay(self.ema_ctrl._scheduled_half_life())
        initial_decay = max(self.ema_ctrl.cfg.decay_min, min(self.ema_ctrl.cfg.decay_max, initial_decay))
        self.ema = ModelEMA(base_model, decay=initial_decay)
        log.info(f"EMA init | decay={self.ema.decay:.6f}")

        # Optional compile
        self.model = base_model
        if toggles.use_compile:
            try:
                self.model = torch.compile(self.model)
                log.info("torch.compile enabled")
            except Exception as e:
                log.warning(f"torch.compile failed: {e}")

        # Optimizer with optional LR boost for global head
        param_groups = []
        global_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'global_head' in name:
                global_params.append(param)
            else:
                other_params.append(param)
        
        if other_params:
            param_groups.append({'params': other_params, 'lr': cfg.lr})
        if global_params and cfg.global_lr_multiplier != 1.0:
            param_groups.append({'params': global_params, 'lr': cfg.lr * cfg.global_lr_multiplier})
        elif global_params:
            param_groups.append({'params': global_params, 'lr': cfg.lr})

        opt_kwargs = dict(weight_decay=cfg.weight_decay, betas=self._OPTIMIZER_BETAS)
        fused_ok = (self.device.type == "cuda")
        try:
            self.optimizer = torch.optim.AdamW(param_groups, fused=fused_ok, **opt_kwargs)
        except Exception:
            self.optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)

        self.scheduler = make_warmup_cosine_scheduler(
            self.optimizer,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
            min_lr_ratio=cfg.min_lr_ratio,
        )

        # Loss function
        self.criterion = HybridLoss(model_cfg, label_smoothing=cfg.label_smoothing)

        # State
        self._global_step = 0
        self._pnorm_cache = 1.0
        self._pnorm_refresh_every = 200
        self._refresh_param_norm()

        # Gating warmup tracking
        self._gating_warmup_done = False

        #Dashboard
        #self.use_dashboard = False  # Set to True to enable dashboard - Remove commented out sections if so.
        #self.dashboard: Optional[TrainingDashboard] = None

        log.info(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_opt_steps}")

    def _maybe_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda" and self.toggles.channels_last:
            return x.to(memory_format=torch.channels_last)
        return x

    def _autocast_ctx(self):
        if self.use_amp and self.amp_dtype is not None:
            return autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)
        return contextlib.nullcontext()

    def _student_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def _refresh_param_norm(self) -> None:
        with torch.no_grad():
            pnorm_sq = 0.0
            for p in self.model.parameters():
                if p.requires_grad:
                    pnorm_sq += float(p.detach().float().pow(2).sum().item())
            self._pnorm_cache = math.sqrt(max(1e-12, pnorm_sq))

    def _compute_grad_norm(self) -> float:
        gnorm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                gnorm_sq += float(p.grad.detach().float().pow(2).sum().item())
        return math.sqrt(max(0.0, gnorm_sq))

    def _update_gating_warmup(self, epoch: int) -> None:
        # Update gating warmup state based on epoch
        warmup_done = epoch >= self.model_cfg.gating_warmup_epochs
        
        if warmup_done and not self._gating_warmup_done:
            log.info(f"Gating warmup complete at epoch {epoch}. Enabling dynamic gating.")
            self._gating_warmup_done = True
        
        # Get the actual model (handle compiled)
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        
        model.set_gating_warmup(not warmup_done)
        
        # Also update EMA model
        ema_model = self.ema.module
        if hasattr(ema_model, '_orig_mod'):
            ema_model = ema_model._orig_mod
        ema_model.set_gating_warmup(not warmup_done)

    def train(self, resume: Optional[Path] = None) -> Path:
        # Main training loop

        self.use_dashboard = False
        self.use_dashboard = None

        start_epoch = 0
        best_acc = -1.0

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        best_path = self.cfg.out_dir / f"{self.cfg.save_name}.best.pt"
        last_path = self.cfg.out_dir / f"{self.cfg.save_name}.last.pt"

        if resume is not None:
            start_epoch, best_acc, meta = load_checkpoint(
                resume, self.model, self.ema.module,
                self.optimizer, self.scheduler, self.scaler, self.device
            )
            if meta:
                self.model_meta.update(meta)
            log.info(f"✓ Resumed from {resume} at epoch={start_epoch}")

            self.ema_ctrl.set_step(int(start_epoch * len(self.train_loader)))
            self._global_step = int(start_epoch * len(self.train_loader))

        # Training header
        log.info("")
        log.info("╔" + "═" * 74 + "╗")
        log.info(f"║{'TRAINING STARTED':^74}║")
        log.info("╠" + "═" * 74 + "╣")
        log.info(f"║  Dataset: {self.cfg.dataset.upper():<20} Epochs: {self.cfg.epochs:<10} Batch: {self.cfg.batch_size:<10}  ║")
        log.info(f"║  MixUp: p={self.cfg.mixup_p:.2f} α={self.cfg.mixup_alpha:.2f}{'':14} Warmup: {self.model_cfg.gating_warmup_epochs} epochs{'':12}  ║")
        log.info("╚" + "═" * 74 + "╝")
        log.info("")


        try:
            for epoch in range(start_epoch, self.cfg.epochs):
                # Update gating warmup state
                self._update_gating_warmup(epoch)
                
                # LR schedule
                lr_used = float(self.optimizer.param_groups[0]["lr"]) # Delete if you pytorch alert is annoying you

                epoch_start = time.time()

                # Train one epoch
                train_metrics = self._train_one_epoch()
                train_metrics['lr'] = lr_used
                
                # Evaluate
                test_metrics = self._eval(self.ema.module)

                improved = test_metrics['acc'] > best_acc
                if improved:
                    best_acc = test_metrics['acc']

                epoch_time = time.time() - epoch_start

                # Traditional logging
                self._log_epoch(epoch, train_metrics, test_metrics, epoch_time, improved)

                #Gating wamrup notification
                if epoch + 1 == self.model_cfg.gating_warmup_epochs:
                    log.info("--" * 76)
                    log.info("✅ Gating warmup period has ended. Dynamic gating is now active.")
                    log.info("--" * 76)

                # Save checkpoints
                save_checkpoint(
                    last_path, self.model, self.ema.module,
                    self.optimizer, self.scheduler, self.scaler,
                    epoch + 1, best_acc, self.cfg, self.model_cfg, self.toggles, self.model_meta
                )

                if improved:
                    save_checkpoint(
                        best_path, self.model, self.ema.module,
                        self.optimizer, self.scheduler, self.scaler,
                        epoch + 1, best_acc, self.cfg, self.model_cfg, self.toggles, self.model_meta
                    )

        except KeyboardInterrupt:
            log.info("\n⚠️ Training interrupted via 'cntrl+c'.")

        #Training complete
        log.info("")
        log.info("╔" + "═" * 74 + "╗")
        log.info(f"║{'TRAINING COMPLETE':^74}║")
        log.info("╚" + "═" * 74 + "╝")
        final_emoji = get_accuracy_emoji(best_acc)
        log.info(f"║  {final_emoji} Best Accuracy: {best_acc:.5f}{'':49}  ║")
        log.info(f"║  📁 Checkpoint: {best_path}  ║")
        log.info("╚" + "═" * 74 + "╝")
        log.info("")

        return best_path
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        epoch_time: float,
        improved: bool,
    ):
        # Log epoch metrics in formatted manner
        # Get emoji based on accuracy
        acc_emoji = get_accuracy_emoji(test_metrics['acc'])
        best_marker = "💎 NEW BEST" if improved else ""
        
        # Calculate ETA (track epoch times)
        if not hasattr(self, '_epoch_times'):
            self._epoch_times = []
        self._epoch_times.append(epoch_time)
        avg_time = sum(self._epoch_times) / len(self._epoch_times)
        eta = format_eta(avg_time, epoch + 1, self.cfg.epochs)
        
        # Header line
        log.info("═" * 76)
        header = f"{acc_emoji} Epoch {epoch+1:02d}/{self.cfg.epochs} │ Time: {epoch_time:.1f}s │ ETA: {eta}"
        if best_marker:
            header = f"{header:<60}{best_marker:>16}"
        log.info(header)
        log.info("─" * 76)
        
        # Accuracy line
        fused_acc = test_metrics.get('acc', 0)
        local_acc = test_metrics.get('local_acc', 0)
        global_acc = test_metrics.get('global_acc', 0)
        disagree = test_metrics.get('disagreement', 0)
        log.info(
            f"Accuracy    │ Fused: {fused_acc:.4f}  "
            f"Local: {local_acc:.4f}  "
            f"Global: {global_acc:.4f}  "
            f"Disagree: {disagree:.1%}"
        )
        
        # Loss line
        train_loss = train_metrics.get('loss', 0)
        test_loss = test_metrics.get('loss', 0)
        fused_loss = train_metrics.get('loss_fused', 0)
        log.info(
            f"Loss        │ Train: {train_loss:.4f}  "
            f"Test: {test_loss:.4f}  "
            f"Fused: {fused_loss:.4f}"
        )
        
        # Gating line (with visual bar)
        gating_local = train_metrics.get('gating_local', 0.5)
        gating_global = train_metrics.get('gating_global', 0.5)
        local_ent = train_metrics.get('local_entropy', 0)
        global_ent = train_metrics.get('global_entropy', 0)
        gating_bar = format_gating_bar(gating_local, gating_global, width=20)
        log.info(
            f"Gating      │ {gating_bar}  "
            f"Ent: {local_ent:.2f}/{global_ent:.2f}"
        )
        
        # Head losses line
        local_ce = train_metrics.get('loss_local', 0)
        global_ce = train_metrics.get('loss_global', 0)
        topo_loss = train_metrics.get('loss_topology', 0)
        log.info(
            f"Heads       │ LocalCE: {local_ce:.4f}  "
            f"GlobalCE: {global_ce:.4f}  "
            f"Topo: {topo_loss:.4f}"
        )
        
        # Adaptive controller line (if active)
        adaptive_local = train_metrics.get('adaptive_local_aux', 0)
        adaptive_global = train_metrics.get('adaptive_global_aux', 0)
        err_div = train_metrics.get('loss_error_diversity', 0)
        
        if adaptive_local > 0 or adaptive_global > 0:
            log.info(
                f"Adaptive    │ AuxW: L={adaptive_local:.2f} G={adaptive_global:.2f}  "
                f"ErrDiv: {err_div:.4f}  "
                f"HeadDrop: {train_metrics.get('head_drop_rate', 0):.1%}"
            )
        else:
            # Legacy diversity metrics
            div_loss = train_metrics.get('loss_diversity', 0)
            ncl_loss = train_metrics.get('loss_ncl', 0)
            if div_loss != 0 or ncl_loss != 0:
                log.info(
                    f"Diversity   │ JS: {div_loss:.4f}  "
                    f"NCL: {ncl_loss:.4f}  "
                    f"HeadDrop: {train_metrics.get('head_drop_rate', 0):.1%}"
                )
        
        # Training accuracy line (optional, can comment out if too verbose)
        train_local_acc = train_metrics.get('train_local_acc', 0)
        train_global_acc = train_metrics.get('train_global_acc', 0)
        train_disagree = train_metrics.get('train_disagreement', 0)
        if train_local_acc > 0:
            log.info(
                f"Train Stats │ Local: {train_local_acc:.4f}  "
                f"Global: {train_global_acc:.4f}  "
                f"Disagree: {train_disagree:.1%}"
            )

    def _train_one_epoch(self) -> Dict[str, float]:
        # Train for one epoch 
        self.model.train()
    
        metrics = {
            'loss': 0.0,
            'count': 0,
            'gating_local': 0.0,
            'gating_global': 0.0,
            'gating_count': 0,  
            'local_entropy': 0.0,
            'global_entropy': 0.0,
            'loss_fused': 0.0,
            'loss_local': 0.0,
            'loss_global': 0.0,
            'loss_topology': 0.0,
            'loss_diversity': 0.0,
            'loss_ncl': 0.0,
            'loss_orthogonal': 0.0,
            'local_correct': 0,
            'global_correct': 0,
            'disagreement': 0,
            'head_drops': 0,
            'adaptive_local_aux': 0.0,
            'adaptive_global_aux': 0.0,
            'local_rescue_rate': 0.0,
            'global_rescue_rate': 0.0,
            'agreement_on_error': 0.0,
        }
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = self._maybe_channels_last(x)
            x_orig = x.clone()

            self.optimizer.zero_grad(set_to_none=True)

            # MixUp
            do_mixup = (random.random() < self.cfg.mixup_p)
            if do_mixup:
                x_in, y_a, y_b, lam = mixup_data(x, y, alpha=self.cfg.mixup_alpha)
                mixup_targets = (y_a, y_b, lam)
            else:
                x_in = x
                y_a = y
                mixup_targets = None

            with self._autocast_ctx():
                outputs = self.model(x_in, return_parts=True)
                loss, loss_dict = self.criterion(
                    outputs, y,
                    mixup_targets=mixup_targets,
                    x_orig=x_orig,
                )

            if not math.isfinite(loss.item()):
                log.warning("Non-finite loss, skipping step")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self._global_step += 1
            if (self._global_step % self._pnorm_refresh_every) == 0:
                self._refresh_param_norm()

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                gnorm = self._compute_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                gnorm = self._compute_grad_norm()
                self.optimizer.step()

            # EMA update
            lr_now = float(self.optimizer.param_groups[0]["lr"])
            r_signal = float(min((lr_now * gnorm) / (self._pnorm_cache + 1e-12), 0.05))
            new_decay = self.ema_ctrl.step(r_signal)
            self.ema.set_decay(new_decay)
            self.ema.update(self.model)

            # Accumulate metrics
            bs = int(x.size(0))
            metrics['loss'] += float(loss.item()) * bs
            metrics['count'] += bs
        
            # Loss components
            for loss_name in ['loss_fused', 'loss_local', 'loss_global', 'loss_topology', 
                            'loss_diversity', 'loss_ncl', 'loss_orthogonal']:
                if loss_name in loss_dict:
                    val = loss_dict[loss_name]
                    if torch.is_tensor(val):
                        val = val.item()
                    metrics[loss_name] += float(val) * bs
        
             # Track head drops
            if 'head_dropped' in outputs:   
                metrics['head_drops'] += 1

            #Track adaptive controller metrics
            if 'adaptive_local_aux' in loss_dict:
                metrics['adaptive_local_aux'] += float(loss_dict['adaptive_local_aux_w'].item9()) * bs
                metrics['adaptive_global_aux'] += float(loss_dict['adaptive_global_aux_w'].item()) * bs
            
            # Training accuracy
            with torch.no_grad():
                target_for_acc = y_a if do_mixup else y
                if 'local_probs' in outputs:
                    local_pred = outputs['local_probs'].argmax(dim=-1)
                    metrics['local_correct'] += int((local_pred == target_for_acc).sum().item())
                if 'global_probs' in outputs:
                    global_pred = outputs['global_probs'].argmax(dim=-1)
                    metrics['global_correct'] += int((global_pred == target_for_acc).sum().item())
                if 'local_probs' in outputs and 'global_probs' in outputs:
                    metrics['disagreement'] += int((local_pred != global_pred).sum().item())
        
            # Gating metrics
            if 'gating_weight_local_mean' in outputs:
                metrics['gating_local'] += float(outputs['gating_weight_local_mean'].item()) * bs
                metrics['gating_global'] += float(outputs['gating_weight_global_mean'].item()) * bs
                metrics['gating_count'] += bs
            if 'local_entropy_mean' in outputs:
                metrics['local_entropy'] += float(outputs['local_entropy_mean'].item()) * bs
            if 'global_entropy_mean' in outputs:
                metrics['global_entropy'] += float(outputs['global_entropy_mean'].item()) * bs

        # Normalize
        count = max(1, metrics['count'])
        gating_count = max(1, metrics['gating_count'])
        num_batches = len(self.train_loader)
    
        return {
            'loss': metrics['loss'] / count,
            'loss_fused': metrics['loss_fused'] / count,
            'loss_local': metrics['loss_local'] / count,
            'loss_global': metrics['loss_global'] / count,
            'loss_topology': metrics['loss_topology'] / count,
            'loss_diversity': metrics['loss_diversity'] / count,
            'loss_ncl': metrics['loss_ncl'] / count,
            'loss_orthogonal': metrics['loss_orthogonal'] / count,
            'gating_local': metrics['gating_local'] / gating_count,
            'gating_global': metrics['gating_global'] / gating_count,
            'local_entropy': metrics['local_entropy'] / count,
            'global_entropy': metrics['global_entropy'] / count,
            'train_local_acc': metrics['local_correct'] / count,
            'train_global_acc': metrics['global_correct'] / count,
            'train_disagreement': metrics['disagreement'] / count,
            'head_drop_rate': metrics['head_drops'] / num_batches,
            'adaptive_local_aux': metrics['adaptive_local_aux'] / count,
            'adaptive_global_aux': metrics['adaptive_global_aux'] / count,
        }
    
    @torch.inference_mode()
    def _eval(self, model_to_eval: nn.Module) -> Dict[str, float]:
        # Evaluate model on test set
        model_to_eval.eval()
        
        metrics = {
            'loss': 0.0,
            'correct': 0,
            'local_correct': 0,
            'global_correct': 0,
            'disagreement': 0,
            'count': 0,
        }

        for x, y in self.test_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = self._maybe_channels_last(x)

            with self._autocast_ctx():
                outputs = model_to_eval(x, return_parts=True)
                
                # Fused prediction
                if 'fused_probs' in outputs:
                    pred = outputs['fused_probs'].argmax(dim=-1)
                elif 'local_probs' in outputs:
                    pred = outputs['local_probs'].argmax(dim=-1)
                else:
                    pred = outputs['global_probs'].argmax(dim=-1)
                
                # Loss
                loss, _ = self.criterion(outputs, y)

            bs = int(x.size(0))
            metrics['loss'] += float(loss.item()) * bs
            metrics['count'] += bs
            metrics['correct'] += int((pred == y).sum().item())
            
            # Per-head accuracy
            if 'local_probs' in outputs:
                local_pred = outputs['local_probs'].argmax(dim=-1)
                metrics['local_correct'] += int((local_pred == y).sum().item())
            
            if 'global_probs' in outputs:
                global_pred = outputs['global_probs'].argmax(dim=-1)
                metrics['global_correct'] += int((global_pred == y).sum().item())
            
            # Disagreement
            if 'local_probs' in outputs and 'global_probs' in outputs:
                local_pred = outputs['local_probs'].argmax(dim=-1)
                global_pred = outputs['global_probs'].argmax(dim=-1)
                metrics['disagreement'] += int((local_pred != global_pred).sum().item())

        count = max(1, metrics['count'])
        return {
            'loss': metrics['loss'] / count,
            'acc': metrics['correct'] / count,
            'local_acc': metrics['local_correct'] / count,
            'global_acc': metrics['global_correct'] / count,
            'disagreement': metrics['disagreement'] / count,
        }