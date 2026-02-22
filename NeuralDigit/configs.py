from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class RuntimeToggles:
    use_amp: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    use_compile: bool = False
    channels_last: bool = True


@dataclass
class ModelConfig:
    # Core architecture
    width: int = 64
    depths: Tuple[int, int, int] = (3, 3, 3)
    dropout: float = 0.05
    drop_path_rate: float = 0.10
    
    # Rectifier (STN)
    use_stn: bool = True
    stn_mode: str = "affine"  # "affine" or "tps" (TPS is extension hook)
    stn_localization_channels: int = 32

    #Multi-scale features
    use_multiscale_heads: bool = True

    #Head specific configs
    use_head_adapters: bool = True
    adapter_hidden_dim: int = 128

    #Local head (CNN)
    local_head_dropout: float = 0.15 #reduce from 0.3
    
    # Global head (Perceiver)
    global_embed_dim: int = 256
    global_num_latents: int = 8
    global_num_heads: int = 4
    global_use_self_attn: bool = True
    global_dropout: float = 0.1
    
    # Fusion
    gating_mode: str = "learned"  #changed from entropy
    gating_weight_floor: float = 0.05
    gating_entropy_reg: float = 0.01
    gating_temperature: float = 0.25

    #Head Dropout for Training
    head_dropout_prob: float = 0.1 # 10% chance to disable one head
    
    # Calibration
    use_calibration: bool = True
    local_temp_init: float = 1.0
    global_temp_init: float = 1.0
    trainable_temps: bool = False
    
    # Auxiliary tasks
    use_aux_topology: bool = True
    aux_topology_type: str = "distance_transform"  # "distance_transform", "skeleton", "endpoints"
    aux_topology_weight: float = 0.1
    
    # Asymmetric loss weights
    aux_local_weight: float = 0.4 #increased - pushes local more
    aux_global_weight: float = 0.25 #decreased - lets global explore

    #Diversity Losses
    diversity_weight: float = 0.0  #testing 
    ncl_weight: float = 0.0
    orthogonal_weight: float = 0.05
    
    # Warmup
    gating_warmup_epochs: int = 3
    
    # Ablation preset (overrides individual settings)
    ablation: str = "hybrid_learned"  # changed from entropy


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    label_smoothing: float
    warmup_epochs: int
    grad_clip_norm: float
    save_name: str
    
    dataset: str
    emnist_split: str
    min_lr_ratio: float
    mixup_p: float
    mixup_alpha: float
    
    # Global head LR boost
    global_lr_multiplier: float = 1.5
    
    # Augmentation toggles
    use_elastic_aug: bool = True
    use_local_erasure: bool = True
    elastic_alpha: float = 2.0
    elastic_sigma: float = 0.08


@dataclass
class AugmentConfig:
    use_affine: bool = True
    affine_degrees: float = 12.0
    affine_translate: Tuple[float, float] = (0.10, 0.10)
    affine_scale: Tuple[float, float] = (0.90, 1.10)
    affine_shear: float = 8.0
    
    use_elastic: bool = True
    elastic_alpha: float = 2.0
    elastic_sigma: float = 0.08
    
    use_local_erasure: bool = True
    erasure_prob: float = 0.3
    erasure_scale: Tuple[float, float] = (0.02, 0.15)
    
    use_noise: bool = True
    noise_std: float = 0.05
    
    use_blur: bool = True
    blur_prob: float = 0.2
    blur_kernel: int = 3


@dataclass
class EMADecayConfig:
    h_final_frac: float = 0.01
    ramp_frac: float = 0.05
    h_min: int = 10
    h_max: int = 200_000
    ramp_min: int = 50
    ramp_max: int = 50_000
    h_start: int = 10
    noise_ema_beta: float = 0.98
    baseline_warmup_updates: int = 200
    noise_strength: float = 1.0
    shrink_max: float = 5.0
    grow_max: float = 2.0
    decay_min: float = 0.90
    decay_max: float = 0.99995