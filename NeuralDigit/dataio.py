from __future__ import annotations
import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from .configs import AugmentConfig, TrainConfig


# Get logger
log = logging.getLogger("ocr_hybrid")

# DATASET SPECIFICATIONS
def _dataset_specs(dataset: str, emnist_split: str) -> Tuple[int, int, Tuple[float, ...], Tuple[float, ...]]:
    # Gets the exact dataset specifications.
    d = dataset.lower()
    
    if d == "mnist":
        return 1, 10, (0.1307,), (0.3081,)
    if d == "fashionmnist":
        return 1, 10, (0.2860,), (0.3530,)
    if d == "usps":
        return 1, 10, (0.2539,), (0.3267,)
    if d == "svhn":
        return 1, 10, (0.4377,), (0.1980,)
    if d == "emnist":
        split = emnist_split.lower()
        if split == "digits":
            return 1, 10, (0.1751,), (0.3332,)
        elif split == "balanced":
            return 1, 47, (0.1751,), (0.3332,)
        elif split == "byclass":
            return 1, 62, (0.1751,), (0.3332,)
        else:
            raise ValueError(f"Unknown EMNIST split: '{emnist_split}'")
    
    raise ValueError(f"Unknown dataset: '{dataset}'")


# AUGMENTATION TRANSFORMS
class ElasticTransform:
    # Elastic deformation transform for OCR augmentation. Approximates handwriting variations by applying smooth random distortions.
    def __init__(self, alpha: float = 2.0, sigma: float = 0.08) -> None:
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        #Args: - x: Image tensor [C, H, W], Returns:- Deformed image tensor [C, H, W]
        if not self.training_mode:
            return x
        
        C, H, W = x.shape
        
        # Generate random displacement fields
        dx = torch.randn(1, H, W) * self.sigma
        dy = torch.randn(1, H, W) * self.sigma
        
        # Smooth with Gaussian (approximate with average pooling)
        kernel = 5
        dx = F.avg_pool2d(F.pad(dx.unsqueeze(0), (kernel//2,)*4, mode='reflect'), 
                          kernel, stride=1).squeeze(0) * self.alpha
        dy = F.avg_pool2d(F.pad(dy.unsqueeze(0), (kernel//2,)*4, mode='reflect'), 
                          kernel, stride=1).squeeze(0) * self.alpha
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid_x = grid_x + dx.squeeze()
        grid_y = grid_y + dy.squeeze()
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Sample
        x_out = F.grid_sample(
            x.unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=False
        ).squeeze(0)
        
        return x_out
    
    training_mode = True


class LocalErasure:
    # Random local erasure (cutout) augmentation.
    def __init__(self, prob: float = 0.3, scale: Tuple[float, float] = (0.02, 0.15)) -> None:
        self.prob = prob
        self.scale = scale
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return x
        
        C, H, W = x.shape
        area = H * W
        
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(0.3, 3.0)
        
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if h < H and w < W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            x = x.clone()
            x[:, top:top+h, left:left+w] = 0
        
        return x


# TRANSFORM BUILDERS
def build_transforms(
    train: bool,
    dataset: str,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    aug_cfg: Optional[AugmentConfig] = None,
) -> transforms.Compose:
    # Build transform pipeline.
    d = dataset.lower()
    ops: List[object] = []

    if d == "svhn":
        ops.append(transforms.Grayscale(num_output_channels=1))
        ops.append(transforms.Resize((28, 28)))
    if d == "usps":
        ops.append(transforms.Resize((28, 28)))

    if train and aug_cfg is not None:
        if aug_cfg.use_affine:
            ops.append(
                transforms.RandomAffine(
                    degrees=aug_cfg.affine_degrees,
                    translate=aug_cfg.affine_translate,
                    scale=aug_cfg.affine_scale,
                    shear=aug_cfg.affine_shear,
                    fill=0,
                )
            )

    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    return transforms.Compose(ops)


# DATASET BUILDERS
def build_datasets(cfg: TrainConfig):
    # Build train and test datasets
    d = cfg.dataset.lower()
    if d == "mnist":
        train_ds = datasets.MNIST(str(cfg.data_dir), train=True, download=True, transform=None)
        test_ds = datasets.MNIST(str(cfg.data_dir), train=False, download=True, transform=None)
        return train_ds, test_ds
    if d == "fashionmnist":
        train_ds = datasets.FashionMNIST(str(cfg.data_dir), train=True, download=True, transform=None)
        test_ds = datasets.FashionMNIST(str(cfg.data_dir), train=False, download=True, transform=None)
        return train_ds, test_ds
    if d == "usps":
        train_ds = datasets.USPS(str(cfg.data_dir), train=True, download=True, transform=None)
        test_ds = datasets.USPS(str(cfg.data_dir), train=False, download=True, transform=None)
        return train_ds, test_ds
    if d == "svhn":
        train_ds = datasets.SVHN(str(cfg.data_dir), split="train", download=True, transform=None)
        test_ds = datasets.SVHN(str(cfg.data_dir), split="test", download=True, transform=None)
        return train_ds, test_ds
    if d == "emnist":
        train_ds = datasets.EMNIST(str(cfg.data_dir), split=cfg.emnist_split, train=True, download=True, transform=None)
        test_ds = datasets.EMNIST(str(cfg.data_dir), split=cfg.emnist_split, train=False, download=True, transform=None)
        return train_ds, test_ds
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


class _TransformDataset(Dataset):
    # Dataset wrapper that applies transforms
    def __init__(
        self,
        base: Dataset,
        transform: transforms.Compose,
        dataset_name: str,
        train: bool = False,
        aug_cfg: Optional[AugmentConfig] = None,
    ) -> None:
        self.base = base
        self.transform = transform
        self.dataset_name = dataset_name.lower()
        self.train = train
        self.aug_cfg = aug_cfg

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)

        y = int(y)
        if self.dataset_name == "svhn" and y == 10:
            y = 0

        x = self.transform(x)
        
        # Additional tensor-level augmentations for training
        if self.train and self.aug_cfg is not None:
            # Elastic transform (simplified)
            if self.aug_cfg.use_elastic and random.random() < 0.3:
                x = self._apply_elastic(x)
            
            # Local erasure
            if self.aug_cfg.use_local_erasure and random.random() < self.aug_cfg.erasure_prob:
                x = self._apply_erasure(x)
            
            # Noise
            if self.aug_cfg.use_noise and random.random() < 0.2:
                x = x + torch.randn_like(x) * self.aug_cfg.noise_std
        
        return x, y
    
    def _apply_elastic(self, x: torch.Tensor) -> torch.Tensor:
        # Apply simplified elastic deformation.
        C, H, W = x.shape
        alpha = self.aug_cfg.elastic_alpha if self.aug_cfg else 2.0
        sigma = self.aug_cfg.elastic_sigma if self.aug_cfg else 0.08
        
        # Random displacement
        dx = (torch.rand(H, W) - 0.5) * 2 * alpha / H
        dy = (torch.rand(H, W) - 0.5) * 2 * alpha / W
        
        # Smooth
        dx = F.avg_pool2d(dx.view(1, 1, H, W), 3, stride=1, padding=1).view(H, W)
        dy = F.avg_pool2d(dy.view(1, 1, H, W), 3, stride=1, padding=1).view(H, W)
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)
        
        x_out = F.grid_sample(
            x.unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=False
        ).squeeze(0)
        
        return x_out
    
    def _apply_erasure(self, x: torch.Tensor) -> torch.Tensor:
        #Apply random erasure.
        C, H, W = x.shape
        scale = self.aug_cfg.erasure_scale if self.aug_cfg else (0.02, 0.15)
        
        area = H * W
        target_area = random.uniform(scale[0], scale[1]) * area
        
        h = int(round(math.sqrt(target_area)))
        w = h
        
        if h < H and w < W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            x = x.clone()
            x[:, top:top+h, left:left+w] = 0
        
        return x


# DATALOADER BUILDERS
def seed_worker(_: int) -> None:
    #Worker init function for reproducible data loading.
    wseed = torch.initial_seed() % 2**32
    random.seed(wseed)
    np.random.seed(wseed)


def build_loaders(
    cfg: TrainConfig,
    device: torch.device,
    aug_cfg: Optional[AugmentConfig] = None,
) -> Tuple[DataLoader, DataLoader, int, int, Tuple[float, ...], Tuple[float, ...]]:
    # Build train and test data loaders.
    in_ch, num_classes, mean, std = _dataset_specs(cfg.dataset, cfg.emnist_split)

    log.info(f"Building datasets for {cfg.dataset.upper()}...")
    raw_train, raw_test = build_datasets(cfg)
    
    train_ds = _TransformDataset(
        raw_train,
        build_transforms(True, cfg.dataset, mean, std, aug_cfg),
        cfg.dataset,
        train=True,
        aug_cfg=aug_cfg,
    )
    test_ds = _TransformDataset(
        raw_test,
        build_transforms(False, cfg.dataset, mean, std, None),
        cfg.dataset,
        train=False,
    )

    pin = device.type == "cuda"
    persistent = cfg.num_workers > 0

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
        drop_last=True,
        generator=g,
        worker_init_fn=seed_worker if persistent else None,
        timeout=60 if cfg.num_workers > 0 else 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(256, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
        drop_last=False,
        generator=g,
        worker_init_fn=seed_worker if persistent else None,
        timeout=60 if cfg.num_workers > 0 else 0,
    )
    
    return train_loader, test_loader, in_ch, num_classes, mean, std


# CORRUPTION BENCHMARKS
def _apply_corruption(x01: torch.Tensor, name: str, sev: int) -> torch.Tensor:
    # Apply corruption to normalized [0,1] images.
    sev = int(sev)
    name = name.lower()

    if name == "clean":
        return x01

    if name == "noise":
        s = [0.05, 0.10, 0.15, 0.20, 0.25][max(0, min(4, sev - 1))]
        return (x01 + torch.randn_like(x01) * s).clamp_(0.0, 1.0)

    if name == "blur":
        k = [3, 5, 7, 9, 11][max(0, min(4, sev - 1))]
        return TF.gaussian_blur(x01, kernel_size=[k, k])

    if name == "rotate":
        deg = [8, 14, 20, 26, 32][max(0, min(4, sev - 1))]
        return TF.rotate(x01, angle=float(deg), fill=0.0)

    if name == "contrast":
        f = [0.6, 0.5, 0.4, 0.3, 0.2][max(0, min(4, sev - 1))]
        return TF.adjust_contrast(x01, contrast_factor=float(f))

    if name == "brightness":
        f = [0.7, 0.6, 0.5, 0.4, 0.3][max(0, min(4, sev - 1))]
        return TF.adjust_brightness(x01, brightness_factor=float(f))

    if name == "cutout":
        s = [6, 8, 10, 12, 14][max(0, min(4, sev - 1))]
        b, c, h, w = x01.shape
        out = x01.clone()
        for i in range(b):
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)
            y1 = max(0, cy - s // 2)
            y2 = min(h, cy + s // 2)
            x1 = max(0, cx - s // 2)
            x2 = min(w, cx + s // 2)
            out[i, :, y1:y2, x1:x2] = 0.0
        return out

    if name == "elastic":
        # Approximate elastic deformation
        b, c, h, w = x01.shape
        alpha = [1.0, 2.0, 3.0, 4.0, 5.0][max(0, min(4, sev - 1))]
        out = x01.clone()
        for i in range(b):
            dx = (torch.rand(h, w, device=x01.device) - 0.5) * 2 * alpha / h
            dy = (torch.rand(h, w, device=x01.device) - 0.5) * 2 * alpha / w
            dx = F.avg_pool2d(dx.view(1, 1, h, w), 3, stride=1, padding=1).view(h, w)
            dy = F.avg_pool2d(dy.view(1, 1, h, w), 3, stride=1, padding=1).view(h, w)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=x01.device),
                torch.linspace(-1, 1, w, device=x01.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)
            out[i:i+1] = F.grid_sample(out[i:i+1], grid, mode='bilinear', padding_mode='border', align_corners=False)
        return out

    if name == "corner_occlusion":
        # Occlude a corner
        s = [4, 6, 8, 10, 12][max(0, min(4, sev - 1))]
        b, c, h, w = x01.shape
        out = x01.clone()
        corners = [(0, 0), (0, w-s), (h-s, 0), (h-s, w-s)]
        for i in range(b):
            cy, cx = corners[random.randint(0, 3)]
            out[i, :, cy:cy+s, cx:cx+s] = 0.0
        return out

    raise ValueError(f"Unknown corruption: {name}")


@torch.inference_mode()
def benchmark_corruptions(
    model: torch.nn.Module,
    device: torch.device,
    dataset: Dataset,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    use_amp: bool,
    channels_last: bool,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Dict[int, float]]:
    # Run corruption benchmark.
    model.eval()
    pin = device.type == "cuda"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    corruptions = ["clean", "noise", "blur", "rotate", "contrast", "brightness", "cutout", "elastic", "corner_occlusion"]
    severities = [1, 3, 5]

    def norm(x01: torch.Tensor) -> torch.Tensor:
        return transforms.Normalize(mean, std)(x01)

    results = {}
    
    for cname in corruptions:
        results[cname] = {}
        for sev in severities:
            total = 0
            correct = 0
            local_correct = 0
            global_correct = 0
            
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x = x.clamp(0.0, 1.0)
                x = _apply_corruption(x, cname, sev)
                x = norm(x)

                if device.type == "cuda" and channels_last:
                    x = x.to(memory_format=torch.channels_last)

                if device.type == "cuda" and use_amp:
                    with autocast(device_type="cuda", enabled=True):
                        outputs = model(x, return_parts=True)
                else:
                    outputs = model(x, return_parts=True)

                if 'fused_probs' in outputs:
                    pred = outputs['fused_probs'].argmax(dim=-1)
                else:
                    pred = outputs.get('local_probs', outputs.get('global_probs')).argmax(dim=-1)
                
                correct += int((pred == y).sum().item())
                total += int(y.numel())
                
                if 'local_probs' in outputs:
                    local_correct += int((outputs['local_probs'].argmax(-1) == y).sum().item())
                if 'global_probs' in outputs:
                    global_correct += int((outputs['global_probs'].argmax(-1) == y).sum().item())

            acc = correct / max(1, total)
            local_acc = local_correct / max(1, total)
            global_acc = global_correct / max(1, total)
            
            results[cname][sev] = acc
            log.info(f"bench | {cname:15s} sev={sev} | Fused: {acc:.5f} | Local: {local_acc:.5f} | Global: {global_acc:.5f}")

    return results