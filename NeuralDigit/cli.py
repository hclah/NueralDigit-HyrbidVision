from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from .configs import ModelConfig, RuntimeToggles, TrainConfig
from .dataio import _dataset_specs, benchmark_corruptions
from .engine import Trainer, get_device, load_checkpoint, log
from .inference import export_to_onnx, predict_topk, preprocess_digit_image
from .modeling import OCRHybridExpert



# ARGUMENT PARSING
def parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OCR Hybrid Expert - Training, Evaluation, Export, and Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode selection
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "export", "predict", "bench"],
        help="Operation mode",
    )
    
    # Paths
    p.add_argument("--data-dir", type=Path, default=Path("./data"), help="Dataset directory")
    p.add_argument("--out-dir", type=Path, default=Path("./runs"), help="Output directory")
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load")
    p.add_argument("--save-name", type=str, default="ocr_hybrid", help="Checkpoint name prefix")
    
    # Dataset
    p.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashionmnist", "emnist", "usps", "svhn"],
        help="Dataset to use",
    )
    p.add_argument(
        "--emnist-split",
        type=str,
        default="balanced",
        choices=["digits", "balanced", "byclass"],
        help="EMNIST split (only used if dataset=emnist)",
    )
    
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--warmup-epochs", type=int, default=3, help="LR warmup epochs")
    p.add_argument("--min-lr-ratio", type=float, default=0.01, help="Min LR ratio for cosine decay")
    p.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing")
    p.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    p.add_argument("--global-lr-mult", type=float, default=1.5, help="LR multiplier for global head")
    
    # MixUp
    p.add_argument("--mixup-p", type=float, default=0.3, help="MixUp probability")
    p.add_argument("--mixup-alpha", type=float, default=0.4, help="MixUp alpha")
    
    # Augmentation
    p.add_argument("--no-elastic", action="store_true", help="Disable elastic augmentation")
    p.add_argument("--no-erasure", action="store_true", help="Disable local erasure augmentation")
    p.add_argument("--elastic-alpha", type=float, default=2.0, help="Elastic transform alpha")
    p.add_argument("--elastic-sigma", type=float, default=0.08, help="Elastic transform sigma")
    
    # Model architecture
    p.add_argument("--width", type=int, default=64, help="Base channel width")
    p.add_argument("--depths", type=int, nargs=3, default=[3, 3, 3], help="Block depths per stage")
    p.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    p.add_argument("--drop-path", type=float, default=0.10, help="Drop path rate")
    
    # STN / Rectifier
    p.add_argument("--no-stn", action="store_true", help="Disable Spatial Transformer Network")
    p.add_argument("--stn-mode", type=str, default="affine", choices=["affine", "tps"], help="STN mode")
    p.add_argument("--stn-channels", type=int, default=32, help="STN localization hidden channels")
    
    # Global head
    p.add_argument("--global-embed-dim", type=int, default=256, help="Global head embedding dim")
    p.add_argument("--global-num-latents", type=int, default=8, help="Number of Perceiver latents")
    p.add_argument("--global-num-heads", type=int, default=4, help="Attention heads")
    p.add_argument("--no-global-self-attn", action="store_true", help="Disable self-attention in global head")
    p.add_argument("--global-dropout", type=float, default=0.1, help="Global head dropout")
    
    # Fusion / Gating
    p.add_argument(
        "--gating-mode",
        type=str,
        default="Learned",
        choices=["fixed", "entropy", "learned"],
        help="Gating mode for fusion",
    )
    p.add_argument("--gating-floor", type=float, default=0.1, help="Minimum gating weight")
    p.add_argument("--gating-entropy-reg", type=float, default=0.01, help="Gating entropy regularization")
    p.add_argument("--gating-temperature", type=float, default=1.0, help="Gating temperature")
    p.add_argument("--gating-warmup-epochs", type=int, default=3, help="Epochs with fixed gating")
    
    # Calibration
    p.add_argument("--no-calibration", action="store_true", help="Disable temperature calibration")
    p.add_argument("--local-temp-init", type=float, default=1.0, help="Initial local temperature")
    p.add_argument("--global-temp-init", type=float, default=1.0, help="Initial global temperature")
    p.add_argument("--trainable-temps", action="store_true", help="Make temperatures trainable")
    
    # Auxiliary tasks
    p.add_argument("--no-aux-topology", action="store_true", help="Disable auxiliary topology task")
    p.add_argument(
        "--aux-topology-type",
        type=str,
        default="distance_transform",
        choices=["distance_transform", "skeleton", "endpoints"],
        help="Topology task type",
    )
    p.add_argument("--aux-topology-weight", type=float, default=0.1, help="Topology loss weight")
    p.add_argument("--aux-local-weight", type=float, default=0.3, help="Local head auxiliary loss weight")
    p.add_argument("--aux-global-weight", type=float, default=0.3, help="Global head auxiliary loss weight")
    
    # Ablation presets
    p.add_argument(
        "--ablation",
        type=str,
        default= "hybrid_learned",
        choices=["baseline_cnn", "global_only", "hybrid_fixed", "hybrid_entropy", "hybrid_learned"],
        help="Ablation preset (overrides individual settings)",
    )
    
    # Runtime toggles
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32")
    p.add_argument("--no-cudnn-bench", action="store_true", help="Disable cuDNN benchmark")
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--no-channels-last", action="store_true", help="Disable channels-last memory format")
    
    # System
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Export options
    p.add_argument("--onnx-path", type=Path, default=None, help="ONNX export path")
    p.add_argument("--onnx-opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-onnx-simplify", action="store_true", help="Skip ONNX simplification")
    p.add_argument("--no-onnx-verify", action="store_true", help="Skip ONNX verification")
    
    # Predict options
    p.add_argument("--image", type=Path, default=None, help="Image path for prediction")
    p.add_argument("--topk", type=int, default=5, help="Top-k predictions to show")
    
    return p.parse_args(args)


# MAIN ENTRY POINT
def main(args=None) -> int:
    """Main entry point."""
    opts = parse_args(args)
    
    # Build configs from args
    model_cfg = ModelConfig(
        width=opts.width,
        depths=tuple(opts.depths),
        dropout=opts.dropout,
        drop_path_rate=opts.drop_path,
        use_stn=not opts.no_stn,
        stn_mode=opts.stn_mode,
        stn_localization_channels=opts.stn_channels,
        global_embed_dim=opts.global_embed_dim,
        global_num_latents=opts.global_num_latents,
        global_num_heads=opts.global_num_heads,
        global_use_self_attn=not opts.no_global_self_attn,
        global_dropout=opts.global_dropout,
        gating_mode=opts.gating_mode,
        gating_weight_floor=opts.gating_floor,
        gating_entropy_reg=opts.gating_entropy_reg,
        gating_temperature=opts.gating_temperature,
        use_calibration=not opts.no_calibration,
        local_temp_init=opts.local_temp_init,
        global_temp_init=opts.global_temp_init,
        trainable_temps=opts.trainable_temps,
        use_aux_topology=not opts.no_aux_topology,
        aux_topology_type=opts.aux_topology_type,
        aux_topology_weight=opts.aux_topology_weight,
        aux_local_weight=opts.aux_local_weight,
        aux_global_weight=opts.aux_global_weight,
        gating_warmup_epochs=opts.gating_warmup_epochs,
        ablation=opts.ablation,
    )
    
    toggles = RuntimeToggles(
        use_amp=not opts.no_amp,
        allow_tf32=not opts.no_tf32,
        cudnn_benchmark=not opts.no_cudnn_bench,
        use_compile=opts.compile,
        channels_last=not opts.no_channels_last,
    )
    
    train_cfg = TrainConfig(
        data_dir=opts.data_dir,
        out_dir=opts.out_dir,
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        lr=opts.lr,
        weight_decay=opts.weight_decay,
        num_workers=opts.num_workers,
        seed=opts.seed,
        label_smoothing=opts.label_smoothing,
        warmup_epochs=opts.warmup_epochs,
        grad_clip_norm=opts.grad_clip_norm,
        save_name=opts.save_name,
        dataset=opts.dataset,
        emnist_split=opts.emnist_split,
        min_lr_ratio=opts.min_lr_ratio,
        mixup_p=opts.mixup_p,
        mixup_alpha=opts.mixup_alpha,
        global_lr_multiplier=opts.global_lr_mult,
        use_elastic_aug=not opts.no_elastic,
        use_local_erasure=not opts.no_erasure,
        elastic_alpha=opts.elastic_alpha,
        elastic_sigma=opts.elastic_sigma,
    )
    
    # Dispatch to mode
    if opts.mode == "train":
        return _run_train(train_cfg, model_cfg, toggles, opts.checkpoint)
    
    elif opts.mode == "eval":
        return _run_eval(train_cfg, model_cfg, toggles, opts.checkpoint)
    
    elif opts.mode == "export":
        return _run_export(train_cfg, model_cfg, toggles, opts)
    
    elif opts.mode == "predict":
        return _run_predict(train_cfg, model_cfg, toggles, opts)
    
    elif opts.mode == "bench":
        return _run_bench(train_cfg, model_cfg, toggles, opts.checkpoint)
    
    else:
        log.error(f"Unknown mode: {opts.mode}")
        return 1


def _run_train(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    toggles: RuntimeToggles,
    resume: Optional[Path],
) -> int:
    #Run training.
    trainer = Trainer(train_cfg, model_cfg, toggles)
    best_path = trainer.train(resume=resume)
    log.info(f"Best checkpoint: {best_path}")
    return 0


def _run_eval(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    toggles: RuntimeToggles,
    checkpoint: Optional[Path],
) -> int:
    #Run evaluation.
    if checkpoint is None:
        log.error("--checkpoint required for eval mode")
        return 1
    
    device = get_device()
    in_ch, num_classes, mean, std = _dataset_specs(train_cfg.dataset, train_cfg.emnist_split)
    
    model = OCRHybridExpert(in_ch=in_ch, num_classes=num_classes, cfg=model_cfg)
    model.to(device)
    
    load_checkpoint(checkpoint, model, None, None, None, None, device)
    model.eval()
    
    # Build test loader
    from .dataio import build_loaders
    _, test_loader, _, _, _, _ = build_loaders(train_cfg, device, None)
    
    # Evaluate
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x, return_parts=True)
            if 'fused_probs' in outputs:
                pred = outputs['fused_probs'].argmax(dim=-1)
            else:
                pred = outputs.get('local_probs', outputs.get('global_probs')).argmax(dim=-1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / max(1, total)
    log.info(f"Test Accuracy: {acc:.5f} ({correct}/{total})")
    
    return 0


def _run_export(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    toggles: RuntimeToggles,
    opts: argparse.Namespace,
) -> int:
    #Run ONNX export.
    if opts.checkpoint is None:
        log.error("--checkpoint required for export mode")
        return 1
    
    device = get_device()
    in_ch, num_classes, mean, std = _dataset_specs(train_cfg.dataset, train_cfg.emnist_split)
    
    model = OCRHybridExpert(in_ch=in_ch, num_classes=num_classes, cfg=model_cfg)
    model.to(device)
    
    load_checkpoint(opts.checkpoint, model, None, None, None, None, device)
    model.eval()
    
    # Determine output path
    onnx_path = opts.onnx_path
    if onnx_path is None:
        onnx_path = opts.checkpoint.with_suffix('.onnx')
    
    export_to_onnx(
        model,
        onnx_path,
        opset_version=opts.onnx_opset,
        simplify=not opts.no_onnx_simplify,
        verify=not opts.no_onnx_verify,
    )
    
    return 0


def _run_predict(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    toggles: RuntimeToggles,
    opts: argparse.Namespace,
) -> int:
    #Run single image prediction.
    if opts.checkpoint is None:
        log.error("--checkpoint required for predict mode")
        return 1
    if opts.image is None:
        log.error("--image required for predict mode")
        return 1
    if not opts.image.exists():
        log.error(f"Image not found: {opts.image}")
        return 1
    
    device = get_device()
    in_ch, num_classes, mean, std = _dataset_specs(train_cfg.dataset, train_cfg.emnist_split)
    
    model = OCRHybridExpert(in_ch=in_ch, num_classes=num_classes, cfg=model_cfg)
    model.to(device)
    
    load_checkpoint(opts.checkpoint, model, None, None, None, None, device)
    model.eval()
    
    # Load image
    from PIL import Image
    img = Image.open(opts.image)
    
    # Predict
    results = predict_topk(
        model, img,
        k=opts.topk,
        device=device,
        mean=mean,
        std=std,
        return_all_heads=True,
    )
    
    log.info(f"Predictions for: {opts.image}")
    log.info("-" * 40)
    
    log.info("Fused predictions:")
    for cls, prob in results['fused']:
        log.info(f"  Class {cls}: {prob:.4f}")
    
    if 'local' in results:
        log.info("Local head predictions:")
        for cls, prob in results['local']:
            log.info(f"  Class {cls}: {prob:.4f}")
    
    if 'global' in results:
        log.info("Global head predictions:")
        for cls, prob in results['global']:
            log.info(f"  Class {cls}: {prob:.4f}")
    
    if 'gating_weights' in results:
        log.info(f"Gating weights: Local={results['gating_weights']['local']:.3f}, Global={results['gating_weights']['global']:.3f}")
    
    return 0


def _run_bench(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    toggles: RuntimeToggles,
    checkpoint: Optional[Path],
) -> int:
    #Run corruption benchmark.
    if checkpoint is None:
        log.error("--checkpoint required for bench mode")
        return 1
    
    device = get_device()
    in_ch, num_classes, mean, std = _dataset_specs(train_cfg.dataset, train_cfg.emnist_split)
    
    model = OCRHybridExpert(in_ch=in_ch, num_classes=num_classes, cfg=model_cfg)
    model.to(device)
    
    if device.type == "cuda" and toggles.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    load_checkpoint(checkpoint, model, None, None, None, None, device)
    model.eval()
    
    # Build test dataset (without transform for corruption benchmark)
    from .dataio import build_datasets
    _, test_ds = build_datasets(train_cfg)
    
    # Run benchmark
    results = benchmark_corruptions(
        model=model,
        device=device,
        dataset=test_ds,
        mean=mean,
        std=std,
        use_amp=toggles.use_amp and device.type == "cuda",
        channels_last=toggles.channels_last,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )
    
    # Summary
    log.info("=" * 60)
    log.info("CORRUPTION BENCHMARK SUMMARY")
    log.info("=" * 60)
    
    for corruption, sev_results in results.items():
        accs = [sev_results[s] for s in sorted(sev_results.keys())]
        mean_acc = sum(accs) / len(accs)
        log.info(f"{corruption:20s}: mean={mean_acc:.4f} | {accs}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())