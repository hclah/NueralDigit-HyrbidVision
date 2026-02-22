from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .modeling import OCRHybridExpert, ONNXExportWrapper


# Get logger
log = logging.getLogger("ocr_hybrid")

# PREPROCESSING
def preprocess_digit_image(
    img: Union[np.ndarray, Image.Image, torch.Tensor],
    mean: Tuple[float, ...] = (0.1307,),
    std: Tuple[float, ...] = (0.3081,),
    target_size: Tuple[int, int] = (28, 28),
    invert: bool = False,
    pad_to_square: bool = True,
) -> torch.Tensor:
    """
    Preprocess a digit image for inference.
    Args:
        img: Input image (numpy array, PIL Image, or torch Tensor)
        mean: Normalization mean (dataset-specific)
        std: Normalization std (dataset-specific)
        target_size: Output size (H, W)
        invert: If True, invert colors (white digit on black -> black on white)
        pad_to_square: If True, pad non-square images to square before resize
    Returns:
        Preprocessed tensor [1, 1, H, W] ready for model input
    """
    # Convert to PIL Image
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
    
    if isinstance(img, np.ndarray):
        # Handle different array formats
        if img.ndim == 3 and img.shape[2] == 3:
            # RGB -> grayscale
            img = np.mean(img, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
        
        # Normalize to [0, 255] uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        img = Image.fromarray(img, mode='L')
    
    if not isinstance(img, Image.Image):
        raise TypeError(f"Cannot convert input of type {type(img)} to PIL Image")
    
    # Convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # Invert if requested (e.g., white digit on black background)
    if invert:
        img = Image.fromarray(255 - np.array(img))
    
    # Pad to square
    if pad_to_square:
        w, h = img.size
        if w != h:
            size = max(w, h)
            new_img = Image.new('L', (size, size), color=0)
            paste_x = (size - w) // 2
            paste_y = (size - h) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
    
    # Resize
    if img.size != target_size[::-1]:  # PIL uses (W, H)
        img = img.resize(target_size[::-1], Image.Resampling.BILINEAR)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    tensor = transform(img)  # [1, H, W]
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    
    return tensor

# PREDICTION
@torch.inference_mode()
def predict_topk(
    model: OCRHybridExpert,
    img: Union[np.ndarray, Image.Image, torch.Tensor],
    k: int = 5,
    device: Optional[torch.device] = None,
    mean: Tuple[float, ...] = (0.1307,),
    std: Tuple[float, ...] = (0.3081,),
    return_all_heads: bool = False,
) -> Union[List[Tuple[int, float]], dict]:
    
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device

    x = preprocess_digit_image(img, mean=mean, std=std)
    x = x.to(device)
    
    if device.type == "cuda":
        try:
            x = x.to(memory_format=torch.channels_last)
        except Exception:
            pass
    
    outputs = model(x, return_parts=True)
    
    def get_topk(probs: torch.Tensor) -> List[Tuple[int, float]]:
        # Extracts top-k from probability tensor.
        probs = probs.squeeze(0)  # [num_classes]
        k_actual = min(k, probs.size(0))
        values, indices = torch.topk(probs, k_actual)
        return [(int(idx.item()), float(val.item())) for idx, val in zip(indices, values)]
    
    if 'fused_probs' in outputs:
        fused_topk = get_topk(outputs['fused_probs'])
    elif 'local_probs' in outputs:
        fused_topk = get_topk(outputs['local_probs'])
    else:
        fused_topk = get_topk(outputs['global_probs'])
    
    if not return_all_heads:
        return fused_topk
    
    result = {'fused': fused_topk}
    
    if 'local_probs' in outputs:
        result['local'] = get_topk(outputs['local_probs'])
    
    if 'global_probs' in outputs:
        result['global'] = get_topk(outputs['global_probs'])
    
    # Include confidence metrics
    if 'local_entropy_mean' in outputs:
        result['local_entropy'] = float(outputs.get('local_entropy_mean', 0))
    if 'global_entropy_mean' in outputs:
        result['global_entropy'] = float(outputs.get('global_entropy_mean', 0))
    if 'gating_weight_local_mean' in outputs:
        result['gating_weights'] = {
            'local': float(outputs['gating_weight_local_mean']),
            'global': float(outputs['gating_weight_global_mean']),
        }
    
    return result


# ONNX EXPORT - for testing inference speed and deployment on another device. (Local head only for best compatibility)
def export_to_onnx(
    model: OCRHybridExpert,
    output_path: Union[str, Path],
    opset_version: int = 17,
    input_shape: Tuple[int, int, int, int] = (1, 1, 28, 28),
    dynamic_batch: bool = True,
    simplify: bool = True,
    verify: bool = True,
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    wrapper = ONNXExportWrapper(model)
    wrapper.eval()
    wrapper.to(device)
    
    # Dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    
    log.info(f"Exporting to ONNX: {output_path}")
    log.info(f"  Opset: {opset_version}")
    log.info(f"  Input shape: {input_shape}")
    log.info(f"  Dynamic batch: {dynamic_batch}")
    
    # Exporting the model
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    
    log.info(f"✓ ONNX export complete: {output_path}")
    
    # Simplify (optional)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            log.info("Running onnx-simplifier...")
            onnx_model = onnx.load(str(output_path))
            simplified, ok = onnx_simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(output_path))
                log.info("✓ ONNX model simplified")
            else:
                log.warning("onnx-simplifier returned check=False, keeping original")
        except ImportError:
            log.info("onnx-simplifier not installed, skipping simplification")
        except Exception as e:
            log.warning(f"onnx-simplifier failed: {e}")
    
    # Verify (optional)
    if verify:
        try:
            import onnx
            import onnxruntime as ort
            
            log.info("Verifying ONNX model...")
            
            # Check model validity
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Run inference test
            sess = ort.InferenceSession(str(output_path))
            test_input = dummy_input.cpu().numpy()
            ort_outputs = sess.run(None, {'input': test_input})
            
            # Compare with PyTorch
            with torch.no_grad():
                torch_output = wrapper(dummy_input).cpu().numpy()
            
            max_diff = float(np.abs(ort_outputs[0] - torch_output).max())
            log.info(f"✓ ONNX verification passed (max diff: {max_diff:.6f})")
            
            if max_diff > 1e-4:
                log.warning(f"Large numerical difference: {max_diff:.6f}")
        
        except ImportError as e:
            log.info(f"Verification skipped (missing dependency: {e})")
        except Exception as e:
            log.warning(f"ONNX verification failed: {e}")
    
    return output_path