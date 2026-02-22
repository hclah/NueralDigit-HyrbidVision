from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import deque
import math


@dataclass
class HeadControllerConfig:
    # Configuration for the Adaptive Head Controller.
    
    # Error-aware diversity
    error_diversity_weight: float = 0.15      # Penalize agreement on errors
    rescue_bonus_weight: float = 0.10         # Reward head that "rescues" wrong prediction
    
    # Adaptive aux weight adjustment
    enable_adaptive_aux: bool = True
    aux_weight_min: float = 0.15              # Minimum aux weight
    aux_weight_max: float = 0.60              # Maximum aux weight
    aux_adaptation_rate: float = 0.02         # How fast aux weights change
    
    # Confidence shaping
    confidence_margin_target: float = 0.15    # Desired confidence gap on correct preds
    confidence_shaping_weight: float = 0.05   # Weight for confidence loss
    
    # History tracking
    history_window: int = 500                 # Samples to track for running stats
    
    # Disagreement bonus on hard samples
    hard_sample_threshold: float = 0.7        # Entropy threshold for "hard" samples
    hard_sample_diversity_bonus: float = 2.0  # Multiplier for diversity on hard samples


class AdaptiveHeadController(nn.Module):
    
    def __init__(self, cfg: HeadControllerConfig = None):
        super().__init__()
        self.cfg = cfg or HeadControllerConfig()
        
        # Running statistics (not parameters, just tracking for analysis)
        self.register_buffer('local_error_rate', torch.tensor(0.5))
        self.register_buffer('global_error_rate', torch.tensor(0.5))
        self.register_buffer('local_rescue_rate', torch.tensor(0.0))
        self.register_buffer('global_rescue_rate', torch.tensor(0.0))
        self.register_buffer('agreement_on_error_rate', torch.tensor(0.0))
        
        # Adaptive aux weights
        self.register_buffer('adaptive_local_aux', torch.tensor(0.35))
        self.register_buffer('adaptive_global_aux', torch.tensor(0.35))
        
        # EMA momentum for stats
        self.stats_momentum = 0.99
        
    def compute_losses(
        self,
        local_logits: torch.Tensor,
        global_logits: torch.Tensor,
        fused_logits: torch.Tensor,
        targets: torch.Tensor,
        local_probs: Optional[torch.Tensor] = None,
        global_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all adaptive head control losses.
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary of individual loss components
        """
        device = local_logits.device
        batch_size = local_logits.size(0)
        
        # Get probabilities if not provided
        if local_probs is None:
            local_probs = F.softmax(local_logits, dim=-1)
        if global_probs is None:
            global_probs = F.softmax(global_logits, dim=-1)
        
        fused_probs = F.softmax(fused_logits, dim=-1)
        
        # Get predictions 
        local_pred = local_logits.argmax(dim=-1)
        global_pred = global_logits.argmax(dim=-1)
        fused_pred = fused_logits.argmax(dim=-1)
        
        # Correctness masks 
        local_correct = (local_pred == targets)
        global_correct = (global_pred == targets)
        fused_correct = (fused_pred == targets)
        
        # Agreement mask 
        heads_agree = (local_pred == global_pred)
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # Diversity Loss on Errors - Attempts to self regulate specialization by penalizing agreement on errors and rewarding when one head "rescues" the other.
        if self.cfg.error_diversity_weight > 0:
            # Find samples where fusion is wrong
            fused_wrong = ~fused_correct
            
            # Penalize agreement on wrong samples
            agreement_on_error = (fused_wrong & heads_agree).float()
            
            # Soft version: use confidence product instead of hard agreement
            local_conf = local_probs.gather(1, local_pred.unsqueeze(1)).squeeze(1)
            global_conf = global_probs.gather(1, global_pred.unsqueeze(1)).squeeze(1)
            
            # High penalty when both confident AND wrong AND agree
            soft_agreement_penalty = local_conf * global_conf * fused_wrong.float() * heads_agree.float()
            
            error_diversity_loss = self.cfg.error_diversity_weight * soft_agreement_penalty.mean()
            loss_dict['loss_error_diversity'] = error_diversity_loss
            total_loss = total_loss + error_diversity_loss
            
            # Track stats
            with torch.no_grad():
                self.agreement_on_error_rate = self.stats_momentum * self.agreement_on_error_rate + \
                                               (1 - self.stats_momentum) * agreement_on_error.mean()
        
        # Rescue Area - Rewards the head that "rescues" the other
        if self.cfg.rescue_bonus_weight > 0:
            # Local rescue: fusion wrong, global wrong, local right
            local_rescue = fused_wrong & ~global_correct & local_correct
            # Global rescue: fusion wrong, local wrong, global right  
            global_rescue = fused_wrong & ~local_correct & global_correct
            
            if local_rescue.any():
                # Get local's probability on the correct class for rescue samples
                local_correct_prob = local_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                # We want to maxamize this, so we minimize -log(prob)
                local_rescue_bonus = -torch.log(local_correct_prob[local_rescue] + 1e-8).mean()
                loss_dict['loss_local_rescue'] = -self.cfg.rescue_bonus_weight * local_rescue_bonus
                total_loss = total_loss - self.cfg.rescue_bonus_weight * local_rescue_bonus
            
            if global_rescue.any():
                global_correct_prob = global_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                global_rescue_bonus = -torch.log(global_correct_prob[global_rescue] + 1e-8).mean()
                loss_dict['loss_global_rescue'] = -self.cfg.rescue_bonus_weight * global_rescue_bonus
                total_loss = total_loss - self.cfg.rescue_bonus_weight * global_rescue_bonus
            
            # Track rescue rates
            with torch.no_grad():
                self.local_rescue_rate = self.stats_momentum * self.local_rescue_rate + \
                                         (1 - self.stats_momentum) * local_rescue.float().mean()
                self.global_rescue_rate = self.stats_momentum * self.global_rescue_rate + \
                                          (1 - self.stats_momentum) * global_rescue.float().mean()
        
        # CONFIDENCE SHAPING 
        if self.cfg.confidence_shaping_weight > 0:
            # On correct predictions, encourage high confidence. On incorrect predictions, encourage LOW confidence (epistemic humility)
            
            local_target_prob = local_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            global_target_prob = global_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            # Correct predictions: push toward high confidence
            if local_correct.any():
                local_conf_loss_correct = -torch.log(local_target_prob[local_correct] + 1e-8).mean()
            else:
                local_conf_loss_correct = torch.tensor(0.0, device=device)
                
            if global_correct.any():
                global_conf_loss_correct = -torch.log(global_target_prob[global_correct] + 1e-8).mean()
            else:
                global_conf_loss_correct = torch.tensor(0.0, device=device)
            # Wrong predictions: push toward LOW confidence (high entropy)
            local_wrong = ~local_correct
            global_wrong = ~global_correct
            
            if local_wrong.any():
                # Maximize entropy when wrong = minimize negative entropy
                local_entropy_wrong = -(local_probs[local_wrong] * torch.log(local_probs[local_wrong] + 1e-8)).sum(dim=-1)
                local_conf_loss_wrong = -local_entropy_wrong.mean()  # We want high entropy
            else:
                local_conf_loss_wrong = torch.tensor(0.0, device=device)
                
            if global_wrong.any():
                global_entropy_wrong = -(global_probs[global_wrong] * torch.log(global_probs[global_wrong] + 1e-8)).sum(dim=-1)
                global_conf_loss_wrong = -global_entropy_wrong.mean()
            else:
                global_conf_loss_wrong = torch.tensor(0.0, device=device)
            
            conf_loss = 0.5 * (local_conf_loss_correct + global_conf_loss_correct) + \
                        0.3 * (local_conf_loss_wrong + global_conf_loss_wrong)
            
            loss_dict['loss_confidence'] = self.cfg.confidence_shaping_weight * conf_loss
            total_loss = total_loss + self.cfg.confidence_shaping_weight * conf_loss
        
        # HARD SAMPLE DIVERSITY BONUS
        if self.cfg.hard_sample_diversity_bonus > 1.0:
            # Identify hard samples by fused entropy
            fused_entropy = -(fused_probs * torch.log(fused_probs + 1e-8)).sum(dim=-1)
            max_entropy = math.log(fused_probs.size(-1))
            normalized_entropy = fused_entropy / max_entropy
            
            hard_samples = normalized_entropy > self.cfg.hard_sample_threshold
            
            if hard_samples.any():
                # Extra diversity push on hard samples, JS divergence between heads on hard samples
                m = 0.5 * (local_probs[hard_samples] + global_probs[hard_samples])
                kl_lm = (local_probs[hard_samples] * (torch.log(local_probs[hard_samples] + 1e-8) - torch.log(m + 1e-8))).sum(dim=-1)
                kl_gm = (global_probs[hard_samples] * (torch.log(global_probs[hard_samples] + 1e-8) - torch.log(m + 1e-8))).sum(dim=-1)
                js_hard = 0.5 * (kl_lm + kl_gm)
                # Negative
                hard_diversity_loss = -0.05 * self.cfg.hard_sample_diversity_bonus * js_hard.mean()
                loss_dict['loss_hard_diversity'] = hard_diversity_loss
                total_loss = total_loss + hard_diversity_loss
        
        # UPDATE ADAPTIVE AUX WEIGHTS 
        if self.cfg.enable_adaptive_aux:
            with torch.no_grad():
                # Track error rates
                local_err = (~local_correct).float().mean()
                global_err = (~global_correct).float().mean()
                
                self.local_error_rate = self.stats_momentum * self.local_error_rate + \
                                        (1 - self.stats_momentum) * local_err
                self.global_error_rate = self.stats_momentum * self.global_error_rate + \
                                         (1 - self.stats_momentum) * global_err
                
                error_diff = self.local_error_rate - self.global_error_rate
                
                # Adjust weights
                local_adj = error_diff * self.cfg.aux_adaptation_rate
                
                self.adaptive_local_aux = torch.clamp(
                    self.adaptive_local_aux + local_adj,
                    self.cfg.aux_weight_min,
                    self.cfg.aux_weight_max
                )
                self.adaptive_global_aux = torch.clamp(
                    self.adaptive_global_aux - local_adj,
                    self.cfg.aux_weight_min,
                    self.cfg.aux_weight_max
                )
        
        loss_dict['total_adaptive'] = total_loss
        
        return total_loss, loss_dict
    
    def get_adaptive_aux_weights(self) -> Tuple[float, float]:
        # Gets the current adaptive auxiliary weights.
        return float(self.adaptive_local_aux), float(self.adaptive_global_aux)
    
    def get_stats(self) -> Dict[str, float]:
        # Gets rhe current controller statistics
        return {
            'local_error_rate': float(self.local_error_rate),
            'global_error_rate': float(self.global_error_rate),
            'local_rescue_rate': float(self.local_rescue_rate),
            'global_rescue_rate': float(self.global_rescue_rate),
            'agreement_on_error_rate': float(self.agreement_on_error_rate),
            'adaptive_local_aux': float(self.adaptive_local_aux),
            'adaptive_global_aux': float(self.adaptive_global_aux),
        }