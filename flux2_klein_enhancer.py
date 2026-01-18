"""
FLUX.2 Klein Conditioning Enhancer

Built from actual diagnostic data:
- Conditioning shape: [batch, 512, 12288]
- Active text region: positions 0-77 (std ~40.7)
- Padding region: positions 77-511 (std ~2.3)
- Image edit mode adds: reference_latents in metadata
- Attention mask: [batch, 512] marks valid positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_management as mm
import gc
from typing import Optional, List, Tuple, Dict, Any


class Flux2KleinEnhancer:
    """
    Conditioning enhancer for FLUX.2 Klein based on empirical analysis.
    
    Key findings from diagnostics:
    - Shape: [1, 512, 12288]
    - Positions 0-77: Active text embeddings (std ~40.7)
    - Positions 77+: Padding/inactive (std ~2.3)
    - Image edit adds 'reference_latents' to metadata
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                
                # === Core Enhancement ===
                "text_enhance": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Enhance text region (0-77). Positive=stronger prompt, Negative=weaker. 0=unchanged."
                }),
                
                "detail_sharpen": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sharpen semantic details in text region. Amplifies differences between tokens."
                }),
                
                "coherence_experimental": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "[EXPERIMENTAL] Self-attention for token coherence. Start low (0.1-0.2). May cause artifacts."
                }),
                
                # === Image Edit Specific ===
                "edit_text_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "[IMAGE EDIT] Scale text influence. <1=preserve original more, >1=follow prompt more."
                }),
                
                "edit_blend_mode": (["none", "boost_text", "preserve_image", "balanced"], {
                    "default": "none",
                    "tooltip": "[IMAGE EDIT] Preset blending strategies for edit mode."
                }),
            },
            "optional": {
                "active_token_end": ("INT", {
                    "default": 77,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "End of active text region. Default 77 based on diagnostics."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed. 0=no seeding."
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print debug info to console."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "enhance"
    CATEGORY = "conditioning/flux2"

    def _get_active_region(self, cond: torch.Tensor, mask: Optional[torch.Tensor], 
                           default_end: int) -> Tuple[int, int]:
        """Determine active region from attention mask or default."""
        start = 0
        
        if mask is not None:
            # Find last non-zero position in mask
            if mask.dim() == 2:
                # [B, S] -> find last 1
                nonzero = mask[0].nonzero()
                if len(nonzero) > 0:
                    end = int(nonzero[-1].item()) + 1
                else:
                    end = default_end
            else:
                end = default_end
        else:
            end = default_end
        
        return start, min(end, cond.shape[1])

    def _enhance_region(self, region: torch.Tensor, enhance: float, 
                        sharpen: float, coherence: float, debug: bool) -> torch.Tensor:
        """Apply enhancements to the active text region."""
        
        original = region.clone()
        B, S, D = region.shape
        
        # === 1. Global Enhancement (scale overall signal) ===
        if abs(enhance) > 0.001:
            # Compute per-token magnitude
            magnitudes = region.norm(dim=-1, keepdim=True)
            mean_mag = magnitudes.mean()
            
            # Scale factor based on enhance value
            # enhance > 0: increase magnitude, < 0: decrease
            scale = 1.0 + enhance * 0.5
            region = region * scale
            
            if debug:
                new_mag = region.norm(dim=-1).mean()
                print(f"  Enhancement: scale={scale:.3f}, mag {mean_mag:.2f} -> {new_mag:.2f}")
        
        # === 2. Detail Sharpening (amplify token differences) ===
        if abs(sharpen) > 0.001:
            # Compute mean across sequence
            seq_mean = region.mean(dim=1, keepdim=True)  # [B, 1, D]
            
            # Deviation from mean = "detail"
            detail = region - seq_mean
            
            # Sharpen: amplify deviations
            # Use tanh to prevent explosion
            if sharpen > 0:
                sharpened_detail = detail * (1.0 + sharpen)
                # Soft clip to prevent extreme values
                sharpened_detail = torch.tanh(sharpened_detail / 50.0) * 50.0 * (1.0 + sharpen)
            else:
                # Negative sharpen = blur/smooth
                sharpened_detail = detail * (1.0 + sharpen)  # sharpen is negative, so this reduces
            
            region = seq_mean + sharpened_detail
            
            if debug:
                old_var = detail.var().item()
                new_var = (region - seq_mean).var().item()
                print(f"  Sharpening: variance {old_var:.2f} -> {new_var:.2f}")
        
        # === 3. Coherence [EXPERIMENTAL] (self-attention for semantic consistency) ===
        if coherence > 0.01 and S > 1:
            # Proper attention with careful scaling
            # Project to lower dimension for stability
            head_dim = 128  # Match Qwen3's head dimension
            num_heads = min(8, D // head_dim)  # Use fewer heads for stability
            proj_dim = num_heads * head_dim  # 1024
            
            # Random but deterministic projections (no learned weights)
            # Use orthogonal-like initialization for stability
            torch.manual_seed(42)  # Deterministic
            
            # Project down: [B, S, 12288] -> [B, S, 1024]
            proj_down = torch.randn(D, proj_dim, device=region.device, dtype=region.dtype)
            proj_down = proj_down / (D ** 0.5)  # Scale by input dim
            
            # Project back up: [B, S, 1024] -> [B, S, 12288]
            proj_up = torch.randn(proj_dim, D, device=region.device, dtype=region.dtype)
            proj_up = proj_up / (proj_dim ** 0.5)  # Scale by proj dim
            
            # Project to lower dim
            region_proj = torch.matmul(region, proj_down)  # [B, S, 1024]
            
            # Normalize before attention (critical for stability)
            proj_norm = region_proj / (region_proj.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Scaled dot-product attention on normalized vectors
            scores = torch.bmm(proj_norm, proj_norm.transpose(1, 2))  # [B, S, S], values in [-1, 1]
            scores = scores / (head_dim ** 0.5)  # Additional scaling
            
            # Temperature to soften attention (prevent winner-take-all)
            temperature = 2.0
            attn_weights = F.softmax(scores / temperature, dim=-1)
            
            # Apply attention
            attn_out_proj = torch.bmm(attn_weights, region_proj)  # [B, S, 1024]
            
            # Project back to full dimension
            attn_out = torch.matmul(attn_out_proj, proj_up)  # [B, S, 12288]
            
            # Match scale of original (prevent magnitude explosion)
            scale_ratio = region.std() / (attn_out.std() + 1e-6)
            attn_out = attn_out * scale_ratio
            
            # Residual blend with conservative strength
            region = region + coherence * (attn_out - region)
            
            # Clean up
            del proj_down, proj_up, region_proj, proj_norm, scores, attn_weights, attn_out_proj, attn_out
            
            if debug:
                print(f"  Coherence [EXPERIMENTAL]: applied with weight {coherence:.2f} (proj_dim={proj_dim})")
        
        # === 4. Preserve mean only (keep intentional magnitude changes) ===
        # Only recenter, don't rescale std - that would undo enhance/sharpen
        orig_mean = original.mean(dim=-1, keepdim=True)
        new_mean = region.mean(dim=-1, keepdim=True)
        
        region = region - new_mean + orig_mean
        
        return region

    def _apply_edit_mode(self, cond: torch.Tensor, meta: Dict, 
                         text_weight: float, blend_mode: str,
                         active_start: int, active_end: int, debug: bool) -> torch.Tensor:
        """Apply image-edit specific modifications."""
        
        is_edit_mode = "reference_latents" in meta and meta["reference_latents"] is not None
        
        if not is_edit_mode:
            if debug:
                print("  Not in image edit mode (no reference_latents)")
            return cond
        
        if debug:
            ref_latents = meta["reference_latents"]
            print(f"  Image edit mode detected: {len(ref_latents)} reference latent(s)")
        
        # Apply blend mode presets
        if blend_mode == "boost_text":
            # Strengthen text signal for stronger edits
            text_weight = max(text_weight, 1.5)
        elif blend_mode == "preserve_image":
            # Weaken text signal to preserve more of original
            text_weight = min(text_weight, 0.5)
        elif blend_mode == "balanced":
            text_weight = 1.0
        
        # Apply text weight scaling to active region only
        if abs(text_weight - 1.0) > 0.001:
            active_region = cond[:, active_start:active_end, :].clone()
            
            # Scale the active region
            # Weight > 1: text dominates more
            # Weight < 1: text influence reduced (image preserved more)
            scaled = active_region * text_weight
            
            # Prevent extreme values
            max_val = active_region.abs().max() * 2.0
            scaled = scaled.clamp(-max_val, max_val)
            
            cond = cond.clone()
            cond[:, active_start:active_end, :] = scaled
            
            if debug:
                print(f"  Edit text weight: {text_weight:.2f} applied to positions {active_start}-{active_end}")
        
        return cond

    def enhance(self, conditioning, text_enhance: float, detail_sharpen: float,
                coherence_experimental: float, edit_text_weight: float, edit_blend_mode: str,
                active_token_end: int = 77, seed: int = 0, debug: bool = False):
        
        # Alias for internal use
        coherence = coherence_experimental
        
        if not conditioning:
            return (conditioning,)
        
        if seed > 0:
            torch.manual_seed(seed)
        
        device = mm.get_torch_device()
        
        enhanced_list = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            if debug:
                print(f"\n=== Flux2KleinEnhancer Item {idx} ===")
                print(f"Input shape: {cond_tensor.shape}")
                print(f"Metadata keys: {list(meta.keys())}")
            
            # Work on device
            cond = cond_tensor.to(device, dtype=torch.float32)
            original_dtype = cond_tensor.dtype
            
            # Get attention mask if available
            attn_mask = meta.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            
            # Determine active region
            active_start, active_end = self._get_active_region(cond, attn_mask, active_token_end)
            
            if debug:
                print(f"Active region: {active_start} to {active_end}")
                active = cond[:, active_start:active_end, :]
                padding = cond[:, active_end:, :]
                print(f"Active region std: {active.std():.4f}")
                if padding.shape[1] > 0:
                    print(f"Padding region std: {padding.std():.4f}")
            
            # === Apply enhancements to active region only ===
            if abs(text_enhance) > 0.001 or abs(detail_sharpen) > 0.001 or coherence > 0.01:
                active_region = cond[:, active_start:active_end, :].clone()
                
                enhanced_region = self._enhance_region(
                    active_region, text_enhance, detail_sharpen, coherence, debug
                )
                
                cond = cond.clone()
                cond[:, active_start:active_end, :] = enhanced_region
            
            # === Apply image edit mode adjustments ===
            cond = self._apply_edit_mode(
                cond, meta, edit_text_weight, edit_blend_mode,
                active_start, active_end, debug
            )
            
            # Back to CPU with original dtype
            result = cond.to("cpu", dtype=original_dtype)
            
            if debug:
                diff = (result - cond_tensor).abs()
                print(f"Output change: mean={diff.mean():.6f}, max={diff.max():.6f}")
            
            enhanced_list.append((result, meta))
            
            # Cleanup
            del cond
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        return (enhanced_list,)


class Flux2KleinEditController:
    """
    Specialized controller for FLUX.2 Klein image editing.
    
    Provides fine-grained control over the balance between
    text prompt (what to change) and reference image (what to keep).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                
                "prompt_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "How strongly the text prompt influences the edit. 0=ignore prompt, 1=normal, >1=stronger"
                }),
                
                "preserve_structure": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Reduce high-frequency changes in conditioning to preserve image structure."
                }),
                
                "token_emphasis_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 76,
                    "step": 1,
                    "tooltip": "Start of token range to emphasize (0-76)"
                }),
                
                "token_emphasis_end": ("INT", {
                    "default": 77,
                    "min": 1,
                    "max": 77,
                    "step": 1,
                    "tooltip": "End of token range to emphasize (1-77)"
                }),
                
                "emphasis_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Strength multiplier for emphasized token range."
                }),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2"

    def control(self, conditioning, prompt_strength: float, preserve_structure: float,
                token_emphasis_start: int, token_emphasis_end: int, 
                emphasis_strength: float, debug: bool = False):
        
        if not conditioning:
            return (conditioning,)
        
        device = mm.get_torch_device()
        result_list = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            cond = cond_tensor.to(device, dtype=torch.float32)
            original_dtype = cond_tensor.dtype
            
            is_edit = "reference_latents" in meta
            
            if debug:
                print(f"\n=== Flux2KleinEditController Item {idx} ===")
                print(f"Edit mode: {is_edit}")
                print(f"Shape: {cond.shape}")
            
            # Active region is 0-77 based on diagnostics
            active_end = 77
            active = cond[:, :active_end, :].clone()
            
            # === 1. Global prompt strength ===
            if abs(prompt_strength - 1.0) > 0.001:
                active = active * prompt_strength
                if debug:
                    print(f"Applied prompt_strength: {prompt_strength}")
            
            # === 2. Preserve structure (low-pass filter) ===
            if preserve_structure > 0.01:
                # Smooth along embedding dimension to reduce high-freq changes
                # This makes the edit more "conservative"
                kernel_size = 5
                padding = kernel_size // 2
                
                # [B, S, D] -> [B, D, S] for conv1d
                active_t = active.transpose(1, 2)
                smoothed = F.avg_pool1d(active_t, kernel_size=kernel_size, 
                                        stride=1, padding=padding)
                smoothed = smoothed.transpose(1, 2)
                
                # Blend original with smoothed
                active = active * (1 - preserve_structure) + smoothed * preserve_structure
                
                if debug:
                    print(f"Applied preserve_structure: {preserve_structure}")
            
            # === 3. Token range emphasis ===
            if abs(emphasis_strength - 1.0) > 0.001:
                start = max(0, min(token_emphasis_start, active_end - 1))
                end = max(start + 1, min(token_emphasis_end, active_end))
                
                active[:, start:end, :] = active[:, start:end, :] * emphasis_strength
                
                if debug:
                    print(f"Emphasized tokens {start}-{end} by {emphasis_strength}")
            
            # Reassemble
            cond = cond.clone()
            cond[:, :active_end, :] = active
            
            result = cond.to("cpu", dtype=original_dtype)
            result_list.append((result, meta))
            
            del cond, active
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        return (result_list,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinEditController": Flux2KleinEditController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinEditController": "FLUX.2 Klein Edit Controller",
}
