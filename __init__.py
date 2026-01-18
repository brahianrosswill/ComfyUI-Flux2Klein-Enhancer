"""
FLUX.2 Klein Conditioning Enhancer v2

Built from empirical diagnostic data:
- Conditioning: [batch, 512, 12288]
- Active text: positions 0-77 (std ~40.7)
- Padding: positions 77-511 (std ~2.3)
- Image edit mode: adds 'reference_latents' to metadata

Nodes:
- Flux2KleinEnhancer: General enhancement for T2I and image edit
- Flux2KleinEditController: Fine-grained control for image editing

Key insight: Only positions 0-77 contain meaningful text embeddings.
The rest is padding. All enhancement targets this active region only.
"""

from .flux2_klein_enhancer import (
    Flux2KleinEnhancer,
    Flux2KleinEditController,
)

NODE_CLASS_MAPPINGS = {
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinEditController": Flux2KleinEditController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinEditController": "FLUX.2 Klein Edit Controller",
}

__version__ = "2.0.0"
