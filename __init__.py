from .flux2_klein_ref_controller import NODE_CLASS_MAPPINGS as REF_NODES, NODE_DISPLAY_NAME_MAPPINGS as REF_NAMES
from .flux2_klein_text_enhancer import NODE_CLASS_MAPPINGS as TEXT_NODES, NODE_DISPLAY_NAME_MAPPINGS as TEXT_NAMES
from .flux2_klein_enhancer import Flux2KleinEnhancer, Flux2KleinDetailController
from .flux2_sectioned_encoder import Flux2KleinSectionedEncoder

NODE_CLASS_MAPPINGS = {
    **REF_NODES,
    **TEXT_NODES,
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **REF_NAMES,
    **TEXT_NAMES,
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller",
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder",
}

__version__ = "2.4.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
