"""Content-Aware Image Resizing via Composition Detection and Composition Rules.
Reproduction of Wang et al., Electronics 2023, 12, 3096."""
from .energy import (gradient_energy, saliency_map, foreground_mask,
                     coseg_importance, importance_for, foreground_center,
                     foreground_bbox)
from .seam import (carve_vertical, carve_horizontal, seam_carve_baseline)
from .composition import (resize, resize_thirds, resize_central,
                          resize_horizontal, resize_symmetric,
                          detect_composition, RULES)
from .metric import quality_index, information_loss, geometric_distortion
from .baselines import cutting_crop_scale

__all__ = [
    "gradient_energy", "saliency_map", "foreground_mask", "coseg_importance",
    "importance_for", "foreground_center", "foreground_bbox",
    "carve_vertical", "carve_horizontal", "seam_carve_baseline",
    "resize", "resize_thirds", "resize_central", "resize_horizontal",
    "resize_symmetric", "detect_composition", "RULES",
    "quality_index", "information_loss", "geometric_distortion",
]
