"""Pet portrait nodes v2: layout/colour nodes (pet_portrait_layout) and the experimental
lost-feature restore nodes (pet_feature_inpaint) in one ComfyUI node pack."""
from .pet_portrait_layout import NODE_CLASS_MAPPINGS as _LAYOUT, NODE_DISPLAY_NAME_MAPPINGS as _LAYOUT_NAMES
from .pet_feature_inpaint import NODE_CLASS_MAPPINGS as _FEATURE, NODE_DISPLAY_NAME_MAPPINGS as _FEATURE_NAMES

NODE_CLASS_MAPPINGS = {**_LAYOUT, **_FEATURE}
NODE_DISPLAY_NAME_MAPPINGS = {**_LAYOUT_NAMES, **_FEATURE_NAMES}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
