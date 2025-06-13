# from .seg_utils import Segmenter, polygon_to_obb, crop_from_obb, save_mask_or_box
# # from .align_utils import orb_align
# from .align_utils import rotate_align
# from .pipeline import process_instance, batch_dir, single_pair
# # __all__ = [
# #     "Segmenter",
# #     "polygon_to_obb",
# #     "crop_from_obb",
# #     "save_mask_or_box",
# #     "orb_align",
# #     "process_instance",
# #     "batch_dir",
# #     "single_pair"
# # ]

# __all__ = [
#     "Segmenter",
#     "polygon_to_obb",
#     "crop_from_obb",
#     "save_mask_or_box",
#     "rotate_align",
#     "process_instance",
#     "batch_dir",
#     "single_pair"
# ]


"""
Convenience re-exports so callers can simply:
    from pipeline import batch_dir_dual, single_pair_dual, ...
"""

from .seg_utils  import (
    Segmenter, polygon_to_obb, crop_from_obb, save_mask_or_box
)
from .align_utils import rotate_align, rotate_align_two_refs
from .pipeline    import (
    process_instance,  batch_dir,  single_pair,
    process_instance_dual, batch_dir_dual, single_pair_dual
)

__all__ = [
    # segmentation
    "Segmenter", "polygon_to_obb", "crop_from_obb", "save_mask_or_box",
    # alignment
    "rotate_align", "rotate_align_two_refs",
    # single-reference high-level
    "process_instance", "batch_dir", "single_pair",
    # dual-reference high-level
    "process_instance_dual", "batch_dir_dual", "single_pair_dual",
]
