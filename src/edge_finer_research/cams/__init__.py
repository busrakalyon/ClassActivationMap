from .epoch_fusion import fuse_epoch_cams

# Torch-backed extractors intentionally are not imported here, so the NumPy
# mathematical reference and its tests remain usable in a CPU/minimal audit environment.
__all__ = ["fuse_epoch_cams"]
