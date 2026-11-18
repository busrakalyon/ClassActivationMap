from .epoch_fusion import fuse_epoch_cams
from .streaming_fusion import (
    StreamingMaxState,
    discover_epoch_checkpoints,
    initial_streaming_state,
    update_streaming_max,
)

# Torch-backed extractors intentionally are not imported here, so the NumPy
# mathematical reference and its tests remain usable in a CPU/minimal audit environment.
__all__ = [
    "StreamingMaxState",
    "discover_epoch_checkpoints",
    "fuse_epoch_cams",
    "initial_streaming_state",
    "update_streaming_max",
]
