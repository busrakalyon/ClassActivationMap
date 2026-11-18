"""Read-only diagnostics for the Experiment-1 EdgeCAM reconstruction.

Nothing in this package is used by the baseline training or CAM scripts.  It
exists so numerical alternatives can be measured before changing the baseline.
"""

from .edgecam_variants import VariantSpec, default_variant_specs

__all__ = ["VariantSpec", "default_variant_specs"]
