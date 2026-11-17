# Data policy

This directory intentionally contains no dataset files.

Raw images, derived masks, SOP artifacts and checkpoints must remain outside the source tree.
Portable CSV manifests under `manifests/` contain relative paths and SHA-256 values. Absolute
roots are supplied with `configs/paths.toml` on each machine.

