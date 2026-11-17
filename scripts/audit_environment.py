from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch
        import torchvision

        report.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            report["cuda_memory_bytes"] = {"free": free, "total": total}
    except ImportError as exc:
        report["torch_import_error"] = str(exc)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
