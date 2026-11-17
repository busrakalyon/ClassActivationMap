from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a public file and optionally verify SHA-256")
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(args.url) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
        actual = digest.hexdigest()
        if args.expected_sha256 and actual.lower() != args.expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch: expected {args.expected_sha256}, found {actual}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"sha256={digest.hexdigest()}  file={destination}")


if __name__ == "__main__":
    main()
