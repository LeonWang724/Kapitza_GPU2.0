"""Codex CUDA Port: run one existing config with the native CUDA executable."""

from __future__ import annotations

import argparse
from pathlib import Path

from cuda_workflow_common import (
    SCRIPT_DIRECTORY,
    locate_executable,
    read_config,
    run_logged,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=SCRIPT_DIRECTORY / "gpe1d.config")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--floquet-mode", choices=("physical",), default="physical",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--allow-nonempty-output", action="store_true")
    arguments = parser.parse_args()

    config = arguments.config.resolve()
    values = read_config(config)
    output = Path(values["output_folder"])
    if not output.is_absolute():
        output = SCRIPT_DIRECTORY / output
    if output.exists() and any(output.glob("*.h5")) and not arguments.allow_nonempty_output:
        raise RuntimeError(
            f"Refusing to mix snapshots in nonempty output directory {output}. "
            "Choose an empty output_folder or pass --allow-nonempty-output explicitly."
        )

    executable = locate_executable(arguments.executable)
    log_path = config.with_suffix(config.suffix + ".cuda.log")
    return run_logged(
        [
            str(executable),
            str(config),
            "--device",
            str(arguments.device),
            "--floquet-mode",
            arguments.floquet_mode,
        ],
        SCRIPT_DIRECTORY,
        log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
