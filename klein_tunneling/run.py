"""Prepare and run optical-lattice Klein-tunneling scans on the existing CUDA solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import default_results, execute, load_settings, locate_executable, prepare, print_plan, query_json, resolve_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=Path(__file__).with_name('settings.json'))
    parser.add_argument('--results', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--backend', choices=('cuda', 'numpy'), default='cuda',
                        help='numpy is a slow CPU validation reference; production defaults to CUDA')
    parser.add_argument('--plan', action='store_true', help='calculate bands and preview the scan without writing files or running CUDA')
    parser.add_argument('--prepare-only', action='store_true', help='write isolated input/config files without requiring a CUDA installation')
    args = parser.parse_args()
    if args.batch_size < 1 or args.device < 0:
        parser.error('batch-size must be positive and device must be nonnegative.')
    settings = load_settings(args.settings)
    second, reports = resolve_model(settings)
    print_plan(settings, second, reports)
    if args.plan:
        return 0
    executable, info, device_info = None, None, None
    if args.backend == 'cuda' and not args.prepare_only:
        executable = locate_executable(args.executable)
        info = query_json(executable, '--version-json')
        device_info = query_json(executable, '--device-info', str(args.device))
    results = (args.results or default_results()).resolve()
    manifest = prepare(settings, results, backend=args.backend, solver_info=info, device_info=device_info)
    print(f'Dataset: {results}')
    if args.prepare_only:
        print('Inputs prepared. Run again using a new results directory to launch evolution.')
        return 0
    execute(results, manifest, executable=executable, device=args.device, batch_size=args.batch_size)
    from .analyze import analyze
    analyze(results/'run_manifest.json')
    print(f'Completed. CSV and figures: {results}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
