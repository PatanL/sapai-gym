#!/usr/bin/env python3
"""Package the shared SAP workspace without credentials or intermediate models.

Run from a workspace containing sibling sapai/ and sapai-gym/ checkouts.
The archive preserves the original runs/ paths required by frozen imports.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


EXPERIMENTS = (
    'claude-planner-20261013-v1',
    'claude-cheap-position-20261112-v1',
    'claude-end-positioner-20261112-v1',
)
SUPPORT = (
    'search-throughput-5x-20260912-v1/reference-code',
    'search-throughput-5x-20260912-v1/reference-source',
    'search-played-teaching-20260912-v1',
    'search-transfer-24h-20260916-v1',
    'search-iterated-campaign-20260917-v1',
    'direct-opening-exploration-20260918-v1',
    'ordinary-rl-shared-20260924-v2',
    'search-scale-campaign-20260911-v1/source-training-frozen-v1',
)
POLICY = 'ordinary-rl-shared-20260924-v2/outputs/chunk-20/update-025772.pt'
MODELS = {
    'update-025772.pt': f'runs/{POLICY}',
    'battle_net.pt': f'runs/{EXPERIMENTS[0]}/out/battle_net.pt',
    'posnet-zeroshot.pt': f'runs/{EXPERIMENTS[2]}/out/posnet-zeroshot.pt',
}
CODE_SUFFIXES = {'.py', '.sh', '.js', '.md', '.rst', '.toml', '.txt'}
SKIP_PARTS = {'.git', '__pycache__', '.pytest_cache', 'node_modules',
              '.venv', 'venv', 'cpu-deps', 'test-deps', 'outputs', 'out',
              'launch-v1', 'launch-early-v2'}
CONTROL_NAMES = {'SPEC.json', 'RECIPE.json', 'REFERENCE_MANIFEST.json'}
SECRET = re.compile(
    rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}'
    rb'|sk-proj-[A-Za-z0-9_-]{35,}'
    rb'|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'
)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_files(root: Path):
    """Keep source and sealed control manifests, excluding run output trees."""
    for p in sorted(root.rglob('*')):
        if any(part in SKIP_PARTS for part in p.relative_to(root).parts):
            continue
        if p.is_file() and (p.suffix in CODE_SUFFIXES or
                            p.name in CONTROL_NAMES or
                            p.name.startswith('MANIFEST') and p.suffix == '.json'):
            yield p


def git_files(root: Path):
    raw = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '--cached', '--others',
         '--exclude-standard', '-z'])
    for value in set(raw.split(b'\0')) - {b''}:
        p = root / value.decode()
        if p.is_file() and p.suffix not in {'.pt', '.zip', '.pyc', '.DS_Store'}:
            if not any(part in SKIP_PARTS for part in p.relative_to(root).parts):
                yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--export-experiments', action='store_true',
                    help='Copy the latest experiment source into this Git repo.')
    args = ap.parse_args()
    root = args.workspace.resolve()
    if args.export_experiments:
        dest = root / 'sapai-gym/research/experiments'
        for name in EXPERIMENTS:
            for p in (root / 'runs' / name).iterdir():
                if p.is_file() and p.suffix in CODE_SUFFIXES:
                    data = p.read_bytes()
                    if SECRET.search(data):
                        raise ValueError(f'Credential marker in {p}')
                    target = dest / name / p.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
        print(f'Exported experiment source to {dest}')
        if args.output_dir is None:
            return
    if args.output_dir is None:
        ap.error('--output-dir is required when packaging')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / 'research-workspace.tar.gz'
    manifest_path = output / 'RELEASE_MANIFEST.json'
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError('Use a new output directory for each immutable release.')
    selected = set()
    for repo in ('sapai', 'sapai-gym'):
        selected.update(git_files(root / repo))
    selected.update(source_files(root / 'scripts'))
    for pattern in ('Dockerfile*', '*.md', '.dockerignore'):
        selected.update(root.glob(pattern))
    for name in (*EXPERIMENTS, *SUPPORT):
        directory = root / 'runs' / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        selected.update(source_files(directory))
    reference = root / 'runs/search-throughput-5x-20260912-v1/REFERENCE_MANIFEST.json'
    selected.add(reference)
    sealed = json.loads(reference.read_text())
    for directory, field in [('reference-code', 'runnerFiles'),
                             ('reference-source', 'sourceFiles')]:
        for name, expected in sealed[field].items():
            p = reference.parent / directory / name
            if digest(p) != expected:
                raise ValueError(f'Changed sealed reference: {p}')
            selected.add(p)
    # Top-level planner measurements are compact; per-game distillation tensors
    # and historical checkpoint collections are deliberately not selected.
    for name in EXPERIMENTS:
        out = root / 'runs' / name / 'out'
        for p in out.iterdir():
            if p.is_file() and p.suffix in {'.json', '.jsonl'}:
                selected.add(p)
        for name2 in ('library.pkl', 'battle_net.pt', 'vocab.json'):
            p = out / name2
            if p.is_symlink():
                selected.add(p)
    selected.add(root / f'runs/{EXPERIMENTS[0]}/out/library.pkl')
    for relative in (
        'corrected-simulator-v30-20260910-v1/IMPLEMENTATION.md',
        'corrected-simulator-v30-20260910-v1/CUDA_IMPLEMENTATION.md',
        'identity-lifetime-fix-20260908-v1/IMPLEMENTATION.md',
        'human-exploit-20260910-v1/food-right-v6/CHANGE.md',
    ):
        selected.add(root / 'runs' / relative)
    training = root / 'runs' / Path(POLICY).parent
    for name in ('config.json', 'league.json', 'REPORT.json',
                 'ORDINARY_ONLY_AUDIT.json', 'RESUME_PARITY_025550.json'):
        selected.add(training / name)
    manifest = {
        'schema': 'sap-rl-share-v1',
        'release_tag': 'research-update-25772-20261112',
        'policy_update': 25772,
        'recorded_decisions': 221367362,
        'positioner_status': 'zero-shot experimental initialization',
        'runtime': 'nvcr.io/nvidia/pytorch:25.09-py3',
        'original_runtime_image': 'sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1',
        'repositories': {}, 'model_assets': {}, 'workspace_files': {},
        'resume_note': 'Optimizer state is included. Exact historical continuation also requires the original replay corpus and resident checkpoint set, which are not included.',
    }
    for repo in ('sapai', 'sapai-gym'):
        manifest['repositories'][repo] = subprocess.check_output(
            ['git', '-C', str(root / repo), 'rev-parse', 'HEAD'], text=True).strip()
    for asset, relative in MODELS.items():
        p = root / relative
        manifest['model_assets'][asset] = {
            'workspace_path': relative, 'bytes': p.stat().st_size,
            'sha256': digest(p),
        }
    with tarfile.open(archive_path, 'w:gz', dereference=False) as archive:
        for p in sorted(selected):
            if not p.is_file():
                raise FileNotFoundError(p)
            relative = p.relative_to(root).as_posix()
            if p.is_symlink():
                target = p.resolve()
                external_models = {root / value for value in MODELS.values()}
                if target not in selected and target not in external_models:
                    raise ValueError(f'Missing symlink target: {relative}')
                manifest['workspace_files'][relative] = {
                    'symlink': str(p.readlink()), 'sha256': digest(target)}
            else:
                if p.suffix in CODE_SUFFIXES | {'.json', '.jsonl'} and SECRET.search(p.read_bytes()):
                    raise ValueError(f'Credential marker in {relative}')
                manifest['workspace_files'][relative] = {
                    'bytes': p.stat().st_size, 'sha256': digest(p)}
            archive.add(p, arcname=relative, recursive=False)
        contents = json.dumps(manifest, indent=2).encode() + b'\n'
        info = tarfile.TarInfo('RELEASE_MANIFEST.json')
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))
    # External manifest also authenticates the archive; its embedded version
    # authenticates the extracted files and external model assets.
    manifest['archive'] = {'name': archive_path.name,
                           'bytes': archive_path.stat().st_size,
                           'sha256': digest(archive_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archive': str(archive_path),
                      'bytes': archive_path.stat().st_size,
                      'workspace_files': len(selected),
                      'manifest': str(manifest_path)}))


if __name__ == '__main__':
    main()
