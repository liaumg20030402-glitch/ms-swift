#!/usr/bin/env python3
"""Safely supplement metadata files in an exported FP8 model directory.

This script never copies model weights and never overwrites files that already
exist in the FP8 directory. In particular, the FP8 config and safetensors index
remain untouched.

Example:
    python resave_quant.py \
        --fp8-path /train21/medcog/permanent/jycai6/jmli27/397b/exps/tjbg_xwkong_32k/v0-20260627-112904/checkpoint-85-resave-FP8
 \
        --metadata-source /train21/medcog/permanent/jycai6/jmli27/397b/exps/tjbg_xwkong_32k/v0-20260627-112904/checkpoint-85-resave
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


METADATA_SUFFIXES = {'.json', '.jinja'}
PROTECTED_FILES = {
    'args.json',
    'config.json',
    'model.safetensors.index.json',
    'quantization_config.json',
}
STRUCTURE_FIELDS = (
    'model_type',
    'text_config.model_type',
    'text_config.hidden_size',
    'text_config.num_hidden_layers',
    'text_config.num_experts',
    'text_config.vocab_size',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Copy missing JSON/Jinja metadata into an exported FP8 model directory without touching weights.')
    parser.add_argument('--fp8-path', required=True, type=Path, help='Exported FP8 Hugging Face model directory.')
    parser.add_argument(
        '--metadata-source',
        required=True,
        type=Path,
        help='BF16 resave/pretrained directory providing missing metadata files.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Show planned changes without copying files.')
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_nested(data: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for key in dotted_key.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_directories(fp8_path: Path, metadata_source: Path) -> None:
    if not fp8_path.is_dir():
        raise NotADirectoryError(f'FP8 directory does not exist: {fp8_path}')
    if not metadata_source.is_dir():
        raise NotADirectoryError(f'Metadata source directory does not exist: {metadata_source}')
    if fp8_path.resolve() == metadata_source.resolve():
        raise ValueError('--fp8-path and --metadata-source must be different directories.')


def validate_fp8_config(fp8_path: Path) -> Dict[str, Any]:
    config_path = fp8_path / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing FP8 config: {config_path}')

    config = load_json(config_path)
    quant_config = config.get('quantization_config') or {}
    if quant_config.get('quant_method') != 'fp8':
        raise ValueError(
            f'Expected quantization_config.quant_method="fp8", got: {quant_config.get("quant_method")!r}')
    return config


def validate_model_compatibility(fp8_config: Dict[str, Any], metadata_source: Path) -> None:
    source_config_path = metadata_source / 'config.json'
    if not source_config_path.is_file():
        print(f'[WARN] Source has no config.json; skipping architecture compatibility check: {source_config_path}')
        return

    source_config = load_json(source_config_path)
    mismatches = []
    for field in STRUCTURE_FIELDS:
        fp8_value = get_nested(fp8_config, field)
        source_value = get_nested(source_config, field)
        if fp8_value is not None and source_value is not None and fp8_value != source_value:
            mismatches.append(f'{field}: FP8={fp8_value!r}, source={source_value!r}')

    if mismatches:
        details = '\n  - '.join(mismatches)
        raise ValueError(f'Metadata source is incompatible with the FP8 model:\n  - {details}')


def validate_weight_index(fp8_path: Path) -> None:
    index_path = fp8_path / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise FileNotFoundError(f'Missing FP8 weight index: {index_path}')

    index = load_json(index_path)
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f'Invalid or empty weight_map: {index_path}')

    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (fp8_path / name).is_file()]
    if missing_shards:
        preview = '\n  - '.join(missing_shards[:20])
        raise FileNotFoundError(f'FP8 index references missing weight shards:\n  - {preview}')

    print(f'[OK] Weight index: {len(weight_map)} tensors, {len(shard_names)} shards, no missing shard files.')


def iter_metadata_files(metadata_source: Path) -> Iterable[Path]:
    for path in sorted(metadata_source.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in METADATA_SUFFIXES:
            yield path


def atomic_copy(src: Path, dst: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f'.{dst.name}.', suffix='.tmp', dir=dst.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(src, temp_path)
        os.replace(temp_path, dst)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def copy_missing_metadata(fp8_path: Path, metadata_source: Path, dry_run: bool) -> None:
    protected_hashes = {
        name: sha256(fp8_path / name) for name in PROTECTED_FILES if (fp8_path / name).is_file()
    }
    copied = []
    skipped_existing = []
    skipped_protected = []

    for src in iter_metadata_files(metadata_source):
        dst = fp8_path / src.name
        if src.name in PROTECTED_FILES:
            skipped_protected.append(src.name)
            continue
        if dst.exists():
            skipped_existing.append(src.name)
            continue

        copied.append(src.name)
        if not dry_run:
            atomic_copy(src, dst)

    for name, expected_hash in protected_hashes.items():
        actual_hash = sha256(fp8_path / name)
        if actual_hash != expected_hash:
            raise RuntimeError(f'Protected FP8 file changed unexpectedly: {name}')

    action = 'Would copy' if dry_run else 'Copied'
    print(f'[{"DRY-RUN" if dry_run else "OK"}] {action} {len(copied)} missing metadata files:')
    for name in copied:
        print(f'  + {name}')
    print(f'[INFO] Skipped {len(skipped_existing)} existing files and {len(skipped_protected)} protected files.')
    if skipped_protected:
        print(f'[INFO] Protected: {", ".join(skipped_protected)}')


def main() -> None:
    args = parse_args()
    fp8_path = args.fp8_path.expanduser().resolve()
    metadata_source = args.metadata_source.expanduser().resolve()

    validate_directories(fp8_path, metadata_source)
    fp8_config = validate_fp8_config(fp8_path)
    validate_model_compatibility(fp8_config, metadata_source)
    validate_weight_index(fp8_path)
    copy_missing_metadata(fp8_path, metadata_source, args.dry_run)

    print('[DONE] FP8 weights, config.json, args.json, and model.safetensors.index.json were not modified.')


if __name__ == '__main__':
    main()
