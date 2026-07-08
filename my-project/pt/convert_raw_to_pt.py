#!/usr/bin/env python3
"""Convert a recursive tree of {"text": ...} JSONL shards to SWIFT PT format.
python convert_text_tree_to_pt.py \
  --input /train21/medcog/permanent/jmli27/pt/dataset/synthetic \
  --output /train21/medcog/permanent/jmli27/pt/dataset/synthetic_pt \
  --workers 8 \
  --strict

  把{"text":"...\n...<ret><end>","data_source":"xxx"}
  转换为：
  {"messages":[{"role":"assistant","content":"..\n..."}]}

"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


TERMINAL_MARKERS = re.compile(r'(?:\s*(?:<ret>|<end>))+\s*\Z', flags=re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSONL file or root directory.')
    parser.add_argument('--output', required=True, help='Output root directory.')
    parser.add_argument('--text-key', default='text', help='Source text field. Default: text.')
    parser.add_argument('--workers', type=int, default=8, help='Number of files converted in parallel.')
    parser.add_argument('--strict', action='store_true', help='Stop a shard at its first malformed row.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output shards.')
    return parser.parse_args()


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_jobs(input_path: Path, output_root: Path):
    input_path = input_path.resolve()
    output_root = output_root.resolve()

    if input_path.is_file():
        if input_path.suffix.lower() != '.jsonl':
            raise ValueError(f'Input file is not JSONL: {input_path}')
        return [(input_path, output_root / input_path.name)]

    if not input_path.is_dir():
        raise FileNotFoundError(f'Input does not exist: {input_path}')

    jobs = []
    for source in sorted(input_path.rglob('*.jsonl')):
        source = source.resolve()
        if is_under(source, output_root):
            continue
        relative = source.relative_to(input_path)
        jobs.append((source, output_root / relative))
    if not jobs:
        raise ValueError(f'No JSONL files found under: {input_path}')
    return jobs


def convert_file(source, target, text_key, strict, overwrite):
    source = Path(source)
    target = Path(target)

    if target.exists() and not overwrite:
        return {
            'source': str(source),
            'target': str(target),
            'skipped_file': 1,
            'input': 0,
            'written': 0,
            'cleaned': 0,
            'empty': 0,
            'errors': 0,
            'error_examples': [],
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    stats = {
        'source': str(source),
        'target': str(target),
        'skipped_file': 0,
        'input': 0,
        'written': 0,
        'cleaned': 0,
        'empty': 0,
        'errors': 0,
        'error_examples': [],
    }

    try:
        with source.open('r', encoding='utf-8') as reader, temporary.open('w', encoding='utf-8') as writer:
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                stats['input'] += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('top-level JSON value must be an object')

                    text = row.get(text_key)
                    if not isinstance(text, str):
                        raise ValueError(f'field {text_key!r} must be a string')

                    text, replacements = TERMINAL_MARKERS.subn('', text)
                    text = text.strip()
                    if replacements:
                        stats['cleaned'] += 1
                    if not text:
                        stats['empty'] += 1
                        continue

                    converted = {'messages': [{'role': 'assistant', 'content': text}]}
                    writer.write(json.dumps(converted, ensure_ascii=False) + '\n')
                    stats['written'] += 1
                except Exception as error:
                    stats['errors'] += 1
                    location = f'{source}:{line_number}: {type(error).__name__}: {error}'
                    if strict:
                        raise RuntimeError(location) from error
                    if len(stats['error_examples']) < 5:
                        stats['error_examples'].append(location)

        os.replace(temporary, target)
        return stats
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()
    jobs = collect_jobs(input_path, output_root)

    print(f'JSONL shards: {len(jobs)}')
    print(f'Output root: {output_root.resolve()}')

    totals = {
        'skipped_file': 0,
        'input': 0,
        'written': 0,
        'cleaned': 0,
        'empty': 0,
        'errors': 0,
    }
    error_examples = []

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(convert_file, source, target, args.text_key, args.strict, args.overwrite)
            for source, target in jobs
        ]
        completed = 0
        for future in as_completed(futures):
            stats = future.result()
            completed += 1
            for key in totals:
                totals[key] += stats[key]
            error_examples.extend(stats['error_examples'])
            print(
                f'[{completed}/{len(jobs)}] {stats["source"]} -> {stats["target"]}; '
                f'written={stats["written"]}, cleaned={stats["cleaned"]}, errors={stats["errors"]}',
                flush=True)

    print('\nSummary')
    for key, value in totals.items():
        print(f'{key}: {value}')

    if error_examples:
        print('\nFirst conversion errors:', file=sys.stderr)
        for example in error_examples[:20]:
            print(f'  {example}', file=sys.stderr)

    if totals['written'] == 0 and totals['skipped_file'] == 0:
        return 1
    if totals['errors']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
