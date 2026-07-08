#!/usr/bin/env python3
'''Convert SWIFT messages-format SFT JSONL to plain-text PT JSONL.'''

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input SFT JSONL file.')
    parser.add_argument('--output', required=True, help='Output PT JSONL file.')
    parser.add_argument('--separator', default='\n', help='Separator used to join message contents.')
    parser.add_argument('--strict', action='store_true', help='Stop when an invalid row is encountered.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f'Input file does not exist: {input_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    errors = 0

    with input_path.open('r', encoding='utf-8') as reader, output_path.open('w', encoding='utf-8') as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                messages = row['messages']
                if not isinstance(messages, list) or not messages:
                    raise ValueError('messages must be a non-empty list')

                contents = []
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError('each message must be an object')
                    content = message.get('content')
                    if not isinstance(content, str):
                        raise ValueError('message content must be a string')
                    content = content.strip()
                    if content:
                        contents.append(content)

                if not contents:
                    raise ValueError('all message contents are empty')

                text = args.separator.join(contents)
                pt_row = {'messages': [{'role': 'assistant', 'content': text}]}
                writer.write(json.dumps(pt_row, ensure_ascii=False) + '\n')
                written += 1
            except Exception as error:
                errors += 1
                message = f'{input_path}:{line_number}: {type(error).__name__}: {error}'
                if args.strict:
                    raise RuntimeError(message) from error
                if errors <= 10:
                    print(message, file=sys.stderr)

    print(f'input: {total}')
    print(f'written: {written}')
    print(f'errors: {errors}')
    print(f'output: {output_path}')

    if written == 0:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
