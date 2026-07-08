#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并多个文件夹/文件中的 swift SFT jsonl 数据到一个大 jsonl。

特点：
- 逐行读写，避免 `cat` 在「文件结尾缺换行」时把两条 JSON 拼到一行的坑；
- 默认校验每行是合法 JSON（坏行跳过并告警），可选只保留含 "messages" 的行；
- 可选按内容去重（不同文件夹常有重复样本）；
- 文件夹会递归收集 *.jsonl；也可直接传单个 jsonl 文件；
- 打印每个文件和总计的行数统计。

用法示例：
  # 合并两个目录 + 一个文件 到 merged.jsonl
  python merge_jsonl.py -o merged.jsonl /data/dir1 /data/dir2 /data/extra.jsonl

  # 同时去重 + 要求每行含 messages 字段
  python merge_jsonl.py -o merged.jsonl --dedup --require-messages /data/dir1 /data/dir2
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def collect_files(inputs, recursive=True):
    """把输入(文件或文件夹)展开成 jsonl 文件列表，保持去重且稳定顺序。"""
    files = []
    seen = set()
    for item in inputs:
        if os.path.isdir(item):
            pattern = os.path.join(item, '**', '*.jsonl') if recursive else os.path.join(item, '*.jsonl')
            matched = sorted(glob.glob(pattern, recursive=recursive))
            if not matched:
                print(f'[WARN] 目录下没有 .jsonl: {item}', file=sys.stderr)
            for f in matched:
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        elif os.path.isfile(item):
            if item not in seen:
                seen.add(item)
                files.append(item)
        else:
            print(f'[WARN] 路径不存在，跳过: {item}', file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description='合并 swift SFT jsonl 数据')
    ap.add_argument('inputs', nargs='+', help='输入的文件夹或 jsonl 文件（可多个）')
    ap.add_argument('-o', '--output', required=True, help='输出的合并 jsonl 路径')
    ap.add_argument('--no-validate', action='store_true', help='不校验 JSON（更快，但坏行会原样写入）')
    ap.add_argument('--require-messages', action='store_true', help='只保留含 "messages" 字段的行')
    ap.add_argument('--dedup', action='store_true', help='按整行内容去重')
    ap.add_argument('--no-recursive', action='store_true', help='文件夹不递归子目录')
    args = ap.parse_args()

    files = collect_files(args.inputs, recursive=not args.no_recursive)
    if not files:
        print('[ERROR] 没有可合并的 jsonl 文件。', file=sys.stderr)
        sys.exit(1)

    print('=' * 60)
    print(f'[INFO] 待合并文件数: {len(files)}')
    for f in files:
        print(f'  - {f}')
    print(f'[INFO] 输出: {args.output}')
    print(f'[INFO] 校验JSON={not args.no_validate}  require_messages={args.require_messages}  dedup={args.dedup}')
    print('=' * 60)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    total_in = 0       # 读到的非空行
    total_out = 0      # 实际写出的行
    bad_json = 0       # JSON 解析失败
    no_msg = 0         # 缺 messages
    dup = 0            # 重复
    seen_hash = set()  # 去重用

    with open(args.output, 'w', encoding='utf-8') as out:
        for f in files:
            file_in = 0
            file_out = 0
            with open(f, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    file_in += 1
                    total_in += 1

                    if not args.no_validate:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            bad_json += 1
                            continue
                        if args.require_messages and 'messages' not in obj:
                            no_msg += 1
                            continue

                    if args.dedup:
                        h = hashlib.md5(line.encode('utf-8')).digest()
                        if h in seen_hash:
                            dup += 1
                            continue
                        seen_hash.add(h)

                    out.write(line + '\n')   # 统一补换行，杜绝拼行
                    file_out += 1
                    total_out += 1
            print(f'  [{os.path.basename(f)}] 读入 {file_in} 行 -> 写出 {file_out} 行')

    print('=' * 60)
    print(f'[OK] 合并完成: {args.output}')
    print(f'     总读入(非空)  : {total_in}')
    print(f'     总写出        : {total_out}')
    if bad_json:
        print(f'     跳过(JSON错误): {bad_json}')
    if no_msg:
        print(f'     跳过(无messages): {no_msg}')
    if dup:
        print(f'     跳过(重复)    : {dup}')
    print('=' * 60)


if __name__ == '__main__':
    main()
