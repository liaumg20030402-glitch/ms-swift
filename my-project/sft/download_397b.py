#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 ModelScope 下载 Qwen3.5-397B-A17B 的指定分片到本机。

默认只下载：
  model.safetensors-00049-of-00094.safetensors

用法：

Windows:
  python download_397b.py
  python download_397b.py --local-dir D:\\models\\Qwen3.5-397B-A17B
  python download_397b.py --files model.safetensors-00049-of-00094.safetensors model.safetensors-00050-of-00094.safetensors

Linux:
  python download_397b.py --local-dir /iflytek/jmli27
  python download_397b.py --local-dir /iflytek/jmli27 --files model.safetensors-00049-of-00094.safetensors

下载整库：
  python download_397b.py --files "*"

无限重试：
  python download_397b.py --max-retry 0

最多重试 10 次：
  python download_397b.py --max-retry 10
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-397B-A17B"


def ensure_modelscope_installed() -> None:
    """确保 modelscope 已安装。"""
    if importlib.util.find_spec("modelscope") is not None:
        return

    print("[INFO] 未检测到 modelscope，正在安装...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "modelscope",
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-397B-A17B shards from ModelScope with retry."
    )

    parser.add_argument(
        "--local-dir",
        type=str,
        default=r"D:\Qwen3.5-397B-A17B",
        help="本地保存目录，默认 D:\\Qwen3.5-397B-A17B",
    )

    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=["model.safetensors-00049-of-00094.safetensors"],
        help='要下载的文件名列表；传 "*" 表示下载整库',
    )

    parser.add_argument(
        "--max-retry",
        type=int,
        default=0,
        help="最大重试次数；0 表示无限重试直到成功",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=30,
        help="每次失败后的等待秒数",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="下载线程数；单文件建议 1",
    )

    parser.add_argument(
        "--http-timeout",
        type=int,
        default=300,
        help="ModelScope HTTP 单次读超时时间，单位秒",
    )

    return parser.parse_args()


def print_file_sizes(local_dir: Path, files: list[str]) -> None:
    """打印指定文件大小。"""
    print()
    print("[OK] 下载完成。各文件大小：")

    if len(files) == 1 and files[0] == "*":
        total_size = 0
        file_count = 0

        for p in local_dir.rglob("*"):
            if p.is_file():
                total_size += p.stat().st_size
                file_count += 1

        print(f"  total files : {file_count}")
        print(f"  total size  : {total_size / (1024 ** 3):.2f} GB")
        return

    for filename in files:
        p = local_dir / filename
        if p.exists():
            gb = p.stat().st_size / (1024 ** 3)
            print(f"  {filename} : {gb:.2f} GB")
        else:
            print(f"  {filename} : <未找到>")


def download_once(
    local_dir: Path,
    files: list[str],
    max_workers: int,
) -> None:
    """执行一次 snapshot_download。"""
    from modelscope import snapshot_download

    # files=["*"] 表示下载整库，此时不传 allow_file_pattern
    if len(files) == 1 and files[0] == "*":
        print("[INFO] allow_file_pattern: <None, download full repo>")
        kwargs = {}
    else:
        print(f"[INFO] allow_file_pattern: {files}")
        kwargs = {
            "allow_file_pattern": files,
        }

    # 部分旧版 modelscope 可能不支持 max_workers，这里做兼容
    try:
        snapshot_download(
            model_id=MODEL_ID,
            local_dir=str(local_dir),
            max_workers=max_workers,
            **kwargs,
        )
    except TypeError as e:
        if "max_workers" not in str(e):
            raise

        print("[WARN] 当前 modelscope 版本可能不支持 max_workers，尝试不传 max_workers 重新下载。")
        snapshot_download(
            model_id=MODEL_ID,
            local_dir=str(local_dir),
            **kwargs,
        )


def main():
    args = parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    files = args.files

    os.environ["MODELSCOPE_HTTP_TIMEOUT"] = str(args.http_timeout)

    print("============================================================")
    print(f"[INFO] Model ID : {MODEL_ID}")
    print(f"[INFO] Local dir: {local_dir}")
    print(f"[INFO] Files    : {', '.join(files)}")
    print(f"[INFO] Workers  : {args.max_workers}")
    print(f"[INFO] Timeout  : {args.http_timeout} s")
    print(f"[INFO] MaxRetry : {'infinite' if args.max_retry == 0 else args.max_retry}")
    print("============================================================")

    local_dir.mkdir(parents=True, exist_ok=True)

    ensure_modelscope_installed()

    attempt = 0

    while True:
        attempt += 1

        print()
        print(f"===== Attempt {attempt} =====")

        try:
            download_once(
                local_dir=local_dir,
                files=files,
                max_workers=args.max_workers,
            )

            print_file_sizes(local_dir, files)
            return

        except KeyboardInterrupt:
            print()
            print("[WARN] 用户手动中断。已下载部分保留，下次重新运行会继续复用。")
            raise

        except Exception as e:
            print()
            print(f"[WARN] 第 {attempt} 次下载失败：{repr(e)}")

            if args.max_retry > 0 and attempt >= args.max_retry:
                print(f"[ERROR] 重试 {args.max_retry} 次后仍失败。")
                sys.exit(1)

            print(f"[WARN] {args.sleep_seconds} 秒后重试，已下载部分会续传/复用...")
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()