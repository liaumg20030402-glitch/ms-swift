"""
将 hendrycks_math 数据集（parquet）转换为 ms-swift 支持的 messages 格式 JSONL。

输出三个文件到 --output-dir：
  - train.jsonl : 所有子集的 train split（已扣除 val），含 assistant（用于 LoRA 训练）
  - val.jsonl   : 从 train 切出的小验证集，含 assistant（训练时监控 eval_loss，与 test 零重叠）
  - test.jsonl  : 所有子集的 test split，含 assistant（用于训练后评测准确率，保留 ground-truth）

每条样本格式（ms-swift messages）：
  {"messages": [
     {"role": "system",    "content": <SYSTEM_PROMPT>},
     {"role": "user",      "content": "Problem: ..."},
     {"role": "assistant", "content": <solution，含 \\boxed{答案}>}
  ]}

用法示例：
  python prepare_data.py \
      --data-dir ../lora/data/hendrycks_math \
      --output-dir ../lora/data/swift_format \
      --val-size 500
"""

import argparse
import glob
import json
import os
import random

import pandas as pd
from tqdm import tqdm

# 与老师 notebook 保持一致的 system prompt
SYSTEM_PROMPT = (
    "You are a mathematical problem-solving assistant. Please provide the final "
    "answer to the following math problem. Your final answer should be enclosed "
    "in \\boxed{}."
)

ALL_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def extract_boxed_answer(text: str):
    r"""提取最后一个 \boxed{...} 的内容，做括号配平（支持嵌套，如 \boxed{\dfrac{1}{33}}）。
    与 evaluate.py 保持一致，避免清洗与评测口径不同。"""
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None  # 括号不配平


def load_split(data_dir: str, subject: str, split: str) -> pd.DataFrame:
    """读取某个子集某个 split 的 parquet 文件。"""
    pattern = os.path.join(data_dir, subject, f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"找不到 parquet: {pattern}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def to_message(problem: str, solution: str, with_assistant: bool = True) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}"},
    ]
    if with_assistant:
        messages.append({"role": "assistant", "content": solution})
    return {"messages": messages}


def write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="hendrycks_math -> ms-swift messages JSONL")
    parser.add_argument("--data-dir", default="/home/lijinmei/swift_lora/data/hendrycks_math", type=str,
                        help="hendrycks_math 根目录（包含各子集文件夹）")
    parser.add_argument("--output-dir", default="/home/lijinmei/swift_lora/data/swift_format", type=str,
                        help="输出目录")
    parser.add_argument("--subjects", type=str, nargs="+", default=ALL_SUBJECTS,
                        help=f"使用哪些子集，默认全部: {ALL_SUBJECTS}")
    parser.add_argument("--val-size", type=int, default=0,
                        help="从 train 切出多少条作为训练期验证集（默认 500，且不超过 train 的 10%）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    train_records, test_records = [], []
    n_drop_train, n_drop_test = 0, 0

    for subject in tqdm(args.subjects, desc="子集"):
        df_train = load_split(args.data_dir, subject, "train")
        df_test = load_split(args.data_dir, subject, "test")

        for _, row in tqdm(df_train.iterrows(), total=len(df_train),
                           desc=f"[{subject}] train", leave=False):
            # 数据清洗：剔除解答中无 \boxed{} 最终答案的样本
            if extract_boxed_answer(row["solution"]) is None:
                n_drop_train += 1
                continue
            train_records.append(to_message(row["problem"], row["solution"], with_assistant=True))

        for _, row in tqdm(df_test.iterrows(), total=len(df_test),
                           desc=f"[{subject}] test", leave=False):
            # test 同样剔除：无 \boxed{} 则无法提取 ground-truth，会干扰准确率统计
            if extract_boxed_answer(row["solution"]) is None:
                n_drop_test += 1
                continue
            test_records.append(to_message(row["problem"], row["solution"], with_assistant=True))

    random.shuffle(train_records)

    # 从 train 中切出一小部分做训练期验证集（与 test 零重叠，方法学更干净）
    n_val = min(args.val_size, len(train_records) // 10)  # 不超过 train 的 10%
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(val_records, os.path.join(args.output_dir, "val.jsonl"))
    write_jsonl(test_records, os.path.join(args.output_dir, "test.jsonl"))

    print("-" * 50)
    print(f"train.jsonl : {len(train_records)} 条 (已扣除 val)")
    print(f"val.jsonl   : {len(val_records)} 条 (从 train 切出，与 test 无重叠)")
    print(f"test.jsonl  : {len(test_records)} 条 (仅用于最终准确率评测)")
    print(f"已剔除无 \\boxed{{}} 样本: train {n_drop_train} 条, test {n_drop_test} 条")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
