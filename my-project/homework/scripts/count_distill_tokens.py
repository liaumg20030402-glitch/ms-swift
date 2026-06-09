"""
统计自蒸馏数据（或任意 messages 格式 jsonl）中 assistant 回复的 token 长度分布。

只需要 tokenizer（不加载模型权重、不需要 GPU），CPU 上几秒跑完。

用法：
  python count_distill_tokens.py \
      --model /home/lijinmei/pretrain_models/Qwen3.5-4B \
      --file  /home/lijinmei/swift_lora/data/swift_format/train_distill.jsonl

  # 想同时对比原始数据：
  python count_distill_tokens.py --file train_distill.jsonl --compare train.jsonl
"""

import argparse
import json
import statistics

from transformers import AutoTokenizer


def assistant_token_lengths(path, tokenizer):
    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for m in rec["messages"]:
                if m["role"] == "assistant":
                    ids = tokenizer(m["content"], add_special_tokens=False)["input_ids"]
                    lengths.append(len(ids))
    return lengths


def report(name, lengths):
    if not lengths:
        print(f"[{name}] 无 assistant 内容")
        return
    lengths_sorted = sorted(lengths)
    p = lambda q: lengths_sorted[min(len(lengths) - 1, int(q * len(lengths)))]
    print(f"== {name} ==")
    print(f"  样本数 : {len(lengths)}")
    print(f"  平均   : {statistics.mean(lengths):.1f} tok")
    print(f"  中位数 : {statistics.median(lengths):.0f} tok")
    print(f"  P90/P99: {p(0.90)} / {p(0.99)} tok")
    print(f"  最小/最大: {min(lengths)} / {max(lengths)} tok")
    print(f"  >=2000 tok 占比: {100*sum(1 for x in lengths if x>=2000)/len(lengths):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/lijinmei/pretrain_models/Qwen3.5-4B",
                    help="只用其 tokenizer（不加载权重）")
    ap.add_argument("--file", default="/home/lijinmei/swift_lora/data/swift_format/train_distill.jsonl")
    ap.add_argument("--compare", default=None, help="可选：另一个 jsonl 一起对比（如原始 train.jsonl）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    report("蒸馏数据" if "distill" in args.file else args.file,
           assistant_token_lengths(args.file, tok))
    if args.compare:
        print()
        report("对比数据 " + args.compare, assistant_token_lengths(args.compare, tok))


if __name__ == "__main__":
    main()
