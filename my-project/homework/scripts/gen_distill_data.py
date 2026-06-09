"""
自蒸馏数据生成（rejection sampling / STaR 思路）。

动机：MATH 原始 solution 是人写的简洁解答，在它上面 SFT 会让强基座变简短、丢失推理能力。
改用「base 模型自己生成的、答对的、详细的解答」当训练数据，强化模型已有的正确推理。

流程：
  对每道 train 题目，用 base 模型采样 n 次（温度采样，保证多样性）
  -> 提取 \boxed 答案，与 gold 比对（复用 evaluate.py 的 answers_match）
  -> 只保留答对的生成，写成 messages 格式的新训练集 train_distill.jsonl

支持多卡数据并行（torchrun），与 evaluate.py 一致。

用法：
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node 5 gen_distill_data.py \
      --train-file /home/lijinmei/swift_lora/data/swift_format/train.jsonl \
      --output-file /home/lijinmei/swift_lora/data/swift_format/train_distill.jsonl \
      --n-gen 4 --keep 1

  之后把 sft_lora.sh 的 TRAIN_FILE 指向 train_distill.jsonl 重训，与原数据做对比实验。
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from tqdm import tqdm

# 复用 evaluate.py 中已实现并验证过的组件
from evaluate import (
    answers_match,
    build_prompt,
    extract_boxed_answer,
    load_model,
    load_test,
    setup_distributed,
)


@torch.no_grad()
def generate_distill(model, tokenizer, items, batch_size, n_gen, max_new_tokens,
                     temperature, top_p, device, keep, out_f, disable_tqdm=False):
    """对每道题采样 n_gen 次，把答对的生成（最多 keep 条/题）边生成边写入 out_f。
    返回 (n_solved, n_kept)。
    """
    pad_id = tokenizer.pad_token_id
    n_solved, n_kept = 0, 0
    pbar = tqdm(range(0, len(items), batch_size), desc="Distill", disable=disable_tqdm)
    for start in pbar:
        batch = items[start:start + batch_size]
        prompts = [build_prompt(tokenizer, pm, enable_thinking=False) for pm, _ in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n_gen,
            pad_token_id=pad_id,
        )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        # texts 长度 = len(batch) * n_gen，按「每个 prompt 的 n_gen 条」分组排列
        for bi, (pm, gold) in enumerate(batch):
            cands = texts[bi * n_gen:(bi + 1) * n_gen]
            n_this = 0
            for gen_text in cands:
                pred = extract_boxed_answer(gen_text)
                if answers_match(gold, pred):
                    # pm 已含 system+user，追加 assistant（base 自己的正确长解答）
                    messages = list(pm) + [{"role": "assistant", "content": gen_text.strip()}]
                    out_f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    n_this += 1
                    n_kept += 1
                    if n_this >= keep:
                        break
            if n_this > 0:
                n_solved += 1
        out_f.flush()  # 边生成边落盘，进程中断也保留已生成结果
        pbar.set_postfix(solved=n_solved, kept=n_kept)
    return n_solved, n_kept


def parse_args():
    p = argparse.ArgumentParser(description="自蒸馏训练数据生成（rejection sampling）")
    p.add_argument("--model", default="/home/lijinmei/pretrain_models/Qwen3.5-4B", help="底模路径")
    p.add_argument("--train-file", default="/home/lijinmei/swift_lora/data/swift_format/train.jsonl",
                   help="原始 train.jsonl（提供题目与 gold 答案）")
    p.add_argument("--output-file", default="/home/lijinmei/swift_lora/data/swift_format/train_distill.jsonl",
                   help="输出的自蒸馏训练集")
    p.add_argument("--num-samples", type=int, default=-1, help="只处理前 N 道题（调试用），-1 表示全部")
    p.add_argument("--n-gen", type=int, default=4, help="每题采样次数（越多解出率越高、越慢）")
    p.add_argument("--keep", type=int, default=1, help="每题最多保留几条答对的生成（>1 为数据增广）")
    p.add_argument("--batch-size", type=int, default=8, help="每批题数（实际序列数 = batch_size × n_gen）")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.8, help="采样温度，0.7~1.0 兼顾多样性与质量")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, device, is_dist, cpu_group = setup_distributed(args.device)
    is_main = (rank == 0)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 复用 load_test：返回 [(prompt_messages, gold), ...]（train.jsonl 与 test 同格式）
    items = load_test(args.train_file, args.num_samples, args.seed)
    shard = items[rank::world_size] if is_dist else items
    if is_main:
        gpu_info = f"{world_size} 卡并行" if is_dist else "单卡"
        print(f"题目数: {len(items)}  {gpu_info}  每题采样: {args.n_gen}  保留/题: {args.keep}  "
              f"采样: T={args.temperature}, top_p={args.top_p}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    # 多卡时各 rank 先写自己的分片文件（边生成边落盘），最后由 rank0 合并
    part_path = f"{args.output_file}.part{rank}" if is_dist else args.output_file

    model = load_model(args.model, "base", device)
    with open(part_path, "w", encoding="utf-8") as out_f:
        n_solved_local, n_kept_local = generate_distill(
            model, tokenizer, shard,
            args.batch_size, args.n_gen, args.max_new_tokens,
            args.temperature, args.top_p, device, args.keep, out_f,
            disable_tqdm=not is_main,
        )
    del model
    torch.cuda.empty_cache()

    # 汇总各卡计数；rank0 合并分片文件
    if is_dist:
        counts = [None] * world_size
        dist.all_gather_object(counts, (n_solved_local, n_kept_local), group=cpu_group)
        dist.barrier()  # 确保各 rank 都写完并关闭分片文件
        n_solved = sum(c[0] for c in counts)
        n_kept = sum(c[1] for c in counts)
        if is_main:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                for r in range(world_size):
                    pp = f"{args.output_file}.part{r}"
                    with open(pp, "r", encoding="utf-8") as fin:
                        fout.write(fin.read())
                    os.remove(pp)
    else:
        n_solved, n_kept = n_solved_local, n_kept_local

    if is_main:
        total = len(items)
        print("-" * 60)
        print(f"题目总数      : {total}")
        print(f"至少解出 1 次 : {n_solved}  (解出率 {100*n_solved/total:.1f}%)")
        print(f"保留训练样本  : {n_kept} 条")
        print(f"已写入        : {args.output_file}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
