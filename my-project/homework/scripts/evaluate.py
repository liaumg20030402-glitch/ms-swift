"""
在 test.jsonl 上评测 Qwen3.5 LoRA 微调效果，统计数学题准确率。

打分逻辑（与老师 notebook 一致）：
  对每道题生成答案 -> 正则提取最后一个 \\boxed{...} -> 归一化 -> 与 ground-truth 比对。

用法：
  # 4 卡数据并行（推荐，全量 test 提速约 4x）：用 torchrun 启动
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node 5 evaluate.py \
      --adapters base \
                 /home/lijinmei/swift_lora/output/v0-20260608-193444/checkpoint-110 \
                 /home/lijinmei/swift_lora/output/v0-20260608-193444/checkpoint-220 \
                 /home/lijinmei/swift_lora/output/v0-20260608-193444/checkpoint-330
                 

  说明：每张卡处理 1/4 的 test 数据，结果自动汇总到 rank0 打印/保存。
        --model / --test-file / --save-dir 均有默认值，可省略。

依赖：transformers / peft / torch（环境里已装），无需 vllm。
"""

import argparse
import json
import os
import random
import re

import torch
import torch.distributed as dist
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False

SYSTEM_PROMPT = (
    "You are a mathematical problem-solving assistant. Please provide the final "
    "answer to the following math problem. Your final answer should be enclosed "
    "in \\boxed{}."
)


# ---------- 打分工具 ----------
def extract_boxed_answer(text: str):
    r"""提取最后一个 \boxed{...} 的内容，做括号配平（支持嵌套，如 \boxed{\dfrac{1}{33}}）。"""
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


def normalize_answer(answer):
    r"""归一化 LaTeX 答案，消除等价写法差异后再比较。
    处理: \dfrac/\tfrac->\frac、去 \left\right、去空格、去 $、统一小写、去末尾标点。
    """
    if answer is None:
        return None
    s = answer
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for tok in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:",
                r"\quad", r"\qquad", "$", "\\\\", r"\ "):
        s = s.replace(tok, "")
    s = re.sub(r"\s+", "", s)          # 去掉所有空白
    s = s.strip().rstrip(".")          # 去末尾句号
    s = s.lower()
    return s if s else None


def answers_match(gold, pred):
    r"""判断预测答案与标准答案是否一致（gold/pred 为 \boxed 内的原始 latex）。
    先用归一化字符串快速判（处理写法差异），再用 math_verify 判数值/符号等价
    （处理 \frac vs 小数、(x+1)(x-1) vs x^2-1 等）。math_verify 未安装时仅用前者。
    """
    if gold is None or pred is None:
        return False
    if normalize_answer(gold) == normalize_answer(pred):
        return True
    if _HAS_MATH_VERIFY:
        try:
            return bool(mv_verify(mv_parse(gold), mv_parse(pred)))
        except Exception:
            return False
    return False


def content_to_text(content) -> str:
    """messages 的 content 可能是 str 或 list，统一取文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def load_test(path: str, num_samples: int, seed: int = 42):
    """读取 test.jsonl，返回 [(prompt_messages, gold_answer), ...]
    num_samples>0 时用固定种子随机抽取（而非取前 N 条），保证可复现且各 checkpoint 抽到同一子集。
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec["messages"]
            # 最后一条 assistant 是 ground-truth 解答，取出 \boxed 答案；其余作为 prompt
            gold = None
            prompt_messages = []
            for msg in messages:
                if msg["role"] == "assistant":
                    # 保留原始 \boxed latex（不在此归一化），交给 answers_match 判等
                    gold = extract_boxed_answer(content_to_text(msg["content"]))
                else:
                    prompt_messages.append({"role": msg["role"], "content": content_to_text(msg["content"])})
            # 确保有 system（数据里已带，这里兜底）
            if not any(m["role"] == "system" for m in prompt_messages):
                prompt_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            items.append((prompt_messages, gold))
    if num_samples and 0 < num_samples < len(items):
        items = random.Random(seed).sample(items, num_samples)
    return items


def build_prompt(tokenizer, prompt_messages, enable_thinking=False):
    """套用 chat template，末尾加 generation prompt；enable_thinking 控制思考/非思考。"""
    try:
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
    except TypeError:
        # 个别模板不接受 enable_thinking 参数
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def evaluate(model, tokenizer, items, batch_size, max_new_tokens, temperature, top_p, device,
             enable_thinking=False, disable_tqdm=False):
    correct, total = 0, 0
    records = []
    pbar = tqdm(range(0, len(items), batch_size), desc="Evaluating", disable=disable_tqdm)
    for start in pbar:
        batch = items[start:start + batch_size]
        prompts = [build_prompt(tokenizer, pm, enable_thinking) for pm, _ in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)  # greedy，准确率可复现

        out = model.generate(**enc, **gen_kwargs)
        # 只取新生成的部分（左 padding，prompt 长度对齐到 enc 的列数）
        gen_only = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        # 逐条统计：生成 token 数；是否被截断（没生成 eos = 跑满 max_new_tokens）
        pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        gen_lens = (gen_only != pad_id).sum(dim=1).tolist()
        has_eos = (gen_only == eos_id).any(dim=1).tolist()

        for k, ((pm, gold), gen_text) in enumerate(zip(batch, texts)):
            pred = extract_boxed_answer(gen_text)
            is_ok = answers_match(gold, pred)
            correct += int(is_ok)
            total += 1
            records.append({
                "query": pm[-1]["content"] if pm else "",
                "gold": gold, "pred": pred, "correct": is_ok,
                "has_boxed": pred is not None,      # 格式正确性：是否输出了 \boxed{}
                "gen_tokens": gen_lens[k],          # 生成长度（token）
                "truncated": not has_eos[k],        # 是否被截断（未生成结束符）
                "generation": gen_text,
            })
        # 进度条实时显示滚动准确率
        pbar.set_postfix(acc=f"{correct/total*100:.2f}%", correct=f"{correct}/{total}")
    acc = correct / total * 100 if total else 0.0
    return acc, correct, total, records


def load_model(base_path, adapter, device):
    # Qwen3.5 是多模态模型，必须用 Qwen3_5ForConditionalGeneration 加载（与 swift 训练时一致），
    # 否则层路径为 model.layers.* 而非 model.language_model.*，LoRA adapter 的 target 匹配不上。
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    if adapter and adapter.lower() != "base":
        model = PeftModel.from_pretrained(model, adapter)
    return model.to(device).eval()


def setup_distributed(default_device):
    """若用 torchrun 启动则初始化数据并行，否则单卡。
    返回 (rank, world_size, device, is_dist, cpu_group)。
    cpu_group 是 gloo 组，专用于在 CPU 上汇总结果，避免占用 GPU 显存。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        cpu_group = dist.new_group(backend="gloo")  # 对象汇总走 CPU，不碰显存
        return rank, world_size, f"cuda:{local_rank}", True, cpu_group
    return 0, 1, default_device, False, None


def gather_records(records_local, world_size, group=None):
    """把各 rank 的逐条结果汇总到所有 rank（返回拼接后的完整列表）。
    通过 gloo group 在 CPU 上汇总，避免 nccl 把序列化数据塞进已占满的 GPU。
    """
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, records_local, group=group)
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5 LoRA 数学准确率评测")
    p.add_argument("--model", default="/home/lijinmei/pretrain_models/Qwen3.5-4B", help="底模路径（本地）")
    p.add_argument("--adapters", nargs="+", default=["base"],
                   help="一个或多个 LoRA checkpoint 路径；用 'base' 表示纯底模 baseline")
    p.add_argument("--test-file", default="/home/lijinmei/swift_lora/data/swift_format/test.jsonl", help="test.jsonl 路径")
    p.add_argument("--num-samples", type=int, default=-1,
                   help="随机抽 N 条评测（固定种子，可复现），-1 表示全量")
    p.add_argument("--seed", type=int, default=42, help="随机抽样种子")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="最大生成长度；太小会在写到 \\boxed 前被截断导致判错")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0=greedy（推荐，准确率可复现）；设 0.7 可对齐老师 notebook")
    p.add_argument("--top-p", type=float, default=1.0, help="仅在 temperature>0 时生效")
    p.add_argument("--thinking", action="store_true",
                   help="开启思考模式（默认非思考，与训练一致且更快）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-dir", default="/home/lijinmei/swift_lora/infer_output", help="可选：保存每个 checkpoint 的逐条预测 jsonl")
    return p.parse_args()


def main():
    args = parse_args()

    # 数据并行：用 torchrun 启动则每张卡跑一个 shard
    rank, world_size, device, is_dist, cpu_group = setup_distributed(args.device)
    is_main = (rank == 0)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = load_test(args.test_file, args.num_samples, args.seed)
    mode = "思考" if args.thinking else "非思考"
    # 思考模式会生成长推理链，token 数不够会截断导致拿不到 \boxed，自动抬高下限
    max_new_tokens = args.max_new_tokens
    if args.thinking and max_new_tokens < 4096:
        max_new_tokens = 4096
        if is_main:
            print(f"[提示] 思考模式已将 max_new_tokens 抬高到 {max_new_tokens}")
    # 每个 rank 取自己的 shard（strided 切分，负载均衡）
    shard = items[rank::world_size] if is_dist else items
    if is_main:
        gpu_info = f"{world_size} 卡数据并行" if is_dist else "单卡"
        judge = "math_verify(数值等价)+字符串" if _HAS_MATH_VERIFY else "仅归一化字符串"
        print(f"评测样本数: {len(items)}  {gpu_info}  模式: {mode}  "
              f"解码: {'greedy' if args.temperature<=0 else f'sample(T={args.temperature})'}  "
              f"判等: {judge}")

    summary = []
    for adapter in args.adapters:
        tag = "baseline(底模)" if adapter.lower() == "base" else adapter
        if is_main:
            print(f"\n{'='*60}\n加载: {tag}\n{'='*60}")
        model = load_model(args.model, adapter, device)
        _, _, _, records_local = evaluate(
            model, tokenizer, shard,
            args.batch_size, max_new_tokens, args.temperature, args.top_p, device,
            enable_thinking=args.thinking, disable_tqdm=not is_main,
        )

        # 先释放模型显存，再做汇总（否则汇总缓冲区会在已占满的 GPU 上 OOM）
        del model
        torch.cuda.empty_cache()

        # 汇总各卡结果（gloo/CPU，不占显存）
        records = gather_records(records_local, world_size, cpu_group) if is_dist else records_local
        total = len(records)
        correct = sum(r["correct"] for r in records)
        acc = correct / total * 100 if total else 0.0
        boxed_rate = 100 * sum(r["has_boxed"] for r in records) / total if total else 0.0
        trunc_rate = 100 * sum(r["truncated"] for r in records) / total if total else 0.0
        avg_len = sum(r["gen_tokens"] for r in records) / total if total else 0.0
        # 在「有 boxed」的样本里，错误率（用于区分「格式对但算错」）
        boxed_n = sum(r["has_boxed"] for r in records)
        boxed_wrong = sum(1 for r in records if r["has_boxed"] and not r["correct"])
        boxed_wrong_rate = 100 * boxed_wrong / boxed_n if boxed_n else 0.0

        if is_main:
            print(f"准确率: {acc:.2f}% ({correct}/{total})  格式率(含boxed): {boxed_rate:.1f}%  "
                  f"boxed但答错: {boxed_wrong_rate:.1f}%  截断率: {trunc_rate:.1f}%  平均长度: {avg_len:.0f} tok")
            summary.append((tag, acc, boxed_rate, boxed_wrong_rate, trunc_rate, avg_len))

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                name = "baseline" if adapter.lower() == "base" else os.path.basename(adapter.rstrip("/"))
                out_path = os.path.join(args.save_dir, f"pred_{name}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"逐条预测已保存: {out_path}")

    if is_main:
        print(f"\n{'='*92}\n汇总对比 (样本数={len(items)})\n{'='*92}")
        print(f"{'checkpoint':<34}{'准确率':>9}{'格式率':>9}{'boxed错':>9}{'截断率':>9}{'平均长度':>10}")
        for tag, acc, boxed_rate, boxed_wrong_rate, trunc_rate, avg_len in summary:
            short = tag if len(tag) <= 32 else "..." + tag[-29:]
            print(f"{short:<34}{acc:>8.2f}%{boxed_rate:>8.1f}%{boxed_wrong_rate:>8.1f}%"
                  f"{trunc_rate:>8.1f}%{avg_len:>10.0f}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
