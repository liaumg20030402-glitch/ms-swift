#!/usr/bin/env bash
set -uo pipefail

# 离线校验 Qwen3.5-397B-A17B 权重目录是否完整（无需联网）。
# 原理：safetensors 文件头记录了每个张量的字节区间，可据此算出"应有文件大小"，
#       与实际大小比对即可抓出 缺失分片 / 没下完的截断文件 / 张量缺失。
#
# 可选的 SHA256 字节级校验（对官方哈希，抓 size 检查发现不了的静默损坏）：
#   1) 外网联网处用 export_sha256_manifest.sh 导出官方 SHA256 清单；
#   2) 把清单随权重带进内网，这里用 HASH_MANIFEST 指向它做比对：
#        HASH_MANIFEST=/path/Qwen3.5-397B-A17B.sha256 bash verify_397b_offline.sh DIR
#   注意：SHA256 要完整读一遍盘（~800GB），较慢；不提供清单时默认跳过。
#   清单格式与 coreutils 的 `sha256sum` 兼容（"<hash>  <文件名>"），可互通。
#
# 用法: bash verify_397b_offline.sh [LOCAL_DIR]

LOCAL_DIR="${1:-/iflytek/jmli27/pretrain_models/Qwen3.5-397B-A17B}"
HASH_MANIFEST="${HASH_MANIFEST:-/iflytek/jmli27/Qwen3.5-397B-A17B.sha256}"   # 官方 SHA256 清单路径；为空则不做哈希校验

echo "============================================================"
echo "[INFO] Local dir: ${LOCAL_DIR}"
echo "[INFO] SHA256 manifest: ${HASH_MANIFEST:-<未提供，跳过哈希校验>}"
echo "============================================================"

LOCAL_DIR="${LOCAL_DIR}" HASH_MANIFEST="${HASH_MANIFEST}" python - <<'PY'
import os, json, struct, glob, sys, hashlib

local_dir = os.environ["LOCAL_DIR"]
manifest_path = os.environ.get("HASH_MANIFEST", "").strip()

def read_header(path):
    """返回 (header_dict, header_len)；文件损坏则抛异常。"""
    with open(path, "rb") as f:
        nbytes = f.read(8)
        if len(nbytes) != 8:
            raise ValueError("文件过小，连 8 字节头长度都不够")
        n = struct.unpack("<Q", nbytes)[0]
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError(f"头部被截断：声明 {n} 字节，实际只读到 {len(raw)}")
        return json.loads(raw), n

def expected_size(header, header_len):
    """根据头部计算文件应有的总字节数。"""
    max_end = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        _, end = v["data_offsets"]
        max_end = max(max_end, end)
    return 8 + header_len + max_end

def sha256_of(path, chunk=64 * 1024 * 1024):
    """流式计算文件 SHA256，避免一次性读入 8GB。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

problems = []   # (文件名, 原因)
checked = 0

# ---------- 1) 结构与大小校验（基于 index.json） ----------
index_path = os.path.join(local_dir, "model.safetensors.index.json")
shard_files = set()

if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_files = set(weight_map.values())
    print(f"[INFO] index.json 引用了 {len(shard_files)} 个分片，{len(weight_map)} 个张量")

    tensors_by_shard = {}
    for tensor, shard in weight_map.items():
        tensors_by_shard.setdefault(shard, set()).add(tensor)

    for shard in sorted(shard_files):
        p = os.path.join(local_dir, shard)
        if not os.path.exists(p):
            problems.append((shard, "缺失：index 引用但本地不存在"))
            continue
        try:
            header, hlen = read_header(p)
            keys = {k for k in header if k != "__metadata__"}
            missing_t = tensors_by_shard[shard] - keys
            if missing_t:
                problems.append((shard, f"缺少 {len(missing_t)} 个张量（文件不完整/错误）"))
                continue
            exp = expected_size(header, hlen)
            act = os.path.getsize(p)
            if act != exp:
                problems.append((shard, f"大小不符：实际 {act} != 应有 {exp}（多半没下完）"))
                continue
            checked += 1
        except Exception as e:
            problems.append((shard, f"头部解析失败：{e}"))
else:
    print("[WARN] 未找到 model.safetensors.index.json（单分片模型或缺失）。")
    print("[WARN] 退化为扫描目录下所有 .safetensors 逐个做完整性校验。")
    shard_files = {os.path.basename(p) for p in glob.glob(os.path.join(local_dir, "*.safetensors"))}
    if not shard_files:
        print("[FAIL] 目录下没有任何 .safetensors 文件。")
        sys.exit(1)
    for shard in sorted(shard_files):
        p = os.path.join(local_dir, shard)
        try:
            header, hlen = read_header(p)
            exp = expected_size(header, hlen)
            act = os.path.getsize(p)
            if act != exp:
                problems.append((shard, f"大小不符：实际 {act} != 应有 {exp}（多半没下完）"))
                continue
            checked += 1
        except Exception as e:
            problems.append((shard, f"头部解析失败：{e}"))

# 必备的非权重文件
for must in ["config.json", "tokenizer_config.json"]:
    if not os.path.exists(os.path.join(local_dir, must)):
        problems.append((must, "缺失：必备配置/分词器文件"))

# ---------- 2) 官方 SHA256 比对（提供了清单时） ----------
hash_problems = []
if manifest_path:
    if not os.path.exists(manifest_path):
        print(f"[FAIL] 指定的 SHA256 清单不存在: {manifest_path}")
        sys.exit(1)
    expected = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 兼容 "<hash>  <name>" 和 "<hash> *<name>"
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts[0], parts[1].lstrip("*").strip()
            expected[os.path.basename(name)] = digest.lower()
    print(f"\n[INFO] 清单含 {len(expected)} 个官方 SHA256，开始逐个比对（需完整读盘，较慢）...")
    targets = sorted(shard_files)
    for i, shard in enumerate(targets, 1):
        p = os.path.join(local_dir, shard)
        if not os.path.exists(p):
            continue  # 缺失已在上面记过
        exp_digest = expected.get(shard)
        if exp_digest is None:
            print(f"  [{i}/{len(targets)}] {shard}: 清单中无此文件，跳过")
            continue
        act_digest = sha256_of(p)
        if act_digest != exp_digest:
            hash_problems.append((shard, f"SHA256 不一致：本地 {act_digest} != 官方 {exp_digest}"))
            print(f"  [{i}/{len(targets)}] {shard}: ✗ 不一致")
        else:
            print(f"  [{i}/{len(targets)}] {shard}: ✓")
else:
    print("\n[INFO] 未提供 HASH_MANIFEST，跳过 SHA256 校验（仅做了结构/大小校验）。")
    print("       如需对官方逐字节校验：外网用 export_sha256_manifest.sh 导出清单，带进来再比对。")

# ---------- 3) 汇总 ----------
all_problems = problems + hash_problems
print(f"\n结构/大小校验通过的分片: {checked} / {len(shard_files)}")
if all_problems:
    print(f"发现 {len(all_problems)} 个问题：")
    for name, why in all_problems:
        print(f"  - {name}: {why}")
    print("\n[FAIL] 目录不完整或存在损坏。请重新下载/续传/重传相关文件。")
    sys.exit(1)
else:
    msg = "结构、大小、张量齐全"
    if manifest_path:
        msg += "，且官方 SHA256 全部一致"
    print(f"\n[OK] {msg}，可以安全加载。")
    sys.exit(0)
PY
