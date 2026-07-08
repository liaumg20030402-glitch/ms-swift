#!/usr/bin/env bash
set -uo pipefail

# 在【能联网的外网】上，从 ModelScope 拉取 Qwen3.5-397B-A17B 每个文件的官方 SHA256，
# 导出成清单文件（格式与 sha256sum / verify_397b_offline.sh 兼容）。
# 把生成的清单随权重一起带进内网，再用 HASH_ALGO=sha256 + HASH_MANIFEST 做离线比对。
#
# 用法:
#   bash export_sha256_manifest.sh [OUT_MANIFEST]
# 例:
#   bash export_sha256_manifest.sh ./Qwen3.5-397B-A17B.sha256

MODEL_ID="Qwen/Qwen3.5-397B-A17B"
OUT="${1:-/iflytek/jmli27/pretrain_models/Qwen3.5-397B-A17B/Qwen3.5-397B-A17B.sha256}"

echo "============================================================"
echo "[INFO] Model ID : ${MODEL_ID}"
echo "[INFO] Output   : ${OUT}"
echo "============================================================"

MODEL_ID="${MODEL_ID}" OUT="${OUT}" python - <<'PY'
import os, re, sys

model_id = os.environ["MODEL_ID"]
out = os.environ["OUT"]

# 与下载脚本一致的忽略规则（这些文件本来就没下，不必校验）
ignore_patterns = [r".*\.md$", r".*\.onnx$", r".*\.gguf$", r".*\.msgpack$", r".*\.h5$"]
def ignored(path):
    return any(re.match(p, path) for p in ignore_patterns)

from modelscope import HubApi
api = HubApi()
files = api.get_model_files(model_id=model_id, recursive=True)

rows = []
no_hash = []
for f in files:
    if f.get("Type") != "blob":
        continue
    path = f["Path"]
    if ignored(path):
        continue
    sha = (f.get("Sha256") or "").strip().lower()
    if not sha:
        no_hash.append(path)
        continue
    # 两个空格分隔，basename 作为文件名，兼容 sha256sum -c 与离线校验脚本
    rows.append(f"{sha}  {os.path.basename(path)}")

if not rows:
    print("[FAIL] 未从 API 取到任何 SHA256，请检查 modelscope 版本/网络。")
    sys.exit(1)

rows.sort(key=lambda x: x.split('  ', 1)[1])
parent = os.path.dirname(out)
if parent:
    os.makedirs(parent, exist_ok=True)   # 自动创建父目录，避免 FileNotFoundError
with open(out, "w") as fp:
    fp.write("\n".join(rows) + "\n")

print(f"[OK] 已写入 {len(rows)} 条 SHA256 -> {out}")
if no_hash:
    print(f"[WARN] {len(no_hash)} 个文件 API 未返回 SHA256（通常是小文件，可忽略）：")
    for p in no_hash:
        print(f"  - {p}")
print("\n下一步：把该清单带进内网，执行")
print(f"  HASH_ALGO=sha256 HASH_MANIFEST=/内网路径/{os.path.basename(out)} \\")
print("    bash verify_397b_offline.sh /内网/权重目录")
PY
