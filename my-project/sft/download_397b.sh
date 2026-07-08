#!/usr/bin/env bash
set -uo pipefail

MODEL_ID="Qwen/Qwen3.5-397B-A17B"
LOCAL_DIR="${1:-/iflytek/jmli27/pretrain_models/Qwen3.5-397B-A17B}"
# 无限重试直到下载成功；如需限制次数，可设置 MAX_RETRY（0 或空表示不限）。
MAX_RETRY="${MAX_RETRY:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
# 多进程并发下载的线程数（modelscope snapshot_download 的 max_workers）。
# 注意: 397B 是 94 个 ~8.5GB 的大分片，并发过高会把带宽切碎，
#       导致单片 60s 读超时(ReadTimeout)反复失败。大文件场景 4~8 最稳，不要开太大。
MAX_WORKERS="${MAX_WORKERS:-1}"

# 单次 HTTP 读超时(秒)。默认 60 对大分片偏紧，网络抖动时调大更稳。
export MODELSCOPE_HTTP_TIMEOUT="${MODELSCOPE_HTTP_TIMEOUT:-120}"

mkdir -p "${LOCAL_DIR}"

# 不建议每次循环里都 pip install，避免下载过程中环境变化
python -m pip install -U modelscope

echo "============================================================"
echo "[INFO] Model ID : ${MODEL_ID}"
echo "[INFO] Local dir: ${LOCAL_DIR}"
if [ "${MAX_RETRY}" -gt 0 ] 2>/dev/null; then
  echo "[INFO] Retry    : ${MAX_RETRY}"
else
  echo "[INFO] Retry    : 无限重试直到成功"
fi
echo "[INFO] Workers  : ${MAX_WORKERS}"
echo "============================================================"

i=0
while true; do
  i=$((i + 1))
  echo
  echo "============================================================"
  echo "[INFO] Attempt ${i}"
  echo "[INFO] Downloading ${MODEL_ID} -> ${LOCAL_DIR}"
  echo "============================================================"

  python - <<PY
from modelscope import snapshot_download

model_id = "${MODEL_ID}"
local_dir = "${LOCAL_DIR}"
max_workers = int("${MAX_WORKERS}")

print(f"Downloading {model_id} -> {local_dir} (max_workers={max_workers})")

snapshot_download(
    model_id=model_id,
    local_dir=local_dir,
    max_workers=max_workers,  # 多进程/多线程并发下载分片，加速大模型下载
    ignore_file_pattern=[
        r".*\\.md$",
        r".*\\.onnx$",
        r".*\\.gguf$",
        r".*\\.msgpack$",
        r".*\\.h5$",
    ],
    allow_file_pattern=[
        "model.safetensors-00049-of-00094.safetensors",
    ],
)

print("Download finished.")
PY

  status=$?

  if [ "${status}" -eq 0 ]; then
    echo
    echo "============================================================"
    echo "[INFO] Download completed successfully."
    echo "============================================================"
    du -sh "${LOCAL_DIR}" || true
    exit 0
  fi

  echo
  echo "[WARN] Download failed with exit code ${status}."
  echo "[WARN] Current local dir size:"
  du -sh "${LOCAL_DIR}" || true

  # 只有在确认没有别人在下的时候，才建议删 lock。
  # 这里默认不自动删，避免影响其他人的下载进程。

  # 达到 MAX_RETRY 次数上限才退出（MAX_RETRY<=0 表示无限重试）。
  if [ "${MAX_RETRY}" -gt 0 ] 2>/dev/null && [ "${i}" -ge "${MAX_RETRY}" ]; then
    echo
    echo "[ERROR] Download failed after ${MAX_RETRY} attempts."
    du -sh "${LOCAL_DIR}" || true
    exit 1
  fi

  echo "[WARN] Retry after ${SLEEP_SECONDS} seconds..."
  sleep "${SLEEP_SECONDS}"
done

