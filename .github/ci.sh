#!/usr/bin/env bash
# CI エントリーポイント (.github/workflows/ci.yml の契約: .github/ci.sh)
# ローカルでも `bash .github/ci.sh` で同じ検証を再現できる。
set -euo pipefail

UV_VERSION="0.10.8"

# ubuntu-latest のシステム Python は PEP 668 により pip install が拒否されるため、
# 一時 venv に uv を固定バージョンで導入する。
tool_venv="$(mktemp -d)/uv-venv"
python3 -m venv "$tool_venv"
"$tool_venv/bin/pip" install --quiet "uv==${UV_VERSION}"
export PATH="$tool_venv/bin:$PATH"

echo "=== uv sync (uv.lock どおりに導入) ==="
uv sync --frozen --extra dev

echo "=== ruff ==="
uv run ruff check src/

echo "=== mypy ==="
uv run mypy src

# 実機 Ollama を必要とするテストは CI では実行しない。
# (tests/test_model_pipeline.py 全件、test_api_server.py の判定系、
#  test_api_evaluate.py の判定系、test_core_pipeline.py の e2e)
# pytest マーカー整備後は `pytest -m "not ollama"` に置き換える。
echo "=== pytest (Ollama 不要のテストのみ) ==="
uv run pytest -p no:cacheprovider \
  tests/test_target_token_mass.py \
  tests/test_request_params.py \
  tests/test_cloud_detection.py \
  tests/test_api_request_params.py \
  tests/test_cloud_backend.py \
  tests/test_cloud_pipeline.py \
  tests/test_core_pipeline.py \
  tests/test_api_server.py::test_api_health \
  tests/test_api_server.py::test_api_vram_metrics \
  tests/test_api_server.py::test_api_models \
  tests/test_api_server.py::test_api_payload_too_large_413 \
  tests/test_api_server.py::test_api_validation_error_422 \
  tests/test_api_evaluate.py::test_evaluate_batch_validation_error \
  -k "not e2e"
