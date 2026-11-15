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

# 指定された Python バージョンがあれば uv に伝播させる
if [ -n "${PYTHON_VERSION:-}" ]; then
  export UV_PYTHON="${PYTHON_VERSION}"
fi

echo "=== Python version ==="
python3 --version
if [ -n "${UV_PYTHON:-}" ]; then
  echo "Target UV_PYTHON: ${UV_PYTHON}"
fi

echo "=== uv lock --check (pyproject.toml と uv.lock の整合) ==="
uv lock --check

echo "=== uv sync (uv.lock どおりに導入) ==="
uv sync --frozen --extra dev

echo "=== ruff ==="
uv run ruff check src/

echo "=== mypy ==="
uv run mypy src

# 実機 Ollama を必要とするテストは @pytest.mark.ollama を付けて CI から除外する。
# 新しいテストは、Ollama が不要ならマーカー無しで自動的に CI の対象になる。
echo "=== pytest (Ollama 不要のテストのみ) ==="
uv run pytest -p no:cacheprovider -m "not ollama"
