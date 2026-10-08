"""クラウド判定のテスト (Issue #20)

VRAM 保護のバイパス可否は、モデル名ではなく実際に使うバックエンドの is_cloud で決まる。
Ollama / FastAPI には依存しない。
"""

from typing import Dict, List, Tuple

import pytest

from jev import JudgePipeline, PayloadTooLargeError, VRAMManager, ZeroDecodeClient
from jev.zero_decode import InferenceBackend, is_cloud_model_name


class _Backend(InferenceBackend):
    def __init__(self, cloud: bool):
        self._cloud = cloud

    @property
    def is_cloud(self) -> bool:
        return self._cloud

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        return {"Yes": -0.05, "No": -3.0}, 1.0


@pytest.mark.parametrize(
    "name",
    ["gpt-4o-mini", "GPT-4o", "o1-mini", "o3-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-pro",
     "text-embedding-3-small"],
)
def test_cloud_api_model_names_are_detected(name: str):
    assert is_cloud_model_name(name) is True


@pytest.mark.parametrize(
    "name",
    ["gpt-oss:20b", "gemini-nano:latest", "qwen3:8b", "phi4-mini:latest", "llama3", "", None],
)
def test_local_or_unknown_model_names_are_not_cloud(name):
    assert is_cloud_model_name(name) is False


def test_local_client_keeps_vram_protection_even_for_cloud_like_model_name():
    """ローカルのバックエンドでは、クラウドらしいモデル名でもリミッターとキューが有効"""
    vm = VRAMManager(hard_char_limit=10)
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=_Backend(cloud=False)), vram_manager=vm)
    with pytest.raises(PayloadTooLargeError):
        pipe.judge_noul("x" * 100, "基準", model="gpt-oss:20b")
    with pytest.raises(PayloadTooLargeError):
        pipe.judge_noul("x" * 100, "基準", model="gpt-4o-mini")


def test_local_client_goes_through_serial_queue():
    vm = VRAMManager()
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=_Backend(cloud=False)), vram_manager=vm)
    pipe.judge_noul("テスト", "基準", model="gpt-oss:20b")
    assert vm.get_metrics()["total_processed"] == 1


def test_cloud_client_bypasses_vram_protection():
    vm = VRAMManager(hard_char_limit=10)
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=_Backend(cloud=True)), vram_manager=vm)
    res = pipe.judge_noul("x" * 100, "基準", model="gpt-4o-mini")
    assert res.status == "SUCCESS"
    assert vm.get_metrics()["total_processed"] == 0
