"""JEV モデル別特性（モデルプロファイル）モジュール

モデル名に紐づく知識（思考モデルか、思考抑制用の Prefill 文字列）を 1 か所に集約する。
従来 ``PromptBuilder`` に直書きされていた判定を、データとして外出ししたもの。
判定結果（どのモデル名にどの Prefill を返すか）は従来と同一。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# 思考モデルに空の思考ブロックを事前注入する Prefill 文字列 (Issue #2, #9 実証成果)
PREFILL_THINK_TAG = "<think>\n\n</think>\n"
PREFILL_THOUGHT_TAG = "<thought>\n\n</thought>\n"


@dataclass(frozen=True)
class ModelProfile:
    """モデル名の部分一致で適用されるモデル特性"""

    name: str
    # 小文字化したモデル名にこの文字列のいずれかが含まれれば、このプロファイルに該当する
    match: Tuple[str, ...]
    # 推論思考（CoT）モデルか
    thinking: bool = False
    # 思考抑制用に assistant ロールへ事前注入する文字列（不要なら None）
    prefill: Optional[str] = None

    def matches(self, model_name: str) -> bool:
        lowered = model_name.lower()
        return any(token in lowered for token in self.match)


# 組み込みプロファイル。先頭から順に評価し、最初に該当したものの prefill を採用する。
_BUILTIN_PROFILES: Tuple[ModelProfile, ...] = (
    ModelProfile(name="qwen3", match=("qwen3",), thinking=True, prefill=PREFILL_THINK_TAG),
    ModelProfile(name="deepseek-r1", match=("deepseek-r1",), thinking=True, prefill=PREFILL_THOUGHT_TAG),
    ModelProfile(name="gemma", match=("gemma",), thinking=False, prefill=PREFILL_THOUGHT_TAG),
)

# 利用者が register_model_profile() で追加したプロファイル（組み込みより優先して評価する）
_custom_profiles: List[ModelProfile] = []


def register_model_profile(profile: ModelProfile) -> None:
    """プロファイルを追加する。組み込みより優先され、後から追加したものが先に評価される。"""
    _custom_profiles.insert(0, profile)


def clear_custom_model_profiles() -> None:
    """register_model_profile() で追加したプロファイルをすべて削除する（主にテスト用）。"""
    _custom_profiles.clear()


def _profiles() -> Tuple[ModelProfile, ...]:
    return (*_custom_profiles, *_BUILTIN_PROFILES)


def resolve_prefill(model_name: str) -> Optional[str]:
    """モデル名に該当する最初のプロファイルの Prefill 文字列を返す（該当なし・Prefill なしは None）。"""
    for profile in _profiles():
        if profile.matches(model_name):
            return profile.prefill
    return None


def is_thinking(model_name: str) -> bool:
    """いずれかの該当プロファイルが思考モデルであれば True。"""
    return any(profile.thinking and profile.matches(model_name) for profile in _profiles())
