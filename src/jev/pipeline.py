"""JEV (Judge & Evaluation via Vectorized-logits) パイプライン統合モジュール

詳細設計書 (JEV-DD-001) 第4章の処理シーケンスに基づく。
5大機能ブロック（Task Interface, VRAM Manager, Prompt Builder, Zero-Decode Client, Result Mapper）を
統合したエンドツーエンドの判定エンジンファサード。
"""

import logging
import sys
import time
from typing import List, Optional

from jev.dto import JudgeRequestDTO, JudgeResponseDTO
from jev.exceptions import JevError, PayloadTooLargeError, QueueTimeoutError
from jev.prompt_builder import PromptBuilder
from jev.result_mapper import ResultMapper
from jev.vram_manager import VRAMManager, default_vram_manager
from jev.zero_decode import ZeroDecodeClient

logger = logging.getLogger("jev.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class JudgePipeline:
    """JEV 推論判定パイプライン公開クラス (JEV-DD-001 第4章)"""

    DEFAULT_PRIMARY_MODEL = "qwen3:8b"
    DEFAULT_LIGHTWEIGHT_MODEL = "phi4-mini:latest"

    def __init__(
        self,
        default_model: str = DEFAULT_PRIMARY_MODEL,
        lightweight_model: str = DEFAULT_LIGHTWEIGHT_MODEL,
        client: Optional[ZeroDecodeClient] = None,
        vram_manager: Optional[VRAMManager] = None,
    ):
        """
        Args:
            default_model: 本番主軸モデル (Tier 1 Primary: qwen3:8b)
            lightweight_model: 本番軽量モデル (Tier 2 Lightweight: phi4-mini:latest)
            client: Zero-Decode クライアント (省略時はデフォルト)
            vram_manager: VRAMリソースマネージャー (省略時はシングルトン)
        """
        self.default_model = default_model
        self.lightweight_model = lightweight_model
        self.client = client or ZeroDecodeClient()
        self.vram_manager = vram_manager or default_vram_manager

    def judge(self, request: JudgeRequestDTO) -> JudgeResponseDTO:
        """型安全な JudgeRequestDTO を受け取り判定を実行する (メインエントリーポイント)

        Args:
            request: 判定リクエストDTO

        Returns:
            JudgeResponseDTO: 型安全な判定結果DTO
        """
        # 1. コンテキスト長リミッター契約の事前検証 (JEV-DD-001 2.4)
        self.vram_manager.validate_payload_limits(request.context_text)

        # モデルの決定
        model_name = request.model or self.default_model

        t_start = time.perf_counter()

        # 2. VRAM Manager による直列実行ロックの獲得 (JEV-DD-001 2.4, Issue #8)
        try:
            with self.vram_manager.acquire():
                if request.task_type == "noul":
                    return self._execute_noul(request, model_name, t_start)
                elif request.task_type == "choice":
                    return self._execute_choice(request, model_name, t_start)
                elif request.task_type == "score":
                    return self._execute_score(request, model_name, t_start)
                elif request.task_type == "multilabel":
                    return self._execute_multilabel(request, model_name, t_start)
                else:
                    raise ValueError(f"Unknown task_type: {request.task_type}")

        except (PayloadTooLargeError, QueueTimeoutError):
            raise
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.exception("Pipeline execution failed: %s", e)
            return JudgeResponseDTO(
                task_type=request.task_type,
                status="ERROR",
                verdict=None,
                latency_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    def _execute_noul(self, request: JudgeRequestDTO, model_name: str, t_start: float) -> JudgeResponseDTO:
        messages, _, _ = PromptBuilder.build_messages(request, model_name)
        logprobs, _ = self.client.forward(model_name, messages)
        total_latency = (time.perf_counter() - t_start) * 1000.0
        return ResultMapper.map_noul(logprobs, total_latency)

    def _execute_choice(self, request: JudgeRequestDTO, model_name: str, t_start: float) -> JudgeResponseDTO:
        # 1. 通常 Forward
        messages_fwd, _, symbol_map_fwd = PromptBuilder.build_messages(request, model_name, swap=False)
        logprobs_fwd, _ = self.client.forward(model_name, messages_fwd)

        logprobs_swap = None
        symbol_map_swap = None

        # 2. A/Bスワップ検証 (Issue #4 仕様)
        if request.swap_verify and len(request.labels) >= 2:
            messages_swap, _, symbol_map_swap = PromptBuilder.build_messages(request, model_name, swap=True)
            logprobs_swap, _ = self.client.forward(model_name, messages_swap)

        total_latency = (time.perf_counter() - t_start) * 1000.0
        return ResultMapper.map_choice(
            logprobs_forward=logprobs_fwd,
            symbol_map_forward=symbol_map_fwd,
            latency_ms=total_latency,
            logprobs_swap=logprobs_swap,
            symbol_map_swap=symbol_map_swap,
        )

    def _execute_score(self, request: JudgeRequestDTO, model_name: str, t_start: float) -> JudgeResponseDTO:
        messages, _, _ = PromptBuilder.build_messages(request, model_name)
        logprobs, _ = self.client.forward(model_name, messages)
        total_latency = (time.perf_counter() - t_start) * 1000.0
        return ResultMapper.map_score(logprobs, total_latency, temperature=request.temperature)

    def _execute_multilabel(self, request: JudgeRequestDTO, model_name: str, t_start: float) -> JudgeResponseDTO:
        labels = request.labels
        if not labels:
            raise ValueError("Multi-Label task requires non-empty request.labels")

        label_results = []
        for lbl in labels:
            messages, _, _ = PromptBuilder.build_messages(request, model_name, single_label_target=lbl)
            logprobs, _ = self.client.forward(model_name, messages)
            label_results.append((lbl, logprobs))

        total_latency = (time.perf_counter() - t_start) * 1000.0
        return ResultMapper.map_multilabel(label_results, total_latency, threshold=0.5)

    # ==========================================
    # ヘルパーショートカットメソッド
    # ==========================================

    def judge_noul(
        self,
        context_text: str,
        rule_definition: str = "",
        model: Optional[str] = None,
    ) -> JudgeResponseDTO:
        """Noul (真偽判定) のショートカットメソッド"""
        req = JudgeRequestDTO(
            task_type="noul",
            context_text=context_text,
            rule_definition=rule_definition,
            model=model,
        )
        return self.judge(req)

    def judge_choice(
        self,
        context_text: str,
        labels: List[str],
        rule_definition: str = "",
        swap_verify: bool = False,
        model: Optional[str] = None,
    ) -> JudgeResponseDTO:
        """Choice (単一選択) のショートカットメソッド"""
        req = JudgeRequestDTO(
            task_type="choice",
            context_text=context_text,
            labels=labels,
            rule_definition=rule_definition,
            swap_verify=swap_verify,
            model=model,
        )
        return self.judge(req)

    def judge_score(
        self,
        context_text: str,
        rule_definition: str = "",
        temperature: float = 1.0,
        model: Optional[str] = None,
    ) -> JudgeResponseDTO:
        """Score (段階評価) のショートカットメソッド"""
        req = JudgeRequestDTO(
            task_type="score",
            context_text=context_text,
            rule_definition=rule_definition,
            temperature=temperature,
            model=model,
        )
        return self.judge(req)

    def judge_multilabel(
        self,
        context_text: str,
        labels: List[str],
        rule_definition: str = "",
        model: Optional[str] = None,
    ) -> JudgeResponseDTO:
        """Multi-Label (複数選択) のショートカットメソッド"""
        req = JudgeRequestDTO(
            task_type="multilabel",
            context_text=context_text,
            labels=labels,
            rule_definition=rule_definition,
            model=model,
        )
        return self.judge(req)
