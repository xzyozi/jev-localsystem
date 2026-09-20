"""JEV (Judge & Evaluation via Vectorized-logits) Local System"""

from jev.dto import (
    ChoiceDetails,
    JudgeRequestDTO,
    JudgeResponseDTO,
    JudgeStatus,
    MultiLabelDetails,
    NoulDetails,
    ScoreDetails,
    TaskType,
)
from jev.exceptions import (
    BackendConnectionError,
    InconclusiveVerdictError,
    JevError,
    PayloadTooLargeError,
    QueueTimeoutError,
    TokenNotFoundError,
)

from jev.prompt_builder import PromptBuilder
from jev.vram_manager import VRAMManager, default_vram_manager
from jev.zero_decode import ZeroDecodeClient

__version__ = "0.1.0"

__all__ = [
    "TaskType",
    "JudgeStatus",
    "JudgeRequestDTO",
    "JudgeResponseDTO",
    "NoulDetails",
    "ChoiceDetails",
    "ScoreDetails",
    "MultiLabelDetails",
    "JevError",
    "PayloadTooLargeError",
    "QueueTimeoutError",
    "InconclusiveVerdictError",
    "TokenNotFoundError",
    "BackendConnectionError",
    "VRAMManager",
    "default_vram_manager",
    "PromptBuilder",
    "ZeroDecodeClient",
]
