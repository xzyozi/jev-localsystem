"""JEV (Judge & Evaluation via Vectorized-logits) Local System"""

from jev.dto import (
    ChoiceDetails,
    EvaluateRequestDTO,
    EvaluateResponseDTO,
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
from jev.model_profiles import ModelProfile, register_model_profile
from jev.pipeline import JudgePipeline
from jev.prompt_builder import PromptBuilder
from jev.prompt_templates import (
    PromptTemplate,
    clear_custom_prompt_templates,
    register_prompt_template,
    resolve_prompt_template,
)
from jev.result_mapper import ResultMapper
from jev.vram_manager import VRAMManager, default_vram_manager
from jev.zero_decode import (
    InferenceBackend,
    OllamaBackend,
    OpenAIBackend,
    ZeroDecodeClient,
)

__version__ = "0.1.0"

__all__ = [
    "TaskType",
    "JudgeStatus",
    "JudgeRequestDTO",
    "JudgeResponseDTO",
    "EvaluateRequestDTO",
    "EvaluateResponseDTO",
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
    "PromptTemplate",
    "register_prompt_template",
    "resolve_prompt_template",
    "clear_custom_prompt_templates",
    "ModelProfile",
    "register_model_profile",
    "InferenceBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "ZeroDecodeClient",
    "ResultMapper",

    "JudgePipeline",
]
