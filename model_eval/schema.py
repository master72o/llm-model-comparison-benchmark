"""
Data Schemas for LLM Model Comparison Benchmark.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class ModelName(str, Enum):
    GPT_4O = "gpt-4o"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    LLAMA_3_70B = "llama-3-70b"
    MISTRAL_LARGE = "mistral-large"
    GEMINI_1_5_PRO = "gemini-1-5-pro"


class TaskDomain(str, Enum):
    REASONING = "Reasoning"
    STRUCTURED_OUTPUT = "Structured_Output"
    MULTILINGUAL = "Multilingual"
    CODE_GEN = "Code_Gen"


@dataclass
class BenchmarkTask:
    __test__ = False

    task_id: str
    domain: TaskDomain
    prompt: str
    reference_answer: str
    require_json: bool = False
    model_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        d_raw = data.get("domain", "Reasoning")
        try:
            dom = TaskDomain(d_raw)
        except ValueError:
            dom = TaskDomain.REASONING

        return cls(
            task_id=data["task_id"],
            domain=dom,
            prompt=data["prompt"],
            reference_answer=data["reference_answer"],
            require_json=data.get("require_json", False),
            model_outputs=data.get("model_outputs", {}),
        )


@dataclass
class TaskEvaluationResult:
    task_id: str
    model_name: ModelName
    domain: TaskDomain
    accuracy_score: float
    format_compliance: float
    reasoning_score: float
    cost_usd: float
    latency_ms: float
    overall_score: float
    response_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_name": self.model_name.value if isinstance(self.model_name, ModelName) else str(self.model_name),
            "domain": self.domain.value if isinstance(self.domain, TaskDomain) else str(self.domain),
            "accuracy_score": round(self.accuracy_score, 4),
            "format_compliance": round(self.format_compliance, 4),
            "reasoning_score": round(self.reasoning_score, 4),
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
            "overall_score": round(self.overall_score, 4),
            "response_text": self.response_text,
        }


class ComparisonDataset:
    """Benchmark Dataset loader."""

    def __init__(self, tasks: List[BenchmarkTask]):
        self.tasks = tasks

    @classmethod
    def from_jsonl(cls, file_path: str) -> "ComparisonDataset":
        import json
        tasks = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(BenchmarkTask.from_dict(json.loads(line)))
        return cls(tasks=tasks)
