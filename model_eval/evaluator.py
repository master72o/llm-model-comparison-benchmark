"""
Model Output Evaluator & Cost Calculator.
"""

import re
import json
from typing import Dict, Any
from model_eval.schema import (
    BenchmarkTask,
    ModelName,
    TaskDomain,
    TaskEvaluationResult,
)


class ModelEvaluator:
    """Evaluates task outputs and computes token costs across LLM providers."""

    MODEL_PRICING = {
        ModelName.GPT_4O: {"input_per_1m": 2.50, "output_per_1m": 10.00},
        ModelName.CLAUDE_3_5_SONNET: {"input_per_1m": 3.00, "output_per_1m": 15.00},
        ModelName.LLAMA_3_70B: {"input_per_1m": 0.60, "output_per_1m": 0.80},
        ModelName.MISTRAL_LARGE: {"input_per_1m": 2.00, "output_per_1m": 6.00},
        ModelName.GEMINI_1_5_PRO: {"input_per_1m": 1.25, "output_per_1m": 5.00},
    }

    @staticmethod
    def evaluate_response(
        task: BenchmarkTask,
        model_name: ModelName,
        raw_output_data: Dict[str, Any]
    ) -> TaskEvaluationResult:
        response_text = raw_output_data.get("output", "")
        latency_ms = raw_output_data.get("latency_ms", 500.0)
        in_tok = raw_output_data.get("input_tokens", 150)
        out_tok = raw_output_data.get("output_tokens", 100)

        # 1. Format Compliance
        if task.require_json:
            try:
                json.loads(response_text.strip())
                format_compliance = 1.0
            except Exception:
                format_compliance = 0.0
        else:
            format_compliance = 1.0

        # 2. Accuracy Score against Reference Answer
        ref_words = set(re.findall(r"\w+", task.reference_answer.lower()))
        out_words = set(re.findall(r"\w+", response_text.lower()))
        if not ref_words:
            accuracy_score = 1.0
        else:
            intersection = out_words.intersection(ref_words)
            accuracy_score = len(intersection) / max(len(ref_words), 1)
            accuracy_score = min(1.0, accuracy_score)

        # 3. Reasoning Score
        reasoning_markers = ["therefore", "because", "step", "thus", "conclude", "since", "="]
        has_markers = sum(1 for m in reasoning_markers if m in response_text.lower())
        reasoning_score = min(1.0, 0.4 + (has_markers * 0.15))

        # 4. Token Cost Calculation
        pricing = ModelEvaluator.MODEL_PRICING.get(model_name, {"input_per_1m": 2.0, "output_per_1m": 6.0})
        cost_usd = (in_tok * pricing["input_per_1m"] / 1_000_000.0) + (out_tok * pricing["output_per_1m"] / 1_000_000.0)

        # 5. Composite Score
        overall_score = (accuracy_score * 0.4) + (format_compliance * 0.3) + (reasoning_score * 0.3)

        return TaskEvaluationResult(
            task_id=task.task_id,
            model_name=model_name,
            domain=task.domain,
            accuracy_score=round(accuracy_score, 4),
            format_compliance=round(format_compliance, 4),
            reasoning_score=round(reasoning_score, 4),
            cost_usd=round(cost_usd, 6),
            latency_ms=latency_ms,
            overall_score=round(overall_score, 4),
            response_text=response_text,
        )
