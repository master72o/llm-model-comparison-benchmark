"""
Tests for Model Evaluator Engine.
"""

from model_eval.schema import BenchmarkTask, ModelName, TaskDomain
from model_eval.evaluator import ModelEvaluator


def test_evaluate_response_json():
    task = BenchmarkTask(
        task_id="b1",
        domain=TaskDomain.STRUCTURED_OUTPUT,
        prompt="Format json",
        reference_answer="{\"status\": \"ok\"}",
        require_json=True,
    )
    raw = {"output": "{\"status\": \"ok\"}", "latency_ms": 300.0, "input_tokens": 50, "output_tokens": 20}
    res = ModelEvaluator.evaluate_response(task, ModelName.GPT_4O, raw)

    assert res.format_compliance == 1.0
    assert res.accuracy_score == 1.0
    assert res.cost_usd > 0.0


def test_evaluate_response_invalid_json():
    task = BenchmarkTask(
        task_id="b2",
        domain=TaskDomain.STRUCTURED_OUTPUT,
        prompt="Format json",
        reference_answer="{\"status\": \"ok\"}",
        require_json=True,
    )
    raw = {"output": "not json", "latency_ms": 300.0, "input_tokens": 50, "output_tokens": 20}
    res = ModelEvaluator.evaluate_response(task, ModelName.GPT_4O, raw)

    assert res.format_compliance == 0.0
