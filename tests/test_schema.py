"""
Tests for Schema module in LLM Model Comparison Benchmark.
"""

from model_eval.schema import BenchmarkTask, ModelName, TaskDomain


def test_benchmark_task_parsing():
    data = {
        "task_id": "bm_001",
        "domain": "Reasoning",
        "prompt": "Test prompt?",
        "reference_answer": "Test answer.",
        "require_json": True,
        "model_outputs": {},
    }
    task = BenchmarkTask.from_dict(data)
    assert task.task_id == "bm_001"
    assert task.domain == TaskDomain.REASONING
    assert task.require_json is True


def test_model_name_enum():
    assert ModelName.GPT_4O.value == "gpt-4o"
    assert ModelName.CLAUDE_3_5_SONNET.value == "claude-3-5-sonnet"
