"""
Tests for Experiment Runner and Metrics in LLM Model Comparison Benchmark.
"""

from model_eval.schema import BenchmarkTask, TaskDomain, ModelName, ComparisonDataset
from model_eval.experiment_runner import ExperimentRunner
from model_eval.metrics import MetricsCalculator


def test_experiment_runner():
    task = BenchmarkTask(
        task_id="b1",
        domain=TaskDomain.REASONING,
        prompt="Math task",
        reference_answer="Answer is 5.",
        require_json=False,
    )
    res_dict = ExperimentRunner.evaluate_task_across_models(task)
    assert ModelName.GPT_4O in res_dict
    assert ModelName.CLAUDE_3_5_SONNET in res_dict
    assert res_dict[ModelName.GPT_4O].overall_score > 0.0


def test_metrics_calculator(tmp_path):
    jsonl = '{"task_id":"b1","domain":"Reasoning","prompt":"p","reference_answer":"a","require_json":false}\n'
    f_path = tmp_path / "dataset.jsonl"
    f_path.write_text(jsonl)

    dataset = ComparisonDataset.from_jsonl(str(f_path))
    all_evals = {m: [] for m in ModelName}
    for t in dataset.tasks:
        res = ExperimentRunner.evaluate_task_across_models(t)
        for m, r in res.items():
            all_evals[m].append(r)

    metrics = MetricsCalculator.calculate_suite_metrics(all_evals)
    assert "gpt-4o" in metrics
    assert metrics["gpt-4o"]["total_tasks"] == 1
