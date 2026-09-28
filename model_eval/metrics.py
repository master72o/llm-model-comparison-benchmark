"""
Metrics Aggregator and Pareto Efficiency Calculator.
"""

import numpy as np
from typing import List, Dict, Any
from model_eval.schema import (
    TaskEvaluationResult,
    ModelName,
    TaskDomain,
)


class ModelMetricsCalculator:
    """Aggregates model performance, latency, cost, and domain breakdowns."""

    @staticmethod
    def calculate_model_summary(evaluations: List[TaskEvaluationResult]) -> Dict[str, Any]:
        if not evaluations:
            return {"total_tasks": 0, "overall_score": 0.0}

        total = len(evaluations)
        scores = [e.overall_score for e in evaluations]
        accuracy = [e.accuracy_score for e in evaluations]
        format_comp = [e.format_compliance for e in evaluations]
        reasoning = [e.reasoning_score for e in evaluations]
        latencies = [e.latency_ms for e in evaluations]
        costs = [e.cost_usd for e in evaluations]

        total_cost = sum(costs)
        cost_per_10k = (total_cost / max(total, 1)) * 10_000

        # Domain breakdown
        domain_metrics = {}
        for dom in TaskDomain:
            dom_evals = [e for e in evaluations if e.domain == dom]
            if dom_evals:
                domain_metrics[dom.value] = round(float(np.mean([e.overall_score for e in dom_evals])), 4)

        overall_mean = round(float(np.mean(scores)), 4)
        quality_efficiency = round(overall_mean / max(cost_per_10k, 0.01), 4)

        return {
            "total_tasks": total,
            "overall_mean_score": overall_mean,
            "mean_accuracy": round(float(np.mean(accuracy)), 4),
            "format_compliance_rate": round(float(np.mean(format_comp)), 4),
            "mean_reasoning_score": round(float(np.mean(reasoning)), 4),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "total_cost_usd": round(total_cost, 6),
            "cost_per_10k_tasks": round(cost_per_10k, 2),
            "quality_efficiency_ratio": quality_efficiency,
            "domain_scores": domain_metrics,
        }

    @classmethod
    def calculate_suite_metrics(
        cls, all_evaluations: Dict[ModelName, List[TaskEvaluationResult]]
    ) -> Dict[str, Any]:
        suite_metrics = {}
        for model_name, evals in all_evaluations.items():
            key = model_name.value if isinstance(model_name, ModelName) else str(model_name)
            suite_metrics[key] = cls.calculate_model_summary(evals)
        return suite_metrics


MetricsCalculator = ModelMetricsCalculator
