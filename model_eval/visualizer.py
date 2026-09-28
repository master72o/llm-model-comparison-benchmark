"""
Visualization Generator for Multi-Model Benchmark.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from model_eval.schema import ModelName, TaskDomain


class Visualizer:
    """Generates comparison charts and Pareto efficiency plots for models."""

    @staticmethod
    def generate_all_figures(
        suite_metrics: Dict[str, Any], output_dir: str = "reports/figures"
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # 1. Overall Performance Bar Chart
        ov_path = os.path.join(output_dir, "model_overall_performance.png")
        Visualizer._plot_overall_performance(suite_metrics, ov_path)
        generated["model_overall_performance"] = ov_path

        # 2. Pareto Cost vs Quality Scatter
        pareto_path = os.path.join(output_dir, "pareto_cost_vs_quality.png")
        Visualizer._plot_pareto_frontier(suite_metrics, pareto_path)
        generated["pareto_cost_vs_quality"] = pareto_path

        # 3. Domain Performance Grouped Bar Chart
        domain_path = os.path.join(output_dir, "domain_performance_radar.png")
        Visualizer._plot_domain_performance(suite_metrics, domain_path)
        generated["domain_performance_radar"] = domain_path

        return generated

    @staticmethod
    def _plot_overall_performance(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(9, 5))
        models = [m.value for m in ModelName]
        labels = [m for m in models]

        scores = [metrics.get(m, {}).get("overall_mean_score", 0.0) * 100 for m in models]
        format_rates = [metrics.get(m, {}).get("format_compliance_rate", 0.0) * 100 for m in models]

        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, scores, width, label="Overall Quality Score (%)", color="#2b5c8f")
        rects2 = ax.bar(x + width/2, format_rates, width, label="Format Compliance Rate (%)", color="#5cb85c")

        ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_title("Frontier & Open-Weights Model Benchmark Performance", fontsize=12, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 115)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for rects in [rects1, rects2]:
            for r in rects:
                h = r.get_height()
                ax.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_pareto_frontier(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(9, 6))
        models = [m.value for m in ModelName]

        costs = [metrics.get(m, {}).get("cost_per_10k_tasks", 0.0) for m in models]
        scores = [metrics.get(m, {}).get("overall_mean_score", 0.0) * 100 for m in models]

        colors = ["#2b5c8f", "#d9534f", "#5cb85c", "#f0ad4e", "#5bc0de"]

        for i, (m, c, s) in enumerate(zip(models, costs, scores)):
            ax.scatter(c, s, color=colors[i % len(colors)], s=150, zorder=5, label=m)
            ax.annotate(
                m,
                (c, s),
                textcoords="offset points",
                xytext=(8, 5),
                ha="left",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_xlabel("Cost per 10,000 Tasks ($)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Overall Quality Score (%)", fontsize=11, fontweight="bold")
        ax.set_title("Pareto Efficiency Frontier: Cost vs Overall Model Quality", fontsize=12, fontweight="bold", pad=15)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_domain_performance(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        domains = [d.value for d in TaskDomain]
        models = [m.value for m in ModelName]

        x = np.arange(len(domains))
        width = 0.15

        colors = ["#2b5c8f", "#d9534f", "#5cb85c", "#f0ad4e", "#5bc0de"]

        for i, (m, color) in enumerate(zip(models, colors)):
            d_scores = [
                metrics.get(m, {}).get("domain_scores", {}).get(d, 0.0) * 100
                for d in domains
            ]
            rects = ax.bar(x + (i - 2) * width, d_scores, width, label=m, color=color)

        ax.set_ylabel("Domain Score (%)", fontsize=11, fontweight="bold")
        ax.set_title("Domain Performance Breakdown (Reasoning, JSON, Multilingual, Code)", fontsize=12, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
