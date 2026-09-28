"""
CLI Interface for LLM Model Comparison Benchmark.
"""

import argparse
import sys
from model_eval.schema import ComparisonDataset, ModelName
from model_eval.experiment_runner import ExperimentRunner
from model_eval.metrics import MetricsCalculator
from model_eval.visualizer import Visualizer
from model_eval.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="LLM Model Comparison Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run multi-model comparison evaluation benchmark")
    run_parser.add_argument("--dataset", required=True, help="Path to JSONL benchmark dataset")
    run_parser.add_argument("--output-dir", default="results", help="Directory to save JSON/CSV outputs")
    run_parser.add_argument("--report-path", default="reports/model_comparison_report.md", help="Path to save Markdown report")
    run_parser.add_argument("--figures-dir", default="reports/figures", help="Directory to save figure plots")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Loading benchmark dataset from: {args.dataset}")
        dataset = ComparisonDataset.from_jsonl(args.dataset)
        print(f"Loaded {len(dataset.tasks)} benchmark tasks.")

        print("Evaluating responses across models (gpt-4o, claude-3-5-sonnet, llama-3-70b, mistral-large, gemini-1-5-pro)...")
        all_evaluations = {m: [] for m in ModelName}

        for task in dataset.tasks:
            res_dict = ExperimentRunner.evaluate_task_across_models(task)
            for m_name, res in res_dict.items():
                all_evaluations[m_name].append(res)

        metrics = MetricsCalculator.calculate_suite_metrics(all_evaluations)
        print("\nModel Performance & Cost Summary:")
        for m_name, m in metrics.items():
            print(f"  {m_name:18s}: Overall Score={m['overall_mean_score']*100:.1f}%, Format Compliance={m['format_compliance_rate']*100:.1f}%, Cost/10k Tasks=${m['cost_per_10k_tasks']:.2f}")

        print(f"\nExporting results to: {args.output_dir}")
        ReportGenerator.export_results(all_evaluations, args.output_dir)

        print(f"Generating visualizations in: {args.figures_dir}")
        Visualizer.generate_all_figures(metrics, args.figures_dir)

        print(f"Generating research report at: {args.report_path}")
        ReportGenerator.generate_markdown_report(metrics, args.report_path)

        print("\nMulti-model comparison benchmark completed successfully!")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
