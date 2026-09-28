# LLM Model Comparison Benchmark (`llm-model-comparison-benchmark`)

> Controlled Multi-Model Evaluation Benchmark Comparing Frontier & Open-Weights LLMs across Reasoning, JSON Compliance, Cost & Latency.

[![CI](https://github.com/user/llm-model-comparison-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/user/llm-model-comparison-benchmark/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Summary

Selecting the optimal Large Language Model (LLM) for production requires balancing **Reasoning Accuracy**, **Structured Output Compliance**, **Latency**, and **API Token Costs**.

`llm-model-comparison-benchmark` provides a controlled, empirical benchmarking suite evaluating 5 leading frontier and open-weights models:
1. `gpt-4o` (OpenAI)
2. `claude-3-5-sonnet` (Anthropic)
3. `llama-3-70b` (Meta / Open-Weights)
4. `mistral-large` (Mistral AI)
5. `gemini-1-5-pro` (Google DeepMind)

---

## Architecture & Benchmark Domains

```
                 ┌───────────────────────────────────────────┐
                 │        Multi-Domain Benchmark Suite       │
                 └─────────────────────┬─────────────────────┘
                                       │
         ┌──────────────────┬──────────┴──────────┬──────────────────┐
         ▼                  ▼                     ▼                  ▼
    Reasoning       Structured Output        Multilingual        Code Generation
 (Logic & Math)       (JSON Schema)          (Translation)       (Algorithms)
         │                  │                     │                  │
         └──────────────────┴──────────┬──────────┴──────────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │   Model Evaluator Engine     │
                       └───────────────┬──────────────┘
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
Accuracy Score (%)            JSON Compliance (%)              Pareto Cost / 10k ($)
```

---

## Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/user/llm-model-comparison-benchmark.git
cd llm-model-comparison-benchmark

# Install in editable mode
pip install -e .

# Run pytest suite
pytest
```

---

## Executing Multi-Model Benchmark CLI

Run the benchmark evaluation CLI:

```bash
python -m model_eval.cli run \
  --dataset data/comparison_benchmark_dataset.jsonl \
  --output-dir results \
  --report-path reports/model_comparison_report.md \
  --figures-dir reports/figures
```

---

## Benchmark Comparison Matrix

| Model Identifier | Overall Score | Format Compliance | Reasoning Score | Avg Latency | Cost / 10k Tasks | Quality Efficiency Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gpt-4o`** | `97.0%` | `100.0%` | `90.0%` | `392ms` | `$1.15` | `84.35` |
| **`claude-3-5-sonnet`** | `98.5%` | `100.0%` | `95.0%` | `368ms` | `$1.38` | `71.38` |
| **`llama-3-70b`** | `94.0%` | `100.0%` | `80.0%` | `268ms` | `$0.24` | `391.67` |
| **`mistral-large`** | `91.0%` | `100.0%` | `70.0%` | `398ms` | `$0.72` | `126.39` |
| **`gemini-1-5-pro`** | `94.5%` | `100.0%` | `81.5%` | `448ms` | `$0.55` | `171.82` |

### Key Benchmark Visualizations

- **Overall Model Performance**: `reports/figures/model_overall_performance.png`
- **Pareto Cost vs Quality Frontier**: `reports/figures/pareto_cost_vs_quality.png`
- **Domain Performance Breakdown**: `reports/figures/domain_performance_radar.png`

---

## Repository Structure

```
llm-model-comparison-benchmark/
├── .github/workflows/ci.yml     # Continuous Integration workflow
├── pyproject.toml               # Package build metadata
├── requirements.txt             # Project dependencies
├── model_eval/                  # Core Python package
│   ├── __init__.py
│   ├── schema.py                # Data models & task domains
│   ├── evaluator.py             # Accuracy, JSON, & cost calculator
│   ├── experiment_runner.py    # Multi-model runner
│   ├── metrics.py               # Aggregator & Quality-Efficiency metric
│   ├── visualizer.py            # Matplotlib figure generator
│   ├── report_generator.py      # Markdown report & JSON/CSV exporter
│   └── cli.py                   # CLI entry point
├── data/
│   ├── README.md
│   └── comparison_benchmark_dataset.jsonl # Benchmark tasks
├── results/                     # JSON & CSV exported results
├── reports/
│   ├── model_comparison_report.md # Generated research report
│   └── figures/                 # Chart graphics
└── tests/                       # Unit and integration tests
```

---

## License

MIT License © 2026 AI Evaluation Engineering Team.
